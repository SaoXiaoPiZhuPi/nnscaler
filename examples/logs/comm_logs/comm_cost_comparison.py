"""
通信开销预测值与实际值对比统计工具
用于比较通信原语的预测时间和实际测量时间
"""

import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


def _piecewise_estimator(xs: List[float], ys: List[float], x: float) -> float:
    """
    Piecewise linear estimator for communication cost prediction.

    Args:
        xs: x coordinates of the points (sizes in MB).
        ys: y coordinates of the points (times in seconds).
        x: x coordinate of the query point (size in MB).

    Returns:
        y coordinate of the query point (time in seconds).
    """
    if x <= xs[0]:
        return ys[0]
    
    # For very large message sizes (>512MB), use linear approximation
    if x >= xs[-1]:
        assert xs[-1] > 0 and ys[-1] > 0, f'Unexpected val x={x}, xs={xs}, ys={ys}'
        if xs[-1] < 512:
            print(f'Warning: Estimation may be inaccurate for x={x} MB, xs={xs[-1]} MB, ys={ys[-1]} s')
        return x / xs[-1] * ys[-1]
    
    for i in range(len(xs) - 1):
        if xs[i] <= x < xs[i + 1]:
            return ys[i] + (x - xs[i]) * (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])
    
    raise RuntimeError(f'x={x}, xs={xs}, ys={ys}, should not reach here')


def load_comm_logs(log_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all communication log files from the directory.
    
    Args:
        log_dir: Directory containing comm_logs_rank*.json files.
    
    Returns:
        List of all communication records.
    """
    all_records = []
    
    for log_file in sorted(log_dir.glob("comm_logs_rank*.json")):
        with open(log_file, 'r') as f:
            records = json.load(f)
            all_records.extend(records)
    
    return all_records


def load_comm_info(comm_dir: Path) -> Dict[str, Dict[str, Tuple[List[float], List[float]]]]:
    """
    Load all communication profiling data from the directory.
    
    Args:
        comm_dir: Directory containing communication profiling JSON files.
    
    Returns:
        Dictionary mapping device_setting to primitive to (sizes, times) tuples.
        Example: {
            'intra_8.json': {
                'all_reduce': ([1, 2, 4, ...], [0.001, 0.002, 0.004, ...]),
                ...
            },
            ...
        }
    """
    comm_info = {}
    
    for comm_file in comm_dir.glob("*.json"):
        with open(comm_file, 'r') as f:
            data = json.load(f)
            comm_info[comm_file.name] = data
    
    return comm_info


def infer_dp_comm_mesh(ranks: List[int], gpus_per_node: int = 8) -> Tuple[int, int]:
    """
    Infer the DP communication mesh (num_nodes, num_gpus_per_node) from ranks.
    
    Args:
        ranks: List of rank numbers involved in the communication.
        gpus_per_node: Number of GPUs per node (default: 8).
    
    Returns:
        Tuple of (num_nodes, num_gpus_per_node).
    
    Examples:
        [0, 1] -> (1, 2)
        [0, 1, 2, 3] -> (1, 4)
        [0, 8] -> (2, 1)
        [0, 1, 8, 9] -> (2, 2)
    """
    if not ranks:
        return (1, 1)
    
    # Calculate node IDs by dividing rank by gpus_per_node
    node_ids = [rank // gpus_per_node for rank in ranks]
    num_nodes = len(set(node_ids))
    
    # Count ranks per node (should be the same for all nodes)
    node_to_ranks = defaultdict(list)
    for rank in ranks:
        node_id = rank // gpus_per_node
        node_to_ranks[node_id].append(rank)
    
    # Get the number of ranks per node (assume all nodes have the same number)
    gpus_per_node_actual = len(list(node_to_ranks.values())[0]) if node_to_ranks else 1
    
    return (num_nodes, gpus_per_node_actual)


def primitive_to_cost(comm_info: Dict, ranks: List[int], byte_size: int, 
                      primitive: str, gpus_per_node: int = 8) -> float:
    """
    Predict communication cost based on comm_info and ranks.
    
    Args:
        comm_info: Dictionary containing communication profiling data.
        ranks: List of rank numbers involved in the communication.
        byte_size: Size of the tensor in bytes.
        primitive: Type of communication primitive (e.g., 'all_reduce', 'all_gather').
        gpus_per_node: Number of GPUs per node (default: 8).
    
    Returns:
        Predicted time in seconds.
    """
    if byte_size == 0 or not ranks:
        return 0.0
    
    size_mb = byte_size / 1024 / 1024
    if primitive == "all gather" or primitive == "all to all":
        size_mb = size_mb * 2
    dp_comm_mesh = infer_dp_comm_mesh(ranks, gpus_per_node)
    (nnodes, gpus_per_node_actual) = dp_comm_mesh
    
    if nnodes == 1:
        device_setting = f'intra_{len(ranks)}.json'
    elif nnodes > 1:
        device_setting = f'inter_{dp_comm_mesh}.json'
    else:
        raise ValueError(f'Invalid dp_comm_mesh: {dp_comm_mesh} provided.')
    
    if device_setting not in comm_info:
        print(f"Warning: device_setting '{device_setting}' not found in comm_info for ranks {ranks}")
        return 0.0
    
    if primitive not in comm_info[device_setting]:
        print(f"Warning: primitive '{primitive}' not found in comm_info['{device_setting}']")
        return 0.0
    
    sizes_in_mb, times_in_s = comm_info[device_setting][primitive]
    est_time = _piecewise_estimator(sizes_in_mb, times_in_s, size_mb)

    # if primitive == "all gather" or primitive == "all to all":
        # return est_time * 2
    
    assert est_time >= 0, f'{primitive} ranks {ranks} comm size: {size_mb} MB, est time: {est_time} s'
    return est_time


def compute_statistics(predicted: List[float], actual: List[float]) -> Dict[str, float]:
    """
    Compute statistical metrics comparing predicted vs actual values.
    
    Args:
        predicted: List of predicted values.
        actual: List of actual values.
    
    Returns:
        Dictionary containing various statistics.
    """
    predicted = np.array(predicted)
    actual = np.array(actual)
    
    errors = predicted - actual
    absolute_errors = np.abs(errors)
    relative_errors = np.abs(errors / (actual + 1e-10))  # Avoid division by zero
    
    stats = {
        'num_samples': len(predicted),
        'mae': float(np.mean(absolute_errors)),  # Mean Absolute Error
        'mape': float(np.mean(relative_errors) * 100),  # Mean Absolute Percentage Error
        'rmse': float(np.sqrt(np.mean(errors ** 2))),  # Root Mean Square Error
        'max_error': float(np.max(absolute_errors)),
        'min_error': float(np.min(absolute_errors)),
        'mean_predicted': float(np.mean(predicted)),
        'mean_actual': float(np.mean(actual)),
        'std_predicted': float(np.std(predicted)),
        'std_actual': float(np.std(actual)),
        'r_squared': float(np.corrcoef(predicted, actual)[0, 1] ** 2) if len(predicted) > 1 else 0.0,
    }
    
    return stats


def analyze_by_primitive(records: List[Dict[str, Any]], comm_info: Dict, 
                         gpus_per_node: int = 8) -> Dict[str, Any]:
    """
    Analyze prediction accuracy by communication primitive.
    
    Args:
        records: List of communication records.
        comm_info: Dictionary containing communication profiling data.
        gpus_per_node: Number of GPUs per node (default: 8).
    
    Returns:
        Dictionary containing analysis results for each primitive.
    """
    primitive_stats = {}
    
    # Group records by primitive type
    primitive_groups = defaultdict(list)
    for record in records:
        primitive_groups[record['op_type']].append(record)
    
    for primitive, group in primitive_groups.items():
        predicted_times = []
        actual_times = []
        errors = []
        filtered_count = 0
        
        for record in group:
            # Skip records where all ranks are the same (e.g., all 0, all 4, etc.)
            if len(set(record['ranks'])) == 1:
                filtered_count += 1
                continue
            
            # Convert actual latency from ms to seconds
            actual_time = record['latency_ms'] / 1000.0
            
            # Predict time using the record's ranks
            predicted_time = primitive_to_cost(
                comm_info, 
                record['ranks'], 
                record['tensor_size_bytes'], 
                primitive,
                gpus_per_node
            )
            
            predicted_times.append(predicted_time)
            actual_times.append(actual_time)
            errors.append(predicted_time - actual_time)
        
        if filtered_count > 0:
            print(f"Filtered {filtered_count} records with all ranks identical from '{primitive}'")
        
        if not predicted_times:
            print(f"Warning: No valid records for primitive '{primitive}' after filtering")
            continue
        
        stats = compute_statistics(predicted_times, actual_times)
        stats['num_samples'] = len(predicted_times)
        stats['primitive'] = primitive
        
        primitive_stats[primitive] = stats
    
    return primitive_stats


def print_statistics(stats: Dict[str, float], title: str = "Statistics"):
    """
    Print statistics in a formatted way.
    
    Args:
        stats: Dictionary containing statistics.
        title: Title for the statistics block.
    """
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    print(f"Number of samples: {stats.get('num_samples', 0)}")
    print(f"Mean Absolute Error (MAE): {stats.get('mae', 0):.6f} s")
    print(f"Mean Absolute Percentage Error (MAPE): {stats.get('mape', 0):.2f}%")
    print(f"Root Mean Square Error (RMSE): {stats.get('rmse', 0):.6f} s")
    print(f"Max Error: {stats.get('max_error', 0):.6f} s")
    print(f"Min Error: {stats.get('min_error', 0):.6f} s")
    print(f"Mean Predicted: {stats.get('mean_predicted', 0):.6f} s")
    print(f"Mean Actual: {stats.get('mean_actual', 0):.6f} s")
    print(f"Std Predicted: {stats.get('std_predicted', 0):.6f} s")
    print(f"Std Actual: {stats.get('std_actual', 0):.6f} s")
    print(f"R-squared: {stats.get('r_squared', 0):.4f}")
    print(f"{'='*80}\n")


def generate_cdf_plots(records: List[Dict[str, Any]], comm_info: Dict, 
                       output_dir: Path, gpus_per_node: int = 8):
    """
    Generate CDF plots for each communication primitive.
    
    Args:
        records: List of communication records.
        comm_info: Dictionary containing communication profiling data.
        output_dir: Directory to save the CDF plots.
        gpus_per_node: Number of GPUs per node (default: 8).
    """
    # Group records by primitive type
    primitive_groups = defaultdict(list)
    for record in records:
        primitive_groups[record['op_type']].append(record)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate CDF plot for each primitive
    for primitive, group in primitive_groups.items():
        # Calculate error percentages
        error_percentages = []
        filtered_count = 0
        message_sizes = {}  # Collect message sizes
        
        for record in group:
            # Skip records where all ranks are the same
            if len(set(record['ranks'])) == 1:
                filtered_count += 1
                continue
            
            # Collect message size
            size_mb = record['tensor_size_bytes'] / 1024 / 1024
            if size_mb not in message_sizes:
                message_sizes[size_mb] = 0
            message_sizes[size_mb] += 1
            
            # Convert actual latency from ms to seconds
            actual_time = record['latency_ms'] / 1000.0
            
            # Predict time using the record's ranks
            predicted_time = primitive_to_cost(
                comm_info, 
                record['ranks'], 
                record['tensor_size_bytes'], 
                primitive,
                gpus_per_node
            )
            
            # Calculate error percentage
            if actual_time > 0:
                error_pct = abs((predicted_time - actual_time) / actual_time) * 100
                error_percentages.append(error_pct)
        
        if filtered_count > 0:
            print(f"Filtered {filtered_count} records with all ranks identical from '{primitive}' (CDF plot)")
        
        # Print message size statistics
        unique_sizes = sorted(message_sizes.keys())
        if unique_sizes:
            print(f"\nMessage sizes for '{primitive}':")
            print(f"  Total unique sizes: {len(unique_sizes)}")
            print(f"  Size range: {min(unique_sizes):.4f} MB - {max(unique_sizes):.4f} MB")
            print(f"  Unique sizes (MB): {', '.join([f'{s:.4f} ({message_sizes[s]} times)' for s in unique_sizes])}")
        
        if not error_percentages:
            print(f"Warning: No valid error data for primitive '{primitive}'")
            continue
        
        # Sort error percentages for CDF
        sorted_errors = np.sort(error_percentages)
        cdf_values = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        
        # Calculate mean error percentage
        mean_error = np.mean(error_percentages)
        
        # Create figure
        plt.figure(figsize=(3, 2))
        plt.plot(sorted_errors, cdf_values, linewidth=2, color='steelblue')
        plt.xlabel('Error Percentage (%)', fontsize=20)
        plt.ylabel('CDF', fontsize=20)
        # plt.title(f'CDF of Prediction Error for {primitive}', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        # plt.xlim(0, max(100, sorted_errors[-1] * 1.1))
        # plt.xlim(0, sorted_errors[-1])
        plt.xlim(0, 100)
        plt.ylim(0, 1.02)
        plt.xticks(fontsize=16)
        plt.yticks([0, 0.25, 0.5, 0.75, 1.0], fontsize=16)
        
        # Add mean error line
        mean_cdf_value = np.sum(sorted_errors <= mean_error) / len(sorted_errors)
        # plt.axvline(x=mean_error, color='green', linestyle='-', alpha=0.7, linewidth=2)
        plt.text(35, 0.5, f'Mean: {mean_error:.1f}%', 
                 fontsize=15, color='green', fontweight='bold')
        
        # Add percentile markers
        percentiles = [90, 99]
        for p in percentiles:
            idx = int(p / 100 * len(sorted_errors))
            if idx < len(sorted_errors):
                plt.axvline(x=sorted_errors[idx], color='red', linestyle='--', alpha=0.5)
                plt.text(sorted_errors[idx]+0.7, 0.02, f'P{p}', rotation=90, 
                         verticalalignment='bottom', fontsize=12, color='red')
        
        # Save figure
        primitive_safe_name = primitive.replace('_', '-').replace(' ', '-')
        output_path = output_dir / f'cdf_{primitive_safe_name}.pdf'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print statistics
        p50 = np.percentile(sorted_errors, 50)
        p90 = np.percentile(sorted_errors, 90)
        p95 = np.percentile(sorted_errors, 95)
        p99 = np.percentile(sorted_errors, 99)
        mean_error = np.mean(error_percentages)
        
        print(f"\nCDF generated for '{primitive}':")
        print(f"  Samples: {len(error_percentages)}")
        print(f"  Mean error: {mean_error:.2f}%")
        print(f"  50th percentile (median): {p50:.2f}%")
        print(f"  90th percentile: {p90:.2f}%")
        print(f"  95th percentile: {p95:.2f}%")
        print(f"  99th percentile: {p99:.2f}%")
        print(f"  Saved to: {output_path}")
    
    print(f"\n{'='*80}")
    print(f"All CDF plots saved to: {output_dir}")
    print(f"{'='*80}\n")

def plot_error_by_size(records: List[Dict[str, Any]], comm_info: Dict, 
                      output_dir: Path, gpus_per_node: int = 8):
    """
    Generate plots showing prediction error by message size for each communication primitive.
    
    This function creates three types of visualizations:
    1. Scatter plot with error bars showing absolute error vs message size
    2. Bar chart showing mean absolute error for size bins
    3. Line plot showing relative error percentage vs message size
    
    Args:
        records: List of communication records.
        comm_info: Dictionary containing communication profiling data.
        output_dir: Directory to save the error plots.
        gpus_per_node: Number of GPUs per node (default: 8).
    """
    # Group records by primitive type
    primitive_groups = defaultdict(list)
    for record in records:
        primitive_groups[record['op_type']].append(record)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots for each primitive
    for primitive, group in primitive_groups.items():
        # Collect data points
        sizes_mb = []
        predicted_times = []
        actual_times = []
        absolute_errors = []
        relative_errors = []
        filtered_count = 0
        
        for record in group:
            # Skip records where all ranks are the same
            if len(set(record['ranks'])) == 1:
                filtered_count += 1
                continue
            
            size_mb = record['tensor_size_bytes'] / 1024 / 1024
            actual_time = record['latency_ms'] / 1000.0
            predicted_time = primitive_to_cost(
                comm_info, 
                record['ranks'], 
                record['tensor_size_bytes'], 
                primitive,
                gpus_per_node
            )
            
            abs_error = abs(predicted_time - actual_time)
            rel_error = abs_error / actual_time * 100 if actual_time > 0 else 0
            
            sizes_mb.append(size_mb)
            predicted_times.append(predicted_time)
            actual_times.append(actual_time)
            absolute_errors.append(abs_error)
            relative_errors.append(rel_error)
        
        if filtered_count > 0:
            print(f"Filtered {filtered_count} records with all ranks identical from '{primitive}' (error plot)")
        
        if not sizes_mb:
            print(f"Warning: No valid data for primitive '{primitive}'")
            continue
        
        sizes_mb = np.array(sizes_mb)
        predicted_times = np.array(predicted_times)
        actual_times = np.array(actual_times)
        absolute_errors = np.array(absolute_errors)
        relative_errors = np.array(relative_errors)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Prediction Error vs Message Size for {primitive}', 
                     fontsize=16, fontweight='bold')
        
        # Subplot 1: Scatter plot - Predicted vs Actual vs Size
        ax1 = axes[0, 0]
        scatter = ax1.scatter(sizes_mb, actual_times, c=absolute_errors, 
                             cmap='RdYlGn_r', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax1.scatter(sizes_mb, predicted_times, c='blue', marker='x', s=80, label='Predicted', alpha=0.6)
        ax1.plot(sizes_mb, actual_times, 'o', color='green', markersize=6, label='Actual', alpha=0.5)
        ax1.set_xlabel('Message Size (MB)', fontsize=12)
        ax1.set_ylabel('Time (s)', fontsize=12)
        ax1.set_title('Predicted vs Actual Communication Time', fontsize=13)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        cbar1 = plt.colorbar(scatter, ax=ax1)
        cbar1.set_label('Absolute Error (s)', fontsize=10)
        
        # Subplot 2: Scatter plot - Absolute Error vs Size
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(sizes_mb, absolute_errors, c=relative_errors, 
                               cmap='viridis', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax2.set_xlabel('Message Size (MB)', fontsize=12)
        ax2.set_ylabel('Absolute Error (s)', fontsize=12)
        ax2.set_title('Absolute Error vs Message Size', fontsize=13)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Relative Error (%)', fontsize=10)
        
        # Subplot 3: Bar chart - Mean absolute error by size bins
        ax3 = axes[1, 0]
        # Create logarithmic bins
        min_size = np.min(sizes_mb)
        max_size = np.max(sizes_mb)
        num_bins = 10
        bin_edges = np.logspace(np.log10(min_size), np.log10(max_size), num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        bin_means = []
        bin_stds = []
        bin_counts = []
        
        for i in range(num_bins):
            mask = (sizes_mb >= bin_edges[i]) & (sizes_mb < bin_edges[i + 1])
            if np.sum(mask) > 0:
                bin_means.append(np.mean(absolute_errors[mask]))
                bin_stds.append(np.std(absolute_errors[mask]))
                bin_counts.append(np.sum(mask))
            else:
                bin_means.append(0)
                bin_stds.append(0)
                bin_counts.append(0)
        
        bin_means = np.array(bin_means)
        bin_stds = np.array(bin_stds)
        bin_counts = np.array(bin_counts)
        
        # Filter out empty bins
        valid_mask = bin_counts > 0
        bin_centers = bin_centers[valid_mask]
        bin_means = bin_means[valid_mask]
        bin_stds = bin_stds[valid_mask]
        bin_counts = bin_counts[valid_mask]
        
        bars = ax3.bar(range(len(bin_centers)), bin_means, yerr=bin_stds, 
                       capsize=5, alpha=0.7, color='steelblue', edgecolor='black')
        ax3.set_xlabel('Message Size Bin (MB)', fontsize=12)
        ax3.set_ylabel('Mean Absolute Error (s)', fontsize=12)
        ax3.set_title('Mean Absolute Error by Size Bins', fontsize=13)
        ax3.set_xticks(range(len(bin_centers)))
        ax3.set_xticklabels([f'{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}' 
                            for i in range(len(bin_centers)) if valid_mask[i]], rotation=45, ha='right', fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add count labels on top of bars
        for i, (bar, count) in enumerate(zip(bars, bin_counts)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bin_stds[i] + bin_means[i]*0.01,
                    f'n={count}', ha='center', va='bottom', fontsize=8)
        
        # Subplot 4: Line plot - Relative error percentage vs size
        ax4 = axes[1, 1]
        # Sort by size for line plot
        sort_idx = np.argsort(sizes_mb)
        sorted_sizes = sizes_mb[sort_idx]
        sorted_rel_errors = relative_errors[sort_idx]
        
        # Create binned averages for smoother line
        bin_rel_means = []
        bin_rel_stds = []
        
        for i in range(num_bins):
            mask = (sizes_mb >= bin_edges[i]) & (sizes_mb < bin_edges[i + 1])
            if np.sum(mask) > 0:
                bin_rel_means.append(np.mean(relative_errors[mask]))
                bin_rel_stds.append(np.std(relative_errors[mask]))
            else:
                bin_rel_means.append(0)
                bin_rel_stds.append(0)
        
        bin_rel_means = np.array(bin_rel_means)
        bin_rel_stds = np.array(bin_rel_stds)
        
         # Use the same valid_mask from subplot 3
        ax4.errorbar(range(len(bin_centers)), bin_rel_means[valid_mask], 
                    yerr=bin_rel_stds[valid_mask], marker='o', linewidth=2, 
                    markersize=8, capsize=5, capthick=2, color='darkorange')
        ax4.set_xlabel('Message Size Bin', fontsize=12)
        ax4.set_ylabel('Mean Relative Error (%)', fontsize=12)
        ax4.set_title('Mean Relative Error by Size Bins', fontsize=13)
        ax4.set_xticks(range(len(bin_centers)))
        ax4.set_xticklabels([f'{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}' 
                            for i in range(len(bin_centers)) if valid_mask[i]], rotation=45, ha='right', fontsize=8)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.axhline(y=np.mean(relative_errors), color='red', linestyle='--', 
                   alpha=0.5, label=f'Mean: {np.mean(relative_errors):.2f}%')
        ax4.legend(fontsize=10)
        
        plt.tight_layout()
        
        # Save figure
        primitive_safe_name = primitive.replace('_', '-').replace(' ', '-')
        output_path = output_dir / f'error_by_size_{primitive_safe_name}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print statistics
        print(f"\nError vs Size plot generated for '{primitive}':")
        print(f"  Samples: {len(sizes_mb)}")
        print(f"  Size range: {np.min(sizes_mb):.4f} - {np.max(sizes_mb):.4f} MB")
        print(f"  Mean absolute error: {np.mean(absolute_errors):.6f} s")
        print(f"  Mean relative error: {np.mean(relative_errors):.2f}%")
        print(f"  Max absolute error: {np.max(absolute_errors):.6f} s at {sizes_mb[np.argmax(absolute_errors)]:.4f} MB")
        print(f"  Max relative error: {np.max(relative_errors):.2f}% at {sizes_mb[np.argmax(relative_errors)]:.4f} MB")
        print(f"  Saved to: {output_path}")
    
    print(f"\n{'='*80}")
    print(f"All error vs size plots saved to: {output_dir}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Compare predicted vs actual communication costs')
    parser.add_argument('--log_dir', type=str,
                        default='/data/haiqwa/zevin_nfs/andy/Auto-Parallelization/nnscaler_group1/nnscaler-h3c/examples/logs/comm_logs',
                        help='Directory containing comm_logs_rank*.json files')
    parser.add_argument('--comm_dir', type=str, 
                        default='/data/haiqwa/zevin_nfs/andy/Auto-Parallelization/nnscaler_group1/nnscaler-h3c/examples/comm_profiler/comm_1',
                        help='Directory containing communication profiling JSON files')
    parser.add_argument('--gpus_per_node', type=int, default=8,
                        help='Number of GPUs per node (default: 8)')
    parser.add_argument('--output', type=str, default='/data/haiqwa/zevin_nfs/andy/Auto-Parallelization/nnscaler_group1/nnscaler-h3c/examples/logs/comm_logs/output.json',
                        help='Output JSON file to save results (optional)')
    parser.add_argument('--plot_dir', type=str, default='/data/haiqwa/zevin_nfs/andy/Auto-Parallelization/nnscaler_group1/nnscaler-h3c/examples/logs/comm_logs/plot',
                        help='Output directory to save CDF plots (optional)')
    
    args = parser.parse_args()
    
    # Load communication logs
    log_dir = Path(args.log_dir)
    print(f"Loading communication logs from {log_dir}...")
    records = load_comm_logs(log_dir)
    print(f"Loaded {len(records)} communication records")
    
    # Load communication profiling data
    comm_dir = Path(args.comm_dir)
    print(f"Loading communication profiling data from {comm_dir}...")
    comm_info = load_comm_info(comm_dir)
    print(f"Loaded {len(comm_info)} communication profiling files:")
    for filename in sorted(comm_info.keys()):
        primitives = list(comm_info[filename].keys())
        print(f"  - {filename}: {len(primitives)} primitives ({', '.join(primitives[:5])}{'...' if len(primitives) > 5 else ''})")
    
    if not comm_info:
        print("\nError: No communication profiling data found.")
        print("Skipping actual analysis...")
        return
    
    # Analysis by primitive
    primitive_stats = analyze_by_primitive(records, comm_info, args.gpus_per_node)
    print("\n" + "="*80)
    print("Statistics by Communication Primitive")
    print("="*80)
    for primitive, stats in sorted(primitive_stats.items()):
        print(f"\n{primitive}:")
        print(f"  Samples: {stats['num_samples']}")
        print(f"  MAE: {stats['mae']:.6f} s")
        print(f"  MAPE: {stats['mape']:.2f}%")
        print(f"  RMSE: {stats['rmse']:.6f} s")
        print(f"  R-squared: {stats['r_squared']:.4f}")
    
    # Generate CDF plots if plot_dir is specified
    if args.plot_dir:
        print("\n" + "="*80)
        print("Generating CDF Plots")
        print("="*80)
        generate_cdf_plots(records, comm_info, Path(args.plot_dir), args.gpus_per_node)

        # print("\n" + "="*80)
        # print("Generating Error vs Size Plots")
        # print("="*80)
        # plot_error_by_size(records, comm_info, Path(args.plot_dir), args.gpus_per_node)
    
    
    # Save results if output file specified
    if args.output:
        results = {
            'by_primitive': primitive_stats,
        }
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()