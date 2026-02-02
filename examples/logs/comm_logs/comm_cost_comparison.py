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


def primitive_to_cost(comm_info: Dict, dev_num: int, byte_size: int, 
                      primitive: str, dp_comm_mesh: Tuple[int, int]) -> float:
    """
    Predict communication cost based on comm_info.
    
    Args:
        comm_info: Dictionary containing communication profiling data.
        dev_num: Number of devices.
        byte_size: Size of the tensor in bytes.
        primitive: Type of communication primitive (e.g., 'all_reduce', 'all_gather').
        dp_comm_mesh: Tuple of (num_nodes, num_gpus_per_node).
    
    Returns:
        Predicted time in seconds.
    """
    if byte_size == 0:
        return 0.0
    
    size_mb = byte_size / 1024 / 1024
    (nnodes, _) = dp_comm_mesh
    
    if nnodes == 1:
        device_setting = f'intra_{dev_num}.json'
    elif nnodes > 1:
        device_setting = f'inter_{dp_comm_mesh}.json'
    else:
        raise ValueError(f'Invalid dp_comm_mesh: {dp_comm_mesh} provided.')
    
    if device_setting not in comm_info:
        print(f"Warning: device_setting '{device_setting}' not found in comm_info")
        return 0.0
    
    if primitive not in comm_info[device_setting]:
        print(f"Warning: primitive '{primitive}' not found in comm_info['{device_setting}']")
        return 0.0
    
    sizes_in_mb, times_in_s = comm_info[device_setting][primitive]
    est_time = _piecewise_estimator(sizes_in_mb, times_in_s, size_mb)
    
    assert est_time >= 0, f'{primitive} {dev_num} comm size: {size_mb} MB, est time: {est_time} s'
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
                         dev_num: int, dp_comm_mesh: Tuple[int, int]) -> Dict[str, Any]:
    """
    Analyze prediction accuracy by communication primitive.
    
    Args:
        records: List of communication records.
        comm_info: Dictionary containing communication profiling data.
        dev_num: Number of devices.
        dp_comm_mesh: Tuple of (num_nodes, num_gpus_per_node).
    
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
        
        for record in group:
            # Convert actual latency from ms to seconds
            actual_time = record['latency_ms'] / 1000.0
            
            # Predict time
            predicted_time = primitive_to_cost(
                comm_info, dev_num, 
                record['tensor_size_bytes'], 
                primitive, dp_comm_mesh
            )
            
            predicted_times.append(predicted_time)
            actual_times.append(actual_time)
            errors.append(predicted_time - actual_time)
        
        stats = compute_statistics(predicted_times, actual_times)
        stats['num_samples'] = len(group)
        stats['primitive'] = primitive
        
        primitive_stats[primitive] = stats
    
    return primitive_stats


def analyze_by_size_range(records: List[Dict[str, Any]], comm_info: Dict, 
                          dev_num: int, dp_comm_mesh: Tuple[int, int],
                          size_ranges: List[Tuple[float, float]] = None) -> Dict[str, Any]:
    """
    Analyze prediction accuracy by tensor size range.
    
    Args:
        records: List of communication records.
        comm_info: Dictionary containing communication profiling data.
        dev_num: Number of devices.
        dp_comm_mesh: Tuple of (num_nodes, num_gpus_per_node).
        size_ranges: List of (min_size_mb, max_size_mb) tuples.
    
    Returns:
        Dictionary containing analysis results for each size range.
    """
    if size_ranges is None:
        size_ranges = [
            (0, 1),           # < 1 MB
            (1, 10),          # 1-10 MB
            (10, 100),        # 10-100 MB
            (100, 512),       # 100-512 MB
            (512, float('inf')),  # > 512 MB
        ]
    
    range_stats = {}
    
    for i, (min_size, max_size) in enumerate(size_ranges):
        range_name = f"{min_size}-{max_size if max_size != float('inf') else 'inf'} MB"
        
        predicted_times = []
        actual_times = []
        
        for record in records:
            size_mb = record['tensor_size_bytes'] / 1024 / 1024
            
            if min_size <= size_mb < max_size:
                actual_time = record['latency_ms'] / 1000.0
                predicted_time = primitive_to_cost(
                    comm_info, dev_num, 
                    record['tensor_size_bytes'], 
                    record['op_type'], dp_comm_mesh
                )
                
                predicted_times.append(predicted_time)
                actual_times.append(actual_time)
        
        if predicted_times:
            stats = compute_statistics(predicted_times, actual_times)
            stats['range'] = range_name
            stats['num_samples'] = len(predicted_times)
            range_stats[range_name] = stats
    
    return range_stats


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


def main():
    parser = argparse.ArgumentParser(description='Compare predicted vs actual communication costs')
    parser.add_argument('--log_dir', type=str, required=True,
                        help='Directory containing comm_logs_rank*.json files')
    parser.add_argument('--dev_num', type=int, default=8,
                        help='Number of devices (default: 8)')
    parser.add_argument('--dp_comm_mesh', type=tuple, default=(1, 8),
                        help='DP communication mesh as tuple (num_nodes, num_gpus_per_node) (default: (1, 8))')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file to save results (optional)')
    
    args = parser.parse_args()
    
    # Load communication logs
    log_dir = Path(args.log_dir)
    print(f"Loading communication logs from {log_dir}...")
    records = load_comm_logs(log_dir)
    print(f"Loaded {len(records)} communication records")
    
    # Assuming comm_info is provided (you need to load or define it)
    # This is a placeholder - you should replace it with actual comm_info
    comm_info = {}
    print("\nNote: Please provide actual comm_info dictionary in the code")
    print("Example comm_info structure:")
    print("""{
    'intra_8.json': {
        'all_reduce': ([1, 2, 4, 8, 16, 32, 64, 128, 256, 512], 
                       [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512]),
        'all_gather': [...],
        ...
    },
    'inter_(2, 4).json': {...}
}""")
    
    if not comm_info:
        print("\nError: comm_info is empty. Please provide actual communication profiling data.")
        print("Skipping actual analysis...")
        return
    
    # Overall analysis
    predicted_times = []
    actual_times = []
    
    for record in records:
        actual_time = record['latency_ms'] / 1000.0
        predicted_time = primitive_to_cost(
            comm_info, args.dev_num, 
            record['tensor_size_bytes'], 
            record['op_type'], args.dp_comm_mesh
        )
        predicted_times.append(predicted_time)
        actual_times.append(actual_time)
    
    overall_stats = compute_statistics(predicted_times, actual_times)
    print_statistics(overall_stats, "Overall Statistics")
    
    # Analysis by primitive
    primitive_stats = analyze_by_primitive(records, comm_info, args.dev_num, args.dp_comm_mesh)
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
    
    # Analysis by size range
    size_range_stats = analyze_by_size_range(records, comm_info, args.dev_num, args.dp_comm_mesh)
    print("\n" + "="*80)
    print("Statistics by Tensor Size Range")
    print("="*80)
    for range_name, stats in sorted(size_range_stats.items()):
        print(f"\n{range_name}:")
        print(f"  Samples: {stats['num_samples']}")
        print(f"  MAE: {stats['mae']:.6f} s")
        print(f"  MAPE: {stats['mape']:.2f}%")
        print(f"  RMSE: {stats['rmse']:.6f} s")
        print(f"  R-squared: {stats['r_squared']:.4f}")
    
    # Save results if output file specified
    if args.output:
        results = {
            'overall': overall_stats,
            'by_primitive': primitive_stats,
            'by_size_range': size_range_stats,
        }
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()