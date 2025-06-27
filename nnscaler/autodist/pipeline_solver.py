#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from .model_graph import ModelGraph, estimate_mem_lower_bound, IntervalInfo
from .spmd_solver import SPMDSolver
from .descs import *
from .autodist_config import AutoDistConfig

import itertools
import torch
import os
import time
import json
import copy
import math
import multiprocessing
import logging
from typing import List, Dict, Tuple
from pathlib import Path

__all__ = [
    'calc_optimal_pp_plan',
    'calc_optimal_pp_plan_dp',
    'calc_optimal_pp_plan_dp_max',
]

_logger = logging.getLogger(__name__)


def _dev_num2mesh_desc(dev_num: int, base_col: int) -> MeshDesc:
    if dev_num <= base_col:
        return MeshDesc(1, dev_num)
    else:
        assert dev_num % base_col == 0
        return MeshDesc(dev_num // base_col, base_col)


def _calc_legal_tp_degrees(max_tp_degree: int) -> List[int]:
    ret = []
    tp_degree = 1
    while tp_degree <= max_tp_degree:
        ret.append(tp_degree)
        tp_degree = tp_degree * 2
    return ret


def split_into_powers_of_two(N: int, s: int, legal_tp_degrees: List[int]):
    """
    This function returns all possible permutations of combinations that sum up to N,
    where each number in the combination is a power of 2, and the combination contains
    exactly s numbers.

    Parameters:
    N (int): The target sum that the combination should equal to.
    s (int): The exact number of elements that the combination should contain.

    Returns:
    list: A list of tuples, where each tuple is a permutation of a valid combination
          that sums up to N and contains exactly s elements, all of which are powers of 2.
    """
    # Generate all powers of 2 that are less than or equal to N
    powers_of_two = []
    power = 1
    while power <= N:
        if power in legal_tp_degrees:
            powers_of_two.append(power)
        power *= 2

    # Recursive function to find all valid combinations
    def find_combinations(target, start, count):
        if count == s:  # If we have already selected s numbers
            if target == 0:  # If the target is 0, it means the combination is valid
                return [
                    []
                ]  # Return an empty list to indicate the combination is complete
            else:
                return [
                ]  # If the target is not 0, it's an invalid combination

        if target <= 0:  # If the target is negative or we haven't selected enough numbers yet
            return []

        combinations = []
        for i in range(start, len(powers_of_two)):
            power = powers_of_two[i]
            if power > target:  # If the current power exceeds the remaining target, skip it
                continue
            # Recursively find combinations by including the current power
            for combo in find_combinations(target - power, i, count + 1):
                combinations.append(
                    [power] +
                    combo)  # Add the current power to the combination
        return combinations

    # Generate all valid combinations that sum up to N and contain exactly s elements
    valid_combinations = find_combinations(N, 0, 0)

    # Sort each combination and generate all unique permutations
    all_permutations = []
    for combo in valid_combinations:
        sorted_combo = sorted(
            combo)  # Sort the combination to ensure consistent permutations
        permutations = set(itertools.permutations(
            sorted_combo))  # Generate unique permutations
        all_permutations.extend(
            permutations)  # Add all unique permutations to the result list

    return all_permutations


def staging_combination(s, pivots_idx):
    """
    This function returns all possible ways to stage a list of integers from 0 to n
    into s segments. Each segment is represented by its starting and ending index.

    Parameters:
    s (int): The number of segments to partition the list into.
    pivots_idx (List(int)): The opterator index of pivots.(ir_cell.id)

    Returns:
    list: A list of tuples, each containing the start and end index of a segment.
    """
    n = len(pivots_idx)

    # Helper function to recursively find all valid partitions
    def staging_helper(start, remaining_segments, current_stage):
        # Base case: if there are no remaining segments to partition
        if remaining_segments == 1:
            current_stage.append(
                (start, n - 1))  # Last segment from start to n
            results.append(
                current_stage)  # Add the current partition to result
            return

        # Try all possible end points for the current segment
        for end in range(start + 1, n - remaining_segments + 1):
            # For each valid partition, recursively partition the remaining numbers
            staging_helper(end, remaining_segments - 1,
                           current_stage + [(start, end)])

    results = []
    staging_helper(0, s, [])
    staged_intervals_combinations = []
    for result in results:
        staged_intervals = []
        for (start, end) in result:
            start_op_index = pivots_idx[start]
            end_op_index = pivots_idx[end] - 1
            interval = (start_op_index, end_op_index)
            staged_intervals.append(interval)
        staged_intervals_combinations.append(staged_intervals)
    return staged_intervals_combinations


def _gen_meshes(cfg: AutoDistConfig) -> List[tuple]:
    '''
    generate all possible meshes (a,b)
    a * b = cfg.plan_ngpus 
    a means uses a servers, b means use b gpus in each server.
    for example:
        plan_ngpus = 8, the runtime_ngus = 4 * 8 = 32
        we need to return meshes:
        (1,8) (2,4) (4,2)
    '''
    plan_ngpus = cfg.mesh_desc.ngpus
    ngpus_per_node = torch.cuda.device_count()
    nnodes = cfg.world_size // ngpus_per_node
    meshes = []
    for a in range(1, nnodes + 1):
        if plan_ngpus % a == 0:
            b = plan_ngpus // a
            if b <= ngpus_per_node:
                meshes.append((a, b))

    # return [(1,8)]
    return meshes


def aggregate_to_groups(lst, b):
    """
    This function aggregates a list of powers of 2 into multiple groups, 
    each with a sum equal to 'b'. Each group will contain elements in 
    the original order, and the function will return two lists:
    1. A list of groups, where each group is a list of numbers.
    2. A list of indices, where each group is a list of the original indices.

    Parameters:
    lst (List[int]): A list of tp_degree, all powers of 2, to be aggregated.
    b (int): A power of 2, representing the sum each group should have.

    Returns:
    Tuple[List[List[int]], List[List[int]]]:
        - The first list contains multiple groups, where each group is a list of integers whose sum equals 'b'.
        - The second list contains the corresponding indices for each group from the original list 'lst'.
    """
    original_order = [(num, i) for i, num in enumerate(lst)]
    groups = []
    indices = []
    current_group = []
    current_indices = []
    current_sum = 0

    for num, original_idx in original_order:
        if current_sum + num > b:
            continue  # 跳过无法加入的元素（需后续处理）
        current_group.append(num)
        current_indices.append(original_idx)
        current_sum += num
        if current_sum == b:
            groups.append(current_group)
            indices.append(current_indices)
            current_group = []
            current_indices = []
            current_sum = 0

    remaining = []
    remaining_indices = []
    for num, idx in original_order:
        if idx not in [i for group in indices for i in group]:
            remaining.append(num)
            remaining_indices.append(idx)
    
    if remaining:
        print(f"Warning: Remaining elements {remaining} could not be grouped.")
    
    return groups, indices


def _collect_tp_intervals(
    model_graph: ModelGraph,
    cfg: AutoDistConfig,
    tp_degree: int,
    stage_num: int,
    interval_groups: List[List[IntervalInfo]],
) -> List[int]:
    '''
    collect intervals for given tp_degree and stage_num
    no need to calculate all possible intervals
        1. some intervals may not fit into the memory
        2. some intervals are sub-optimal
            we want to make pipeline stages as balanced as possible
            ideally, we want to make the time of each stage equal.
            to be robust, we can constrain the average time of each stage
            is within a certain range, like no more than 200% of the global
            average time
        3. some intervals are identical (exactly the same ops and topology)

    Args:
        model_graph: the graph in AutoDist
        cfg: the AutoDistConfig
        tp_degree: the tensor parallelism degree
        stage_num: the pipeline stage number
        interval_groups: a list of groups. identical intervals are in a group
        spmd_solver: the solver for tensor parallelism

    Returns:
        selected_groups: the indices of selected interval groups
    '''

    def calc_min_mem(start, end):
        param_mem, buffer_mem, activation_mem = model_graph.query_mem(
            start, end)
        if cfg.zero_stage == 1:
            zero_group_size = tp_degree * cfg.world_size // cfg.mesh_desc.ngpus // cfg.zero_ngroups
        elif cfg.zero_stage == 0:
            zero_group_size = tp_degree
        else:
            raise RuntimeError(f'invalid zero stage {cfg.zero_stage}')
        return estimate_mem_lower_bound(
            param_mem=param_mem,
            buffer_mem=buffer_mem,
            activation_mem=activation_mem * stage_num,
            plan_ngpus=tp_degree,
            zero_group_size=zero_group_size,
            cfg=cfg,
        )

    idxs = [0] + model_graph.get_pipeline_pivots() + [model_graph.op_num]
    global_fw_span = model_graph.query_fw_span(
        0,
        model_graph.op_num - 1) / model_graph.autodist_config.mesh_desc.ngpus
    min_fw_span = global_fw_span * cfg.max_pipeline_unbalance_ratio
    max_fw_span = global_fw_span / cfg.max_pipeline_unbalance_ratio
    selected_groups = []
    for i, group in enumerate(interval_groups):
        start, end = group[0].start, group[0].end
        if calc_min_mem(start, end) > cfg.memory_constraint:
            continue
        local_fw_span = model_graph.query_fw_span(start, end) / tp_degree
        if local_fw_span < min_fw_span or local_fw_span > max_fw_span:
            continue
        selected_groups.append(i)
    return selected_groups


def _compute_tp_info(
    model_graph: ModelGraph,
    cfg: AutoDistConfig,
    legal_tp_degrees: List[int],
) -> Dict[Tuple[int, int, int, int, str], SPMDSearchOutput]:
    '''
    Pre-compute the optimal spmd plan and store the result in a dict.
    The key of the dict is (tp_degree, stage_num, start, end),
    which means the optimal spmd plan for the interval [start, end]
    with tp_degree devices and stage_num pipeline stages.

    Args:
        model_graph: the graph in AutoDist
        cfg: the AutoDistConfig
        legal_tp_degrees: the legal tensor parallelism device numbers

    Returns:
        tp_info: the dict that stores the optimal spmd plan for each interval
    '''

    _logger.info('start to compute tp info')
    start_time = time.time()
    interval_groups = model_graph.group_pipeline_intervals()
    # if there is no solution for (tp_degree, stage_num, start, end),
    # there is no solution for (tp_degree, stage_num + 1, start, end)
    no_solution_states = set()
    count = 0

    def process_case(device_num, stage_num):
        print(f"device_num, stage_num:{device_num, stage_num}")
        selected_group_idxs = _collect_tp_intervals(
            model_graph,
            cfg,
            device_num,
            stage_num,
            interval_groups,
        )
        print(f"selected_group_idxs:{selected_group_idxs}")
        intervals = []
        for i in selected_group_idxs:
            start, end = interval_groups[i][0].start, interval_groups[i][0].end
            if (start, end, device_num) in no_solution_states:
                continue
            intervals.append((start, end))
            _logger.info(
                f'process case: tp {device_num}, s {stage_num}, interval:{(start, end)}'
            )
        print(f"intervals:{intervals}")
        if not intervals or (device_num == cfg.mesh_desc.ngpus
                             and stage_num > 1):
            return [], [], [], []
        # postpone the initialization of SPMDSolver to save time
        cur_cfg = copy.deepcopy(cfg)
        # In current parallel profiler's implementation, the profiling is divided into
        # following steps:
        #   1. searialize the input graph
        #   2. lauch the multi-process profiling by python's spawn method
        #   3. each process loads the serialized graph and do profiling
        #   4. transport the profiling result back to the main process
        # It helps to reduce the profiling time when the graph has not been met before.
        # But the procedure itself has a large overhead.
        # In PipelineSolver, the SPMDSolver is constructed and used to search the optimal
        # plan for multiple times. For given `tp_degree`, cases that need to be profiled
        # are the same. As a result, we set `cfg.parallel_profile` to True at the first time
        # and set it to False for the rest of the time.
        if stage_num == 1:
            cur_cfg.parallel_profile = True
        else:
            cur_cfg.parallel_profile = False
        cur_cfg.world_size = cfg.world_size // cfg.mesh_desc.ngpus * device_num
        cur_cfg.mesh_desc = _dev_num2mesh_desc(device_num, cfg.mesh_desc.col)
        import time
        start_time = time.time()
        dp_group_meshes = _gen_meshes(cfg)
        solvers = []
        solver_rets = []
        for dp_group_mesh in dp_group_meshes:
            if device_num > dp_group_mesh[1]:
                solver = None
                solver_ret = None
            else:
                solver = SPMDSolver(graph=model_graph,
                                    mesh_desc=cur_cfg.mesh_desc,
                                    autodist_config=cur_cfg,
                                    stage_num=stage_num,
                                    dp_group_mesh=dp_group_mesh)
                solver_ret = solver.solve(intervals, 1)
            solvers.append(solver)
            solver_rets.append(solver_ret)
        total_time = time.time() - start_time
        return dp_group_meshes, solvers, solver_rets, intervals

    def _calc_upper_bound(tp_degree: int):
        # bubble time percentage <= bubble_ratio:
        # (stage_num - 1) / (stage_num - 1 + micro_batch_num) <= bubble_ratio
        # stage_num <= 1 + bubble_ratio * micro_batch_num / (1 - bubble_ratio)
        bubble_ratio = cfg.max_pipeline_bubble_ratio
        micro_batch_num = cfg.update_freq
        upper_bound = math.floor(bubble_ratio /
                                 (1 - bubble_ratio) * micro_batch_num + 1)
        logging.info(
            f'pipeline的上限是:{min(cfg.mesh_desc.ngpus - tp_degree + 1, upper_bound)}'
        )
        return min(cfg.mesh_desc.ngpus - tp_degree + 1, upper_bound)

    # intervals in a same group share a distributed plan. To make the code generation
    # correct, we need to adjust the plan for each interval based on each offset with
    # respect to the first interval in the group.
    def shift_plan(solver, spmd_desc, offset: int, shifted_start: int,
                   shifted_end: int):
        assert offset >= 0, f'invalid offset {offset}'
        if offset == 0:
            return spmd_desc
        new_spmd_desc = copy.deepcopy(spmd_desc)
        new_partition_descs = dict()
        plan = list()
        shifted_idx = shifted_start
        for k, v in spmd_desc.desc.partition_descs.items():
            new_partition_descs[k + offset] = v
            plan.append((shifted_idx, solver.node_desc2idx(shifted_idx, v)))
            shifted_idx += 1
        assert shifted_idx == shifted_end + 1, f'expect {shifted_end + 1}, got {shifted_idx}'
        new_spmd_desc.desc.partition_descs = new_partition_descs
        new_spmd_desc.desc.analysis = solver.analyze_plan(plan)
        return new_spmd_desc

    tp_info = {}
    for tp_degree in legal_tp_degrees:
        # for stage_num in range(1, _calc_upper_bound(tp_degree) + 1):
        # for stage_num in range(1, 5):
        for stage_num in range(1, 5):
            dp_group_meshes, solvers, solver_rets, intervals = process_case(
                tp_degree, stage_num)
            for dp_group_mesh, solver, solver_ret in zip(
                    dp_group_meshes, solvers, solver_rets):
                if dp_group_mesh[1] < tp_degree:
                    continue
                for interval, spmd_descs in zip(intervals, solver_ret):
                    start, end = interval
                    if spmd_descs:
                        for group in interval_groups:
                            if group[0].start == start and group[0].end == end:
                                # iso -> isomorphic
                                for iso_interval in group:
                                    iso_start, iso_end = iso_interval.start, iso_interval.end
                                    offset = model_graph.operator_list[iso_start].ir_cell.cid - \
                                            model_graph.operator_list[start].ir_cell.cid
                                    tp_info[(tp_degree, stage_num, iso_start,
                                             iso_end,
                                             dp_group_mesh)] = shift_plan(
                                                 solver, spmd_descs[0], offset,
                                                 iso_start, iso_end)
                    else:
                        no_solution_states.add((start, end, tp_degree))
                        _logger.info(
                            f'fail to find a valid plan for {start}, {end}')
    _logger.info('finish computing tp info')
    _logger.info(f'pipeline中tp_info求解总时间: {time.time() - start_time}')
    return tp_info


def find_node_id(target, indices):
    for node_id, sublist in enumerate(indices):
        if target in sublist:
            return node_id
    return -1


def calc_pp_comm_cost(model_graph: ModelGraph,
                      staged_intervals: List[Tuple[int, int]],
                      spmd_outs: List[TensorParallelDesc],
                      indices: List[List[int]]) -> float:

    pp_comm_cost = 0
    stage_num = len(spmd_outs)
    for stage_id in range(stage_num - 1):
        (_, end) = staged_intervals[stage_id]
        (start, _) = staged_intervals[stage_id + 1]
        src_stage_spmd_desc = spmd_outs[stage_id].desc.partition_descs
        dst_stage_spmd_desc = spmd_outs[stage_id + 1].desc.partition_descs
        if find_node_id(stage_id,
                        indices) == find_node_id(stage_id + 1, indices):
            pp_comm_cost += model_graph.calc_pivot_comm_cost(
                list(src_stage_spmd_desc.items()),
                list(dst_stage_spmd_desc.items()), "intra")
        else:
            pp_comm_cost += model_graph.calc_pivot_comm_cost(
                list(src_stage_spmd_desc.items()),
                list(dst_stage_spmd_desc.items()), "inter")

    return pp_comm_cost


#max版本
def calc_optimal_pp_plan_dp_max(
    model_graph: ModelGraph, autodist_config: AutoDistConfig
) -> Dict[Tuple[int, Tuple, Tuple, Tuple], float]:
    '''
    目标：对于每个max_time，找到最小的sum_time.
    '''
    legal_tp_degrees = _calc_legal_tp_degrees(
        min(8, autodist_config.mesh_desc.col))
    ngpus = autodist_config.mesh_desc.ngpus
    tp_info = _compute_tp_info(model_graph, autodist_config, legal_tp_degrees)
    pp_idxs = [0] + model_graph.get_pipeline_pivots() + [model_graph.op_num]
    dp_group_meshes = _gen_meshes(autodist_config)
    micro_batch_num = autodist_config.update_freq

    best_plan = None
    best_pp_cost = float("inf")

    # Dynamic programming table T
    # T is a dictionary where the key is a tuple (s, pp, tp, i)
    # s: stage number
    # pp: total number of devices used up to this stage
    # tp: number of devices used for the current stage
    # i: start operator index for the current stage
    # The value is a list [optimal_time, previous_state]
    # optimal_time: the optimal time for the given state
    # previous_state: the previous state (s-1, pp-tp, tp', j) that leads to the optimal time
    T = {}
    for dp_group_mesh in dp_group_meshes:
        T[dp_group_mesh] = {}
        # print(f'dp_group_mesh:{dp_group_mesh}')
        for s in range(1, ngpus + 1):  # 第s个stage
            for pp in range(s, ngpus + 1):  # 目前一共用了pp个gpu
                for tp in range(1, pp - s + 1 + 1):  # 当前stage用了tp个gpu
                    if tp > dp_group_mesh[1] or tp not in legal_tp_degrees:
                        continue
                    for ii in range(len(pp_idxs) - 1 - 1, 0 - 1, -1):
                        i = pp_idxs[ii]
                        cur_idx = (s, pp, tp, i)
                        T[dp_group_mesh][cur_idx] = (-1, None, None)
                        if tp == pp and s == 1:
                            tp_idx = (tp, s, i, model_graph.op_num - 1,
                                      dp_group_mesh)
                            if tp_idx in tp_info:
                                t = tp_info[tp_idx].all_time
                                T[dp_group_mesh][cur_idx] = (
                                    t, None,
                                    tp_info[tp_idx].desc.partition_descs)
                            # if(T[dp_group_mesh][cur_idx][0] != -1):
                            #      print(f'[s, pp, tp, i]:{cur_idx}, T:{T[dp_group_mesh][cur_idx]}')

                            continue

                        for jj in range(len(pp_idxs) - 1 - 1, ii, -1):
                            j = pp_idxs[jj]
                            next_pp = pp - tp
                            for next_tp in range(1, next_pp - (s - 1) + 1 + 1):
                                if next_tp not in legal_tp_degrees:
                                    continue
                                prev_idx = (s - 1, next_pp, next_tp, j)
                                if prev_idx not in T[dp_group_mesh] or T[
                                        dp_group_mesh][prev_idx][0] == -1:
                                    continue
                                tp_idx = (tp, s, i, j - 1, dp_group_mesh)
                                if tp_idx not in tp_info:
                                    continue

                                # merge T[prev_idx] into T[cur_idx]

                                prev_t = T[dp_group_mesh][prev_idx]
                                max_time = prev_t[0]

                                max_time = max(max_time,
                                               tp_info[tp_idx].all_time)

                                # comm_time = 0
                                # cur_stage_desc = tp_info[tp_idx].desc.partition_descs
                                # next_stage_desc = prev_t[2]

                                # comm_time += model_graph.calc_pivot_comm_cost(
                                #     list(cur_stage_desc.items())[-1],
                                #     list(next_stage_desc.items())[0], "intra")
                                # 目前不考虑 comm

                                if T[dp_group_mesh][cur_idx][0] != -1 and T[
                                        dp_group_mesh][cur_idx][0] <= max_time:
                                    continue

                                T[dp_group_mesh][cur_idx] = (
                                    max_time, prev_idx,
                                    tp_info[tp_idx].desc.partition_descs)

                        # if(T[dp_group_mesh][cur_idx][0] != -1):
                        #     print(f'[s, pp, tp, i]:{cur_idx}, T:{T[dp_group_mesh][cur_idx]}')

    best_time = float('inf')
    best_state = None
    micro_batch_num = autodist_config.update_freq
    for dp_group_mesh in dp_group_meshes:
        for stage_num in range(1, ngpus + 1):
            for pp_dev_num in range(stage_num, ngpus + 1):
                for tp_degree in range(1, pp_dev_num - stage_num + 1 + 1):
                    if tp_degree not in legal_tp_degrees:
                        continue

                    cur_idx = (stage_num, pp_dev_num, tp_degree, 0)
                    if cur_idx not in T[dp_group_mesh]:
                        continue

                    if T[dp_group_mesh][cur_idx][0] == -1:
                        continue

                    cur_time = T[dp_group_mesh][cur_idx][0] * (
                        micro_batch_num - 1 + stage_num)

                    logging.info(
                        f'mesh:{dp_group_mesh}, [s, pp, tp, i]:{cur_idx}, All_time: {cur_time}'
                    )
                    if best_time > cur_time:
                        best_time, best_state = cur_time, (dp_group_mesh,
                                                           cur_idx)

    _logger.info(
        f'best time/s: {best_time}, state (s, pp, tp, i): {best_state}')
    if best_state == None:
        raise RuntimeError('fail to find a valid pipeline plan')

    spmd_outs = []
    tp_list = []

    def build_answer(s, pp, tp, i, dp_group_mesh):
        print(f's:{s} pp:{pp} tp:{tp} i:{i} dp_group_mesh:{dp_group_mesh}')
        _1, prev_idx, _2 = T[dp_group_mesh][(s, pp, tp, i)]
        if prev_idx == None:
            tp_idx = (tp, s, i, model_graph.op_num - 1, dp_group_mesh)
        else:
            j_plus_1 = prev_idx[3]
            tp_idx = (tp, s, i, j_plus_1 - 1, dp_group_mesh)
        spmd_outs.append(tp_info[tp_idx])
        print(
            f"stage weight update time:{tp_info[tp_idx].weight_update_time},dp_size:{tp_info[tp_idx].dp_size}"
        )
        tp_list.append(tp)
        if prev_idx != None:
            build_answer(*prev_idx, dp_group_mesh)

    build_answer(*best_state[1], best_state[0])

    tp_groups, indices = aggregate_to_groups(tp_list, best_state[0][1])

    print(
        f'tp_list:{tp_list} b:{best_state[0][1]} tp_groups:{tp_groups} indices:{indices}'
    )

    spmd_descs = [spmd_out.desc for spmd_out in spmd_outs]
    pp_desc = PipelineParallelDesc(spmd_descs, [], autodist_config.mesh_desc)
    stage_mems = [spmd_out.memory for spmd_out in spmd_outs]
    stage_all_times = [spmd_out.all_time for spmd_out in spmd_outs]
    stage_comp_times = [spmd_out.comp_time for spmd_out in spmd_outs]
    return PipelineSearchOutput(best_state[0], tp_groups, indices, pp_desc,
                                best_pp_cost, stage_mems, stage_all_times,
                                stage_comp_times)


#sum+max动态规划版本
def calc_optimal_pp_plan_dp(
    model_graph: ModelGraph, autodist_config: AutoDistConfig
) -> Dict[Tuple[int, Tuple, Tuple, Tuple], float]:
    '''
    目标：对于每个max_time，找到最小的sum_time.
    '''
    legal_tp_degrees = _calc_legal_tp_degrees(
        min(8, autodist_config.mesh_desc.col))
    ngpus = autodist_config.mesh_desc.ngpus
    tp_info = _compute_tp_info(model_graph, autodist_config, legal_tp_degrees)
    for key, value in tp_info.items():
        print(
            f"tp_degree, stage_num, start, end, dp_group_mesh:{key}, time:{value.all_time}"
        )
    pp_idxs = [0] + model_graph.get_pipeline_pivots() + [model_graph.op_num]
    dp_group_meshes = _gen_meshes(autodist_config)
    micro_batch_num = autodist_config.update_freq

    best_plan = None
    best_pp_cost = float("inf")

    # Dynamic programming table T
    # T is a dictionary where the key is a tuple (s, pp, tp, i)
    # s: stage number
    # pp: total number of devices used up to this stage
    # tp: number of devices used for the current stage
    # i: start operator index for the current stage
    # The value is a list [optimal_time, previous_state]
    # optimal_time: the optimal time for the given state
    # previous_state: the previous state (s-1, pp-tp, tp', j) that leads to the optimal time
    T = {}
    for dp_group_mesh in dp_group_meshes:
        T[dp_group_mesh] = {}
        print(f'dp_group_mesh:{dp_group_mesh}')
        for s in range(1, ngpus + 1):  # 第s个stage
            for pp in range(s, ngpus + 1):  # 目前一共用了pp个gpu
                for tp in range(1, pp - s + 1 + 1):  # 当前stage用了tp个gpu
                    if tp > dp_group_mesh[1] or tp not in legal_tp_degrees:
                        continue
                    for ii in range(len(pp_idxs) - 1 - 1, 0 - 1, -1):
                        i = pp_idxs[ii]
                        cur_idx = (s, pp, tp, i)
                        T[dp_group_mesh][cur_idx] = [
                        ]  # T中存的是Max = x的最小 的sum_time的一个单调队列
                        if tp == pp and s == 1:
                            tp_idx = (tp, s, i, model_graph.op_num - 1,
                                      dp_group_mesh)
                            if tp_idx in tp_info:
                                t = tp_info[tp_idx].all_time
                                T[dp_group_mesh][cur_idx].append(
                                    (t, t, None, -1,
                                     tp_info[tp_idx].desc.partition_descs))
                            # if(len(T[dp_group_mesh][cur_idx]) > 0):
                            #     print(f'[s, pp, tp, i]:{cur_idx}, len(T): {len(T[dp_group_mesh][cur_idx])}')
                            continue

                        for jj in range(len(pp_idxs) - 1 - 1, ii, -1):
                            j = pp_idxs[jj]
                            next_pp = pp - tp
                            for next_tp in range(1, next_pp - (s - 1) + 1 + 1):
                                if next_tp not in legal_tp_degrees:
                                    continue
                                prev_idx = (s - 1, next_pp, next_tp, j)
                                if prev_idx not in T[dp_group_mesh] or len(
                                        T[dp_group_mesh][prev_idx]) == 0:
                                    continue
                                tp_idx = (tp, s, i, j - 1, dp_group_mesh)
                                if tp_idx not in tp_info:
                                    continue

                                # merge T[prev_idx] into T[cur_idx]

                                for k in range(
                                        0, len(T[dp_group_mesh][prev_idx])):
                                    prev_t = T[dp_group_mesh][prev_idx][k]
                                    max_time = prev_t[0]
                                    sum_time = prev_t[1]

                                    max_time = max(max_time,
                                                   tp_info[tp_idx].all_time)
                                    sum_time += tp_info[tp_idx].all_time

                                    comm_time = 0
                                    cur_stage_desc = tp_info[
                                        tp_idx].desc.partition_descs
                                    next_stage_desc = prev_t[4]

                                    # comm_time += model_graph.calc_pivot_comm_cost(
                                    #     list(cur_stage_desc.items()),
                                    #     list(next_stage_desc.items()), "intra")

                                    sum_time += comm_time
                                    T[dp_group_mesh][cur_idx].append(
                                        (max_time, sum_time, prev_idx, k,
                                         tp_info[tp_idx].desc.partition_descs))
                        # sort T[cur_idx] by max_time
                        T[dp_group_mesh][cur_idx].sort(key=lambda x: x[0])
                        # maintain the monotonicity of T[cur_idx]
                        new_T = []
                        for k in range(1, len(T[dp_group_mesh][cur_idx])):
                            if len(new_T) != 0 and T[dp_group_mesh][cur_idx][
                                    k][1] >= new_T[-1][1]:
                                continue
                            new_T.append(T[dp_group_mesh][cur_idx][k])
                        T[dp_group_mesh][cur_idx] = new_T
                        # if(len(T[dp_group_mesh][cur_idx]) > 0):
                        #     print(f'[s, pp, tp, i]:{cur_idx}, len(T): {len(T[dp_group_mesh][cur_idx])}')

    best_time = float('inf')
    best_state = None
    micro_batch_num = autodist_config.update_freq
    for dp_group_mesh in dp_group_meshes:
        for stage_num in range(1, ngpus + 1):
            for pp_dev_num in range(stage_num, ngpus + 1):
                for tp_degree in range(1, pp_dev_num - stage_num + 1 + 1):
                    if tp_degree not in legal_tp_degrees:
                        continue

                    cur_idx = (stage_num, pp_dev_num, tp_degree, 0)
                    if cur_idx not in T[dp_group_mesh]:
                        continue
                    for t in range(0, len(T[dp_group_mesh][cur_idx])):
                        cur_time = T[dp_group_mesh][cur_idx][t][0] * (
                            micro_batch_num -
                            1) + T[dp_group_mesh][cur_idx][t][1]
                        logging.info(
                            f'mesh:{dp_group_mesh}, [s, pp, tp, i]:{cur_idx}, All_time: {cur_time}'
                        )
                        if best_time > cur_time:
                            best_time, best_state = cur_time, (dp_group_mesh,
                                                               cur_idx, t)

    _logger.info(
        f'best time/s: {best_time}, state (s, pp, tp, i): {best_state}')
    if best_state == None:
        raise RuntimeError('fail to find a valid pipeline plan')

    spmd_outs = []
    tp_list = []

    def build_answer(s, pp, tp, i, dp_group_mesh, idx):
        print(
            f's:{s} pp:{pp} tp:{tp} i:{i} dp_group_mesh:{dp_group_mesh} idx:{idx}'
        )
        _1, _2, prev_idx, _3, _4 = T[dp_group_mesh][(s, pp, tp, i)][idx]
        if prev_idx == None:
            tp_idx = (tp, s, i, model_graph.op_num - 1, dp_group_mesh)
        else:
            j_plus_1 = prev_idx[3]
            tp_idx = (tp, s, i, j_plus_1 - 1, dp_group_mesh)
        spmd_outs.append(tp_info[tp_idx])
        tp_list.append(tp)
        if prev_idx != None:
            build_answer(*prev_idx, dp_group_mesh, _3)

    build_answer(*best_state[1], best_state[0], best_state[2])

    tp_groups, indices = aggregate_to_groups(tp_list, best_state[0][1])

    print(
        f'tp_list:{tp_list} b:{best_state[0][1]} tp_groups:{tp_groups} indices:{indices}'
    )

    spmd_descs = [spmd_out.desc for spmd_out in spmd_outs]
    pp_desc = PipelineParallelDesc(spmd_descs, [], autodist_config.mesh_desc)
    stage_mems = [spmd_out.memory for spmd_out in spmd_outs]
    stage_all_times = [spmd_out.all_time for spmd_out in spmd_outs]
    stage_comp_times = [spmd_out.comp_time for spmd_out in spmd_outs]
    return PipelineSearchOutput(best_state[0], tp_groups, indices, pp_desc,
                                best_pp_cost, stage_mems, stage_all_times,
                                stage_comp_times)


# max+sum动态规划版本, dp_cost 单独计算
def calc_optimal_pp_plan_max_sum_dpsum(
    model_graph: ModelGraph, autodist_config: AutoDistConfig
) -> Dict[Tuple[int, Tuple, Tuple, Tuple], float]:
    '''
    目标：对于每个max_time，找到最小的sum_time + dp_cost.
    '''
    legal_tp_degrees = _calc_legal_tp_degrees(
        min(8, autodist_config.mesh_desc.col))
    ngpus = autodist_config.mesh_desc.ngpus
    tp_info = _compute_tp_info(model_graph, autodist_config, legal_tp_degrees)
    for key, value in tp_info.items():
        print(
            f"tp_degree, stage_num, start, end, dp_group_mesh:{key}, time:{value.all_time}"
        )
    pp_idxs = [0] + model_graph.get_pipeline_pivots() + [model_graph.op_num]
    dp_group_meshes = _gen_meshes(autodist_config)
    micro_batch_num = autodist_config.update_freq

    best_plan = None
    best_pp_cost = float("inf")

    # Dynamic programming table T
    # T is a dictionary where the key is a tuple (s, pp, tp, i)
    # s: stage number
    # pp: total number of devices used up to this stage
    # tp: number of devices used for the current stage
    # i: start operator index for the current stage
    # The value is a list [optimal_time, previous_state]
    # optimal_time: the optimal time for the given state
    # previous_state: the previous state (s-1, pp-tp, tp', j) that leads to the optimal time
    T = {}
    for dp_group_mesh in dp_group_meshes:
        T[dp_group_mesh] = {}
        print(f'dp_group_mesh:{dp_group_mesh}')
        for s in range(1, ngpus + 1):  # 第s个stage
            for pp in range(s, ngpus + 1):  # 目前一共用了pp个gpu
                for tp in range(1, pp - s + 1 + 1):  # 当前stage用了tp个gpu
                    if tp > dp_group_mesh[1] or tp not in legal_tp_degrees:
                        continue
                    for ii in range(len(pp_idxs) - 1 - 1, 0 - 1, -1):
                        i = pp_idxs[ii]
                        cur_idx = (s, pp, tp, i)
                        T[dp_group_mesh][cur_idx] = [
                        ]  # T中存的是Max = x的最小 的sum_time的一个单调队列
                        if tp == pp and s == 1:
                            tp_idx = (tp, s, i, model_graph.op_num - 1,
                                      dp_group_mesh)
                            if tp_idx in tp_info:
                                t1 = (tp_info[tp_idx].all_time -
                                      tp_info[tp_idx].weight_update_time / autodist_config.update_freq)
                                t2 = t1 + tp_info[tp_idx].weight_update_time 
                                T[dp_group_mesh][cur_idx].append(
                                    (t1, t2, None, -1,
                                     tp_info[tp_idx].desc.partition_descs))
                            # if(len(T[dp_group_mesh][cur_idx]) > 0):
                            #     print(f'[s, pp, tp, i]:{cur_idx}, len(T): {len(T[dp_group_mesh][cur_idx])}')
                            continue

                        for jj in range(len(pp_idxs) - 1 - 1, ii, -1):
                            j = pp_idxs[jj]
                            next_pp = pp - tp
                            for next_tp in range(1, next_pp - (s - 1) + 1 + 1):
                                if next_tp not in legal_tp_degrees:
                                    continue
                                prev_idx = (s - 1, next_pp, next_tp, j)
                                if prev_idx not in T[dp_group_mesh] or len(
                                        T[dp_group_mesh][prev_idx]) == 0:
                                    continue
                                tp_idx = (tp, s, i, j - 1, dp_group_mesh)
                                if tp_idx not in tp_info:
                                    continue

                                # merge T[prev_idx] into T[cur_idx]

                                for k in range(
                                        0, len(T[dp_group_mesh][prev_idx])):
                                    prev_t = T[dp_group_mesh][prev_idx][k]
                                    max_time = prev_t[0]
                                    sum_time = prev_t[1]

                                    dp_time = tp_info[tp_idx].weight_update_time / autodist_config.update_freq

                                    max_time = max(
                                        max_time,
                                        tp_info[tp_idx].all_time - dp_time)
                                    sum_time += tp_info[tp_idx].all_time - dp_time

                                    sum_time += tp_info[tp_idx].weight_update_time

                                    # comm_time = 0
                                    # cur_stage_desc = tp_info[
                                    #     tp_idx].desc.partition_descs
                                    # next_stage_desc = prev_t[4]

                                    # comm_time += model_graph.calc_pivot_comm_cost(
                                    #     list(cur_stage_desc.items()),
                                    #     list(next_stage_desc.items()), "intra")

                                    # sum_time += comm_time
                                    T[dp_group_mesh][cur_idx].append(
                                        (max_time, sum_time, prev_idx, k,
                                         tp_info[tp_idx].desc.partition_descs))
                        # sort T[cur_idx] by max_time
                        T[dp_group_mesh][cur_idx].sort(key=lambda x: x[0])
                        # maintain the monotonicity of T[cur_idx]
                        new_T = []
                        for k in range(1, len(T[dp_group_mesh][cur_idx])):
                            if len(new_T) != 0 and T[dp_group_mesh][cur_idx][
                                    k][1] >= new_T[-1][1]:
                                continue
                            new_T.append(T[dp_group_mesh][cur_idx][k])
                        T[dp_group_mesh][cur_idx] = new_T
                        # if(len(T[dp_group_mesh][cur_idx]) > 0):
                        #     print(f'[s, pp, tp, i]:{cur_idx}, len(T): {len(T[dp_group_mesh][cur_idx])}')

    best_time = float('inf')
    best_state = None
    micro_batch_num = autodist_config.update_freq
    for dp_group_mesh in dp_group_meshes:
        for stage_num in range(1, ngpus + 1):
            for pp_dev_num in range(stage_num, ngpus + 1):
                for tp_degree in range(1, pp_dev_num - stage_num + 1 + 1):
                    if tp_degree not in legal_tp_degrees:
                        continue

                    cur_idx = (stage_num, pp_dev_num, tp_degree, 0)
                    if cur_idx not in T[dp_group_mesh]:
                        continue
                    for t in range(0, len(T[dp_group_mesh][cur_idx])):
                        cur_time = T[dp_group_mesh][cur_idx][t][0] * (
                            micro_batch_num -
                            1) + T[dp_group_mesh][cur_idx][t][1]
                        logging.info(
                            f'mesh:{dp_group_mesh}, [s, pp, tp, i]:{cur_idx}, All_time: {cur_time}'
                        )
                        if best_time > cur_time:
                            best_time, best_state = cur_time, (dp_group_mesh,
                                                               cur_idx, t)

    _logger.info(
        f'best time/s: {best_time}, state (s, pp, tp, i): {best_state}')
    if best_state == None:
        raise RuntimeError('fail to find a valid pipeline plan')

    spmd_outs = []
    tp_list = []

    def build_answer(s, pp, tp, i, dp_group_mesh, idx):
        print(
            f's:{s} pp:{pp} tp:{tp} i:{i} dp_group_mesh:{dp_group_mesh} idx:{idx}'
        )
        _1, _2, prev_idx, _3, _4 = T[dp_group_mesh][(s, pp, tp, i)][idx]
        if prev_idx == None:
            tp_idx = (tp, s, i, model_graph.op_num - 1, dp_group_mesh)
        else:
            j_plus_1 = prev_idx[3]
            tp_idx = (tp, s, i, j_plus_1 - 1, dp_group_mesh)
        spmd_outs.append(tp_info[tp_idx])
        logging.info(
            f'stage weight_update_time:{micro_batch_num * tp_info[tp_idx].weight_update_time},',
            f'dp_size:{tp_info[tp_idx].dp_size/1024}GB')
        tp_list.append(tp)
        if prev_idx != None:
            build_answer(*prev_idx, dp_group_mesh, _3)

    build_answer(*best_state[1], best_state[0], best_state[2])

    tp_groups, indices = aggregate_to_groups(tp_list, best_state[0][1])

    print(
        f'tp_list:{tp_list} b:{best_state[0][1]} tp_groups:{tp_groups} indices:{indices}'
    )

    spmd_descs = [spmd_out.desc for spmd_out in spmd_outs]
    pp_desc = PipelineParallelDesc(spmd_descs, [], autodist_config.mesh_desc)
    stage_mems = [spmd_out.memory for spmd_out in spmd_outs]
    stage_all_times = [spmd_out.all_time for spmd_out in spmd_outs]
    stage_comp_times = [spmd_out.comp_time for spmd_out in spmd_outs]
    return PipelineSearchOutput(best_state[0], tp_groups, indices, pp_desc,
                                best_pp_cost, stage_mems, stage_all_times,
                                stage_comp_times)


#sum+max遍历版本，dp_cost单独算
def calc_optimal_pp_plan_max_dpsum(
    model_graph: ModelGraph, autodist_config: AutoDistConfig
) -> Dict[Tuple[int, Tuple, Tuple, Tuple], float]:
    legal_tp_degrees = _calc_legal_tp_degrees(
        min(8, autodist_config.mesh_desc.col))
    ngpus = autodist_config.mesh_desc.ngpus
    tp_info = _compute_tp_info(model_graph, autodist_config, legal_tp_degrees)
    for key, value in tp_info.items():
        print(
            f"tp_degree, stage_num, start, end, dp_group_mesh:{key}, time:{value.all_time}"
        )
    pp_idxs = [0] + model_graph.get_pipeline_pivots() + [model_graph.op_num]
    dp_group_meshes = _gen_meshes(autodist_config)
    micro_batch_num = autodist_config.update_freq

    best_plan = None
    best_pp_cost = float("inf")
    for stage_num in range(1, 5):
        tp_combinations = split_into_powers_of_two(ngpus, stage_num,
                                                   legal_tp_degrees)
        for tp_list in tp_combinations:
            staged_intervals_combinations = staging_combination(
                stage_num, pp_idxs)
            for staged_intervals in staged_intervals_combinations:
                for dp_group_mesh in dp_group_meshes:
                    if max(tp_list) > dp_group_mesh[1]:
                        continue

                    no_solution_flag = 0
                    stage_time_list = []
                    stage_dptime_list = []
                    spmd_outs = []
                    for stage_idx in range(stage_num):
                        tp = tp_list[stage_idx]
                        (start, end) = staged_intervals[stage_idx]
                        if (tp, stage_num - stage_idx, start, end,
                                dp_group_mesh) not in tp_info:
                            no_solution_flag = 1
                            break

                        dp_time = tp_info[
                            (tp, stage_num - stage_idx, start, end,
                             dp_group_mesh)].weight_update_time / 4
                        stage_time_list.append(
                            tp_info[tp, stage_num - stage_idx, start, end,
                                    dp_group_mesh].all_time - dp_time)

                        stage_dptime_list.append(dp_time)
                        spmd_outs.append(tp_info[(tp, stage_num - stage_idx,
                                                  start, end, dp_group_mesh)])

                    if no_solution_flag:
                        continue

                    max_stage_time = max(stage_time_list)
                    pp_comp_cost = max_stage_time * (
                        micro_batch_num -
                        1) + sum(stage_time_list) + sum(stage_dptime_list) * 4
                    # pp_comp_cost = max_stage_time * (micro_batch_num +
                    #                                  stage_num - 1) + sum(
                    #                                      stage_dptime_list)*4

                    tp_groups, indices = aggregate_to_groups(
                        tp_list, dp_group_mesh[1])
                    pp_comm_cost = 0
                    # pp_comm_cost = calc_pp_comm_cost(model_graph,
                    #                                  staged_intervals,
                    #                                  spmd_outs, indices)
                    print(
                        f"stage_num:{stage_num} max_stage_time:{max_stage_time} sum_time:{sum(stage_time_list)} pp_comp_cost:{pp_comp_cost} pp_comm_cost:{pp_comm_cost}"
                    )

                    total_pp_cost = pp_comm_cost + pp_comp_cost
                    # print(f'curr_cost:{total_pp_cost},best_cost:{best_pp_cost}')
                    if total_pp_cost < best_pp_cost:
                        best_pp_cost = total_pp_cost
                        best_plan = (stage_num, tp_list, staged_intervals,
                                     dp_group_mesh, tp_groups, indices,
                                     best_pp_cost)

    if best_plan == None:
        raise RuntimeError('fail to find a valid pipeline plan')
    else:
        (stage_num, tp_list, staged_intervals, dp_group_mesh, tp_groups,
         indices, best_pp_cost) = best_plan
        _logger.info(
            f'best time/s: {best_pp_cost}, plan: (stage_num = {stage_num}, tp_list = {tp_list}, '
            f'staged_intervals = {staged_intervals}, dp_group_mesh = {dp_group_mesh}), tp_groups:{tp_groups} ,indices:{indices}'
        )

    spmd_outs = []
    for stage_id in range(stage_num):
        start, end = staged_intervals[stage_id]
        tp = tp_list[stage_id]
        index = (tp, stage_num - stage_id, start, end, dp_group_mesh)
        spmd_outs.append(tp_info[index])
        logging.info(
            f'stage_id:{stage_id}, weight_update_time:{ tp_info[index].weight_update_time}, dp_size:{tp_info[index].dp_size/1024/1024/1024}GB'
        )

    spmd_descs = [spmd_out.desc for spmd_out in spmd_outs]
    pp_desc = PipelineParallelDesc(spmd_descs, [], autodist_config.mesh_desc)
    stage_mems = [spmd_out.memory for spmd_out in spmd_outs]
    stage_all_times = [spmd_out.all_time for spmd_out in spmd_outs]
    stage_comp_times = [spmd_out.comp_time for spmd_out in spmd_outs]

    return PipelineSearchOutput(dp_group_mesh, tp_groups, indices, pp_desc,
                                best_pp_cost, stage_mems, stage_all_times,
                                stage_comp_times)


#sum+max遍历版本，dp_cost单独算
def calc_optimal_pp_plan(
    model_graph: ModelGraph, autodist_config: AutoDistConfig
) -> Dict[Tuple[int, Tuple, Tuple, Tuple], float]:
    legal_tp_degrees = _calc_legal_tp_degrees(
        min(8, autodist_config.mesh_desc.col))
    ngpus = autodist_config.mesh_desc.ngpus
    tp_info = _compute_tp_info(model_graph, autodist_config, legal_tp_degrees)
    for key, value in tp_info.items():
        print(
            f"tp_degree, stage_num, start, end, dp_group_mesh:{key}, time:{value.all_time}"
        )
    pp_idxs = [0] + model_graph.get_pipeline_pivots() + [model_graph.op_num]
    dp_group_meshes = _gen_meshes(autodist_config)
    micro_batch_num = autodist_config.update_freq

    best_plan = None
    best_pp_cost = float("inf")
    for stage_num in range(1, 5):
        tp_combinations = split_into_powers_of_two(ngpus, stage_num,
                                                   legal_tp_degrees)
        for tp_list in tp_combinations:
            staged_intervals_combinations = staging_combination(
                stage_num, pp_idxs)
            for staged_intervals in staged_intervals_combinations:
                for dp_group_mesh in dp_group_meshes:
                    if max(tp_list) > dp_group_mesh[1]:
                        continue

                    no_solution_flag = 0
                    stage_time_list = []
                    stage_dptime_list = []
                    spmd_outs = []
                    for stage_idx in range(stage_num):
                        tp = tp_list[stage_idx]
                        (start, end) = staged_intervals[stage_idx]
                        if (tp, stage_num - stage_idx, start, end,
                                dp_group_mesh) not in tp_info:
                            no_solution_flag = 1
                            break

                        dp_time = tp_info[(tp, stage_num - stage_idx, start,
                                           end,
                                           dp_group_mesh)].weight_update_time

                        stage_time_list.append(
                            tp_info[tp, stage_num - stage_idx, start, end,
                                    dp_group_mesh].all_time - dp_time)
                        stage_dptime_list.append(dp_time *
                                                 autodist_config.update_freq)
                        spmd_outs.append(tp_info[(tp, stage_num - stage_idx,
                                                  start, end, dp_group_mesh)])

                    if no_solution_flag:
                        continue

                    max_stage_time = max(stage_time_list)
                    pp_comp_cost = max_stage_time * (
                        micro_batch_num -
                        1) + sum(stage_time_list) + sum(stage_dptime_list)

                    tp_groups, indices = aggregate_to_groups(
                        tp_list, dp_group_mesh[1])
                    pp_comm_cost = 0
                    # pp_comm_cost = calc_pp_comm_cost(model_graph,
                    #                                  staged_intervals,
                    #                                  spmd_outs, indices)
                    print(
                        f"stage_num:{stage_num} max_stage_time:{max_stage_time} sum_time:{sum(stage_time_list)} pp_comp_cost:{pp_comp_cost} pp_comm_cost:{pp_comm_cost}"
                    )

                    total_pp_cost = pp_comm_cost + pp_comp_cost
                    if total_pp_cost < best_pp_cost:
                        best_pp_cost = total_pp_cost
                        best_plan = (stage_num, tp_list, staged_intervals,
                                     dp_group_mesh, tp_groups, indices,
                                     best_pp_cost)

    if best_plan == None:
        raise RuntimeError('fail to find a valid pipeline plan')
    else:
        (stage_num, tp_list, staged_intervals, dp_group_mesh, tp_groups,
         indices, best_pp_cost) = best_plan
        _logger.info(
            f'best time/s: {best_pp_cost}, plan: (stage_num = {stage_num}, tp_list = {tp_list}, '
            f'staged_intervals = {staged_intervals}, dp_group_mesh = {dp_group_mesh}), tp_groups:{tp_groups} ,indices:{indices}'
        )

    spmd_outs = []
    for stage_id in range(stage_num):
        start, end = staged_intervals[stage_id]
        tp = tp_list[stage_id]
        index = (tp, stage_num - stage_id, start, end, dp_group_mesh)
        spmd_outs.append(tp_info[index])
        logging.info(
            f'stage_id:{stage_id}, weight_update_time:{micro_batch_num * tp_info[index].weight_update_time}, dp_size:{tp_info[index].dp_size/1024}GB'
        )

    spmd_descs = [spmd_out.desc for spmd_out in spmd_outs]
    pp_desc = PipelineParallelDesc(spmd_descs, [], autodist_config.mesh_desc)
    stage_mems = [spmd_out.memory for spmd_out in spmd_outs]
    stage_all_times = [spmd_out.all_time for spmd_out in spmd_outs]
    stage_comp_times = [spmd_out.comp_time for spmd_out in spmd_outs]

    return PipelineSearchOutput(dp_group_mesh, tp_groups, indices, pp_desc,
                                best_pp_cost, stage_mems, stage_all_times,
                                stage_comp_times)


def alter_partition(
        spmd_descs: List[TensorParallelDesc]) -> List[TensorParallelDesc]:
    '''
    alter node partition
    '''

    return spmd_descs


def calc_optimal_pp_plan_backup(
        model_graph: ModelGraph,
        autodist_config: AutoDistConfig) -> PipelineSearchOutput:
    '''
    T: dynamic programming table
    T[s, pp, tp, i, dp_mode]: optimal time of a pipeline state, where
        - s: stage number
        - pp: device number used for this state
        - tp: device number used for the 1st pipeline stage in this state
        - i: start operator index
    Transitions of T:
        - leaf state:
            tp == pp and s == 1: means current tp is the last one in the pipeline
            T[1, pp, tp, i, dp_mode] = tp[tp][s][i][end_op_idx][dp_mode]
        - non-leaf state:
            T[s, pp, tp, i, dp_mode] = min(max(T[s-1, pp-tp, tp', j+1, dp_mode], tp[tp][s][i][j][dp_mode]))
    store optimal path during dynamic programming in T as well
    '''
    # TODO: based on experience, tensor parallelism should <= 8
    legal_tp_degrees = _calc_legal_tp_degrees(
        min(8, autodist_config.mesh_desc.col))
    start_time = time.time()
    tp_info = _compute_tp_info(model_graph, autodist_config, legal_tp_degrees)
    total_time = time.time() - start_time
    print(f'tp_info求解时间:{total_time}')
    legal_tp_degrees = _calc_legal_tp_degrees(
        min(8, autodist_config.mesh_desc.col))
    ngpus = autodist_config.mesh_desc.ngpus
    pp_idxs = [0] + model_graph.get_pipeline_pivots() + [model_graph.op_num]

    T = {}
    for s in range(1, ngpus + 1):
        for pp in range(s, ngpus + 1):
            for tp in range(1, pp - s + 1 + 1):
                if tp not in legal_tp_degrees:
                    continue
                for ii in range(len(pp_idxs) - 1 - 1, 0 - 1, -1):
                    i = pp_idxs[ii]
                    cur_idx = (s, pp, tp, i)
                    T[cur_idx] = [float('inf'), (-1, -1, -1, -1, '')]

                    if tp == pp and s == 1:
                        tp_idx = (tp, s, i, model_graph.op_num - 1)
                        if tp_idx in tp_info:
                            T[cur_idx][0] = tp_info[tp_idx].all_time
                        continue

                    for jj in range(len(pp_idxs) - 1 - 1, ii, -1):
                        j = pp_idxs[jj]
                        next_pp = pp - tp
                        for next_tp in range(1, next_pp - (s - 1) + 1 + 1):
                            if next_tp not in legal_tp_degrees:
                                continue
                            prev_idx = (s - 1, next_pp, next_tp, j)
                            if prev_idx not in T:
                                continue
                            prev_tp_idx = (tp, s, i, j - 1)
                            if prev_tp_idx not in tp_info:
                                continue
                            lhs, _ = T[prev_idx]
                            rhs = tp_info[prev_tp_idx].all_time
                            val = max(lhs, rhs)
                            if T[cur_idx][0] > val:
                                T[cur_idx] = [val, prev_idx]

    # print(f'All plan for pp_solver :\n')
    # for key,value in T.items():
    #     if(value[1] != (-1,-1,-1,-1)):
    #         print(f'[s, pp, tp, i]:{key} time:{value}')
    best_time = float('inf')
    best_state = (-1, -1, -1, -1)
    micro_batch_num = autodist_config.update_freq
    for stage_num in range(1, ngpus + 1):
        for pp_dev_num in range(stage_num, ngpus + 1):
            for tp_degree in range(1, pp_dev_num - stage_num + 1 + 1):
                if tp_degree not in legal_tp_degrees:
                    continue

                cur_idx = (stage_num, pp_dev_num, tp_degree, 0)
                if cur_idx not in T:
                    continue
                cur_time = T[cur_idx][0] * (micro_batch_num - 1 + stage_num)
                logging.info(f'[s, pp, tp, i]:{cur_idx}, All_time: {cur_time}')
                if best_time > cur_time:
                    best_time, best_state = cur_time, cur_idx

    _logger.info(
        f'best time/s: {best_time}, state (s, pp, tp, i): {best_state}')
    if best_state == (-1, -1, -1, -1):
        raise RuntimeError('fail to find a valid pipeline plan')

    spmd_outs = []

    def build_answer(s, pp, tp, i):
        _, prev_idx = T[(s, pp, tp, i)]
        if prev_idx[0] == -1:
            tp_idx = (tp, s, i, pp_idxs[-1] - 1)
        else:
            j_plus_1 = prev_idx[3]
            tp_idx = (tp, s, i, j_plus_1 - 1)
        spmd_outs.append(tp_info[tp_idx])
        if prev_idx[0] != -1:
            build_answer(*prev_idx)

    build_answer(*best_state)

    spmd_descs = [spmd_out.desc for spmd_out in spmd_outs]
    pp_desc = PipelineParallelDesc(spmd_descs, [], autodist_config.mesh_desc)
    stage_mems = [spmd_out.memory for spmd_out in spmd_outs]
    stage_all_times = [spmd_out.all_time for spmd_out in spmd_outs]
    stage_comp_times = [spmd_out.comp_time for spmd_out in spmd_outs]
    return PipelineSearchOutput(pp_desc, best_time, stage_mems,
                                stage_all_times, stage_comp_times)
