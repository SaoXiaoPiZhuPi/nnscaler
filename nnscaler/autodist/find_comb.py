import itertools
from typing import List, Dict, Tuple
import torch
def split_into_powers_of_two(N, s):
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
        powers_of_two.append(power)
        power *= 2

    # Recursive function to find all valid combinations
    def find_combinations(target, start, count):
        if count == s:  # If we have already selected s numbers
            if target == 0:  # If the target is 0, it means the combination is valid
                return [[]]  # Return an empty list to indicate the combination is complete
            else:
                return []  # If the target is not 0, it's an invalid combination

        if target <= 0:  # If the target is negative or we haven't selected enough numbers yet
            return []

        combinations = []
        for i in range(start, len(powers_of_two)):
            power = powers_of_two[i]
            if power > target:  # If the current power exceeds the remaining target, skip it
                continue
            # Recursively find combinations by including the current power
            for combo in find_combinations(target - power, i, count + 1):
                combinations.append([power] + combo)  # Add the current power to the combination
        return combinations

    # Generate all valid combinations that sum up to N and contain exactly s elements
    valid_combinations = find_combinations(N, 0, 0)
    
    # Sort each combination and generate all unique permutations
    all_permutations = []
    for combo in valid_combinations:
        sorted_combo = sorted(combo)  # Sort the combination to ensure consistent permutations
        permutations = set(itertools.permutations(sorted_combo))  # Generate unique permutations
        all_permutations.extend(permutations)  # Add all unique permutations to the result list
    
    return all_permutations

def staging_combination(n, s):
    """
    This function returns all possible ways to stage a list of integers from 0 to n
    into s segments. Each segment is represented by its starting and ending index.

    Parameters:
    n (int): The last number in the list (0 to n).
    s (int): The number of segments to partition the list into.

    Returns:
    list: A list of tuples, each containing the start and end index of a segment.
    """
    
    # Helper function to recursively find all valid partitions
    def staging_helper(start, remaining_segments, current_stage):
        # Base case: if there are no remaining segments to partition
        if remaining_segments == 1:
            current_stage.append((start, n-1))  # Last segment from start to n
            result.append(current_stage)  # Add the current partition to result
            return
        
        # Try all possible end points for the current segment
        for end in range(start+1, n - remaining_segments + 1):
            # For each valid partition, recursively partition the remaining numbers
            staging_helper(end, remaining_segments - 1, current_stage + [(start, end)])

    result = []
    staging_helper(0, s, [])
    
    return result

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

    # 处理剩余元素（可选回填逻辑）
    remaining = []
    remaining_indices = []
    for num, idx in original_order:
        if idx not in [i for group in indices for i in group]:
            remaining.append(num)
            remaining_indices.append(idx)
    
    # 尝试回填剩余元素（更复杂的算法需在此扩展）
    if remaining:
        print(f"Warning: Remaining elements {remaining} could not be grouped.")
    
    return groups, indices

def _gen_meshes(plan_ngpus) -> List[tuple]:
    '''
    generate all possible meshes (a,b)
    a * b = cfg.plan_ngpus 
    a means uses a servers, b means use b gpus in each server.
    for example:
        plan_ngpus = 8, the runtime_ngus = 4 * 8 = 32
        we need to return meshes:
        (1,8) (2,4) (4,2)
    '''
    plan_ngpus = 16
    ngpus_per_node = 8
    nnodes = 16 // ngpus_per_node
    meshes = []
    for a in range(1, nnodes + 1):
        if plan_ngpus % a == 0:
            b = plan_ngpus // a
            if b <= ngpus_per_node:
                meshes.append((a, b))

    # return [(1,8)]
    return meshes

# # 示例
# lst = [4, 4]
# b = 4
# groups, indices = aggregate_to_groups(lst, b)

# # 打印每组的和以及对应的原始索引位置
# for group, index in zip(groups, indices):
#     print(f"Group: {group}, Indices: {index}")
print(aggregate_to_groups([2,2,2,2,8,8,4,4], 8))
