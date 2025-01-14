from data_loader import TSP, SolutionTSP
from itertools import pairwise, groupby, permutations, product
from time import time

from assignment1.nearest_neighbor_at_any import nearest_neighbor_at_any_solve

INF = 10**18


def segment_joiner_solve(
        tsp: TSP,
        initial_segment_size: int = 5,
        # todo - should_use_local_search ???
) -> SolutionTSP:
    def calculate_segment_cost(segment: list[int]) -> int:
        return (sum(tsp.additional_costs[n] for n in segment) +
                sum(tsp.distances_matrix[s][d] for s, d in pairwise(segment)))

    # todo - this may not be optimal at all (?) - what if joining_segment should be in between?
    #   e.g. segments that form a "T" ??? -> (talk with mom) "1 common division place" (???)
    #   note - LS could (maybe) mitigate this problem
    def calculate_optimal_segments_join(base_segment: list[int], joining_segment: list[int]) -> tuple[list[int], int]:
        """
        Strips joining segment off nodes common with base segment and joins it at its start or end (whichever is better)
        """
        # 1. identify common parts
        # 2. divide segments into
        #   - separate parts from base
        #   - common parts
        #   - separate parts from joining
        # 3. arrange all parts into best configuration
        common_nodes = [n for n in joining_segment if n in base_segment]

        def divide_segment_by_common_nodes(segment: list[int]) -> list[list[int]]:
            result, temp = [], []
            for key, group in groupby(segment, lambda x: x in common_nodes):
                group_list = list(group)
                if key:  # Common part
                    if temp:  # Add the preceding non-common part
                        result.append(temp)
                        temp = []
                    result.append(group_list)
                else:  # Non-common part
                    temp.extend(group_list)
            if temp:  # Add remaining non-common part
                result.append(temp)

            return result

        # note - does reversing each one by one make sense?
        base_segment_parts = divide_segment_by_common_nodes(base_segment)
        joining_segment_parts = [p for p in divide_segment_by_common_nodes(joining_segment)
                                 if p[0] not in common_nodes]  # common parts come from base segment
        base_segment_parts_variants = [[p, p[::-1]] for p in base_segment_parts]
        joining_segment_parts_variants = [[p, p[::-1]] for p in joining_segment_parts]

        best_arrangement, best_arrangement_cost = [], INF
        for permutation in permutations(base_segment_parts_variants + joining_segment_parts_variants):
            for combination in product(*permutation):
                arrangement = [node for part in combination for node in part]
                arrangement_cost = tsp.calculate_total_objective_function(arrangement)
                if arrangement_cost < best_arrangement_cost:
                    best_arrangement, best_arrangement_cost = arrangement, arrangement_cost

        return best_arrangement, best_arrangement_cost

    # note - potentially redundant segments (same in nodes -> same in edges?)
    initial_segments = sorted([
        nearest_neighbor_at_any_solve(tsp=tsp, starting_node=n, desired_number_of_nodes=initial_segment_size).nodes
        for n in tsp.nodes
    ], key=lambda s: calculate_segment_cost(s))

    t0 = time()
    solution = initial_segments.pop()
    # fixme - better divide and conquer method
    while len(solution) < tsp.get_required_number_of_nodes_in_solution():
        print(f'{len(solution)}, time: {time() - t0}')
        solution, _ = calculate_optimal_segments_join(base_segment=solution, joining_segment=initial_segments.pop())

    def remove_worst_node(solution: list[int]) -> list[int]:
        nodes_with_neighbors = [solution[-1]] + solution + [solution[0]]
        worst_node, worst_node_change = -1, INF
        for i in range(1, len(nodes_with_neighbors)-1):
            start, removed, end = nodes_with_neighbors[i-1], nodes_with_neighbors[i], nodes_with_neighbors[i+1]
            node_removal_change = tsp.removal_changes[start][end][removed]
            if node_removal_change < worst_node_change:
                worst_node, worst_node_change = removed, node_removal_change
        solution.remove(worst_node)

        return solution

    while len(solution) > tsp.get_required_number_of_nodes_in_solution():
        solution = remove_worst_node(solution)

    return tsp.calculate_solution(solution)

    # note - go only by cost? (too greedy?)
    # todo - join by only considering "additional cost", since parts of segments are still similar
    #   (too much favorism of common nodes?) <- BUT this can actually be a good thing

    # todo - instead of joining one huge cycle, cluster many independently (and join them one by one in each step)?

    # note - if (for some reason) not enough nodes, nn_at_any starting from starting_nodes (of current solution) ?
    #   ^ impossible, because initial_segments contain all nodes (worst case - we go through all initial segments)


if __name__ == "__main__":
    tsp = TSP.load_tspa(data_folder='../data')
    t0 = time()
    solution = segment_joiner_solve(tsp)
    t1 = time()
    print(f'execution_time: {t1 - t0}')
    print(solution)
    tsp.visualize_solution(solution=solution, method_name='segment joiner')
