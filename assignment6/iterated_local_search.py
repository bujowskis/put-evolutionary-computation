from data_loader import TSP, SolutionTSP
from random import seed, sample, shuffle, choice
from time import time

from assignment3.local_search_types import (
    LocalSearchType,
    StartingSolutionType,
    IntraRouteMovesType,
)
from assignment5.local_search_with_deltas import local_search_with_deltas_solve
from assignment5.local_search_no_deltas import local_search_no_deltas_solve


def iterated_local_search_solve(
        tsp: TSP,
        local_search_type: LocalSearchType = LocalSearchType.STEEPEST,
        starting_solution_type: StartingSolutionType = StartingSolutionType.RANDOM,
        intra_route_move_type: IntraRouteMovesType = IntraRouteMovesType.TWO_EDGES,
        starting_node: int = None,  # in case of RANDOM start - initial seed,
        timeout_after_seconds: int = 47,
) -> tuple[SolutionTSP, int]:
    number_of_local_search_runs = 0

    if starting_node and starting_solution_type == StartingSolutionType.RANDOM:
        seed(starting_node)

    solution, _ = local_search_with_deltas_solve(
        tsp=tsp,
        local_search_type=local_search_type,
        starting_solution_type=starting_solution_type,
        intra_route_move_type=intra_route_move_type)
    number_of_local_search_runs += 1

    # todo - EA crossover-inspired?
    # solution2, _ = local_search_with_deltas_solve(tsp=tsp,
    #                                               local_search_type=local_search_type,
    #                                               starting_solution_type=starting_solution_type,
    #                                               intra_route_move_type=intra_route_move_type)
    # number_of_local_search_runs += 1

    def perturbate_crossover(parent1: SolutionTSP, parent2: SolutionTSP,
                             splits: int = 5) -> tuple[SolutionTSP, SolutionTSP]:
        split_points = sorted([parent1.nodes.index(node)
                               for node in sample(parent1.nodes, splits)])
        previous_split_point = 0
        splices_parent1, splices_parent2 = [], []
        for split_point in split_points:
            splices_parent1.append(parent1.nodes[previous_split_point:split_point])
            splices_parent2.append(parent2.nodes[previous_split_point:split_point])
            previous_split_point = split_point
        splices_parent1.append(parent1.nodes[previous_split_point:])
        splices_parent2.append(parent2.nodes[previous_split_point:])

        child1, child2 = [], []
        for i, (splice_parent1, splice_parent2) in enumerate(zip(splices_parent1, splices_parent2)):
            if i % 2 == 0:
                child1.extend(splice_parent1)
                child2.extend(splice_parent2)
            else:
                child1.extend(splice_parent2)
                child2.extend(splice_parent1)
        print(f'{child1}, {child2}')

        return tsp.calculate_solution(child1), tsp.calculate_solution(child2)

    def perturbate_node_exchanges(input_solution: SolutionTSP, exchanges: int = 30) -> SolutionTSP:
        # internal(?) node exchanges?
        # any node exchanges?
        exchanges *= 2
        nodes = input_solution.nodes
        perturbation_nodes = sample(tsp.nodes, exchanges)
        for i in list(range(exchanges))[::2]:
            node1, node2 = perturbation_nodes[i], perturbation_nodes[i + 1]
            if node1 in nodes and node2 in nodes:
                place1, place2 = nodes.index(node1), nodes.index(node2)
                nodes[place1] = node2
                nodes[place2] = node1
            elif node1 in nodes:
                nodes[nodes.index(node1)] = node2
            elif node2 in nodes:
                nodes[nodes.index(node2)] = node1
        perturbed_solution = tsp.calculate_solution(nodes)

        return perturbed_solution

    def perturbate_reverse_segments(input_solution: SolutionTSP, reversed_segments: int = 4) -> SolutionTSP:
        nodes = input_solution.nodes
        split_points = sorted([nodes.index(node)
                               for node in sample(nodes, reversed_segments*2)])
        # print(split_points)
        previous_split_point = 0
        splices = []
        for split_point in split_points:
            splices.append(nodes[previous_split_point:split_point])
            previous_split_point = split_point
        splices.append(nodes[previous_split_point:])

        result = []
        for i, splice in enumerate(splices):
            result.extend(splice if i % 2 == 0 else splice[::-1])
        # print(f'{len(result)}, {result}')

        return tsp.calculate_solution(result)

    def perturbate_shuffle_segments(input_solution: SolutionTSP, segments_no: int = 5):
        nodes = input_solution.nodes
        split_points = sorted([nodes.index(node)
                               for node in sample(nodes, segments_no)])
        # print(split_points)
        previous_split_point = 0
        segments = []
        for split_point in split_points:
            segments.append(nodes[previous_split_point:split_point])
            previous_split_point = split_point
        segments.append(nodes[previous_split_point:])

        result = []
        shuffle(segments)
        for segment in segments:
            result.extend(segment)
        # print(f'{len(result)}, {result}')

        return tsp.calculate_solution(result)

    def perturbate_segment_exchange(input_solution: SolutionTSP, segment_size: int = 5, exchanges: int = 5):
        nodes = input_solution.nodes
        for _ in range(exchanges):
            segment1_start = nodes.index(choice(nodes))
            segment1 = {(segment1_start + i) % len(nodes): nodes[(segment1_start + i) % len(nodes)]
                        for i in range(segment_size)}
            leading_overlap = {i if i >= 0 else i + len(nodes)
                               for i in [segment1_start - 1 - j for j in range(segment_size - 1)]}
            non_overlapping_choices = list(set(range(len(nodes))) - set(segment1) - leading_overlap)
            segment2_start = choice(non_overlapping_choices)
            segment2 = {(segment2_start + i) % len(nodes): nodes[(segment2_start + i) % len(nodes)]
                        for i in range(segment_size)}
            for (segment1_idx, segment1_node), (segment2_idx, segment2_node) in zip(segment1.items(), segment2.items()):
                nodes[segment1_idx] = segment2_node
                nodes[segment2_idx] = segment1_node

        return tsp.calculate_solution(nodes)

    def perturbate_exchange_low_cost_nodes(input_solution: SolutionTSP, exchanges: int = 5):
        nodes_in_cycle = input_solution.nodes
        nodes_outside_cycle = list(set(tsp.nodes) - set(nodes_in_cycle))
        highest_cost_nodes_in_cycle = sorted(nodes_in_cycle, key=lambda n: tsp.additional_costs[n],
                                             reverse=True)
        lowest_cost_nodes_outside_cycle = sorted(nodes_outside_cycle, key=lambda n: tsp.additional_costs[n])

        for i in range(exchanges):
            exchange_cycle_index = nodes_in_cycle.index(highest_cost_nodes_in_cycle[i])
            nodes_in_cycle[exchange_cycle_index] = lowest_cost_nodes_outside_cycle[i]

        return tsp.calculate_solution(nodes_in_cycle)

    def perturbate_exchange_low_nn_cost(input_solution: SolutionTSP, exchanges: int = 5, neighborhood_size: int = 10):
        # todo - add "should be preserved?" logic
        nodes_in_cycle = input_solution.nodes
        nodes_outside_cycle = list(set(tsp.nodes) - set(nodes_in_cycle))

        # note - these could be calculated beforehand, once
        closest_nodes = {
            node: sorted([destination for destination in tsp.nodes
                          if destination is not node],
                         key=lambda destination: tsp.total_move_costs[node][destination])
            for node in tsp.nodes
        }
        neighborhood_costs = {
            node: sum(closest_nodes_list[:neighborhood_size])
            for node, closest_nodes_list in closest_nodes.items()
        }

        highest_cost_nodes_in_cycle = sorted(nodes_in_cycle, key=lambda n: neighborhood_costs[n],
                                             reverse=True)
        lowest_cost_nodes_outside_cycle = sorted(nodes_outside_cycle, key=lambda n: neighborhood_costs[n])

        for i in range(exchanges):
            exchange_cycle_index = nodes_in_cycle.index(highest_cost_nodes_in_cycle[i])
            nodes_in_cycle[exchange_cycle_index] = lowest_cost_nodes_outside_cycle[i]

        return tsp.calculate_solution(nodes_in_cycle)

    def perturbate_random_node_exchanges(exchanges: int = 15) -> SolutionTSP:
        # internal(?) node exchanges?
        # any node exchanges?
        exchanges *= 2
        nodes = solution.nodes
        perturbation_nodes = sample(tsp.nodes, exchanges)
        # print(perturbation_nodes)
        for i in list(range(exchanges))[::2]:
            node1, node2 = perturbation_nodes[i], perturbation_nodes[i + 1]
            if node1 in nodes and node2 in nodes:
                place1, place2 = nodes.index(node1), nodes.index(node2)
                nodes[place1] = node2
                nodes[place2] = node1
            elif node1 in nodes:
                nodes[nodes.index(node1)] = node2
            elif node2 in nodes:
                nodes[nodes.index(node2)] = node1
        perturbed_solution = tsp.calculate_solution(nodes)

        return perturbed_solution

    # todo - crossover, then mutate?
    # while True:
    #     new_solution1, new_solution2 = perturbate_crossover(solution, solution2)
    #     # new_solution1, new_solution2 = perturbate_node_exchanges(new_solution1), perturbate_node_exchanges(new_solution2)
    #     new_solution1, _ = local_search_with_deltas_solve(
    #         tsp=tsp,
    #         local_search_type=local_search_type,
    #         starting_solution_type=starting_solution_type,
    #         intra_route_move_type=intra_route_move_type,
    #         starting_solution=new_solution1,
    #     )
    #     number_of_local_search_runs += 1
    #     new_solution2, _ = local_search_with_deltas_solve(
    #         tsp=tsp,
    #         local_search_type=local_search_type,
    #         starting_solution_type=starting_solution_type,
    #         intra_route_move_type=intra_route_move_type,
    #         starting_solution=new_solution2,
    #     )
    #     number_of_local_search_runs += 1
    #     best_so_far = list(sorted([solution, solution2], key=lambda s: s.objective_function))
    #     new_solutions = list(sorted([new_solution1, new_solution2], key=lambda s: s.objective_function))
    #     if new_solutions[0].objective_function < best_so_far[0].objective_function:
    #         solution = new_solutions[0]
    #         if new_solutions[1].objective_function < best_so_far[1].objective_function:
    #             solution2 = new_solutions[1]
    #

    t0 = time()
    while True:
        # new_solution = perturbate_random_node_exchanges(exchanges=10)
        # new_solution = perturbate_exchange_low_nn_cost(solution, exchanges=5, neighborhood_size=5)
        # new_solution = perturbate_exchange_low_cost_nodes(solution, exchanges=5)
        new_solution = perturbate_segment_exchange(solution, segment_size=3, exchanges=5)  # <- just this for TSPB
        # new_solution = perturbate_shuffle_segments(solution, segments_no=50)  # <- just this for TSPA
        # new_solution = perturbate_shuffle_segments(new_solution, segments_no=10)  # <- just this for TSPA
        # new_solution = perturbate_shuffle_segments(new_solution, segments_no=5)
        # new_solution = perturbate_node_exchanges(solution, exchanges=10)
        # new_solution = perturbate_node_exchanges(new_solution, exchanges=10)
        # new_solution = perturbate_reverse_segments(new_solution, reversed_segments=3)
        # print(new_solution.nodes)
        new_solution, _ = local_search_with_deltas_solve(
            tsp=tsp,
            local_search_type=local_search_type,
            starting_solution_type=starting_solution_type,
            intra_route_move_type=intra_route_move_type,
            starting_solution=new_solution,
        )
        number_of_local_search_runs += 1
        # print(f'{new_solution.objective_function} < {solution.objective_function}')
        if new_solution.objective_function < solution.objective_function:
            solution = new_solution

        if time() - t0 > timeout_after_seconds:
            break

    return solution, number_of_local_search_runs
    # return best_so_far[0], number_of_local_search_runs
