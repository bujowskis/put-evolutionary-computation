import copy
from math import ceil

from data_loader import TSP, SolutionTSP
from random import seed
from numpy.random import choice
from time import time

from assignment3.local_search_types import (
    LocalSearchType,
    StartingSolutionType,
    IntraRouteMovesType,
)
# from assignment1.random_solution import random_solve
from assignment1.nearest_neighbor_at_any import nearest_neighbor_at_any_solve
from assignment5.local_search_with_deltas import local_search_with_deltas_solve


def large_scale_neighborhood_search_solve(
        tsp: TSP,
        local_search_type: LocalSearchType = LocalSearchType.STEEPEST,
        starting_solution_type: StartingSolutionType = StartingSolutionType.RANDOM,
        intra_route_move_type: IntraRouteMovesType = IntraRouteMovesType.TWO_EDGES,
        starting_node: int = None,  # in case of RANDOM start - initial seed,
        timeout_after_seconds: int = 47,  # average running time of MSLS run
        should_use_local_search: bool = False,
) -> tuple[SolutionTSP, int]:
    number_of_main_loop_runs = 0

    if starting_node and starting_solution_type == StartingSolutionType.RANDOM:
        seed(starting_node)

    # if should_use_local_search:
    #     solution, _ = local_search_with_deltas_solve(
    #         tsp=tsp,
    #         local_search_type=local_search_type,
    #         starting_solution_type=starting_solution_type,
    #         intra_route_move_type=intra_route_move_type,
    #         starting_node=starting_node,
    #     )
    # else:
    #     solution = random_solve(tsp=tsp, initial_seed=starting_node)
    solution, _ = local_search_with_deltas_solve(
        tsp=tsp,
        local_search_type=local_search_type,
        starting_solution_type=starting_solution_type,
        intra_route_move_type=intra_route_move_type,
        starting_node=starting_node,
    )

    percentage_destroyed: int = 25
    nodes_to_destroy = ceil(len(solution.nodes) * percentage_destroyed / 100)
    neighborhood_size = 10
    # neighborhood costs are respectively, weights for picking the given node for destruction
    costs_to_all_nodes = {
        node: [
            tsp.additional_costs[node] + tsp.distances_matrix[node][destination] + tsp.additional_costs[destination]
            for destination in tsp.nodes
            if destination is not node
        ]
        for node in tsp.nodes
    }
    neighborhood_costs = {
        node: sum(sorted(costs)[:neighborhood_size])
        for node, costs in costs_to_all_nodes.items()
    }

    # neighborhood_costs = [
    #     sum(sorted([
    #         tsp.additional_costs[node] + tsp.distances_matrix[node][destination] + tsp.additional_costs[destination]
    #         for destination in tsp.nodes
    #         if destination is not node
    #     ]))[:neighborhood_size]
    #     for node in tsp.nodes
    # ]

    def destroy(input_solution: SolutionTSP) -> list[int]:
        nodes = input_solution.nodes.copy()
        nodes_costs = [neighborhood_costs[node] for node in nodes]
        total_cost = sum(nodes_costs)
        nodes_probabilities = [node_cost / total_cost for node_cost in nodes_costs]
        chosen_nodes = choice(nodes, size=nodes_to_destroy, p=nodes_probabilities, replace=False)
        # print(f'{nodes}, {len(nodes)}, {len(set(nodes))}')
        # print(f'{chosen_nodes}, {len(chosen_nodes)}, {len(set(chosen_nodes))}')
        for node in chosen_nodes:
            nodes.remove(node)

        return nodes

    def repair(destroyed_solution_nodes: list[int]) -> SolutionTSP:
        return nearest_neighbor_at_any_solve(
            tsp=tsp,
            starting_nodes=destroyed_solution_nodes
        )

    t0 = time()
    while True:
        number_of_main_loop_runs += 1
        destroyed_solution_nodes = destroy(solution)
        repaired_solution = repair(destroyed_solution_nodes)

        if should_use_local_search:
            repaired_solution, _ = local_search_with_deltas_solve(
                tsp=tsp,
                local_search_type=local_search_type,
                starting_solution_type=starting_solution_type,
                intra_route_move_type=intra_route_move_type,
                starting_solution=repaired_solution,
            )

        if repaired_solution.objective_function < solution.objective_function:
            solution = copy.deepcopy(repaired_solution)

        if time() - t0 > timeout_after_seconds:
            return solution, number_of_main_loop_runs
