# todo - nn_at_any creates full solution (all nodes) -> LS -> remove nodes greedily until desired length
from data_loader import TSP, SolutionTSP
from random import shuffle
from time import time

from assignment3.local_search_types import (
    LocalSearchType,
    IntraRouteMovesType,
)
from assignment5.local_search_with_deltas import local_search_with_deltas_solve
from assignment5.local_search_no_deltas import local_search_no_deltas_solve

INF = 10**18


# todo - ls (but keep same nodes)???
def cycle_stripper_long_running_solve(
        tsp: TSP,
        ls_interval: int = 5,
        should_run_ls_at_end: bool = True,
        timeout_after_seconds: int = 47,  # average running time of MSLS run
) -> tuple[SolutionTSP, int]:
    best_solution, best_objective = None, INF

    number_of_main_loop_runs: int = 0
    t0 = time()
    while time() - t0 < timeout_after_seconds:
        number_of_main_loop_runs += 1
        nodes = tsp.nodes.copy()
        shuffle(nodes)
        solution, _ = local_search_with_deltas_solve(
            tsp=tsp,
            local_search_type=LocalSearchType.STEEPEST,
            intra_route_move_type=IntraRouteMovesType.TWO_EDGES,
            starting_solution=tsp.calculate_solution(nodes),
        )
        nodes = solution.nodes

        def remove_worst_node(nodes: list[int]) -> list[int]:
            nodes_with_neighbors = [nodes[-1]] + nodes + [nodes[0]]
            worst_node, worst_node_change = -1, INF
            options = dict()
            for i in range(1, len(nodes_with_neighbors)-1):
                start, removed, end = nodes_with_neighbors[i-1], nodes_with_neighbors[i], nodes_with_neighbors[i+1]
                node_removal_change = tsp.removal_changes[start][end][removed]
                options[removed] = int(node_removal_change)
                if node_removal_change < worst_node_change:
                    worst_node, worst_node_change = removed, node_removal_change
            nodes.remove(worst_node)

            return nodes

        for i in range(len(nodes) - tsp.get_required_number_of_nodes_in_solution()):
            nodes = remove_worst_node(nodes)
            if ls_interval > 0 and i % ls_interval == 0:
                solution, _ = local_search_with_deltas_solve(
                    tsp=tsp,
                    local_search_type=LocalSearchType.STEEPEST,
                    intra_route_move_type=IntraRouteMovesType.TWO_EDGES,
                    starting_solution=tsp.calculate_solution(nodes)
                )
                nodes = solution.nodes

        if should_run_ls_at_end:
            solution, _ = local_search_with_deltas_solve(
                tsp=tsp,
                local_search_type=LocalSearchType.STEEPEST,
                intra_route_move_type=IntraRouteMovesType.TWO_EDGES,
                starting_solution=tsp.calculate_solution(nodes),
            )
        else:
            solution = tsp.calculate_solution(nodes)

        if solution.objective_function < best_objective:
            best_solution, best_objective = solution, best_objective

    return best_solution, number_of_main_loop_runs


if __name__ == "__main__":
    tsp = TSP.load_tspa(data_folder='../data')
    t0 = time()
    solution, _ = cycle_stripper_long_running_solve(tsp, ls_interval=-1, should_run_ls_at_end=True)
    t1 = time()
    print(f'execution_time: {t1 - t0}')
    print(len(solution.nodes))
    print(solution)
    tsp.visualize_solution(solution=solution, method_name='cycle stripper long running')
