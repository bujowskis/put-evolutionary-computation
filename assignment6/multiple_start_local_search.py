from data_loader import TSP, SolutionTSP
from random import seed

from assignment3.local_search_types import (
    LocalSearchType,
    StartingSolutionType,
    IntraRouteMovesType,
)
from assignment5.local_search_with_deltas import local_search_with_deltas_solve


def multiple_start_local_search_solve(
        tsp: TSP,
        number_of_starts: int = 200,
        local_search_type: LocalSearchType = LocalSearchType.STEEPEST,
        starting_solution_type: StartingSolutionType = StartingSolutionType.RANDOM,
        intra_route_move_type: IntraRouteMovesType = IntraRouteMovesType.TWO_EDGES,
        starting_node: int = None,  # in case of RANDOM start - initial seed
) -> tuple[SolutionTSP, dict]:
    if starting_node and starting_solution_type == StartingSolutionType.RANDOM:
        seed(starting_node)
    solutions: list[tuple[SolutionTSP, dict]] = [
        local_search_with_deltas_solve(tsp=tsp,
                                       local_search_type=local_search_type,
                                       starting_solution_type=starting_solution_type,
                                       intra_route_move_type=intra_route_move_type)
        for _ in range(number_of_starts)
    ]

    return min(solutions, key=lambda solution: solution[0].objective_function)
