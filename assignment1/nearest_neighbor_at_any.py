from data_loader import TSP, SolutionTSP
from random import choice
from itertools import pairwise
from typing import Tuple
from time import time


def nearest_neighbor_at_any_solve(
        tsp: TSP, starting_node: int = None,
        starting_nodes: list[int] | None = None,
        except_edges: set[tuple[int, int]] | None = None
) -> SolutionTSP:
    if starting_nodes:
        chosen_nodes = starting_nodes
    elif starting_node is None:
        chosen_nodes = [choice(tsp.nodes)]
    else:
        chosen_nodes = [starting_node]

    if except_edges is None:
        except_edges = set()

    remaining_nodes = tsp.get_nodes(without_nodes=chosen_nodes)

    # todo - into TSP? (pass remaining nodes?)
    # note - could be done better by avoiding unnecessary calculations - only need to update
    # actually touching the currently removed node(?)
    def get_smallest_move_cost_node_directly_from(node: int) -> Tuple[int, int]:
        remaining_nodes_move_costs_from_node = [
            tsp.total_move_costs[node][destination_node]
            for destination_node in remaining_nodes
        ]
        smallest_move_cost = min(remaining_nodes_move_costs_from_node)
        # first occurrence in case of tie
        smallest_move_cost_node_idx = remaining_nodes_move_costs_from_node.index(smallest_move_cost)

        return remaining_nodes[smallest_move_cost_node_idx], smallest_move_cost

    # todo - into TSP? (pass remaining nodes?)
    # note - could be done better by avoiding unnecessary calculations - only need to update
    # actually touching the currently removed node(?)
    def get_smallest_move_cost_node_between(start_node: int, end_node: int):
        remaining_nodes_move_costs_between = [
            tsp.insertion_costs[start_node][end_node][inserted_node]
            for inserted_node in remaining_nodes
        ]
        smallest_move_cost = min(remaining_nodes_move_costs_between)
        # first occurrence in case of tie
        smallest_move_cost_node_idx = remaining_nodes_move_costs_between.index(smallest_move_cost)

        return remaining_nodes[smallest_move_cost_node_idx], smallest_move_cost

    required_number_of_nodes = tsp.get_required_number_of_nodes_in_solution()
    while len(chosen_nodes) < required_number_of_nodes and remaining_nodes:
        best_from_start = (0, get_smallest_move_cost_node_directly_from(chosen_nodes[0]))
        best_from_end = (-1, get_smallest_move_cost_node_directly_from(chosen_nodes[-1]))
        best_choices_between = [
            ((start, end), get_smallest_move_cost_node_between(start, end))
            for start, end in pairwise(chosen_nodes)
            if (start, end) not in except_edges
        ]

        best_place, (next_node, _) = min([best_from_start, best_from_end] + best_choices_between, key=lambda x: x[1][1])
        if best_place == 0:
            chosen_nodes.insert(0, next_node)
        elif best_place == -1:
            chosen_nodes.append(next_node)
        else:
            _, end = best_place
            chosen_nodes.insert(chosen_nodes.index(end), next_node)
        remaining_nodes.remove(next_node)

    return tsp.calculate_solution(chosen_nodes)


if __name__ == "__main__":
    tsp = TSP.load_tspa(data_folder='../data')
    t0 = time()
    solution = nearest_neighbor_at_any_solve(tsp)
    t1 = time()
    print(f'execution_time: {t1 - t0}')
    print(solution)
    print(solution.nodes_in_excel_format())
