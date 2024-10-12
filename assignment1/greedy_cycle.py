from data_loader import TSP, SolutionTSP
from random import sample
from itertools import pairwise
from typing import List
from time import time


def greedy_cycle_solve(tsp: TSP, starting_nodes: List[int] = None) -> SolutionTSP:
    if starting_nodes is None:
        starting_nodes = sample(tsp.nodes, 2)

    chosen_nodes = starting_nodes
    remaining_nodes = tsp.get_nodes(without_nodes=chosen_nodes)

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
        best_choices_between = [
            ((start, end), get_smallest_move_cost_node_between(start, end))
            for start, end in pairwise(chosen_nodes + [chosen_nodes[0]])  # also pairwise between start and end
        ]

        (_, end), (next_node, _) = min(best_choices_between, key=lambda x: x[1][1])
        chosen_nodes.insert(chosen_nodes.index(end), next_node)
        remaining_nodes.remove(next_node)

    return tsp.calculate_solution(chosen_nodes)


if __name__ == "__main__":
    tsp = TSP.load_tspa(data_folder='../data')
    t0 = time()
    solution = greedy_cycle_solve(tsp)
    t1 = time()
    print(f'execution_time: {t1 - t0}')
    print(solution)
    print(solution.nodes_in_excel_format())
