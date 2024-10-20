from data_loader import TSP, SolutionTSP
from random import choice
from itertools import pairwise
from time import time
from typing import List


INF = 10**18


def greedy_2_regret_weighted_objective_solve(tsp: TSP, starting_node: int = None) -> SolutionTSP:
    if starting_node is None:
        starting_node = choice(tsp.nodes)

    chosen_nodes = [starting_node]
    remaining_nodes = tsp.get_nodes(without_nodes=chosen_nodes)

    def get_smallest_move_cost_node_between(start_node: int, end_node: int, current_remaining_nodes: List):
        remaining_nodes_move_costs_between = [
            tsp.insertion_costs[start_node][end_node][inserted_node]
            for inserted_node in current_remaining_nodes
        ]
        smallest_move_cost = min(remaining_nodes_move_costs_between)
        # first occurrence in case of tie
        smallest_move_cost_node_idx = remaining_nodes_move_costs_between.index(smallest_move_cost)

        return current_remaining_nodes[smallest_move_cost_node_idx], smallest_move_cost

    required_number_of_nodes = tsp.get_required_number_of_nodes_in_solution()
    while len(chosen_nodes) < required_number_of_nodes and remaining_nodes:
        best_choices_1_ahead_between = {
            ((start, end), get_smallest_move_cost_node_between(start, end, remaining_nodes))
            for start, end in pairwise(chosen_nodes + [chosen_nodes[0]])  # also pairwise between start and end
        }

        best_choice_2_ahead_between = ((-1, -1), (-1, INF))
        for (start, end), (next_node, next_node_cost) in best_choices_1_ahead_between:
            chosen_nodes_after_1_choice = chosen_nodes.copy()
            chosen_nodes_after_1_choice.insert(chosen_nodes_after_1_choice.index(end), next_node)
            remaining_nodes_after_1_choice = remaining_nodes.copy()
            remaining_nodes_after_1_choice.remove(next_node)

            best_choice_2_ahead_between_from_current = ((-1, -1), (-1, INF))
            for start2, end2 in pairwise(chosen_nodes_after_1_choice + [chosen_nodes_after_1_choice[0]]):
                smallest_move_cost_node_from_current, smallest_move_cost_from_current =\
                    get_smallest_move_cost_node_between(start2, end2, remaining_nodes_after_1_choice)
                smallest_move_cost_from_current += next_node_cost
                # + weighted sum part
                smallest_move_cost_from_current += next_node_cost
                smallest_move_cost_from_current /= 2

                if smallest_move_cost_from_current < best_choice_2_ahead_between_from_current[1][1]:
                    best_choice_2_ahead_between_from_current =\
                        ((start2, end2), (smallest_move_cost_node_from_current, smallest_move_cost_from_current))

            if best_choice_2_ahead_between_from_current[1][1] < best_choice_2_ahead_between[1][1]:
                best_choice_2_ahead_between =\
                    ((start, end), (next_node, best_choice_2_ahead_between_from_current[1][1]))

        (_, end), (next_node, _) = best_choice_2_ahead_between
        chosen_nodes.insert(chosen_nodes.index(end), next_node)
        remaining_nodes.remove(next_node)

    return tsp.calculate_solution(chosen_nodes)


if __name__ == "__main__":
    tsp = TSP.load_tspa(data_folder='../data')
    t0 = time()
    solution = greedy_2_regret_weighted_objective_solve(tsp, starting_node=0)
    t1 = time()
    print(f'execution_time: {t1 - t0}')
    print(solution)
    print(solution.nodes_in_excel_format())
    tsp.visualize_solution(solution, 'greedy 2-regret weighted objective')
