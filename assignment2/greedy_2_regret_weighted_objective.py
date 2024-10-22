from data_loader import TSP, SolutionTSP
from random import choice
from itertools import pairwise
from time import time
import heapq

INF = 10**18


def greedy_2_regret_weighted_objective_solve(tsp: TSP, starting_node: int = None) -> SolutionTSP:
    if starting_node is None:
        starting_node = choice(tsp.nodes)

    chosen_nodes = [starting_node]
    remaining_nodes = tsp.get_nodes(without_nodes=chosen_nodes)

    required_number_of_nodes = tsp.get_required_number_of_nodes_in_solution()
    while len(chosen_nodes) < required_number_of_nodes and remaining_nodes:
        possible_insertions = list(pairwise(chosen_nodes + [chosen_nodes[0]]))

        insertion_costs = [list() for _ in possible_insertions]
        best_insertion_places = [(INF, (-1, -1)) for _ in remaining_nodes]  # (insertion_cost, (start, end))
        for insertion_place_idx, next_node in enumerate(remaining_nodes):
            for insertion_costs_idx, (start, end) in enumerate(possible_insertions):
                insertion_cost = tsp.insertion_costs[start][end][next_node]

                heapq.heappush(insertion_costs[insertion_costs_idx],
                               (tsp.insertion_costs[start][end][next_node], next_node))

                if insertion_cost < best_insertion_places[insertion_place_idx][0]:
                    best_insertion_places[insertion_place_idx] = (insertion_cost, (start, end))

        highest_weighted_objective = (-INF, -1, (-1, -1))  # (regret, next_node, (start, end))
        for insertion_costs_idx, _ in enumerate(possible_insertions):
            best_1st_move_cost, best_1st_move_node = heapq.heappop(insertion_costs[insertion_costs_idx])
            best_2nd_move_cost, best_2nd_move_node = heapq.heappop(insertion_costs[insertion_costs_idx])
            regret = best_2nd_move_cost - best_1st_move_cost
            best_1st_move_node_insertion_cost, best_1st_move_node_insertion_place =\
                best_insertion_places[remaining_nodes.index(best_1st_move_node)]
            weighted_objective = (regret - best_1st_move_node_insertion_cost) / 2  # "-" because then we maximize
            if weighted_objective > highest_weighted_objective[0]:
                highest_weighted_objective = (weighted_objective, best_1st_move_node,
                                              best_1st_move_node_insertion_place)

        _, next_node, (_, end) = highest_weighted_objective
        chosen_nodes.insert(chosen_nodes.index(end), next_node)
        remaining_nodes.remove(next_node)

    return tsp.calculate_solution(chosen_nodes)


if __name__ == "__main__":
    tsp = TSP.load_tspa(data_folder='../data')
    t0 = time()
    solution = greedy_2_regret_weighted_objective_solve(tsp)
    t1 = time()
    print(f'execution_time: {t1 - t0}')
    print(solution)
    print(solution.nodes_in_excel_format())
    tsp.visualize_solution(solution, 'greedy 2-regret')
