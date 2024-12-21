from data_loader import TSP, SolutionTSP
from random import seed, choice
import heapq
from enum import Enum
from time import time
from itertools import pairwise
from numpy.random import choice as choice_multiple

from assignment1.nearest_neighbor_at_any import nearest_neighbor_at_any_solve
from assignment3.local_search_types import (
    LocalSearchType,
    StartingSolutionType,
    IntraRouteMovesType,
)
from assignment5.local_search_with_deltas import local_search_with_deltas_solve


class RecombinationOperator(Enum):
    RANDOM = 1
    HEURISTIC = 2


def evolutionary_algorithm_solve(
        tsp: TSP,
        local_search_type: LocalSearchType = LocalSearchType.STEEPEST,
        starting_solution_type: StartingSolutionType = StartingSolutionType.RANDOM,
        intra_route_move_type: IntraRouteMovesType = IntraRouteMovesType.TWO_EDGES,
        starting_node: int = None,  # in case of RANDOM start - initial seed,
        timeout_after_seconds: int = 47,  # average running time of MSLS run
        should_use_local_search: bool = False,
        population_size: int = 20,
        recombination_operator: RecombinationOperator = RecombinationOperator.HEURISTIC,
) -> tuple[SolutionTSP, int]:
    number_of_main_loop_runs = 0

    if starting_node and starting_solution_type == StartingSolutionType.RANDOM:
        seed(starting_node)

    population: list[SolutionTSP] = list()

    def push_to_population_if_viable(solution: SolutionTSP, population: list[SolutionTSP]) -> list[SolutionTSP]:
        """Works on objective function value, not comparing entire solution"""
        for solution_in_population in population:
            if solution.objective_function == solution_in_population.objective_function:
                break
        heapq.heappush(population, solution)
        return population[:20]

    while len(population) is not population_size:
        candidate_solution, _ = local_search_with_deltas_solve(
            tsp=tsp,
            local_search_type=local_search_type,
            starting_solution_type=starting_solution_type,
            intra_route_move_type=intra_route_move_type,
        )
        population = push_to_population_if_viable(
            solution=candidate_solution,
            population=population,
        )

    def locate_common_nodes_and_edges(
            parent1: SolutionTSP,
            parent2: SolutionTSP
    ) -> tuple[set[int], set[tuple[int, int]]]:
        common_nodes = {node for node in parent1.nodes if node in parent2.nodes}

        parent1_edges = set(pairwise(parent1.nodes + [parent1.nodes[0]]))
        parent2_edges = set(pairwise(parent2.nodes + [parent2.nodes[0]]))
        common_edges = set()
        for parent1_edge in parent1_edges:
            if parent1_edge in parent2_edges or parent1_edge[::-1] in parent2_edges:
                common_edges.add(parent1_edge)
                common_edges.add(parent1_edge[::-1])

        return common_nodes, common_edges

    def recombine_with_random(parent1: SolutionTSP, parent2: SolutionTSP) -> SolutionTSP:
        goal_length = len(parent1.nodes)
        common_nodes, common_edges = locate_common_nodes_and_edges(parent1, parent2)
        # safeguard against infinite loops
        if common_nodes == goal_length:
            raise Exception('parents have all the same nodes, which leads to infinite loops')
        
        # note - it suffices to remove uncommon nodes - common edges will also be preserved
        #   common_edges are important in the repairing part
        stripped_solution = parent1.nodes.copy()
        for node in parent1.nodes:
            if node not in common_nodes:
                stripped_solution.remove(node)

        # while adding indices may change -> store nodes before which new node may be inserted
        insertion_places_before_node: list[int] = list()
        for edge in pairwise([stripped_solution[-1]] + stripped_solution):
            if edge not in common_edges:
                insertion_places_before_node.append(edge[1])

        nodes_outside_cycle = [n for n in tsp.nodes if n not in stripped_solution]
        while len(stripped_solution) != goal_length:
            # note - no need to remove this insertion place, it's still valid (in random approach)
            chosen_place_node = choice(insertion_places_before_node)
            chosen_node_outside_cycle = nodes_outside_cycle.pop(choice(range(len(nodes_outside_cycle))))
            stripped_solution.insert(stripped_solution.index(chosen_place_node), chosen_node_outside_cycle)

        recombined_solution = tsp.calculate_solution(stripped_solution)
        if should_use_local_search:
            recombined_solution, _ = local_search_with_deltas_solve(
                tsp=tsp,
                local_search_type=local_search_type,
                starting_solution_type=starting_solution_type,
                intra_route_move_type=intra_route_move_type,
                starting_solution=recombined_solution,
            )

        return recombined_solution

    def recombine_with_heuristic(parent1: SolutionTSP, parent2: SolutionTSP) -> SolutionTSP:
        goal_length = len(parent1.nodes)
        common_nodes, common_edges = locate_common_nodes_and_edges(parent1, parent2)
        # safeguard against infinite loops
        if common_nodes == goal_length:
            raise Exception('parents have all the same nodes, which leads to infinite loops')

        # note - it suffices to remove uncommon nodes - common edges will also be preserved
        #   common_edges are important in the repairing part
        stripped_solution = parent1.nodes.copy()
        for node in parent1.nodes:
            if node not in common_nodes:
                stripped_solution.remove(node)

        recombined_solution = nearest_neighbor_at_any_solve(
            tsp=tsp,
            starting_nodes=stripped_solution,
            except_edges=common_edges,
        )
        if should_use_local_search:
            recombined_solution, _ = local_search_with_deltas_solve(
                tsp=tsp,
                local_search_type=local_search_type,
                starting_solution_type=starting_solution_type,
                intra_route_move_type=intra_route_move_type,
                starting_solution=recombined_solution,
            )

        return recombined_solution

    match recombination_operator:
        case RecombinationOperator.RANDOM:
            recombine = recombine_with_random
        case RecombinationOperator.HEURISTIC:
            recombine = recombine_with_heuristic
        case _:
            raise Exception('unsupported recombination_operator')

    t0 = time()
    while time() - t0 < timeout_after_seconds:
        number_of_main_loop_runs += 1
        parents_indices = choice_multiple(range(population_size), 2)
        child = recombine(parent1=population[parents_indices[0]],
                          parent2=population[parents_indices[1]])
        population = push_to_population_if_viable(child, population)

    return population[0], number_of_main_loop_runs
