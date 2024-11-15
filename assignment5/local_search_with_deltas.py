from data_loader import TSP, SolutionTSP
from random import choice
from itertools import combinations, pairwise
import heapq
from typing import List, Tuple
from time import time

from assignment1.random_solution import random_solve
from assignment1.nearest_neighbor_at_any import nearest_neighbor_at_any_solve
from assignment3.local_search_types import (
    LocalSearchType,
    StartingSolutionType,
    IntraRouteMovesType,
    MoveType,
)


def local_search_with_deltas_solve(
        tsp: TSP,
        local_search_type: LocalSearchType = LocalSearchType.STEEPEST,
        starting_solution_type: StartingSolutionType = StartingSolutionType.RANDOM,
        intra_route_move_type: IntraRouteMovesType = IntraRouteMovesType.TWO_NODES,
        starting_node: int = None,  # in case of RANDOM start - initial seed
) -> tuple[SolutionTSP, dict]:
    return LocalSearchWithDeltas(tsp=tsp,
                                 local_search_type=local_search_type,
                                 starting_solution_type=starting_solution_type,
                                 intra_route_move_type=intra_route_move_type,
                                 starting_node=starting_node).solve()

#TODO - check both ways AND if value is still the same?? (not value same - but if neighbors are also the same)
class LocalSearchWithDeltas:
    def __init__(self, tsp: TSP,
                 local_search_type: LocalSearchType = LocalSearchType.STEEPEST,
                 starting_solution_type: StartingSolutionType = StartingSolutionType.RANDOM,
                 intra_route_move_type: IntraRouteMovesType = IntraRouteMovesType.TWO_NODES,
                 starting_node: int = None,  # in case of RANDOM start - initial seed
                 ):
        self.tsp: TSP = tsp
        self.local_search_type: LocalSearchType = local_search_type
        self.starting_solution_type: StartingSolutionType = starting_solution_type
        self.intra_route_move_type: IntraRouteMovesType = intra_route_move_type
        self.starting_node: int = starting_node

        self.initial_solution: SolutionTSP
        match starting_solution_type:
            case StartingSolutionType.RANDOM:
                self.initial_solution = random_solve(tsp, initial_seed=starting_node)
            case StartingSolutionType.GREEDY:
                self.initial_solution = nearest_neighbor_at_any_solve(tsp, starting_node=starting_node)
            case _:
                raise Exception('no such starting_solution_type')

        self.cycle = self.initial_solution.nodes
        self.objective = self.initial_solution.objective_function
        self.not_selected_nodes = [node for node in tsp.nodes if node not in self.cycle]
        self.last_cycle_idx = len(self.cycle) - 1
        self.moves: List[Tuple[int, int, MoveType, Tuple]] = list()
        self.preserved_moves: List[Tuple[int, int, MoveType, Tuple]] = list()

        self.inter_nodes_exchanges_count: int = 0
        self.intra_two_nodes_count: int = 0
        self.intra_two_edges_count: int = 0
        self.moves_evaluated_count: int = 0

    def solve(self) -> tuple[SolutionTSP, dict]:
        self.initialize_moves()
        while self.moves:
            self.make_move_if_possible()

        total_moves = self.inter_nodes_exchanges_count + self.intra_two_nodes_count + self.intra_two_edges_count

        return self.tsp.calculate_solution(self.cycle), {
            'total_moves': total_moves,
            'moves_evaluated': self.moves_evaluated_count,
            'inter_nodes_exchanges_count': self.inter_nodes_exchanges_count,
            'intra_two_nodes_count': self.intra_two_nodes_count,
            'intra_two_edges_count': self.intra_two_edges_count,
            'inter_nodes_exchanges_percentage': None if total_moves == 0 else round(self.inter_nodes_exchanges_count / total_moves, 2),
            'intra_two_nodes_percentage': None if total_moves == 0 else round(self.intra_two_nodes_count / total_moves, 2),
            'intra_two_edges_percentage': None if total_moves == 0 else round(self.intra_two_edges_count / total_moves, 2),
        }

    def add_preserved_moves_back(self):
        self.moves = self.preserved_moves + self.moves
        self.preserved_moves = list()

    def add_move_if_improves(self, move: Tuple[int, MoveType, Tuple]):
        self.moves_evaluated_count += 1
        objective_change, move_type, move_specification = move
        if objective_change >= 0:
            return
        move_on_queue = (objective_change, self.moves_evaluated_count, move_type, move_specification)
        if self.local_search_type.STEEPEST:
            heapq.heappush(self.moves, move_on_queue)
        else:
            self.moves.append(move_on_queue)

    def add_move_without_evaluation(self, move: Tuple[int, MoveType, Tuple]):
        objective_change, move_type, move_specification = move
        move_on_queue = (objective_change, self.moves_evaluated_count, move_type, move_specification)
        self.moves.append(move_on_queue)

    def get_connected_nodes(self, node: int):
        place_in_cycle = self.cycle.index(node)
        if place_in_cycle == 0:
            return self.cycle[-1], self.cycle[1]
        elif place_in_cycle == self.last_cycle_idx:
            return self.cycle[-2], self.cycle[0]
        else:
            return self.cycle[place_in_cycle - 1], self.cycle[place_in_cycle + 1]

    def initialize_moves(self):
        # - generate all possible inter nodes exchange
        for node_in_cycle in self.cycle:
            self.add_inter_nodes_moves(node_in_cycle=node_in_cycle, neighbors=self.get_connected_nodes(node_in_cycle))
        # - generate all possible relevant intra moves todo - could try using both at the same time
        if self.intra_route_move_type == IntraRouteMovesType.TWO_NODES:
            for node1, node2 in combinations(self.cycle, 2):
                node1_neighbors, node2_neighbors = self.get_connected_nodes(node1), self.get_connected_nodes(node2)
                self.add_intra_nodes(node1=node1, node1_neighbors=node2_neighbors,
                                     node2=node2, node2_neighbors=node2_neighbors,)
        if self.intra_route_move_type == IntraRouteMovesType.TWO_EDGES:
            for edge1_nodes, edge2_nodes in combinations(pairwise(self.cycle + [self.cycle[-1]]), 2):
                self.add_intra_edges(edge1_nodes=edge1_nodes, edge2_nodes=edge2_nodes)

    def make_move_if_possible(self):
        objective_change, moves_evaluated_count, move_type, move_specification = self.get_move()
        match move_type:
            case MoveType.INTER_NODES_EXCHANGE:
                old_node, new_node, move_neighbors = move_specification
                try:
                    current_neighbors = self.get_connected_nodes(old_node)
                except ValueError:
                    return
                # check if still valid
                if (old_node in self.cycle) and (new_node not in self.cycle) and \
                   (move_neighbors == current_neighbors):
                    if self.local_search_type == LocalSearchType.GREEDY:
                        objective_change = self.calculate_inter_nodes_move_objective_change(old_node, new_node, move_neighbors)
                        self.moves_evaluated_count += 1
                        if objective_change >= 0:
                            return
                    # make move
                    exchange_idx = self.cycle.index(old_node)
                    self.cycle[exchange_idx] = new_node
                    self.not_selected_nodes.remove(new_node)
                    self.not_selected_nodes.append(old_node)
                    self.objective += objective_change
                    self.inter_nodes_exchanges_count += 1

                    # add resulting new moves
                    self.add_preserved_moves_back()
                    self.add_inter_nodes_moves(node_in_cycle=new_node, neighbors=move_neighbors)
                    if self.intra_route_move_type == IntraRouteMovesType.TWO_NODES:
                        for node2 in self.cycle:
                            if node2 == new_node:
                                continue
                            node2_neighbors = self.get_connected_nodes(node2)
                            self.add_intra_nodes(node1=new_node, node1_neighbors=move_neighbors,
                                                 node2=node2, node2_neighbors=node2_neighbors)
                    if self.intra_route_move_type == IntraRouteMovesType.TWO_EDGES:
                        for new_edge_nodes in ((move_neighbors[0], new_node), (new_node, move_neighbors[1])):
                            for edge_nodes in pairwise(self.cycle + [self.cycle[-1]]):
                                if new_edge_nodes == edge_nodes:
                                    continue
                                self.add_intra_edges(edge1_nodes=new_edge_nodes, edge2_nodes=edge_nodes)
            case MoveType.INTRA_TWO_NODES:
                node1, node2, node1_neighbors, node2_neighbors = move_specification
                try:
                    current_node1_neighbors = self.get_connected_nodes(node1)
                    current_node2_neighbors = self.get_connected_nodes(node2)
                except ValueError:
                    return
                # check if still valid
                if (node1 in self.cycle) and (node2 in self.cycle) and \
                   (node1_neighbors == current_node1_neighbors) and (node2_neighbors == current_node2_neighbors):
                    if self.local_search_type == LocalSearchType.GREEDY:
                        objective_change = self.calculate_intra_nodes_objective_change(node1, node1_neighbors, node2, node2_neighbors)
                        self.moves_evaluated_count += 1
                        if objective_change >= 0:
                            return
                    # make move
                    node1_idx, node2_idx = self.cycle.index(node1), self.cycle.index(node2)
                    self.cycle[node1_idx], self.cycle[node2_idx] = node2, node1
                    self.objective += objective_change
                    self.intra_two_nodes_count += 1
                    # update neighbors
                    current_node1_neighbors = self.get_connected_nodes(node1)
                    current_node2_neighbors = self.get_connected_nodes(node2)
                    # add resulting new moves
                    self.add_preserved_moves_back()
                    self.add_inter_nodes_moves(node_in_cycle=node1, neighbors=node2_neighbors)
                    self.add_inter_nodes_moves(node_in_cycle=node2, neighbors=node1_neighbors)
                    if self.intra_route_move_type == IntraRouteMovesType.TWO_NODES:
                        for other_node in self.cycle:
                            if (other_node == node1) or (other_node == node2):
                                continue  # doesn't make sense to repeat the same exact move
                            other_node_neighbors = self.get_connected_nodes(other_node)
                            self.add_intra_nodes(node1=node1, node1_neighbors=node2_neighbors,
                                                 node2=other_node, node2_neighbors=other_node_neighbors)
                            self.add_intra_nodes(node1=node2, node1_neighbors=node1_neighbors,
                                                 node2=other_node, node2_neighbors=other_node_neighbors)
                    if self.intra_route_move_type == IntraRouteMovesType.TWO_EDGES:
                        for edge in pairwise(self.cycle + [self.cycle[-1]]):
                            for node1_edge in ((current_node2_neighbors[0], node1), (node1, current_node2_neighbors[1])):
                                if edge == node1_edge:
                                    continue
                                self.add_intra_edges(edge1_nodes=node1_edge, edge2_nodes=edge)
                            for node2_edge in ((current_node1_neighbors[0], node2), (node2, current_node1_neighbors[1])):
                                if edge == node2_edge:
                                    continue
                                self.add_intra_edges(edge1_nodes=node2_edge, edge2_nodes=edge)
            case MoveType.INTRA_TWO_EDGES:
                edge1_nodes, edge2_nodes = move_specification
                cycle_edges = list(pairwise(self.cycle + [self.cycle[-1]]))
                # check if still valid
                if (((edge1_nodes in cycle_edges) and (edge2_nodes in cycle_edges)) or
                        ((edge1_nodes[::-1] in cycle_edges) and (edge2_nodes[::-1] in cycle_edges))):
                    if self.local_search_type == LocalSearchType.GREEDY:
                        objective_change = self.calculate_intra_edges_objective_change(edge1_nodes, edge2_nodes)
                        self.moves_evaluated_count += 1
                        if objective_change >= 0:
                            return
                    # make move
                    edge1_idx, edge2_idx = self.cycle.index(edge1_nodes[0]), self.cycle.index(edge2_nodes[0])
                    if edge1_idx < edge2_idx:
                        left_idx, right_idx = edge1_idx, edge2_idx
                    else:
                        left_idx, right_idx = edge2_idx, edge1_idx
                    left, middle, right = [], [], []
                    for i, node in enumerate(self.cycle):
                        if i <= left_idx:
                            left.append(node)
                        elif i <= right_idx:
                            middle.append(node)
                        else:
                            right.append(node)
                    middle = middle[::-1]
                    self.cycle = left + middle + right
                    self.objective += objective_change
                    self.intra_two_edges_count += 1

                    # add resulting new moves
                    self.add_preserved_moves_back()
                    # inter nodes exchanges do not stay the same - neighbors at "shifting points" changed
                    try:
                        l, ml, mr, r = left[-1], middle[0], middle[-1], right[0]
                    except IndexError:
                        if len(left) == 0:
                            l, ml, mr, r = middle[0], middle[-1], right[0], right[-1]
                        elif len(right) == 0:
                            l, ml, mr, r = left[0], left[-1], middle[0], middle[-1]
                        else:
                            raise Exception('wrong index error stuff')
                    for node in (l, ml, mr, r):
                        self.add_inter_nodes_moves(node, self.get_connected_nodes(node))
                    # edges on "shifting points" changed
                    for new_edge_nodes in ((l, ml), (mr, r)):
                        for edge_nodes in pairwise(self.cycle + [self.cycle[-1]]):
                            if new_edge_nodes == edge_nodes:
                                continue
                            self.add_intra_edges(edge1_nodes=new_edge_nodes, edge2_nodes=edge_nodes)
                # note - wrong relative order is handled after flipping them right
                if (((edge1_nodes[::-1] in cycle_edges) and (edge2_nodes in cycle_edges)) or
                        ((edge1_nodes in cycle_edges) and (edge2_nodes[::-1] in cycle_edges))):
                    self.preserved_moves.append((objective_change, moves_evaluated_count, move_type, move_specification))

    def calculate_intra_nodes_objective_change(self, node1, node1_neighbors, node2, node2_neighbors) -> int:
        # ..., node1_neighbors[0], node1, node2, node2_neighbors[1]
        # 1                - n1n[0]_n1   - n1_n2
        # 2                              - n1_n2 (2x!) - n2_n2n[1]
        # =
        # ..., node1_neighbors[0], node2, node1, node2_neighbors[1]
        # 3                + n1n[0]_n2   + n2_n2 (!)
        # 4                              + n1_n1 (!) + n1_n2n[1]
        # correct = - n1n[0]_n1 - n1_n2 - n2_n2n[1]
        #           + n1n[0]_n2 + n2_n1 + n1_n2n[1]
        if node1 == node2_neighbors[0]:
            return - self.tsp.distances_matrix[node1_neighbors[0]][node1] \
                   - self.tsp.distances_matrix[node1][node2] \
                   - self.tsp.distances_matrix[node2][node2_neighbors[1]] \
                   + self.tsp.distances_matrix[node1_neighbors[0]][node2] \
                   + self.tsp.distances_matrix[node2][node1] \
                   + self.tsp.distances_matrix[node1][node2_neighbors[1]]
        # ..., node2_neighbors[0], node2, node1, node1_neighbors[1]
        # 1                - n2n[0]_n2   - n2_n1
        # 2                              - n2_n1 (2x!) - n1_n1n[1]
        # ..., node2_neighbors[0], node1, node2, node1_neighbors[1]
        # 3                + n2n[0]_n1   + n1_n1 (!)
        # 4                              + n2_n2 (!) + n2_n1n[1]
        # correct = - n2n[0]_n2 - n2_n1 - n1_n1n[1]
        #           + n2n[0]_n1 + n1_n2 + n2_n1n[1]
        if node2 == node1_neighbors[0]:
            return - self.tsp.distances_matrix[node2_neighbors[0]][node2] \
                   - self.tsp.distances_matrix[node2][node1] \
                   - self.tsp.distances_matrix[node1][node1_neighbors[1]] \
                   + self.tsp.distances_matrix[node2_neighbors[0]][node1] \
                   + self.tsp.distances_matrix[node1][node2] \
                   + self.tsp.distances_matrix[node2][node1_neighbors[1]]

        return - (self.tsp.distances_matrix[node1_neighbors[0]][node1] +
                  self.tsp.distances_matrix[node1][node1_neighbors[1]]) \
               - (self.tsp.distances_matrix[node2_neighbors[0]][node2] +
                  self.tsp.distances_matrix[node2][node2_neighbors[1]]) \
               + (self.tsp.distances_matrix[node1_neighbors[0]][node2] +
                  self.tsp.distances_matrix[node2][node1_neighbors[1]]) \
               + (self.tsp.distances_matrix[node2_neighbors[0]][node1] +
                  self.tsp.distances_matrix[node1][node2_neighbors[1]])

    def add_intra_nodes(self, node1, node1_neighbors, node2, node2_neighbors):
        if self.local_search_type == LocalSearchType.STEEPEST:
            objective_change = self.calculate_intra_nodes_objective_change(node1, node1_neighbors, node2, node2_neighbors)
            self.add_move_if_improves((
                objective_change, MoveType.INTRA_TWO_NODES,
                (node1, node2, node1_neighbors, node2_neighbors)
            ))
        else:
            self.add_move_without_evaluation((
                0, MoveType.INTRA_TWO_NODES,
                (node1, node2, node1_neighbors, node2_neighbors)
            ))

    def calculate_intra_edges_objective_change(self, edge1_nodes, edge2_nodes) -> int:
        return - self.tsp.distances_matrix[edge1_nodes[0]][edge1_nodes[1]] \
               - self.tsp.distances_matrix[edge2_nodes[0]][edge2_nodes[1]] \
               + self.tsp.distances_matrix[edge1_nodes[0]][edge2_nodes[0]] \
               + self.tsp.distances_matrix[edge1_nodes[1]][edge2_nodes[1]]

    def add_intra_edges(self, edge1_nodes, edge2_nodes):
        """
        1 2-3 4 5 6-7 8 9

        ___         _____
        1 2-6 5 4 3-7 8 9
            _______
        ^reversed
        note - need to introduce only 2 edges - at reversal points
        """
        if self.local_search_type == LocalSearchType.STEEPEST:
            objective_change = self.calculate_intra_edges_objective_change(edge1_nodes, edge2_nodes)
            self.add_move_if_improves((
                objective_change, MoveType.INTRA_TWO_EDGES,
                (edge1_nodes, edge2_nodes)
            ))
        else:
            self.add_move_without_evaluation((
                0, MoveType.INTRA_TWO_EDGES,
                (edge1_nodes, edge2_nodes)
            ))

    def calculate_inter_nodes_move_objective_change(
            self, node_in_cycle: int, non_cycle_node: int, neighbors: tuple[int, int]) -> int:
        return - (self.tsp.additional_costs[node_in_cycle] +
                  self.tsp.distances_matrix[neighbors[0]][node_in_cycle] +
                  self.tsp.distances_matrix[node_in_cycle][neighbors[1]]) \
               + (self.tsp.additional_costs[non_cycle_node] +
                  self.tsp.distances_matrix[neighbors[0]][non_cycle_node] +
                  self.tsp.distances_matrix[non_cycle_node][neighbors[1]])

    def add_inter_nodes_moves(self, node_in_cycle: int, neighbors: tuple[int, int]):
        for non_cycle_node in self.not_selected_nodes:
            if self.local_search_type == LocalSearchType.STEEPEST:
                objective_change = self.calculate_inter_nodes_move_objective_change(node_in_cycle, non_cycle_node, neighbors)
                self.add_move_if_improves((
                    objective_change, MoveType.INTER_NODES_EXCHANGE,
                    (node_in_cycle, non_cycle_node, neighbors)
                ))
            else:
                self.add_move_without_evaluation((
                    0, MoveType.INTER_NODES_EXCHANGE,
                    (node_in_cycle, non_cycle_node, neighbors)
                ))

    def get_move(self) -> Tuple[int, int, MoveType, Tuple]:
        match self.local_search_type:
            case LocalSearchType.STEEPEST:
                return heapq.heappop(self.moves)
            case LocalSearchType.GREEDY:
                return self.moves.pop(choice(range(len(self.moves))))
            case _:
                raise Exception('no such local_search_type')


if __name__ == "__main__":
    tsp = TSP.load_tspa(data_folder='../data')
    t0 = time()
    solution, stats = local_search_with_deltas_solve(
        tsp,
        starting_solution_type=StartingSolutionType.RANDOM,
        local_search_type=LocalSearchType.STEEPEST,
        intra_route_move_type=IntraRouteMovesType.TWO_EDGES,
    )
    t1 = time()
    print(f'execution_time: {t1 - t0}')
    print(stats)
    print(solution)
    tsp.visualize_solution(solution, method_name='local_search')
