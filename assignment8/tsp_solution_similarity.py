from data_loader import SolutionTSP
from itertools import pairwise


def measure_common_edges(
        baseline_solution: SolutionTSP,
        compared_solution: SolutionTSP,
):
    baseline_edges = set(pairwise(baseline_solution.nodes + [baseline_solution.nodes[-1]]))
    compared_edges = set(pairwise(compared_solution.nodes + [compared_solution.nodes[-1]]))

    common_edges_number = 0
    for compared_edge in compared_edges:
        if compared_edge in baseline_edges or compared_edge[::-1] in baseline_edges:
            common_edges_number += 1

    return common_edges_number


def measure_common_nodes(
        baseline_solution: SolutionTSP,
        compared_solution: SolutionTSP,
):
    baseline_nodes = set(baseline_solution.nodes)
    compared_nodes = set(compared_solution.nodes)

    return len(baseline_nodes.intersection(compared_nodes))
