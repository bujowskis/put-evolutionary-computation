from data_loader import SolutionTSP
from itertools import pairwise


def measure_common_edges(
        baseline_solution: SolutionTSP,
        compared_solution: SolutionTSP,
):
    baseline_edges = set(pairwise(baseline_solution.nodes + [baseline_solution.nodes[-1]]))
    compared_edges = set(pairwise(compared_solution.nodes + [compared_solution.nodes[-1]]))

    return len(baseline_edges.intersection(compared_edges))


def measure_common_nodes(
        baseline_solution: SolutionTSP,
        compared_solution: SolutionTSP,
):
    baseline_nodes = set(baseline_solution.nodes)
    compared_nodes = set(compared_solution.nodes)

    return len(baseline_nodes.intersection(compared_nodes))
