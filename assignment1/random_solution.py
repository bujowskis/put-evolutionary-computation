from data_loader import TSP, SolutionTSP
from random import sample, seed
from time import time


def random_solve(tsp: TSP, initial_seed: int = None) -> SolutionTSP:
    if initial_seed is not None:
        seed(initial_seed)
    required_number_of_nodes = tsp.get_required_number_of_nodes_in_solution()
    chosen_nodes = sample([i for i in tsp.nodes], required_number_of_nodes)
    return tsp.calculate_solution(chosen_nodes)


if __name__ == "__main__":
    tsp = TSP.load_tspa(data_folder='../data')
    t0 = time()
    solution = random_solve(tsp)
    t1 = time()
    print(f'execution_time: {t1 - t0}')
    print(solution)
    print(solution.nodes_in_excel_format())
