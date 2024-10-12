from data_loader import TSP, SolutionTSP
from random import sample


def random_solve(tsp: TSP) -> SolutionTSP:
    required_number_of_nodes = tsp.get_required_number_of_nodes_in_solution()
    chosen_nodes = sample([i for i in range(len(tsp.raw_data))], required_number_of_nodes)
    return tsp.calculate_solution(chosen_nodes)


if __name__ == "__main__":
    tsp = TSP.load_tspa(data_folder='../data')
    solution = random_solve(tsp)
    print(solution)
    print(solution.nodes_in_excel_format())
