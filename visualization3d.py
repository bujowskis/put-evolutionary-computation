from data_loader import TSP
from assignment1.nearest_neighbor_at_any import nearest_neighbor_at_any_solve


if __name__ == "__main__":
    tsp = TSP.load_tspa()
    solution = nearest_neighbor_at_any_solve(tsp)
    tsp.visualize_solution3d(solution, 'NN any (some solution)', 'experiments_results-3d-nn-any.png')
