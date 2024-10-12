from assignment1.random_solution import random_solve
from assignment1.nearest_neighbor_at_end import nearest_neighbor_at_end_solve
from assignment1.nearest_neighbor_at_any import nearest_neighbor_at_any_solve
from assignment1.greedy_cycle import greedy_cycle_solve
from data_loader import TSP


if __name__ == '__main__':
    tsp = TSP.load_tspa()
    solution = random_solve(tsp)
    tsp.visualize_solution(solution, 'random')
    #tsp.visualize_solution(solution, method_name='random', path_to_save='experiments_results/random.png')

