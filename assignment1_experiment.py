from assignment1.random_solution import random_solve
from assignment1.nearest_neighbor_at_end import nearest_neighbor_at_end_solve
from assignment1.nearest_neighbor_at_any import nearest_neighbor_at_any_solve
from assignment1.greedy_cycle import greedy_cycle_solve
from data_loader import TSP

from time import time
import pickle
from numpy import mean
import numpy
import json

EXPERIMENTS_RESULTS_FOLDER = 'experiments_results/assignment1'

if __name__ == '__main__':
    t_start = time()

    problems = {'TSPA': TSP.load_tspa(), 'TSPB': TSP.load_tspb()}

    for problem_name, tsp in problems.items():
        solutions = []
        times = []
        for i in range(len(tsp.nodes)):
            t0 = time()
            solution = random_solve(tsp, initial_seed=i)
            t1 = time()
            solutions.append(solution)
            times.append(t1 - t0)

        best = min(solutions)
        worst = max(solutions)

        print('random')
        print(best)
        print(best.nodes_in_excel_format())

        with open(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}-random-best.pkl', 'wb') as file:
            pickle.dump(best, file)

        with open(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}-random-best.json', 'w') as json_file:
            json.dump({
                'min_objective_function': best.objective_function,
                'max_objective_function': worst.objective_function,
                'avg_objective_function': mean([s.objective_function for s in solutions]),
                'best_solution_nodes': best.nodes,
                'min_time': min(times),
                'max_time': max(times),
                'avg_time': mean(times)
            }, json_file, indent=4, default=lambda x: int(x) if isinstance(x, numpy.int64) else x)

        tsp.visualize_solution(best, method_name=f'{problem_name} random',
                               path_to_save=f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}-random-best.png')

    for problem_name, tsp in problems.items():
        solutions = []
        times = []
        for i in range(len(tsp.nodes)):
            t0 = time()
            solution = nearest_neighbor_at_end_solve(tsp, starting_node=i)
            t1 = time()
            solutions.append(solution)
            times.append(t1 - t0)

        best = min(solutions)
        worst = max(solutions)

        print('nn-end')
        print(best)
        print(best.nodes_in_excel_format())

        with open(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}-nn-end-best.pkl', 'wb') as file:
            pickle.dump(best, file)

        with open(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}-nn-end-best.json', 'w') as json_file:
            json.dump({
                'min_objective_function': best.objective_function,
                'max_objective_function': worst.objective_function,
                'avg_objective_function': mean([s.objective_function for s in solutions]),
                'best_solution_nodes': best.nodes,
                'min_time': min(times),
                'max_time': max(times),
                'avg_time': mean(times)
            }, json_file, indent=4, default=lambda x: int(x) if isinstance(x, numpy.int64) else x)

        tsp.visualize_solution(best, method_name=f'{problem_name} NN end',
                               path_to_save=f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}-nn-end-best.png')

    for problem_name, tsp in problems.items():
        solutions = []
        times = []
        for i in range(len(tsp.nodes)):
            t0 = time()
            solution = nearest_neighbor_at_any_solve(tsp, starting_node=i)
            t1 = time()
            solutions.append(solution)
            times.append(t1 - t0)

        best = min(solutions)
        worst = max(solutions)

        print('nn-any')
        print(best)
        print(best.nodes_in_excel_format())

        with open(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}-nn-any-best.pkl', 'wb') as file:
            pickle.dump(best, file)

        with open(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}-nn-any-best.json', 'w') as json_file:
            json.dump({
                'min_objective_function': best.objective_function,
                'max_objective_function': worst.objective_function,
                'avg_objective_function': mean([s.objective_function for s in solutions]),
                'best_solution_nodes': best.nodes,
                'min_time': min(times),
                'max_time': max(times),
                'avg_time': mean(times)
            }, json_file, indent=4, default=lambda x: int(x) if isinstance(x, numpy.int64) else x)

        tsp.visualize_solution(best, method_name=f'{problem_name} NN any',
                               path_to_save=f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}-nn-any-best.png')

    for problem_name, tsp in problems.items():
        solutions = []
        times = []
        for i in range(len(tsp.nodes) - 1):
            t0 = time()
            solution = greedy_cycle_solve(tsp, starting_node=i)
            t1 = time()
            solutions.append(solution)
            times.append(t1 - t0)

        best = min(solutions)
        worst = max(solutions)

        print('greedy-cycle')
        print(best)
        print(best.nodes_in_excel_format())

        with open(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}-greedy-cycle-best.pkl', 'wb') as file:
            pickle.dump(best, file)

        with open(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}-greedy-cycle-best.json', 'w') as json_file:
            json.dump({
                'min_objective_function': best.objective_function,
                'max_objective_function': worst.objective_function,
                'avg_objective_function': mean([s.objective_function for s in solutions]),
                'best_solution_nodes': best.nodes,
                'min_time': min(times),
                'max_time': max(times),
                'avg_time': mean(times)
            }, json_file, indent=4, default=lambda x: int(x) if isinstance(x, numpy.int64) else x)

        tsp.visualize_solution(best, method_name=f'{problem_name} greedy cycle',
                               path_to_save=f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}-greedy-cycle-best.png')

    t_end = time()
    print(f'duration of whole experiment: {t_end - t_start}')
