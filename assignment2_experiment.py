from assignment2.greedy_2_regret import greedy_2_regret_solve
from assignment2.greedy_2_regret_weighted_objective import greedy_2_regret_weighted_objective_solve
from data_loader import TSP

from time import time
import pickle
from numpy import mean
import numpy
import json

EXPERIMENTS_RESULTS_FOLDER = 'experiments_results/assignment2'

if __name__ == '__main__':
    t_start = time()

    problems = {'TSPA': TSP.load_tspa(), 'TSPB': TSP.load_tspb()}
    print('(data loaded)')
    run, total_time = 1, 0

    for problem_name, tsp in problems.items():
        solutions = []
        times = []
        for i in range(len(tsp.nodes)):
            t0 = time()
            solution = greedy_2_regret_solve(tsp, starting_node=i)
            t1 = time()
            solutions.append(solution)
            times.append(t1 - t0)
            total_time += t1 - t0
            print(f'run:\t{run},\ttime: {times[-1]},\ttotal_time: {total_time}')
            run += 1

        best = min(solutions)
        worst = max(solutions)

        print('greedy-2-regret')
        print(best)
        print(best.nodes_in_excel_format())

        with open(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/pkl/{problem_name}-greedy-2-regret-best.pkl', 'wb') as file:
            pickle.dump(best, file)

        with open(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/json/{problem_name}-greedy-2-regret-best.json', 'w') as json_file:
            json.dump({
                'min_objective_function': best.objective_function,
                'max_objective_function': worst.objective_function,
                'avg_objective_function': mean([s.objective_function for s in solutions]),
                'best_solution_nodes': best.nodes,
                'min_time': min(times),
                'max_time': max(times),
                'avg_time': mean(times)
            }, json_file, indent=4, default=lambda x: int(x) if isinstance(x, numpy.int64) else x)

        tsp.visualize_solution(
            best, method_name=f'{problem_name} greedy-2-regret',
            path_to_save=f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/png/{problem_name}-greedy-2-regret-best.png')

    for problem_name, tsp in problems.items():
        solutions = []
        times = []
        for i in range(len(tsp.nodes)):
            t0 = time()
            solution = greedy_2_regret_weighted_objective_solve(tsp, starting_node=i)
            t1 = time()
            solutions.append(solution)
            times.append(t1 - t0)
            total_time += t1 - t0
            print(f'run:\t{run},\ttime: {times[-1]},\ttotal_times: {total_time}')
            run += 1

        best = min(solutions)
        worst = max(solutions)

        print('greedy-2-regret-weighted')
        print(best)
        print(best.nodes_in_excel_format())

        with open(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/pkl/{problem_name}-greedy-2-regret-weighted-best.pkl', 'wb') as file:
            pickle.dump(best, file)

        with open(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/json/{problem_name}-greedy-2-regret-weighted-best.json', 'w') as json_file:
            json.dump({
                'min_objective_function': best.objective_function,
                'max_objective_function': worst.objective_function,
                'avg_objective_function': mean([s.objective_function for s in solutions]),
                'best_solution_nodes': best.nodes,
                'min_time': min(times),
                'max_time': max(times),
                'avg_time': mean(times)
            }, json_file, indent=4, default=lambda x: int(x) if isinstance(x, numpy.int64) else x)

        tsp.visualize_solution(
            best, method_name=f'{problem_name} greedy-2-regret-weighted',
            path_to_save=f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/png/{problem_name}-greedy-2-regret-weighted-best.png')

    t_end = time()
    print(f'duration of whole experiment: {t_end - t_start}')
