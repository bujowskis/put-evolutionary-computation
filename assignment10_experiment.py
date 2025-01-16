from assignment10.cycle_stripper import cycle_stripper_solve
from data_loader import TSP

from time import time
import pickle
from numpy import mean
import numpy
import json

EXPERIMENTS_RESULTS_FOLDER = 'experiments_results/assignment10'

if __name__ == '__main__':
    t_start = time()

    problems = {'TSPA': TSP.load_tspa(), 'TSPB': TSP.load_tspb()}
    print('(data loaded)')
    run, total_time = 1, 0

    for problem_name, tsp in problems.items():
        for ls_interval in [-1]:  # [1, 3, 5, 10]:
            solutions, times = [], []
            for i in range(len(tsp.nodes)):
                t0 = time()
                solution = cycle_stripper_solve(tsp=tsp, ls_interval=ls_interval, should_run_ls_at_end=False)
                t1 = time()
                solutions.append(solution)
                times.append(t1 - t0)
                total_time += t1 - t0
                print(f'run:\t{run},\ttime: {times[-1]},\ttotal_time: {total_time},\tobjective_function: {solution.objective_function}')
                run += 1

            best, worst = min(solutions), max(solutions)

            # print(f'Cycle stripper - deltas, ls_interval={ls_interval}')
            # print(f'Cycle stripper - deltas, no main ls')
            print(f'Cycle stripper - deltas, no ls')
            print(best)

            # with open(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/pkl/{problem_name}-cycle_stripper-deltas-{ls_interval}-best.pkl', 'wb') as file:
            # with open(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/pkl/{problem_name}-cycle_stripper-deltas-no-main-ls-best.pkl', 'wb') as file:
            with open(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/pkl/{problem_name}-cycle_stripper-deltas-no-ls-best.pkl', 'wb') as file:
                pickle.dump(best, file)

            # with open(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/json/{problem_name}-cycle_stripper-deltas-{ls_interval}.json', 'w') as json_file:
            # with open(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/json/{problem_name}-cycle_stripper-deltas-no-main-ls.json', 'w') as json_file:
            with open(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/json/{problem_name}-cycle_stripper-deltas-no-ls.json', 'w') as json_file:
                json.dump({
                    'min_objective_function': best.objective_function,
                    'max_objective_function': worst.objective_function,
                    'avg_objective_function': mean([s.objective_function for s in solutions]),
                    'best_solution_nodes': best.nodes,
                    'min_time': min(times),
                    'max_time': max(times),
                    'avg_time': mean(times),
                }, json_file, indent=4, default=lambda x: int(x) if isinstance(x, numpy.int64) else x)

            tsp.visualize_solution(
                best,
                method_name=f'{problem_name} Cycle stripper - ls_interval={ls_interval}',
                # path_to_save=f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/png/{problem_name}-cycle_stripper-deltas-{ls_interval}-best.png'
                # path_to_save=f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/png/{problem_name}-cycle_stripper-deltas-no-main-ls-best.png'
                path_to_save=f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/png/{problem_name}-cycle_stripper-deltas-no-ls-best.png'
            )

    t_end = time()
    print(f'duration of whole experiment: {t_end - t_start}')
