from assignment10.cycle_stripper_long_running import cycle_stripper_long_running_solve
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
        for ls_interval in [1, 3, 5, 10]:  # [-1]:
            solutions, times, numbers_of_main_loop_runs = [], [], []
            for i in range(20):
                t0 = time()
                solution, number_of_main_loop_runs = cycle_stripper_long_running_solve(tsp=tsp, ls_interval=ls_interval, should_run_ls_at_end=True)
                t1 = time()
                solutions.append(solution)
                numbers_of_main_loop_runs.append(number_of_main_loop_runs)
                times.append(t1 - t0)
                total_time += t1 - t0
                print(f'run:\t{run},\ttime: {times[-1]},\ttotal_time: {total_time},\tobjective_function: {solution.objective_function}')
                run += 1

            best, worst = min(solutions), max(solutions)

            print(f'Cycle stripper - long_running, ls_interval={ls_interval}')
            # print(f'Cycle stripper - long_running, no main ls')
            # print(f'Cycle stripper - long_running, no ls')
            print(best)

            with open(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/pkl/{problem_name}-cycle_stripper-long_running-{ls_interval}-best.pkl', 'wb') as file:
            # with open(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/pkl/{problem_name}-cycle_stripper-long_running-no-main-ls-best.pkl', 'wb') as file:
            # with open(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/pkl/{problem_name}-cycle_stripper-long_running-no-ls-best.pkl', 'wb') as file:
                pickle.dump(best, file)

            with open(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/json/{problem_name}-cycle_stripper-long_running-{ls_interval}.json', 'w') as json_file:
            # with open(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/json/{problem_name}-cycle_stripper-long_running-no-main-ls.json', 'w') as json_file:
            # with open(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/json/{problem_name}-cycle_stripper-long_running-no-ls.json', 'w') as json_file:
                json.dump({
                    'min_objective_function': best.objective_function,
                    'max_objective_function': worst.objective_function,
                    'avg_objective_function': mean([s.objective_function for s in solutions]),
                    'best_solution_nodes': best.nodes,
                    'min_time': min(times),
                    'max_time': max(times),
                    'avg_time': mean(times),
                    'min_number_of_main_loop_runs': min(numbers_of_main_loop_runs),
                    'max_number_of_main_loop_runs': max(numbers_of_main_loop_runs),
                    'avg_number_of_main_loop_runs': mean(numbers_of_main_loop_runs),
                }, json_file, indent=4, default=lambda x: int(x) if isinstance(x, numpy.int64) else x)

            tsp.visualize_solution(
                best,
                method_name=f'{problem_name} Cycle stripper - long_running, ls_interval={ls_interval}',
                path_to_save=f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/png/{problem_name}-cycle_stripper-long_running-{ls_interval}-best.png'
                # path_to_save=f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/png/{problem_name}-cycle_stripper-long_running-no-main-ls-best.png'
                # path_to_save=f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/png/{problem_name}-cycle_stripper-long_running-no-ls-best.png'
            )

    t_end = time()
    print(f'duration of whole experiment: {t_end - t_start}')
