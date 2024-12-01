from assignment7.large_scale_neighborhood_search import large_scale_neighborhood_search_solve
from assignment3.local_search_types import (
    LocalSearchType,
    StartingSolutionType,
    IntraRouteMovesType,
)
from data_loader import TSP

from time import time
import pickle
from numpy import mean
import numpy
import json

EXPERIMENTS_RESULTS_FOLDER = 'experiments_results/assignment7'

if __name__ == '__main__':
    t_start = time()

    problems = {'TSPA': TSP.load_tspa(), 'TSPB': TSP.load_tspb()}
    print('(data loaded)')
    run, total_time = 1, 0

    local_search_type = LocalSearchType.STEEPEST
    starting_solution_type = StartingSolutionType.RANDOM
    intra_route_move_type = IntraRouteMovesType.TWO_EDGES

    for problem_name, tsp in problems.items():
        for should_use_local_search in [False, True]:
            solutions, times, all_stats, numbers_of_main_loop_runs = [], [], [], []
            for i in range(20):
                t0 = time()
                solution, number_of_main_loop_runs = large_scale_neighborhood_search_solve(
                    tsp,
                    local_search_type=local_search_type,
                    starting_solution_type=starting_solution_type,
                    intra_route_move_type=intra_route_move_type,
                    starting_node=i,
                    should_use_local_search=should_use_local_search,
                )
                t1 = time()
                solutions.append(solution)
                numbers_of_main_loop_runs.append(number_of_main_loop_runs)
                times.append(t1 - t0)
                total_time += t1 - t0
                print(f'run:\t{run},\ttime: {times[-1]},\ttotal_time: {total_time},\tobjective_function: {solution.objective_function}')
                run += 1

            best, worst = min(solutions), max(solutions)

            print(f'Large-Scale Neighborhood Search - LS={should_use_local_search}')
            print(best)

            with open(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/pkl/{problem_name}-lsns-{should_use_local_search}-best.pkl', 'wb') as file:
                pickle.dump(best, file)

            with open(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/json/{problem_name}-lsns-{should_use_local_search}.json', 'w') as json_file:
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
                method_name=f'{problem_name} Large-Scale Neighborhood Search - LS={should_use_local_search}',
                path_to_save=f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/png/{problem_name}-lsns-{should_use_local_search}-best.png'
            )

    t_end = time()
    print(f'duration of whole experiment: {t_end - t_start}')
