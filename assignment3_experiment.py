from assignment3.local_search import local_search_solve
from assignment3.local_search_types import (
    LocalSearchType,
    StartingSolutionType,
    IntraRouteMovesType,
)
from data_loader import TSP

from time import time
from itertools import product
import pickle
from numpy import mean
import numpy
import json
import pandas as pd

EXPERIMENTS_RESULTS_FOLDER = 'experiments_results/assignment3'

if __name__ == '__main__':
    t_start = time()

    problems = {'TSPA': TSP.load_tspa(), 'TSPB': TSP.load_tspb()}
    print('(data loaded)')
    run, total_time = 1, 0

    config_combinations = list(product(LocalSearchType, StartingSolutionType))
    intra_route_move_type = IntraRouteMovesType.TWO_NODES

    for local_search_type, starting_solution_type in config_combinations:
        lst, sst, irmt = str(local_search_type).split('.')[1], str(starting_solution_type).split('.')[1], str(intra_route_move_type).split('.')[1]
        short_readable_config = f'({", ".join([lst, sst, irmt])})'
        short_file_config = f'({"-".join([lst, sst, irmt])})'
        print(f'{local_search_type} {starting_solution_type} {intra_route_move_type}')
        for problem_name, tsp in problems.items():
            solutions, times, all_stats = [], [], []
            for i in range(len(tsp.nodes)):
                t0 = time()
                solution, stats = local_search_solve(
                    tsp,
                    local_search_type=local_search_type,
                    starting_solution_type=starting_solution_type,
                    intra_route_move_type=intra_route_move_type,
                    starting_node=i,
                )
                t1 = time()
                solutions.append(solution)
                times.append(t1 - t0)
                all_stats.append(stats)
                total_time += t1 - t0
                print(f'run:\t{run},\ttime: {times[-1]},\ttotal_time: {total_time}')
                run += 1

            best, worst = min(solutions), max(solutions)

            print(f'Local Search {short_readable_config}')
            print(best)
            #print(best.nodes_in_excel_format())

            with open(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/pkl/{problem_name}-ls-{short_file_config}-best.pkl', 'wb') as file:
                pickle.dump(best, file)

            ls_stats_df = pd.DataFrame(all_stats)
            with open(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/json/{problem_name}-ls-{short_file_config}.json', 'w') as json_file:
                json.dump({
                    'min_objective_function': best.objective_function,
                    'max_objective_function': worst.objective_function,
                    'avg_objective_function': mean([s.objective_function for s in solutions]),
                    'best_solution_nodes': best.nodes,
                    'min_time': min(times),
                    'max_time': max(times),
                    'avg_time': mean(times),
                    'ls_stats_min': ls_stats_df.min().to_dict(),
                    'ls_stats_max': ls_stats_df.max().to_dict(),
                    'ls_stats_avg': ls_stats_df.mean().to_dict(),
                }, json_file, indent=4, default=lambda x: int(x) if isinstance(x, numpy.int64) else x)

            tsp.visualize_solution(
                best, method_name=f'{problem_name} Local Search {short_readable_config}',
                path_to_save=f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/png/{problem_name}-ls-{short_file_config}-best.png')

    t_end = time()
    print(f'duration of whole experiment: {t_end - t_start}')
