from assignment6.multiple_start_local_search import multiple_start_local_search_solve
from assignment6.iterated_local_search import iterated_local_search_solve
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

EXPERIMENTS_RESULTS_FOLDER = 'experiments_results/assignment6'

if __name__ == '__main__':
    t_start = time()

    problems = {'TSPA': TSP.load_tspa(), 'TSPB': TSP.load_tspb()}
    print('(data loaded)')
    run, total_time = 1, 0

    local_search_type = LocalSearchType.STEEPEST
    starting_solution_type = StartingSolutionType.RANDOM
    intra_route_move_type = IntraRouteMovesType.TWO_EDGES

    # for problem_name, tsp in problems.items():
    #     solutions, times, all_stats = [], [], []
    #     for i in range(20):
    #         t0 = time()
    #         solution, _ = multiple_start_local_search_solve(
    #             tsp,
    #             local_search_type=local_search_type,
    #             starting_solution_type=starting_solution_type,
    #             intra_route_move_type=intra_route_move_type,
    #             starting_node=i,
    #             number_of_starts=200,
    #         )
    #         t1 = time()
    #         solutions.append(solution)
    #         times.append(t1 - t0)
    #         total_time += t1 - t0
    #         print(f'run:\t{run},\ttime: {times[-1]},\ttotal_time: {total_time}')
    #         run += 1
    #
    #     best, worst = min(solutions), max(solutions)
    #
    #     print(f'Multiple start local search')
    #     print(best)
    #
    #     with open(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/pkl/{problem_name}-msls-best.pkl', 'wb') as file:
    #         pickle.dump(best, file)
    #
    #     with open(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/json/{problem_name}-msls.json', 'w') as json_file:
    #         json.dump({
    #             'min_objective_function': best.objective_function,
    #             'max_objective_function': worst.objective_function,
    #             'avg_objective_function': mean([s.objective_function for s in solutions]),
    #             'best_solution_nodes': best.nodes,
    #             'min_time': min(times),
    #             'max_time': max(times),
    #             'avg_time': mean(times),
    #         }, json_file, indent=4, default=lambda x: int(x) if isinstance(x, numpy.int64) else x)
    #
    #     tsp.visualize_solution(
    #         best, method_name=f'{problem_name} Multiple Start Local Search',
    #         path_to_save=f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/png/{problem_name}-msls-best.png')

    for problem_name, tsp in problems.items():
        solutions, times, all_stats, number_of_local_search_runs_list = [], [], [], []
        for i in range(20):
            t0 = time()
            solution, number_of_local_search_runs = iterated_local_search_solve(
                tsp,
                local_search_type=local_search_type,
                starting_solution_type=starting_solution_type,
                intra_route_move_type=intra_route_move_type,
                starting_node=i,
            )
            t1 = time()
            solutions.append(solution)
            number_of_local_search_runs_list.append(number_of_local_search_runs)
            times.append(t1 - t0)
            total_time += t1 - t0
            print(f'run:\t{run},\ttime: {times[-1]},\ttotal_time: {total_time}')
            run += 1

        best, worst = min(solutions), max(solutions)

        print(f'Iterated Local Search')
        print(best)

        with open(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/pkl/{problem_name}-ils-best.pkl', 'wb') as file:
            pickle.dump(best, file)

        with open(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/json/{problem_name}-ils.json', 'w') as json_file:
            json.dump({
                'min_objective_function': best.objective_function,
                'max_objective_function': worst.objective_function,
                'avg_objective_function': mean([s.objective_function for s in solutions]),
                'best_solution_nodes': best.nodes,
                'min_time': min(times),
                'max_time': max(times),
                'avg_time': mean(times),
                'min_number_of_local_search_runs': min(number_of_local_search_runs_list),
                'max_number_of_local_search_runs': max(number_of_local_search_runs_list),
                'avg_number_of_local_search_runs': mean(number_of_local_search_runs_list),
            }, json_file, indent=4, default=lambda x: int(x) if isinstance(x, numpy.int64) else x)

        tsp.visualize_solution(
            best, method_name=f'{problem_name} Iterated Local Search',
            path_to_save=f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/png/{problem_name}-ils-best.png')

    t_end = time()
    print(f'duration of whole experiment: {t_end - t_start}')
