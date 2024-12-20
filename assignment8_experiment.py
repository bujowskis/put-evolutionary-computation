from assignment8.tsp_solution_similarity import (
    measure_common_edges,
    measure_common_nodes,
)
from assignment5.local_search_with_deltas import local_search_with_deltas_solve
from assignment3.local_search_types import (
    LocalSearchType,
    StartingSolutionType,
    IntraRouteMovesType,
)
from data_loader import TSP

from time import time
from numpy import mean
import pandas as pd

EXPERIMENTS_RESULTS_FOLDER = 'experiments_results/assignment8'

if __name__ == '__main__':
    t_start = time()

    problems = {'TSPA': TSP.load_tspa(), 'TSPB': TSP.load_tspb()}
    print('(data loaded)')
    run, total_time = 1, 0

    local_search_type = LocalSearchType.STEEPEST
    starting_solution_type = StartingSolutionType.RANDOM
    intra_route_move_type = IntraRouteMovesType.TWO_EDGES

    for problem_name, tsp in problems.items():
        solutions, times = [], []
        for i in range(1000):
            t0 = time()
            solution, number_of_main_loop_runs = local_search_with_deltas_solve(
                tsp,
                local_search_type=local_search_type,
                starting_solution_type=starting_solution_type,
                intra_route_move_type=intra_route_move_type,
                starting_node=i,
            )
            t1 = time()
            solutions.append(solution)
            times.append(t1 - t0)
            total_time += t1 - t0
            print(f'run:\t{run},\ttime: {times[-1]},\ttotal_time: {total_time},\tobjective_function: {solution.objective_function}')
            run += 1

        # note - 100 nodes and 100 edges are max
        sorted_solutions = sorted(solutions)

        # calculations for best
        # best_solution = sorted_solutions[0]
        # sorted_solutions_without_best = sorted_solutions.copy()[1:]
        # edges_similarities_to_best = [
        #     measure_common_edges(baseline_solution=best_solution, compared_solution=s)
        #     for s in sorted_solutions_without_best
        # ]
        # nodes_similarities_to_best = [
        #     measure_common_nodes(baseline_solution=best_solution, compared_solution=s)
        #     for s in sorted_solutions_without_best
        # ]
        # df_best = pd.DataFrame({
        #     "objective_function": [s.objective_function for s in sorted_solutions_without_best],
        #     "edges_similarity": edges_similarities_to_best,
        #     "nodes_similarity": nodes_similarities_to_best,
        # })
        # df_best.to_csv(f"{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/best.csv")

        # calculations for avg
        edges_similarities_avg = [
            mean([measure_common_edges(baseline_solution=s, compared_solution=compared_solution)
                  for compared_solution in sorted_solutions
                  if compared_solution is not s])
            for s in sorted_solutions
        ]
        nodes_similarities_avg = [
            mean([measure_common_nodes(baseline_solution=s, compared_solution=compared_solution)
                  for compared_solution in sorted_solutions
                  if compared_solution is not s])
            for s in sorted_solutions
        ]
        df_avg = pd.DataFrame({
            "objective_function": [s.objective_function for s in sorted_solutions],
            "edges_similarity": edges_similarities_avg,
            "nodes_similarity": nodes_similarities_avg,
        })
        df_avg.to_csv(f"{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/avg.csv")

    t_end = time()
    print(f'duration of whole experiment: {t_end - t_start}')
