import pandas as pd
import matplotlib.pyplot as plt
from itertools import product


def plot_similarities(problem_name: str, similarity: str, baseline_solution: str):
    df = pd.read_csv(f'../experiments_results/assignment8/{problem_name}/{baseline_solution}.csv')

    plt.figure(figsize=(10, 6))
    plt.scatter(df['objective_function'], df[similarity], marker='o', s=20, label=similarity)

    plt.title(f'{problem_name} - {baseline_solution} similarity')
    plt.xlabel('Objective Function')
    plt.ylabel(similarity)
    plt.grid(True, linestyle='--', alpha=0.6)
    # plt.legend()

    plt.savefig(f'../experiments_results/assignment8/{problem_name}/{baseline_solution}_{similarity}.png')


if __name__ == "__main__":
    problem_names = ['TSPA', 'TSPB']
    similarities = ['nodes_similarity', 'edges_similarity']
    baseline_solutions = ['best', 'avg']

    for problem_name, similarity, baseline_solution in product(problem_names, similarities, baseline_solutions):
        plot_similarities(problem_name=problem_name, similarity=similarity, baseline_solution=baseline_solution)
