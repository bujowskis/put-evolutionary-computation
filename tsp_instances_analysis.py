from data_loader import TSP
import matplotlib.pyplot as plt
import json
import numpy


EXPERIMENTS_RESULTS_FOLDER = 'experiments_results/instances_analysis'

if __name__ == '__main__':
    problems = {
        'TSPA': TSP.load_tspa(),
        'TSPB': TSP.load_tspb()
    }

    for problem_name, tsp in problems.items():
        print(problem_name)

        min_additional_cost = tsp.additional_costs.min()
        max_additional_cost = tsp.additional_costs.max()
        avg_additional_cost = tsp.additional_costs.mean()

        print(f'Minimum additional cost: {min_additional_cost}')
        print(f'Maximum additional cost: {max_additional_cost}')
        print(f'Average additional cost: {avg_additional_cost}')

        with open(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/statistics.json', 'w') as json_file:
            json.dump({
                'max_additional_cost': max_additional_cost,
                'min_additional_cost': min_additional_cost,
                'avg_additional_cost': avg_additional_cost,
            }, json_file, indent=4, default=lambda x: int(x) if isinstance(x, numpy.int64) else x)

        # histogram of additional costs
        plt.figure(figsize=(10, 6))
        plt.hist(tsp.additional_costs, bins=20, color='skyblue', edgecolor='black')
        plt.title(f'Distribution of Additional Costs - {problem_name}')
        plt.xlabel('Additional Cost')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/additional-costs-distribution.png')

        # heatmap of distances_matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(tsp.distances_matrix, cmap='viridis', interpolation='none')
        plt.colorbar(label='Distance')
        plt.title(f'Heatmap of Distance Matrix - {problem_name}')
        plt.xlabel('node')
        plt.ylabel('node')
        plt.savefig(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/distance-matrix-heatmap.png')

        # heatmap of total_move_costs
        plt.figure(figsize=(10, 8))
        plt.imshow(tsp.total_move_costs, cmap='inferno', interpolation='none')
        plt.colorbar(label='Move Cost')
        plt.title(f'Heatmap of Total Move Costs - {problem_name}')
        plt.xlabel('node')
        plt.ylabel('node')
        plt.savefig(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/total-move-costs-heatmap.png')

        # mean along 3rd dimension of insertion costs
        mean_insertion_costs = numpy.mean(tsp.insertion_costs, axis=-1)

        # Plot and save heatmap of the mean insertion costs
        plt.figure(figsize=(10, 8))
        plt.imshow(mean_insertion_costs, cmap='plasma', interpolation='none')
        plt.colorbar(label='Mean Insertion Cost')
        plt.title(f'Mean Insertion Costs Heatmap - {problem_name}')
        plt.xlabel('node')
        plt.ylabel('node')
        plt.savefig(f'{EXPERIMENTS_RESULTS_FOLDER}/{problem_name}/mean-insertion-costs-heatmap.png')

