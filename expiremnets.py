import csv

from training import *
from SearchAlgorithms import *


def run_experiment(w, b, heuristic_name, heuristic, states, algo="A*", T=1000000):
    runtimes = []
    path_lengths = []
    expansions = []

    for state in states:
        start_time = time.time()
        if algo == "A*":
            path, num_expansions = AStar(state, heuristic.get_h_values, T)
        elif  algo == "WA*":
            path, num_expansions= WAStar(state, w, heuristic.get_h_values, T)
        else: # elif algo == "BWA*":
            path, num_expansions = BWAS(state, w, b, heuristic.get_h_values, T)
        print(num_expansions)
        end_time = time.time()
        if path is not None:
            runtimes.append(end_time - start_time)
            path_lengths.append(len(path) - 1)
            expansions.append(num_expansions)

    avg_runtime = round(sum(runtimes) / len(runtimes) if runtimes else float('inf'), 2)
    avg_path_length = sum(path_lengths) / len(path_lengths) if path_lengths else float('inf')
    avg_expansions = round(sum(expansions) / len(expansions) if expansions else float('inf'), 2)

    return avg_runtime, avg_path_length, avg_expansions


def main():
    w_values = [1]
    #b_values = [1,100]
    #w_values = [1]
    b_values = [1]
    Bootstrapping = BootstrappingHeuristic(11, 4, path="path_to_trained_agent")
    Bootstrapping.load_model()
    heuristics = [
        ("basic", BaseHeuristic(11, 4)),
        ("learned-bootstrap", Bootstrapping),

    ]

    test_states = Bootstrapping.generate_random_states_test()[:100]

    results = []
    for w in w_values:
        for b in b_values:
            for heuristic_name, heuristic in heuristics:
                avg_runtime, avg_path_length, avg_expansions = run_experiment(w, b, heuristic_name, heuristic,
                                                                              test_states, algo="A*")
                print(heuristic_name)
                results.append([w, b, heuristic_name, avg_runtime, avg_path_length, avg_expansions])

    # Define the file path
    file_path = "results.csv"

    # Write the results to a CSV file
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["W", "B", "Heuristic", "Avg. Runtime", "Avg. Path Length", "Avg. # Expansions"])
        writer.writerows(results)

    print(f"CSV file saved to {file_path}")


if __name__ == "__main__":
    main()
