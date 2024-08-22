import copy
import os
import random
import time

from tqdm import tqdm

from SearchAlgorithms import *
from heuristics import BootstrappingHeuristic, BaseHeuristic
from topspin import TopSpinState

def bootstrappingTraining(bootstrapping_heuristic, algorithm='A*', weight=5, batch=10, t_number=1_000, num_iterations=200, number_of_states_samples=1_000):
    """
    Train the bootstrapping heuristic using the base heuristic
    :param bootstrapping_heuristic: heuristics.BootstrappingHeuristic
    :param algorithm: algorithm to use for generating data
    :param weight: int, default 5 (weight for the heuristic)
    :param batch: int default 10 (batch size)
    :param t_number: int default 1000 (maximum number of expansions)
    :param num_iterations: int default 200 (number of iterations)
    :param number_of_states_samples: int default 1000 (number of states to sample)
    :return: None
    """

    def generate_data_set():
        """
        Generate training data for the bootstrapping heuristic
        :return: tuple
        """
        inputs = []
        outputs = []

        states = bootstrapping_heuristic.generate_random_states()
        states = random.sample(states, number_of_states_samples)
        b = BaseHeuristic(bootstrapping_heuristic.n, bootstrapping_heuristic.k)
        for state in states:
            W, B, T = weight, batch, t_number
            if algorithm == 'A*':
                path, expansions = AStar(state, b.get_h_values, T)
            elif algorithm == 'BWA*':
                path, expansions = BWAS(state, W, B, b.get_h_values, T)
            else:
                path, expansions = WAStar(state, W, b.get_h_values, T)
            if path is None:
                while path is None:
                    T *= 2
                    path, expansions = BWAS(state, W, B, b.get_h_values, T)

            n = len(path)
            for i, node in enumerate(path):
                if i == n - 1:
                    break
                inputs.append(TopSpinState(node, bootstrapping_heuristic.k))
                outputs.append(n - (i + 1))

        indices = np.random.permutation(len(inputs))
        inputs_shuffle = np.array(inputs)[indices]
        outputs_shuffle = np.array(outputs)[indices]

        return inputs_shuffle, outputs_shuffle

    prev_loss = float('inf')
    counter = 0
    for _ in tqdm(range(num_iterations)):  # add early stopping
        if os.path.exists(bootstrapping_heuristic.path):
            bootstrapping_heuristic.load_model()

        input_data, output_labels = generate_data_set()
        loss = bootstrapping_heuristic.train_model(input_data, output_labels, epochs=100)
        if prev_loss <= loss:
            counter += 1

        prev_loss = loss

        if counter == 100:
            break

        bootstrapping_heuristic.save_model()
