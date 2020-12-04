import os

import numpy as np

import music
from algorithms import ESN, MarkovChain


def main():
    # Define all the relevant algorithms here
    # TODO: Fill the parameters for the constructor
    algorithm_list = [MarkovChain(), ESN()]

    # TODO: Get the music input vector

    # Loop over all algorithms
    for algorithm in algorithm_list:
        # TODO: Fill in the relevant parameters
        algorithm.fit()

        # TODO: Visualize the results of each algorithm
        algorithm.predict()


if __name__ == "__main__":
    main()
