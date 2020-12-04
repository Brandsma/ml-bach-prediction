import os

import numpy as np

import music
from algorithms import ESN, MarkovChain


def main():
    # Define all the relevant algorithms here
    # TODO: Fill the parameters for the constructor
    algos = {"Markov" : MarkovChain(), 
            "ESN": ESN()}

    # Get the music input vector time series
    original_score = music.load_midi("sample/unfin.mid")
    score_vector_ts = music.to_vector_ts(original_score)

    # Loop over all algorithms
    for algorithm in algorithm_list:
        # TODO: Fill in the relevant parameters
        algorithm.fit()

        # TODO: Visualize the results of each algorithm
        algorithm.predict()


if __name__ == "__main__":
    main()
