import os

import numpy as np

import logger
import music
from algorithms.markov_chain_model import MarkovChainList
from midi2img import midi2image


def multiple_algorithms():
    # Define all the relevant algorithms here
    algorithm_list = {"Markov": MarkovChainList()}

    # Get the music input vector time series
    original_score = music.load_midi("sample/unfin.mid")
    score_vector_ts = music.to_vector_ts(original_score)

    # Loop over all algorithms
    for algorithm_name in algorithm_list:
        log.info("Currently fitting and predicting using %s", algorithm_name)
        # Set the current algorithm
        algorithm = algorithm_list[algorithm_name]

        # TODO: Fill in the relevant parameters
        algorithm.fit(score_vector_ts, score_vector_ts[1 : len(score_vector_ts)])

        # TODO: Visualize the results of each algorithm
        future_states = algorithm.predict(score_vector_ts)

        # Show the resulting midi file
        # music.from_vector_ts(future_states).show("midi")


def single_use_function():
    # Get the music input vector time series
    voice = 3
    original_score = music.load_midi("sample/unfin.mid")
    score_vector_ts = music.to_vector_ts(original_score)

    mcl = MarkovChainList()

    mcl.fit(score_vector_ts)

    # predicted states to numpy array
    predicted_states = mcl.model[voice].generate_states(score_vector_ts[0, voice], 128)
    predicted_states = np.array([predicted_states]).T

    predicted = music.from_vector_ts(predicted_states)
    fp = predicted.write("midi", fp="markov_chain_output_voice_3.mid")

    midi2image(
        "markov_chain_output_voice_3.mid", "markov_chain_output_voice_3.png", 0, 128
    )


def main():
    single_use_function()


if __name__ == "__main__":
    log = logger.setup_logger(__name__)
    log.info("Starting...")

    # Run the main
    main()
