import numpy as np
from algorithm.algorithm_base import PredictionModel


def generate_markov_chain(X):
    markov_chain_list = []
    return markov_chain_list


class MarkovChain(PredictionModel):
    def __init__(self, transition_matrix, states):
        """
        Initialize the MarkovChain instance.

        Parameters
        ----------
        transition_matrix: 2-D array
            A 2-D array representing the probabilities of change of
            state in the Markov Chain.

        states: 1-D array
            An array representing the states of the Markov Chain. It
            needs to be in the same order as transition_matrix.
        """
        self.transition_matrix = np.atleast_2d(transition_matrix)
        self.states = states
        self.index_dict = {
            self.states[index]: index for index in range(len(self.states))
        }
        self.state_dict = {
            index: self.states[index] for index in range(len(self.states))
        }
        self.model = []

    def generate_markov_chain_for_one_voice(self, X, voice=3):
        selected_voice = np.array([int(x) for x in X[:, voice]])

        # Retrieve all the states where we can go to
        states = np.unique(selected_voice)
        # Build a transition matrix where every probability is zero initially
        transition_matrix = np.zeros((len(states), len(states)))

        # Find all occurences of the transition matrix
        index_dict = {states[index]: index for index in range(len(transition_matrix))}
        for idx, note in enumerate(selected_voice):
            if idx + 1 >= len(selected_voice):
                continue
            transition_matrix[
                index_dict[note], index_dict[selected_voice[idx + 1]]
            ] += 1

        # Normalize the rows of the matrix so each probability row sums to 1
        for idx, row in enumerate(transition_matrix):
            transition_matrix[idx] = row / np.linalg.norm(row, ord=1)

        # Create the markov chain using the states and the transition matrix
        return MarkovChain(transition_matrix, states)

    def fit(self, X, y):
        log.info(np.shape(X))
        log.info(np.shape(y))

        # for idx in range(4):
        #     self.model.append(self.generate_markov_chain_for_one_voice(X, idx))

    def next_state(self, current_state):
        """
        Returns the state of the random variable at the next time
        instance.

        Parameters
        ----------
        current_state: str
            The current state of the system.
        """
        return np.random.choice(
            self.states, p=self.transition_matrix[self.index_dict[current_state], :]
        )

    def generate_states(self, current_state, no=10):
        """
        Generates the next states of the system.

        Parameters
        ----------
        current_state: str
            The state of the current random variable.

        no: int
            The number of future states to generate.
        """
        future_states = []
        for i in range(no):
            next_state = self.next_state(current_state)
            future_states.append(next_state)
            current_state = next_state
        return future_states

    def predict(self, X):
        pass
