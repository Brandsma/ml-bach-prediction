""" This module contains an abstract class for a prediction model. """

from abc import ABC, abstractmethod


def PredictionModel(ABC):
    def __init__():
        pass

    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError(
            "fit function for prediction model has not been implemented"
        )

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError(
            "predict function for prediction model has not been implemented"
        )
