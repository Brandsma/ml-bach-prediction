""" This module contains an ESN """

import importlib.util
import os

import numpy as np
from easyesn import PredictionESN

from algorithm_base import PredictionModel


class ESN(PredictionModel):
    """ This is a simple wrapper function for the easyesn library """

    def __init__(
        self,
        transient_time=100,
        verbose=True,
        n_reservoir=500,
        leaking_rate=0.2,
        spectral_radius=0.2,
        regression_parameters=[1e-2],
    ):
        super().__init__()
        # TODO: Add some input validation
        self.transient_time = transient_time
        self.verbose = 1 if self.verbose else 0
        self.model = None
        self.n_reservoir = n_reservoir
        self.leaking_rate = leaking_rate
        self.spectral_radius = spectral_radius
        self.regression_parameters = regression_parameters

    def fit(self, X, y):
        # If we have cupy, then we'll use that
        # cupy is GPU accelerated calculations, which would make the code much faster
        if (spec := importlib.util.find_spec("cupy")) is not None:
            os.environ["EASYESN_BACKEND"] = "cupy"
        else:
            print(
                "Note: if you have CUDA drivers setup, then \
            install the relevant cupy version (cupy-XX)"
            )
            os.environ["EASYESN_BACKEND"] = "numpy"

        print(np.shape(X))

        # self.model = PredictionESN(
        #     n_input=1,
        #     n_output=1,
        #     n_reservoir=self.n_reservoir,
        #     leakingRate=self.leaking_rate,
        #     spectralRadius=self.spectral_radius,
        #     regressionParameters=self.regression_parameters,
        # )

    def predict(self, X):
        pass


if __name__ == "__main__":
    esn = ESN()
    esn.fit()
