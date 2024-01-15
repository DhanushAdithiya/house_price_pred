import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import logging

from sklearn.linear_model import LinearRegression

class ModelTrainer(ABC):

    @abstractmethod
    def train(self, X_train, y_train):
        pass


class LinearRegressionModel(ModelTrainer):

    def train(self, X_train, y_train, **kwargs) :

        try:
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            logging.info("Linear Regression Fitting Completed")
            return lr

        except Exception as e:
            logging.error(f"Error occured while fitting the model {e}")
            raise e 