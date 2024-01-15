import numpy as np
import pandas as pd
import logging
from abc import ABC, abstractmethod


from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np,ndarray):
        pass



class MSE(Evaluation):

    def calculate_scores(self, y_true: np.ndarray, y_pred:  np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_pred=y_pred, y_true= y_true)
            logging.info(f"MSE: {mse}")
            return mse

        except Exception as e:
            logging.error(f"Error while calculating MSE {e}")
            raise e

class R2Score(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculatin r2 Score")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2 Score {round(r2 * 100)}%")
            return r2
        except Exception as e:
            logging.error(f"Error while calculating R2 Score {e}")
            raise e