import logging
import pandas as pd
from zenml import step

from src.model_training import LinearRegressionModel, ModelTrainer
from sklearn.base import RegressorMixin


@step
def train_data(X_train: pd.DataFrame, y_train: pd.Series, ) -> RegressorMixin:
    try:
        model = LinearRegressionModel()
        trained_model = model.train(X_train,y_train )       
        return trained_model
    except Exception as e:
        logging.error("Could not train model {e}")
        raise e