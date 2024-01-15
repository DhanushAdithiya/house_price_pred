import logging
import pandas as pd
from zenml import step
from typing import Tuple
from typing_extensions import Annotated
from src.evaluation_data import  MSE, R2Score

from sklearn.base import RegressorMixin


@step
def evaluate_model(model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Annotated[float, "MSE"],Annotated[float, "R2 Score"]]:

    try:
        predicted = model.predict(X_test)
        MSE_class = MSE()
        mse = MSE_class.calculate_scores(y_test, predicted)


        R2_class = R2Score()
        r2 = R2_class.calculate_scores(y_test, predicted)

        return mse, r2
    except Exception as e:
        logging.error(f"error occured while evaluation {e}")