import logging
import pandas as pd
from zenml import step

from src.data_cleaning import DataCleaning, DataStrategy, split_data, preprocess_strategy
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    try:
        process_data = preprocess_strategy()
        data_cleaning = DataCleaning(df, process_data)
        processed_data = data_cleaning.handle_data()


        data_split = split_data()
        data_cleaning = DataCleaning(processed_data, data_split)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        
        logging.info("data cleaning completed")
        return  X_train, X_test, y_train, y_test 

    except Exception as e:
        logging.log(f"Error in cleaning data {e}")
        raise e 