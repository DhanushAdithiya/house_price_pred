import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from abc import ABC, abstractmethod
from typing import Union


class DataStrategy(ABC):

    @abstractmethod
    def handle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class preprocess_strategy(DataStrategy):
    def handle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            le = LabelEncoder() 
            df.drop(['prefarea'], axis=1, inplace=True)
            cat = []
            for cols in df.columns:
                if not pd.api.types.is_numeric_dtype(df[cols]):
                    cat.append(cols)
            
            for column in cat:
                df[column] = le.fit_transform(df[column])
            
            return df
        except Exception as e:
            logging.error(f"Error Occured while preprocessing data: {e}")
            raise e



class split_data(DataStrategy):
    def handle_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            X = df.drop(['price'], axis=1)
            y = df['price']

            # TODO Get test_size from a yaml.config file later
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            return X_train, X_test, y_train, y_test 

        except Exception as e:
            logging.error(f"error occured while spliting data: {e}")
            raise e


class DataCleaning:

    def __init__(self, df: pd.DataFrame, strategy: DataStrategy) -> None:
        self.data = df
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame,pd.Series]:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in handling data {e}")
            raise e
