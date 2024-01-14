from zenml import pipeline
from steps.data_ingestion import ingest_data
from steps.data_processing import clean_data
from steps.model_train import train_data
from steps.evaluation import evaluate_model
import logging

@pipeline
def training_pipeline(data_path: str):
    logging.info("Training Pipeline Started")
    df = ingest_data(data_path)
    cX_train, X_test, y_train, y_test  = clean_data(df)
    

