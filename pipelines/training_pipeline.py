from zenml import pipeline
from steps.data_ingestion import ingest_data
from steps.data_processing import clean_data
from steps.model_train import train_data
from steps.evaluation import evaluate_model
import logging

@pipeline(enable_cache=False)
def training_pipeline(data_path: str):
    logging.info("Training Pipeline Started")
    df = ingest_data(data_path)
    X_train, X_test, y_train, y_test  = clean_data(df)
    model = train_data(X_train, y_train)
    mse = evaluate_model(model, X_test, y_test)
    logging.info(f"MSE OF THE MODEL {mse}")

    

