import math
import optuna
import pathlib
import pickle
import mlflow
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from prefect import flow, task
import os

@task(name="Read Data")
def read_data(file_path: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    df = pd.read_parquet(file_path)

    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df

@task(name="Add Features")
def add_features(df_train: pd.DataFrame, df_val: pd.DataFrame):
    """Add features to the model"""
    df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
    df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]

    categorical = ["PU_DO"]  #'PULocationID', 'DOLocationID']
    numerical = ["trip_distance"]

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values
    return X_train, X_val, y_train, y_val, dv


@task(name="Train Challenger Model")
def train_challenger_model(X_train, X_val, y_train, y_val, dv, experiment_id: str) -> str:
    """Entrena un modelo con un set de hiperparámetros fijo (Challenger) y lo registra."""
    
    params = {
        "max_depth": 10,
        "learning_rate": 0.1,
        "reg_alpha": 0.001,
        "reg_lambda": 0.0001,
        "min_child_weight": 1.0,
        "objective": "reg:squarederror",
        "seed": 42,
    }

    with mlflow.start_run(experiment_id=experiment_id, run_name="Challenger Model Training"):
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)
        
        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, "validation")],
            early_stopping_rounds=10,
            verbose_eval=False
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_params(params)
        
        mlflow.xgboost.log_model(booster, "model", input_example=X_val[:5], signature=None)
        
        return mlflow.active_run().info.run_id

@task(name="Evaluate and Promote Model")
def evaluate_and_promote(X_val, y_val, challenger_run_id: str):
    """Compara el Challenger con el Champion y promueve al mejor como el nuevo @champion."""

    MODEL_NAME = "workspace.default.nyc-taxi-model-prefect"
    client = MlflowClient()
    
    challenger_run = client.get_run(challenger_run_id)
    challenger_rmse = challenger_run.data.metrics["rmse"]
    print(f"Challenger RMSE: {challenger_rmse:.4f} (Run ID: {challenger_run_id})")

    try:
        champion_version = client.get_model_version_by_alias(MODEL_NAME, "champion")
        champion_run_id = champion_version.run_id
        
        champion_run = client.get_run(champion_run_id)
        champion_rmse = champion_run.data.metrics["rmse"]
        print(f"Current Champion RMSE: {champion_rmse:.4f} (Version: {champion_version.version})")

    except Exception as e:
        print(f"No se encontró un Champion existente (o error de acceso): {e}. Promoviendo Challenger por defecto.")
        champion_rmse = float('inf') 
        champion_version = None


    # --- 3. Comparación y Promoción ---
    
    if challenger_rmse < champion_rmse:
        print("Challenger supero al champion.")
        
        model_uri = f"runs:/{challenger_run_id}/model"
        
        try:
            client.get_registered_model(name=MODEL_NAME)
        except Exception:
            client.create_registered_model(name=MODEL_NAME)

        new_model_version = client.create_model_version(
            name=MODEL_NAME, 
            source=model_uri, 
            run_id=challenger_run_id
        )
        
        client.set_registered_model_alias(
            name=MODEL_NAME, 
            alias="champion", 
            version=new_model_version.version
        )
        
        print(f"✅ Nuevo Champion: Versión {new_model_version.version} con RMSE {challenger_rmse:.4f}")

        if champion_version:
             client.set_registered_model_alias(
                name=MODEL_NAME, 
                alias="previous_champion", 
                version=champion_version.version
            )
    else:
        print("El Champion actual mantiene su título.")

@flow(name="Challenger Champion Flow")
def challenger_champion_flow(year: int, month_train: str, month_val: str) -> None:
    """The main challenger-champion pipeline"""
    
    train_path = f"../data/green_tripdata_{year}-{month_train}.parquet"
    val_path = f"../data/green_tripdata_{year}-{month_val}.parquet"
    
    load_dotenv(override=True)
    EXPERIMENT_NAME = "/Users/estebangmzv@gmail.com/nyc-taxi-experiment-prefect"
    MODEL_NAME = "main.default.nyc-taxi-model-prefect" # Nombre completo de Unity Catalog
    
    mlflow.set_tracking_uri("databricks")
    experiment = mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id

    df_train = read_data(train_path)
    df_val = read_data(val_path)
    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)

    challenger_run_id = train_challenger_model(X_train, X_val, y_train, y_val, dv, experiment_id)
    
    evaluate_and_promote(X_val, y_val, challenger_run_id)


if __name__ == "__main__":
    challenger_champion_flow(year=2025, month_train="01", month_val="02")