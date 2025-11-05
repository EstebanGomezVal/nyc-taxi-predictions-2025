import os
import math
import optuna
import pathlib
import pickle
import mlflow
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from optuna.samplers import TPESampler
from mlflow.models.signature import infer_signature
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from prefect import flow, task
import time

EXPERIMENT_NAME = "/Users/estebangmzv@gmail.com/nyc-taxi-experiment-prefect"
MODEL_NAME_UC = "main.default.nyc-taxi-model-prefect" 

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

@task(name="Tune Model Family")
def tune_model_family(X_train, X_val, y_train, y_val, model_family: str):
    """Realiza la optimización de hiperparámetros para una familia de modelos (RF o GB)."""
    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.trial.Trial):
        with mlflow.start_run(nested=True):
            mlflow.set_tag("model_family", model_family)
            
            if model_family == "random_forest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 30),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
                    "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                    "random_state": 42,
                    "n_jobs": -1
                }
                model = RandomForestRegressor(**params)
            
            elif model_family == "gradient_boosting":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "max_depth": trial.suggest_int("max_depth", 2, 10),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                    "random_state": 42
                }
                model = GradientBoostingRegressor(**params)
            
            else:
                raise ValueError(f"Modelo no soportado: {model_family}")

            mlflow.log_params(params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            rmse = root_mean_squared_error(y_val, y_pred)
            mlflow.log_metric("rmse", rmse)

        return rmse

    with mlflow.start_run(run_name=f"{model_family.replace('_', ' ').title()} Hyperparameter Optimization (Optuna)"):
        study.optimize(objective, n_trials=3) 
        
        best_params = study.best_params
        best_params["random_state"] = 42
        
        if model_family == "random_forest":
             best_params["n_jobs"] = -1
        
        return best_params

@task(name="Train Final Challenger Model")
def train_final_challenger(X_train, X_val, y_train, y_val, dv, best_params: dict, model_family: str) -> str:
    """Entrena el modelo final con los mejores hiperparámetros y registra el run."""
    
    with mlflow.start_run(run_name=f"Challenger Model: {model_family.title()}") as run:
        
        mlflow.log_params(best_params)
        mlflow.set_tags({"model_family": model_family, "challenger_status": "Candidate"})
        
        if model_family == "random_forest":
            model = RandomForestRegressor(**best_params)
        elif model_family == "gradient_boosting":
            model = GradientBoostingRegressor(**best_params)
        else:
            raise ValueError(f"Modelo no soportado: {model_family}")
        
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        # Guardar preprocesador (artefacto)
        pathlib.Path("preprocessor").mkdir(exist_ok=True)
        with open("preprocessor/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("preprocessor/preprocessor.b", artifact_path="preprocessor")

        # Registrar modelo (artefacto)
        feature_names = dv.get_feature_names_out()
        input_example = pd.DataFrame(X_val[:5].toarray(), columns=feature_names)
        signature = infer_signature(input_example, y_val[:5])

        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=input_example,
            signature=signature,
        )
        return run.info.run_id

@task(name="Compare and Promote Champion")
def compare_and_promote(experiment_id: str):
    """Busca el mejor modelo de todo el experimento, lo promueve como @champion, 
       y asigna el alias @challenger al segundo mejor."""
    
    client = MlflowClient()
    
    all_best_runs_df = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="metrics.rmse IS NOT NULL", 
        order_by=["metrics.rmse ASC"],
    )
    
    champion_run = all_best_runs_df.iloc[0]
    champion_run_id = champion_run["run_id"]
    
    model_uri = f"runs:/{champion_run_id}/model"
    
    try:
        client.get_registered_model(name=MODEL_NAME_UC)
    except Exception as e:
        if "CATALOG_DOES_NOT_EXIST" in str(e) or "INVALID_PARAMETER_VALUE" in str(e):
             return
        client.create_registered_model(name=MODEL_NAME_UC)
        
    new_model_version = client.create_model_version(
        name=MODEL_NAME_UC, 
        source=model_uri, 
        run_id=champion_run_id
    )
    
    client.set_registered_model_alias(
        name=MODEL_NAME_UC, 
        alias="champion", 
        version=new_model_version.version
    )
    

@flow(name="Challenger Comparison and Promotion Flow")
def challenger_comparison_flow(year: int, month_train: str, month_val: str) -> None:
    """The main flow to train two challenger models (RF/GB) and promote the best."""
    
    train_path = f"../data/green_tripdata_{year}-{month_train}.parquet"
    val_path = f"../data/green_tripdata_{year}-{month_val}.parquet"
    
    load_dotenv(override=True)
    
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id

    df_train = read_data(train_path)
    df_val = read_data(val_path)
    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)
    
    
    best_params_rf = tune_model_family(X_train, X_val, y_train, y_val, "random_forest")
    best_params_gb = tune_model_family(X_train, X_val, y_train, y_val, "gradient_boosting")
    
    train_final_challenger(X_train, X_val, y_train, y_val, dv, best_params_rf, "random_forest")
    train_final_challenger(X_train, X_val, y_train, y_val, dv, best_params_gb, "gradient_boosting")
    
    compare_and_promote(experiment_id)

if __name__ == "__main__":
    challenger_comparison_flow(year=2025, month_train="01", month_val="02")