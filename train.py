import os

import logging
import pickle
import yaml
import json

from mlflow.tracking import MlflowClient
import mlflow

import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import requests

from prophet import Prophet
from bs4 import BeautifulSoup
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

CONFIG_PATH = os.path.join('config/params.yaml')
CONFIG = yaml.safe_load(open('config/params.yaml'))['train']

PATH_DATA = "data/moscow.csv"
PATH_UNIQUE_VALUES = "config/unique_values.json"

RAND = CONFIG['random_state']
TEST_SIZE = CONFIG['test_size']
PATH_MODEL = CONFIG['path_model']

DROP_COLS = ["date", "time", "geo_lat", "geo_lon", "region"]
CATEGORICAL_FEATURES = ["building_type", "object_type"]
NUMERIC_FEATURES = ["level", "levels", "rooms", "area", "kitchen_area"]

logging.basicConfig(filename='log/app.log', filemode='w+', format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG)


def get_version_model(config_name, client):
    """
    Get model last version from MLFlow
    """
    dict_push = {}
    for count, value in enumerate(client.search_model_versions(f"name='{config_name}'")):
        # All model versions
        dict_push[count] = value
    return dict(list(dict_push.items())[0][1])['version']


def load_and_train_data(path_data, drop_cols):
    df = pd.read_csv(path_data)
    df = df.drop(columns=drop_cols)
    # df[categorical_features] = df[categorical_features].astype(str)

    # Remove outliers
    df = df[df.price.between(df.price.quantile(0.05), df.price.quantile(0.95))]
    df = df[df.area.between(df.area.quantile(0.01), df.area.quantile(0.99))]
    df = df[df.rooms > -2]

    y = df["price"]
    X = df.drop(columns="price", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=TEST_SIZE,
                                                        shuffle=True,
                                                        random_state=RAND)

    X_train_, X_val, y_train_, y_val = train_test_split(X_train,
                                                        y_train,
                                                        test_size=TEST_SIZE,
                                                        random_state=RAND)

    eval_set = [(X_val, y_val)]

    data_dict = {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "X_val": X_val,
        "y_train": y_train,
        "y_test": y_test,
        "y_val": y_val
    }

    # Create pipeline
    preprocessor = make_column_transformer(
        (StandardScaler(), NUMERIC_FEATURES),
        (OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL_FEATURES),
    )
    cat_optuna = CatBoostRegressor(n_estimators=300,
                                   learning_rate=0.14576781861855528,
                                   max_depth=12,
                                   loss_function='MAE',
                                   eval_metric='MAE',
                                   random_state=10,
                                   allow_writing_files=False
                                   )
    model = Pipeline([('columnTransformer', preprocessor),
                      ('cat', cat_optuna)])

    # Train model
    model.fit(X_train, y_train)
    y_prediction = model.predict(X_test)

    return data_dict, model, mean_absolute_error(y_test, y_prediction)


def main():
    logging.info('Fitting the model')

    # MLFlow tracking
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(CONFIG['name_experiment'])
    with (mlflow.start_run()):
        data, model, mae = load_and_train_data(PATH_DATA, DROP_COLS)

        with open(PATH_MODEL, 'wb') as f:
            pickle.dump(model, f)

        print(f'MAE score: {mae}')

        # Model and parameters logging
        mlflow.log_param('MAE', mae)
        mlflow.sklearn.log_model(model,
                                 artifact_path='model_lr',
                                 registered_model_name=f"{CONFIG['model_lr']}")
        mlflow.log_artifact(local_path='./train.py',
                            artifact_path='code')
        mlflow.end_run()

    # Get model last version and save to files
    client = MlflowClient()
    last_version_lr = get_version_model(CONFIG['model_lr'], client)

    yaml_file = yaml.safe_load(open(CONFIG_PATH))
    yaml_file['predict']["version_lr"] = int(last_version_lr)

    with open(CONFIG_PATH, 'w') as fp:
        yaml.dump(yaml_file, fp, encoding='UTF-8', allow_unicode=True)

    # Save unique values
    dict_unique = {key: data["X"][key].unique().tolist() for key in data["X"].columns}

    with open(PATH_UNIQUE_VALUES, "w") as file:
        json.dump(dict_unique, file)

    return data, model, mae

if __name__ == "__main__":
    main()
