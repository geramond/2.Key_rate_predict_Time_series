import os

import logging
import pickle
import yaml
import json

import mlflow
from mlflow.tracking import MlflowClient

import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np

# from prophet import Prophet
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from src import load_data

import warnings
warnings.filterwarnings('ignore')


CONFIG_PATH = os.path.join('config/params.yaml')
CONFIG = yaml.safe_load(open('config/params.yaml'))['train']

PATH_DATA = "data/key_rate.csv"

RAND = CONFIG['random_state']
TEST_SIZE = CONFIG['test_size']
PATH_MODEL = CONFIG['path_model']



# logging.basicConfig(filename='log/app.log', filemode='w+', format='%(asctime)s : %(levelname)s : %(message)s',
#                     level=logging.DEBUG)


def get_version_model(config_name, client):
    """
    Get model last version from MLFlow
    """
    dict_push = {}
    for count, value in enumerate(client.search_model_versions(f"name='{config_name}'")):
        # All model versions
        dict_push[count] = value
    return dict(list(dict_push.items())[0][1])['version']


def create_features(data_full, col_datetime, cat_type):
    """Creates time series features"""

    data = data_full.copy()
    data['weekday'] = data[col_datetime].dt.day_name().astype(cat_type)
    data['quarter'] = data['datetime'].dt.quarter
    data['month'] = data[col_datetime].dt.month
    data['year'] = data[col_datetime].dt.year
    data['date_offset'] = (data[col_datetime].dt.month * 100 +
                           data.datetime.dt.day - 320) % 1300

    data['season'] = pd.cut(data['date_offset'], [0, 300, 602, 900, 1300],
                            labels=['Spring', 'Summer', 'Fall', 'Winter'])
    return data


def load_and_train_data(path_data):
    # df = pd.read_csv(path_data, sep='\t')
    df = load_data.load_data(PATH_DATA)

    df['Дата'] = df['Дата'].apply(lambda x: load_data.check(x))
    df['Дата'] = pd.to_datetime(df['Дата'].astype(str), format='%d.%m.%Y')
    df = df.sort_values(by='Дата').reset_index(drop=True)
    df.columns = ['datetime', 'key_rate']

    cat_type = CategoricalDtype(categories=[
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',
        'Sunday'
    ],
        ordered=True)

    df_features = create_features(data_full=df, col_datetime='datetime', cat_type=cat_type)

    df.columns = ['ds', 'y']
    SIZE = int(df.shape[0] * TEST_SIZE)

    train_df = df[:-SIZE]
    test_df = df[-SIZE:]

    # Train model
    model = NeuralProphet()
    model.fit(train_df, freq="D", epochs=10)

    # result predict dataset
    df_test_null = test_df.copy()
    df_test_null['y'] = None

    df_predict = pd.concat([train_df, df_test_null])
    predict = model.predict(df_predict)

    mae = mean_absolute_error(y_true=test_df['y'],
                              y_pred=predict['yhat1'][train_df.shape[0]:]
                              )

    mape = mean_absolute_percentage_error(y_true=test_df['y'],
                                          y_pred=predict['yhat1'][train_df.shape[0]:]
                                          )

    # Future predict
    model_full = NeuralProphet()
    model_full.fit(df)

    future = model_full.make_future_dataframe(df, periods=365, n_historic_predictions=180)
    forecast = model_full.predict(future)

    return model, mae, mape, forecast


def main():
    logging.info('Fitting the model')

    # MLFlow tracking
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(CONFIG['name_experiment'])
    with (mlflow.start_run()):
        model, mae, mape, forecast = load_and_train_data(PATH_DATA)

        with open(PATH_MODEL, 'wb') as f:
            pickle.dump(model, f)

        print(f'MAE score: {mae}')
        print(f'MAPE score: {mape}')

        # Model and parameters logging
        mlflow.log_param('MAE', mae)
        mlflow.log_param('MAPE', mape)
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
    # dict_unique = {key: data["X"][key].unique().tolist() for key in data["X"].columns}

    # with open(PATH_UNIQUE_VALUES, "w") as file:
    #     json.dump(dict_unique, file)

    return model, mae, mape, forecast

if __name__ == "__main__":
    main()

# load_and_train_data(PATH_DATA)
