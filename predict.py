import logging
import pickle
import yaml

import mlflow

import pandas as pd

from src import load_data


CONFIG = yaml.safe_load(open('config/params.yaml'))['predict']
PATH_DATA = "data/key_rate.csv"
TEST_SIZE = CONFIG['test_size']


logging.basicConfig(filename='log/app.log', filemode='w+', format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG)


def predict_rate(df, periods=365, n_predict=180):
    logging.info('Predict key rate')
    logging.info('Loading model last version')

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    model_uri_lr = f"models:/{CONFIG['model_lr']}/{CONFIG['version_lr']}"
    model = mlflow.sklearn.load_model(model_uri_lr)

    logging.info(f"Loaded {CONFIG['model_lr']}{CONFIG['version_lr']}")

    df['Дата'] = df['Дата'].apply(lambda x: load_data.check(x))
    df['Дата'] = pd.to_datetime(df['Дата'].astype(str), format='%d.%m.%Y')
    # df['Дата'] = pd.to_datetime(df['Дата'].astype(str), format='%m.%d.%Y')
    df = df.sort_values(by='Дата').reset_index(drop=True)
    # df.columns = ['datetime', 'key_rate']

    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df.ds)
    df['ds'] = df['ds'].dt.strftime('%m.%d.%Y')

    model.restore_trainer()
    future = model.make_future_dataframe(df, periods=periods, n_historic_predictions=n_predict)
    forecast = model.predict(future)

    # return forecast, model.plot(forecast), model.plot_components(forecast)
    result = (model, forecast)
    return result


def main():
    # Download last saved models from MLFlow
    logging.info('Loading model last version')
    logging.info(f"Loaded {CONFIG['model_lr']}{CONFIG['version_lr']}")

    df = load_data.load_data(PATH_DATA)
    forecast, plot, components = predict_rate(df)

    print(forecast)
    print('Hello!')


if __name__ == "__main__":
    main()
