import logging
import pickle
import yaml

import mlflow

import pandas as pd


CONFIG = yaml.safe_load(open('config/params.yaml'))['predict']
PATH_DATA = "data/key_rate.csv"


logging.basicConfig(filename='log/app.log', filemode='w+', format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG)


def predict_rate(df, periods=365, n_predict=180):
    logging.info('Predict key rate')
    logging.info('Loading model last version')

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    model_uri_lr = f"models:/{CONFIG['model_lr']}/{CONFIG['version_lr']}"
    model = mlflow.sklearn.load_model(model_uri_lr)

    logging.info(f"Loaded {CONFIG['model_lr']}{CONFIG['version_lr']}")

    future = model.make_future_dataframe(df, periods=periods, n_historic_predictions=n_predict)
    forecast = model.predict(future)

    return forecast, model.plot(forecast), model.plot_components(forecast)


def main():
    # Download last saved models from MLFlow
    logging.info('Loading model last version')
    logging.info(f"Loaded {CONFIG['model_lr']}{CONFIG['version_lr']}")

    df = pd.read_csv(PATH_DATA)
    forecast, plot, components = predict_rate(df)

    print(forecast)
    print('Hello!')


if __name__ == "__main__":
    main()
