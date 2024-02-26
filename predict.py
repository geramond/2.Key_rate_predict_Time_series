import logging
import pickle
import yaml

import mlflow

import pandas as pd


CONFIG = yaml.safe_load(open('config/params.yaml'))['predict']


logging.basicConfig(filename='log/app.log', filemode='w+', format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG)


def predict_price(model: pickle, dict_data: dict):
    logging.info('Predict house price')
    logging.info('Loading model last version')

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    model_uri_lr = f"models:/{CONFIG['model_lr']}/{CONFIG['version_lr']}"
    model = mlflow.sklearn.load_model(model_uri_lr)

    logging.info(f"Loaded {CONFIG['model_lr']}{CONFIG['version_lr']}")

    data_predict = pd.DataFrame([dict_data])
    predict = model.predict(data_predict)

    return predict


def main():
    # Download last saved models from MLFlow
    logging.info('Loading model last version')

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    model_uri_lr = f"models:/{CONFIG['model_lr']}/{CONFIG['version_lr']}"
    model = mlflow.sklearn.load_model(model_uri_lr)

    logging.info(f"Loaded {CONFIG['model_lr']}{CONFIG['version_lr']}")

    dict_data = {
        "building_type": 2,
        "object_type": 11,
        "level": 10,
        "levels": 23,
        "rooms": 5,
        "area": 100,
        "kitchen_area": 30,
    }

    data_predict = pd.DataFrame([dict_data])
    predict = model.predict(data_predict)

    print(predict)
    print(type(predict))
    # print(type(float(predict)))
    print('Hello!')

if __name__ == "__main__":
    main()
