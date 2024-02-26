import os

import yaml
import json

import mlflow
from fastapi import FastAPI
import streamlit as st

from src import load_data
import train
import predict


# PATH_DATA = "data/moscow.csv"
# PATH_UNIQUE_VALUES = 'config/unique_values.json'

CONFIG = yaml.safe_load(open('config/params.yaml'))['predict']


app = FastAPI()


@st.cache_data
@app.post('/get_train')
def get_train():
    data, model, mae = train.main()

    result = {
        "mae": f"{mae}"
    }

    return result


@st.cache_data
@app.post('/get_predict')
def get_predict(dict_data: dict):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    model_uri_lr = f"models:/{CONFIG['model_lr']}/{CONFIG['version_lr']}"
    model = mlflow.sklearn.load_model(model_uri_lr)

    result = predict.predict_price(model, dict_data)

    return result


def main():
    st.set_page_config(layout="wide")
    st.header('House prices in Moscow')

    with open(PATH_UNIQUE_VALUES) as file:
        dict_unique = json.load(file)

    df = load_data.load_data(PATH_DATA)
    df = load_data.transform_data(df)
    st.write(df[:4])

    st.map(data=df, latitude="geo_lat", longitude="geo_lon", color='label_colors')

    st.markdown(
        """
        ### Fields describe 
            - Building_type - Facade type:
                0 - Other.
                1 - Panel.
                2 - Monolithic.
                3 - Brick.
                4 - Blocky.
                5 - Wooden.
                
            - Object_type - Apartment type.
                1 - Secondary real estate market.
                11 - New building.

            - Level - Apartment floor
            - Levels - Number of storeys
            - Rooms - the number of living rooms.
                If the value is '-1', then it means 'studio apartment'
            - Area - the total area of the apartment
            - Kitchen_area - Kitchen area
            - Price - Price in rubles
    """
    )

    # Features
    building_type = st.sidebar.selectbox('Building type', (dict_unique['building_type']))
    object_type = st.sidebar.selectbox("Object type", (dict_unique["object_type"]))
    level = st.sidebar.slider(
        "Level", min_value=min(dict_unique["level"]), max_value=max(dict_unique["level"])
    )
    levels = st.sidebar.slider(
        "Levels", min_value=min(dict_unique["levels"]), max_value=max(dict_unique["levels"])
    )
    rooms = st.sidebar.selectbox("Rooms", (dict_unique["rooms"]))
    area = st.sidebar.slider(
        "Area", min_value=min(dict_unique["area"]), max_value=max(dict_unique["area"])
    )
    kitchen_area = st.sidebar.slider(
        "Kitchen area",
        min_value=min(dict_unique["kitchen_area"]),
        max_value=max(dict_unique["kitchen_area"])
    )

    dict_data = {
        "building_type": building_type,
        "object_type": object_type,
        "level": level,
        "levels": levels,
        "rooms": rooms,
        "area": area,
        "kitchen_area": kitchen_area,
    }

    button_train = st.button("Train")
    if button_train:
        result_train = get_train()
        result_train = result_train['mae']
        st.success(f"MAE score: {round(float(result_train), 2)} rub")


    button_predict = st.button("Predict")

    if button_predict:
        result_predict = get_predict(dict_data)
        st.success(f"{round(float(result_predict), 2)} rub")

    button_mlflow = st.button("MLFlow")
    if button_mlflow:
        mlflow_cmd = "mlflow server --host localhost --port 5000 --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root mlflow"
        os.system(f"{mlflow_cmd}")


if __name__ == '__main__':
    main()

# TODO:
#   - CREATE modules: train, test, main
#   - Docker
#   - MLFlow, Airflow
#   - FastAPI
#   - Streamlit
#   - Graphics, Predicts
#   - SCALE: choose rates (key_rate, currencies,...), choose periods to train
