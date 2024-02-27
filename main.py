import os
import yaml

from fastapi import FastAPI
import streamlit as st

from src import load_data
import train
import predict


CONFIG = yaml.safe_load(open('config/params.yaml'))['predict']
PATH_DATA = "data/key_rate.csv"

app = FastAPI()


@st.cache_data
@app.post('/load_data')
def download_data():
    result = load_data.download_data()

    return result

@st.cache_data
@app.post('/get_train')
def get_train():
    model, mae, mape, forecast = train.main()

    result = {
        "mae": f"{mae}",
        "mape": f"{mape}"
    }

    return result


@st.cache_data
@app.post('/get_predict')
def get_predict(df):

    forecast, plot, components = predict.predict_rate(df)

    return forecast


def main():
    st.set_page_config(layout="wide")
    st.header('Key rate predict')

    df = download_data.load_data(PATH_DATA)
    st.write(df[:4])

    st.markdown(
        """
        ### Fields describe 

        """
    )

    # Features
    # building_type = st.sidebar.selectbox('Building type', (dict_unique['building_type']))
    # object_type = st.sidebar.selectbox("Object type", (dict_unique["object_type"]))
    # level = st.sidebar.slider(
    #     "Level", min_value=min(dict_unique["level"]), max_value=max(dict_unique["level"])
    # )
    # levels = st.sidebar.slider(
    #     "Levels", min_value=min(dict_unique["levels"]), max_value=max(dict_unique["levels"])
    # )
    # rooms = st.sidebar.selectbox("Rooms", (dict_unique["rooms"]))
    # area = st.sidebar.slider(
    #     "Area", min_value=min(dict_unique["area"]), max_value=max(dict_unique["area"])
    # )
    # kitchen_area = st.sidebar.slider(
    #     "Kitchen area",
    #     min_value=min(dict_unique["kitchen_area"]),
    #     max_value=max(dict_unique["kitchen_area"])
    # )

    button_download_data = st.button("Download data")
    if button_download_data:
        result_load_data = download_data()
        st.success(f"{result_load_data}")


    button_train = st.button("Train")
    if button_train:
        result_train = get_train()
        # result_train = result_train['mae']
        st.success(f"MAE score: {round(float(result_train['mae']), 2)}")
        st.success(f"MAPE score: {round(float(result_train['mape']), 2)}")


    button_predict = st.button("Predict")

    if button_predict:
        result_predict = get_predict(df)
        st.success(f"{result_predict}")

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
#   - Graphics: key_rate(time), boxplots, plot_components
#   - Predicts: next_year key_rate(time), forecast, plot_components
#   - SCALE: choose rates (key_rate, currencies,...), choose periods to train
