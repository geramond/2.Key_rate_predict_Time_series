import logging

import pandas as pd
import seaborn as sns

import requests
from bs4 import BeautifulSoup


PATH_DATA = "data/key_rate.csv"

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def load_data(path):
    """Load data from path"""
    data = pd.read_csv(path)
    # for demonstration
    data = data.sample(5000)
    return data


def download_data(URL, path_data):
    data = requests.get(URL).text

    beautiful_soup = BeautifulSoup(data, "html.parser")
    tables = beautiful_soup.find_all("table")

    df = pd.read_html(str(tables))[0]
    df.iloc[:, 1:] /= 100

    df.to_csv(path_data, index=False, sep='\t', encoding='utf-8')

    return 'Success'

