import os
import pandas as pd

USUAL_PATH = "/data"
IRIS_NAME = "Iris.csv"

def load_data(path, name):
    csv_path = os.path.join(path, name)
    return pd.read_csv(csv_path)

def load_iris_data(path = USUAL_PATH, name = IRIS_NAME):
    return load_data(path, name)

iris = load_iris_data()

print("end")