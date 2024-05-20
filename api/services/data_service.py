import pandas as pd
import os

class DataService:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = pd.read_parquet(data_path)

    def get_unique_values(self, features):
        unique_values = {}
        for feature in features:
            if not self.data[feature].dtype.kind in 'iufc': # https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html
                unique_values[feature] = self.data[feature].unique().tolist()
        return unique_values
