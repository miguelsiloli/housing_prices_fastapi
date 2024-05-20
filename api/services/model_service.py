import pandas as pd
import joblib
import numpy as np
from typing import Dict

class ModelService:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.model.predict(features)

    def preprocess(self, data: Dict) -> pd.DataFrame:
        df = pd.DataFrame([data])
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype == 'bool':
                df[col] = df[col].astype('category')
        return df