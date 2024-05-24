from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Dict
from fastapi import FastAPI, HTTPException
import os
from api.services.data_service import DataService
from api.services.model_service import ModelService
from mangum import Mangum

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'model.joblib'))
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'house_price_data_20-05-2024.parquet'))

# Initialize services
model_service = ModelService(model_path=model_path)
data_service = DataService(data_path=data_path)

class HouseFeatures(BaseModel):
    home_type: str
    garage: bool
    home_size: str
    floor: int
    elevator: bool
    municipality: str
    parish: str
    neighborhood: str
    home_area: float

def preprocess_input(data: Dict) -> pd.DataFrame:
    df = pd.DataFrame(data)
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'bool' or df[col].dtype == 'string':
            df[col] = df[col].astype('category')
    return df

app = FastAPI()

@app.post("/predict")
async def predict(features: HouseFeatures):
    try:
        data = features.dict()
        processed_data = model_service.preprocess(data)
        prediction = model_service.predict(processed_data)
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/features")
async def get_unique_features():
    try:
        selected_features = ['municipality', 'parish', 'neighborhood', 'garage', 'home_type', 'home_size', 'home_area', 'floor', 'elevator']
        unique_values = data_service.get_unique_values(selected_features)
        return unique_values
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
