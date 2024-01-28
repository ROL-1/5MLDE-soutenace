import joblib
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from dotenv import load_dotenv
import os

from WineModel import WineInput

load_dotenv()
API_KEY = os.getenv("SERVER_API_KEY")

app = FastAPI()

model = joblib.load('./model/model.pkl')

api_key_header = APIKeyHeader(name="access_token", auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=403, detail="Accès non autorisé"
        )

@app.post("/predict/", dependencies=[Depends(get_api_key)])
async def predict(wine: WineInput):
    data = {
        "type": wine.type,
        "fixed acidity": wine.fixed_acidity,
        "volatile acidity": wine.volatile_acidity,
        "citric acid": wine.citric_acid,
        "residual sugar": wine.residual_sugar,
        "chlorides": wine.chlorides,
        "free sulfur dioxide": wine.free_sulfur_dioxide,
        "total sulfur dioxide": wine.total_sulfur_dioxide,
        "density": wine.density,
        "pH": wine.pH,
        "sulphates": wine.sulphates,
        "alcohol": wine.alcohol
    }

    example = pd.DataFrame([data])
    prediction = model.predict(example)

    prediction_int = int(round(prediction[0]))

    return prediction_int
