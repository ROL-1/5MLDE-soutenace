import streamlit as st
import requests
from dotenv import load_dotenv
import os
from typing import Dict

load_dotenv()

API_URL: str = os.getenv("SERVER_API_URL", "http://localhost:8000")
API_KEY: str = os.getenv("SERVER_API_KEY", "")

def predict_wine_quality(data: Dict) -> requests.Response:
    """
    Make a POST request to the prediction API with the provided data.

    Parameters:
    data (Dict): A dictionary containing wine features.

    Returns:
    requests.Response: The response from the prediction API.
    """
    headers = {"access_token": API_KEY}
    response = requests.post(f'{API_URL}/predict', json=data, headers=headers)
    return response

def main() -> None:
    """
    Main function to render the Streamlit UI and handle the prediction logic.
    """
    st.title("Wine Quality Prediction")

    wine_features = {
        "type": st.selectbox("Type", ["red", "white"]),
        "fixed_acidity": st.number_input("Fixed Acidity", min_value=0.0, max_value=15.0, step=0.1),
        "volatile_acidity": st.number_input("Volatile Acidity", min_value=0.0, max_value=2.0, step=0.01),
        "citric_acid": st.number_input("Citric Acid", min_value=0.0, max_value=1.0, step=0.01),
        "residual_sugar": st.number_input("Residual Sugar", min_value=0.0, max_value=50.0, step=0.1),
        "chlorides": st.number_input("Chlorides", min_value=0.0, max_value=0.5, step=0.001),
        "free_sulfur_dioxide": st.number_input("Free Sulfur Dioxide", min_value=0.0, max_value=70.0, step=1.0),
        "total_sulfur_dioxide": st.number_input("Total Sulfur Dioxide", min_value=0.0, max_value=300.0, step=1.0),
        "density": st.number_input("Density", min_value=0.0, max_value=2.0, step=0.0001),
        "pH": st.number_input("pH", min_value=0.0, max_value=4.0, step=0.01),
        "sulphates": st.number_input("Sulphates", min_value=0.0, max_value=2.0, step=0.01),
        "alcohol": st.number_input("Alcohol", min_value=0.0, max_value=20.0, step=0.1)
    }

    if st.button("Predict"):
        response = predict_wine_quality(wine_features)
        if response.status_code == 200:
            prediction = response.json()
            st.write(f"Prediction of the quality: {prediction}")
        else:
            st.error("Error in prediction")

if __name__ == "__main__":
    main()
