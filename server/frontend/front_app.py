import streamlit as st
import requests
from dotenv import load_dotenv
import os

load_dotenv()

API_URL = os.getenv("SERVER_API_URL")
API_KEY = os.getenv("SERVER_API_KEY")

st.title("Prédiction de la qualité du vin")

type = st.selectbox("Type", ["red", "white"])
fixed_acidity = st.number_input("Acidité fixe", min_value=0.0, max_value=15.0, step=0.1)
volatile_acidity = st.number_input("Acidité volatile", min_value=0.0, max_value=2.0, step=0.01)
citric_acid = st.number_input("Acide citrique", min_value=0.0, max_value=1.0, step=0.01)
residual_sugar = st.number_input("Sucre résiduel", min_value=0.0, max_value=50.0, step=0.1)
chlorides = st.number_input("Chlorures", min_value=0.0, max_value=0.5, step=0.001)
free_sulfur_dioxide = st.number_input("Dioxide de soufre libre", min_value=0.0, max_value=70.0, step=1.0)
total_sulfur_dioxide = st.number_input("Dioxide de soufre total", min_value=0.0, max_value=300.0, step=1.0)
density = st.number_input("Densité", min_value=0.0, max_value=2.0, step=0.0001)
pH = st.number_input("pH", min_value=0.0, max_value=4.0, step=0.01)
sulphates = st.number_input("Sulfates", min_value=0.0, max_value=2.0, step=0.01)
alcohol = st.number_input("Alcool", min_value=0.0, max_value=20.0, step=0.1)

if st.button("Prédire"):
    data = {
        "type": type,
        "fixed_acidity": fixed_acidity,
        "volatile_acidity": volatile_acidity,
        "citric_acid": citric_acid,
        "residual_sugar": residual_sugar,
        "chlorides": chlorides,
        "free_sulfur_dioxide": free_sulfur_dioxide,
        "total_sulfur_dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol
    }

    headers = {"access_token": API_KEY}
    response = requests.post(API_URL, json=data, headers=headers)
    if response.status_code == 200:
        prediction = response.json()
        st.write(f"Prédiction de la qualité : {prediction}")
    else:
        st.error("Erreur lors de la prédiction")

