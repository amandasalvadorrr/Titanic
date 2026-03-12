import streamlit as st
import joblib
import numpy as np

model = joblib.load("modelo.pkl")

st.title("🚢 Titanic Survival Predictor")

st.write("Preveja se um passageiro sobreviveria")

pclass = st.selectbox(
    "Classe do passageiro",
    [1,2,3]
)

sexo = st.selectbox(
    "Sexo",
    ["Masculino","Feminino"]
)

idade = st.slider(
    "Idade",
    0,80,30
)

sibsp = st.number_input(
    "Número de irmãos/cônjuges",
    0,10,0
)

fare = st.number_input(
    "Tarifa paga",
    0.0,500.0,50.0
)

if sexo == "Masculino":
    sexo = 0
else:
    sexo = 1

if st.button("Prever"):

    dados = np.array([[pclass,sexo,idade,sibsp,fare]])

    resultado = model.predict(dados)

    if resultado[0] == 1:
        st.success("✅ O passageiro sobreviveria")
    else:
        st.error("❌ O passageiro não sobreviveria")