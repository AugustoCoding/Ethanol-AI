import streamlit as st
import pandas as pd

# Título do app
st.title('⚗️Ethanol AI')

# Escolha da biomassa
st.sidebar.header("Biomassa")
biomassa = st.sidebar.selectbox("Selecione um tipo", ['Bagaço de Cana-de-Açúcar', 'Palha da Cana-de-Açúcar', 'Palha de milho'])

# Ajustando a composição da biomassa

st.sidebar.header('Composição')
celulose = st.sidebar.number_input("Porcentagem de Celulose", min_value=0.0, max_value=100.0, format="%.2f")
lignina = st.sidebar.number_input("Porcentagem de Lignina", min_value=0.0, max_value=100.0, format="%.2f")
hemicelulose = st.sidebar.number_input("Porcentagem de Hemicelulose", min_value=0.0, max_value=100.0, format="%.2f")
cinzas = st.sidebar.number_input("Porcentagem de cinzas", min_value=0.0, max_value=100.0, format="%.2f")

# Se a biomassa escolhida foi Bagaço de Cana-de-Açúcar
if biomassa == "Bagaço de Cana-de-Açúcar":
    st.header("Parâmetros")
    st.write(f"Selecione os parâmetros para definir a sua condição de operação para {biomassa}.")
    bagaco1 = st.number_input("Parâmetro 1 - Bagaço")
    bagaco2 = st.number_input("Parâmetro 2 - Bagaço")
    bagaco3 = st.number_input("Parâmetro 3 - Bagaço")
    bagaco4 = st.selectbox("Parâmetro 4 - Bagaço", options=["Opção 1", "Opção 2", "Opção 3"])
    bagaco5 = st.selectbox("Parâmetro 5 - Bagaço", options=["Opção A", "Opção B", "Opção C"])
    bagaco6 = st.selectbox("Parâmetro 6 - Bagaço", options=["Sim", "Não"])

# Se a biomassa escolhida foi Palha da Cana-de-Açúcar
if biomassa == "Palha da Cana-de-Açúcar":
    palha1 = st.sidebar.number_input("Parâmetro 1 - Palha Cana")
    palha2 = st.sidebar.number_input("Parâmetro 2 - Palha Cana")
    palha3 = st.sidebar.number_input("Parâmetro 3 - Palha Cana")
    palha4 = st.sidebar.selectbox("Parâmetro 4 - Palha Cana", options=["Opção 1", "Opção 2", "Opção 3"])
    palha5 = st.sidebar.selectbox("Parâmetro 5 - Palha Cana", options=["Opção A", "Opção B", "Opção C"])
    palha6 = st.sidebar.selectbox("Parâmetro 6 - Palha Cana", options=["Sim", "Não"])

# Se a biomassa escolhida foi Palha de Milho
if biomassa == "Palha de milho":
    milho1 = st.sidebar.number_input("Parâmetro 1 - Milho")
    milho2 = st.sidebar.number_input("Parâmetro 2 - Milho")
    milho3 = st.sidebar.number_input("Parâmetro 3 - Milho")
    milho4 = st.sidebar.selectbox("Parâmetro 4 - Milho", options=["Opção 1", "Opção 2", "Opção 3"])
    milho5 = st.sidebar.selectbox("Parâmetro 5 - Milho", options=["Opção A", "Opção B", "Opção C"])
    milho6 = st.sidebar.selectbox("Parâmetro 6 - Milho", options=["Sim", "Não"])
