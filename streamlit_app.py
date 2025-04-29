import streamlit as st
import pandas as pd

# Título do app
st.title('⚗️Ethanol AI')

# Informação principal
st.info('Aqui é possível prever as melhores condições para produzir o seu etanol de segunda geração.')

# Os parâmetros do modelo são inseridos na sidebar
st.sidebar.title("Parâmetros")
st.sidebar.write("Preencha os campos abaixo para calcular o rendimento máximo aplicável nas condições disponíveis.")

# Escolha da biomassa
st.sidebar.header("Biomassa")
biomassa = st.sidebar.selectbox('Selecione', ['Bagaço de Cana-de-Açúcar', 'Palha da Cana-de-Açúcar', 'Palha de milho'])
st.write(f"Você escolheu: {biomassa}")

# Ajustando a composição da biomassa

st.sidebar.header('Composição')
celulose = st.sidebar.number_input("Porcentagem de Celulose", min_value=0.0, max_value=100.0, format="%.2f")
lignina = st.sidebar.number_input("Porcentagem de Lignina", min_value=0.0, max_value=100.0, format="%.2f")
hemicelulose = st.sidebar.number_input("Porcentagem de Hemicelulose", min_value=0.0, max_value=100.0, format="%.2f")

#Apresentar os campos de input corretos para cada biomassa escolhida

st.sidebar.header("Parâmetos do modelo")

# Se a biomassa escolhida foi Bagaço de Cana-de-Açúcar

if biomassa == "Bagaço de Cana-de-Açúcar":
  st.sidebar.number_input("Parâmetro 1")
  st.sidebar.number_input("Parâmetro 2")
  st.sidebar.number_input("Parâmetro 3")
  st.sidebar.selectbox("Parâmetro 4")
  st.sidebar.selectbox("Parâmetro 5")
  st.sidebar.selectbox("Parâmetro 6")
  
# Exemplo de campo de input
valor_input = st.sidebar.number_input("Concentração de Enzimas")
st.write(f"Concentração de Enzimas: {valor_input}")

# Exemplo de slider na sidebar
valor_slider = st.sidebar.slider('Escolha um valor', 0, 100, 50)
st.write(f"Você selecionou o valor: {valor_slider}")
