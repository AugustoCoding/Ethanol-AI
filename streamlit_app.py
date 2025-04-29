import streamlit as st
import pandas as pd

# Título do app
st.title('⚗️ Ethanol AI')

# Informação principal
st.info('Aqui é possível prever as melhores condições para produzir o seu etanol de segunda geração.')

# Adicionando uma sidebar
st.sidebar.header("Parâmetros")
st.sidebar.title("Parâmetros")
st.sidebar.write("Aqui você pode ajustar os parâmetros do modelo.")

# Escolha da biomassa
biomassa = st.sidebar.selectbox('Biomassa', ['Bagaço de Cana-de-Açúcar', 'Palha da Cana-de-Açúcar', 'Palha de milho'])
st.write(f"Você escolheu: {biomassa}")

# Exemplo de campo de input
valor_input = st.sidebar.number_input("Concentração de Enzimas")
st.write(f"Concentração de Enzimas: {valor_input}")

# Exemplo de slider na sidebar
valor_slider = st.sidebar.slider('Escolha um valor', 0, 100, 50)
st.write(f"Você selecionou o valor: {valor_slider}")
