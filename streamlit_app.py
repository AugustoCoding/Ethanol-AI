import streamlit as st
import pandas as pd

# Título do app
st.title('⚗️ Ethanol AI')

# Informação principal
st.info('Aqui é possível prever as melhores condições para produzir o seu etanol de segunda geração.')

# Adicionando uma sidebar
st.sidebar.title("Configurações")
st.sidebar.write("Aqui você pode ajustar as configurações do app.")

# Exemplo de slider na sidebar
valor_slider = st.sidebar.slider('Escolha um valor', 0, 100, 50)
st.write(f"Você selecionou o valor: {valor_slider}")

# Exemplo de seleção na sidebar
opcao = st.sidebar.selectbox('Escolha uma opção', ['Opção 1', 'Opção 2', 'Opção 3'])
st.write(f"Você escolheu: {opcao}")
