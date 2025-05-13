import streamlit as st
import pandas as pd

# Configurando o layout para modo "wide"
st.set_page_config(layout="wide")

# Título do app
st.title('⚗️Ethanol AI (Software em desenvolvimento)')

# Informação principal
st.write("O Ethanol AI é uma ferramenta útil para estudar o comportamento de diferentes processos de produção de etanol de segunda geração, quando submetidos a diferentes condições operacionais. Aqui, você pode testar diferentes combinações de parâmetros de reação, encontrando essencialmente o rendimento máximo possível para cada situação.")
st.markdown("<hr style='border: 1px solid #ccc;' />", unsafe_allow_html=True)

# Etapa de Pré-Tratamento
st.markdown(
    "<h1 style='font-size:50px;'>Pré-Tratamento</h1>",
    unsafe_allow_html=True
)
st.write("Nesta seção, introduza os dados relevantes ao cálculo do redimento do Pré-Tratamento.")

# Criando colunas "Parâmetros" e "Resultados"
col1, spacer, col2, spacer2, col3 = st.columns([10, 2, 10, 2, 10])

# Personalizando a coluna Bimomassa(col1)
with col1:
    # Escolha das características do modelo
    st.header("Dados Iniciais")
    st.write("Informe os dados principais de sua simulação.")
    biomassa = st.selectbox("Selecione um tipo de biomassa", ['Bagaço de Cana-de-Açúcar', 'Palha da Cana-de-Açúcar'])
    pretratamento = st.selectbox("Selecione um tipo de Pré-Tratamento", ['Ácido/Básico', 'Explosão a Vapor', 'Organossolve'])
    celulose = st.number_input("Porcentagem de Celulose (0,00 - 100,00) (%)", min_value=0.0, max_value=100.0, format="%.2f")
    lignina = st.number_input("Porcentagem de Lignina (0,00 - 100,00) (%)", min_value=0.0, max_value=100.0, format="%.2f")
    hemicelulose = st.number_input("Porcentagem de Hemicelulose (0,00 - 100,00) (%)", min_value=0.0, max_value=100.0, format="%.2f")
    cinzas = st.number_input("Porcentagem de cinzas (0,00 - 100,00) (%)", min_value=0.0, max_value=100.0, format="%.2f")

# Personalizando a coluna Parâmetros do Pré-Tratamento (col2)
with col2:
    
    st.header("Parâmetros do Pré-Tratamento")
    st.write(f"Informe os parâmetros do pré tratamento para {biomassa}.")
    
    # Se a biomassa escolhida foi Bagaço de Cana-de-Açúcar
    if biomassa == "Bagaço de Cana-de-Açúcar":
        bagaco1 = st.number_input("Parâmetro 1 - Bagaço")
        bagaco2 = st.number_input("Parâmetro 2 - Bagaço")
        bagaco3 = st.number_input("Parâmetro 3 - Bagaço")
        bagaco4 = st.selectbox("Parâmetro 4 - Bagaço", options=["Opção 1", "Opção 2", "Opção 3"])
        bagaco5 = st.selectbox("Parâmetro 5 - Bagaço", options=["Opção A", "Opção B", "Opção C"])
        bagaco6 = st.checkbox("Presença do reagente x")

    # Se a biomassa escolhida foi Palha da Cana-de-Açúcar
    if biomassa == "Palha da Cana-de-Açúcar":
        palha1 = st.number_input("Parâmetro 1 - Palha Cana")
        palha2 = st.number_input("Parâmetro 2 - Palha Cana")
        palha3 = st.number_input("Parâmetro 3 - Palha Cana")
        palha4 = st.selectbox("Parâmetro 4 - Palha Cana", options=["Opção 1", "Opção 2", "Opção 3"])
        palha5 = st.selectbox("Parâmetro 5 - Palha Cana", options=["Opção A", "Opção B", "Opção C"])
        palha6 = st.selectbox("Parâmetro 6 - Palha Cana", options=["Sim", "Não"])


# Selecionando o modelo de pré-tratamento

import pickle

if biomassa == "Bagaço de Cana-de-Açúcar" and pretratamento == "Ácido/Básico":

    # Carregar o modelo de machine learning empacotado com pickle
    try:
        with open('modelo_pretratamento.pkl', 'rb') as file:
            modelo = pickle.load(file)
        
        # Preparar os dados de entrada para o modelo
        dados_entrada = [[bagaco1, bagaco2, bagaco3, int(bagaco4[-1]), int(bagaco5[-1]), int(bagaco6)]]
        
        # Fazer a previsão com o modelo carregado
        rendimento_previsto = modelo.predict(dados_entrada)[0]
        
        # Exibir o resultado previsto
        st.success(f"Rendimento previsto pelo modelo: {rendimento_previsto:.2f}%")
    except FileNotFoundError:
        st.error("O arquivo do modelo 'modelo_pretratamento.pkl' não foi encontrado.")
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar o modelo: {e}")

if biomassa == "Palha da Cana-de-Açúcar" and pretratamento == "Ácido/Básico":
    # Carregar o modelo de machine learning empacotado com pickle
    try:
        with open('modelo_pretratamento.pkl', 'rb') as file:
            modelo = pickle.load(file)
        
        # Preparar os dados de entrada para o modelo
        dados_entrada = [[palha1, palha2, palha3, int(palha4[-1]), int(palha5[-1]), int(palha6)]]
        
        # Fazer a previsão com o modelo carregado
        rendimento_previsto = modelo.predict(dados_entrada)[0]
        
        # Exibir o resultado previsto
        st.success(f"Rendimento previsto pelo modelo: {rendimento_previsto:.2f}%")
    except FileNotFoundError:
        st.error("O arquivo do modelo 'modelo_pretratamento.pkl' não foi encontrado.")
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar o modelo: {e}")

if biomassa == "Bagaço de Cana-de-Açúcar" and pretratamento == "Explosão a Vapor":
    # Carregar o modelo de machine learning empacotado com pickle
    try:
        with open('modelo_pretratamento.pkl', 'rb') as file:
            modelo = pickle.load(file)
        
        # Preparar os dados de entrada para o modelo
        dados_entrada = [[bagaco1, bagaco2, bagaco3, int(bagaco4[-1]), int(bagaco5[-1]), int(bagaco6)]]
        
        # Fazer a previsão com o modelo carregado
        rendimento_previsto = modelo.predict(dados_entrada)[0]
        
        # Exibir o resultado previsto
        st.success(f"Rendimento previsto pelo modelo: {rendimento_previsto:.2f}%")
    except FileNotFoundError:
        st.error("O arquivo do modelo 'modelo_pretratamento.pkl' não foi encontrado.")
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar o modelo: {e}")

if biomassa == "Palha da Cana-de-Açúcar" and pretratamento == "Explosão a Vapor":
    # Carregar o modelo de machine learning empacotado com pickle
    try:
        with open('modelo_pretratamento.pkl', 'rb') as file:
            modelo = pickle.load(file)
        
        # Preparar os dados de entrada para o modelo
        dados_entrada = [[palha1, palha2, palha3, int(palha4[-1]), int(palha5[-1]), int(palha6)]]
        
        # Fazer a previsão com o modelo carregado
        rendimento_previsto = modelo.predict(dados_entrada)[0]
        
        # Exibir o resultado previsto
        st.success(f"Rendimento previsto pelo modelo: {rendimento_previsto:.2f}%")
    except FileNotFoundError:
        st.error("O arquivo do modelo 'modelo_pretratamento.pkl' não foi encontrado.")
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar o modelo: {e}")

if biomassa == "Bagaço de Cana-de-Açúcar" and pretratamento == "Organossolve":
    # Carregar o modelo de machine learning empacotado com pickle
    try:
        with open('modelo_pretratamento.pkl', 'rb') as file:
            modelo = pickle.load(file)
        
        # Preparar os dados de entrada para o modelo
        dados_entrada = [[bagaco1, bagaco2, bagaco3, int(bagaco4[-1]), int(bagaco5[-1]), int(bagaco6)]]
        
        # Fazer a previsão com o modelo carregado
        rendimento_previsto = modelo.predict(dados_entrada)[0]
        
        # Exibir o resultado previsto
        st.success(f"Rendimento previsto pelo modelo: {rendimento_previsto:.2f}%")
    except FileNotFoundError:
        st.error("O arquivo do modelo 'modelo_pretratamento.pkl' não foi encontrado.")
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar o modelo: {e}")

if biomassa == "Palha da Cana-de-Açúcar" and pretratamento == "Organossolve":
    # Carregar o modelo de machine learning empacotado com pickle
    try:
        with open('modelo_pretratamento.pkl', 'rb') as file:
            modelo = pickle.load(file)
        
        # Preparar os dados de entrada para o modelo
        dados_entrada = [[palha1, palha2, palha3, int(palha4[-1]), int(palha5[-1]), int(palha6)]]
        
        # Fazer a previsão com o modelo carregado
        rendimento_previsto = modelo.predict(dados_entrada)[0]
        
        # Exibir o resultado previsto
        st.success(f"Rendimento previsto pelo modelo: {rendimento_previsto:.2f}%")
    except FileNotFoundError:
        st.error("O arquivo do modelo 'modelo_pretratamento.pkl' não foi encontrado.")
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar o modelo: {e}")


# Personalizando a coluna Resultados do Pré-Tratamento (col3)
from plotly import graph_objs as go
with col3:
    st.header("Resultados do Pré-Tratamento")
    st.write(f"Aqui é possível ver os resultados obtidos para a etapa de Pré-Tratamento de {biomassa}. Alterne a disposição do gráfico para visualizar mais relações entre as variáveis.")
    st.button("Calcular Rendimento", key="pretratamento_resultados")
    st.metric(label="🔍 **Rendimento Previsto (%)**", value= "86%", delta="+6%", help="Este é o rendimento previsto para as condições selecionadas.")
    # Criando um gráfico de exemplo
    fig = go.Figure(data=[go.Bar(x=['Categoria 1', 'Categoria 2', 'Categoria 3'], y=[10, 20, 30])])
    fig.update_layout(title="Exemplo de Gráfico", xaxis_title="Categorias", yaxis_title="Valores")
    
    # Exibindo o gráfico
    st.plotly_chart(fig, key = "pretratamento_grafico")

st.markdown("<hr style='border: 1px solid #ccc;' />", unsafe_allow_html=True)

# Próxima Etapa: Hidrólise

st.markdown(
    "<h1 style='font-size:50px;'>Hidrólise</h1>",
    unsafe_allow_html=True
)
st.write("Nesta seção, introduza os dados relevantes ao cálculo do redimento da Hidrólise.")
# Criando colunas "Parâmetros" e "Resultados"
col4, spacer3, col5, spacer4, col6 = st.columns([10, 2, 10, 2, 10])

# Personalizando a coluna Dados do Pré-Tratamento (col4)
with col4:
    st.header("Dados do Pré-Tratamento")
    st.write("Informe os dados obtidos no pré-tratamento.")
    celulose1 = st.number_input("Porcentagem de Celulose Restante", min_value=0.0, max_value=100.0, format="%.2f")
    lignina1 = st.number_input("Porcentagem de Lignina Restante", min_value=0.0, max_value=100.0, format="%.2f")
    hemicelulose1 = st.number_input("Porcentagem de Hemicelulose Restante", min_value=0.0, max_value=100.0, format="%.2f")
    cinzas1 = st.number_input("Porcentagem de cinzas Restante", min_value=0.0, max_value=100.0, format="%.2f")

# Personalizando a coluna Parâmetros da Hidrólise (col5)
with col5:
    st.header("Parâmetros da Hidrólise")
    st.write(f"Informe os parâmetros da hidrólise para definir a sua condição de operação de {biomassa}.")
    catalizador = st.selectbox("Catalizador", ['Tipo 1', 'Tipo 2', 'Tipo 3'])
    # Se o catalizador utilizado foi Tipo 1
    if catalizador == "Tipo 1":
        bagaco11 = st.number_input("Parâmetro 11 - Bagaço")
        bagaco21 = st.number_input("Parâmetro 21 - Bagaço")
        bagaco31 = st.number_input("Parâmetro 31 - Bagaço")
        bagaco41 = st.selectbox("Parâmetro 41 - Bagaço", options=["Opção 1", "Opção 2", "Opção 3"])
        bagaco51 = st.selectbox("Parâmetro 51- Bagaço", options=["Opção A", "Opção B", "Opção C"])
        bagaco61 = st.selectbox("Parâmetro 61 - Bagaço", options=["Sim", "Não"])

    # Se o catalizador utilizado foi Tipo 2
    if catalizador == "Tipo 2":
        palha11 = st.number_input("Parâmetro 11 - Palha Cana")
        palha21 = st.number_input("Parâmetro 21 - Palha Cana")
        palha31 = st.number_input("Parâmetro 31 - Palha Cana")
        palha41= st.selectbox("Parâmetro 41 - Palha Cana", options=["Opção 1", "Opção 2", "Opção 3"])
        palha51 = st.selectbox("Parâmetro 51 - Palha Cana", options=["Opção A", "Opção B", "Opção C"])
        palha61 = st.selectbox("Parâmetro 61 - Palha Cana", options=["Sim", "Não"])
    
    # Se o catalizador utilizado foi Tipo 3
    if catalizador == "Tipo 3":
        milho11 = st.number_input("Parâmetro 11 - Milho")
        milho21 = st.number_input("Parâmetro 21 - Milho")
        milho31 = st.number_input("Parâmetro 31 - Milho")
        milho41 = st.selectbox("Parâmetro 41 - Milho", options=["Opção 1", "Opção 2", "Opção 3"])
        milho51 = st.selectbox("Parâmetro 51 - Milho", options=["Opção A", "Opção B", "Opção C"])
        milho61 = st.selectbox("Parâmetro 61 - Milho", options=["Sim", "Não"])

# Personalizando a coluna Resultados da Hidrólise (col6)
with col6:
    st.header("Resultados da Hidrólise")
    st.write(f"Aqui é possível ver os resultados obtidos para a etapa de Hidrólise de {biomassa}. Alterne a disposição do gráfico para visualizar mais relações entre as variáveis.")
    st.button("Calcular Rendimento", key="hidrolise_resultados")
    st.metric(label="🔍 **Rendimento Previsto (%)**", value= "91%", delta="+5%", help="Este é o rendimento previsto para as condições selecionadas.")
    # Criando um gráfico de exemplo
    fig1 = go.Figure(data=[go.Bar(x=['Categoria 1', 'Categoria 2', 'Categoria 3'], y=[10, 20, 30])])
    fig1.update_layout(title="Exemplo de Gráfico", xaxis_title="Categorias", yaxis_title="Valores")
    
    # Exibindo o gráfico
    st.plotly_chart(fig1, key = "hidrolise_grafico")