import streamlit as st
import pandas as pd

# Configurando o layout para modo "wide"
st.set_page_config(layout="wide")

# Título do app
st.title('⚗️Ethanol AI (Software em desenvolvimento)')

# Informação principal
st.write("O Ethanol AI é uma ferramenta útil para estudar o comportamento de diferentes processos de produção de etanol de segunda geração, quando submetidos a diferentes condições operacionais. Aqui, você pode testar diferentes combinações de parâmetros de reação, encontrando essencialmente o rendimento máximo possível para cada situação.")
st.markdown("<hr style='border: 1px solid #ccc;' />", unsafe_allow_html=True)

# Criando colunas "Parâmetros" e "Resultados"
col1, spacer, col2, spacer2, col3 = st.columns([10, 2, 10, 2, 10])

# Personalizando a coluna Bimomassa(col1)
with col1:
    # Escolha da biomassa
    st.header("Biomassa")
    st.write("Informe os dados de sua biomassa.")
    biomassa = st.selectbox("Selecione um tipo", ['Bagaço de Cana-de-Açúcar', 'Palha da Cana-de-Açúcar', 'Palha de milho'])
    celulose = st.number_input("Porcentagem de Celulose", min_value=0.0, max_value=100.0, format="%.2f")
    lignina = st.number_input("Porcentagem de Lignina", min_value=0.0, max_value=100.0, format="%.2f")
    hemicelulose = st.number_input("Porcentagem de Hemicelulose", min_value=0.0, max_value=100.0, format="%.2f")
    cinzas = st.number_input("Porcentagem de cinzas", min_value=0.0, max_value=100.0, format="%.2f")

# Personalizando a coluna Parâmetros de Reação (col2)
with col2:
    
    # Se a biomassa escolhida foi Bagaço de Cana-de-Açúcar
    if biomassa == "Bagaço de Cana-de-Açúcar":
        st.header("Parâmetros de Reação")
        st.write(f"Informe os parâmetros da reação para definir a sua condição de operação de {biomassa}.")
        bagaco1 = st.number_input("Parâmetro 1 - Bagaço")
        bagaco2 = st.number_input("Parâmetro 2 - Bagaço")
        bagaco3 = st.number_input("Parâmetro 3 - Bagaço")
        bagaco4 = st.selectbox("Parâmetro 4 - Bagaço", options=["Opção 1", "Opção 2", "Opção 3"])
        bagaco5 = st.selectbox("Parâmetro 5 - Bagaço", options=["Opção A", "Opção B", "Opção C"])
        bagaco6 = st.selectbox("Parâmetro 6 - Bagaço", options=["Sim", "Não"])

    # Se a biomassa escolhida foi Palha da Cana-de-Açúcar
    if biomassa == "Palha da Cana-de-Açúcar":
        st.header("Parâmetros de Reação")
        st.write(f"Informe os parâmetros da reação para definir a sua condição de operação de {biomassa}.")
        palha1 = st.number_input("Parâmetro 1 - Palha Cana")
        palha2 = st.number_input("Parâmetro 2 - Palha Cana")
        palha3 = st.number_input("Parâmetro 3 - Palha Cana")
        palha4 = st.selectbox("Parâmetro 4 - Palha Cana", options=["Opção 1", "Opção 2", "Opção 3"])
        palha5 = st.selectbox("Parâmetro 5 - Palha Cana", options=["Opção A", "Opção B", "Opção C"])
        palha6 = st.selectbox("Parâmetro 6 - Palha Cana", options=["Sim", "Não"])
    
    # Se a biomassa escolhida foi Palha de Milho
    if biomassa == "Palha de milho":
        st.header("Parâmetros de Reação")
        st.write(f"Informe os parâmetros da reação para definir a sua condição de operação de {biomassa}.")
        milho1 = st.number_input("Parâmetro 1 - Milho")
        milho2 = st.number_input("Parâmetro 2 - Milho")
        milho3 = st.number_input("Parâmetro 3 - Milho")
        milho4 = st.selectbox("Parâmetro 4 - Milho", options=["Opção 1", "Opção 2", "Opção 3"])
        milho5 = st.selectbox("Parâmetro 5 - Milho", options=["Opção A", "Opção B", "Opção C"])
        milho6 = st.selectbox("Parâmetro 6 - Milho", options=["Sim", "Não"])

# Personalizando a coluna Resultados (col3)
from plotly import graph_objs as go
with col3:
    st.header("Resultados")
    st.write("Observe os resultados obtidos nessa coluna. Alterne a disposição do gráfico para visualizar mais relações.")
    st.metric(label="🔍 **Rendimento Previsto (%)**", value= "86%", delta="Boas Condições", help="Este é o rendimento previsto para as condições selecionadas.")
    # Criando um gráfico de exemplo
    fig = go.Figure(data=[go.Bar(x=['Categoria 1', 'Categoria 2', 'Categoria 3'], y=[10, 20, 30])])
    fig.update_layout(title="Exemplo de Gráfico", xaxis_title="Categorias", yaxis_title="Valores")
    
    # Exibindo o gráfico
    st.plotly_chart(fig)
