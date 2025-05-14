import streamlit as st
import pandas as pd
import pickle
from plotly import graph_objs as go

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
    pretratamento = st.selectbox("Selecione um tipo de Pré-Tratamento", ['Ácido', 'Básico', 'Organossolve', 'Hidrotérmico'])
    celulose = st.number_input("Porcentagem de Celulose (0,00 - 100,00) (%)", min_value=0.0, max_value=100.0, format="%.2f")
    lignina = st.number_input("Porcentagem de Lignina (0,00 - 100,00) (%)", min_value=0.0, max_value=100.0, format="%.2f")
    hemicelulose = st.number_input("Porcentagem de Hemicelulose (0,00 - 100,00) (%)", min_value=0.0, max_value=100.0, format="%.2f")
    cinzas = st.number_input("Porcentagem de cinzas (0,00 - 100,00) (%)", min_value=0.0, max_value=100.0, format="%.2f")

# Personalizando a coluna Parâmetros do Pré-Tratamento (col2)
with col2:
    
    st.header("Parâmetros do Pré-Tratamento")
    st.write(f"Informe os parâmetros do pré tratamento para {biomassa}.")
    
    # Se a biomassa escolhida foi Bagaço de Cana-de-Açúcar e o pré-tratamento é Ácido
    if biomassa == "Bagaço de Cana-de-Açúcar" and pretratamento == "Ácido":
        bagaco1 = st.number_input("Parâmetro 1 - Bagaço (Ácido)")
        bagaco2 = st.number_input("Parâmetro 2 - Bagaço (Ácido)")
        bagaco3 = st.number_input("Parâmetro 3 - Bagaço (Ácido)")
        bagaco4 = st.selectbox("Parâmetro 4 - Bagaço (Ácido)", options=["Opção 1", "Opção 2", "Opção 3"])
        bagaco5 = st.selectbox("Parâmetro 5 - Bagaço (Ácido)", options=["Opção A", "Opção B", "Opção C"])
        bagaco6 = st.checkbox("Presença do reagente x (Ácido)")

    # Se a biomassa escolhida foi Bagaço de Cana-de-Açúcar e o pré-tratamento é Básico
    if biomassa == "Bagaço de Cana-de-Açúcar" and pretratamento == "Básico":
        bagaco1 = st.number_input("Parâmetro 1 - Bagaço (Básico)")
        bagaco2 = st.number_input("Parâmetro 2 - Bagaço (Básico)")
        bagaco3 = st.number_input("Parâmetro 3 - Bagaço (Básico)")
        bagaco4 = st.selectbox("Parâmetro 4 - Bagaço (Básico)", options=["Opção 1", "Opção 2", "Opção 3"])
        bagaco5 = st.selectbox("Parâmetro 5 - Bagaço (Básico)", options=["Opção A", "Opção B", "Opção C"])
        bagaco6 = st.checkbox("Presença do reagente x (Básico)")

    # Se a biomassa escolhida foi Palha da Cana-de-Açúcar e o pré-tratamento é Ácido
    if biomassa == "Palha da Cana-de-Açúcar" and pretratamento == "Ácido":
        palha1 = st.number_input("Parâmetro 1 - Palha Cana (Ácido)")
        palha2 = st.number_input("Parâmetro 2 - Palha Cana (Ácido)")
        palha3 = st.number_input("Parâmetro 3 - Palha Cana (Ácido)")
        palha4 = st.selectbox("Parâmetro 4 - Palha Cana (Ácido)", options=["Opção 1", "Opção 2", "Opção 3"])
        palha5 = st.selectbox("Parâmetro 5 - Palha Cana (Ácido)", options=["Opção A", "Opção B", "Opção C"])
        palha6 = st.selectbox("Parâmetro 6 - Palha Cana (Ácido)", options=["Sim", "Não"])

    # Se a biomassa escolhida foi Palha da Cana-de-Açúcar e o pré-tratamento é Básico
    if biomassa == "Palha da Cana-de-Açúcar" and pretratamento == "Básico":
        palha1 = st.number_input("Parâmetro 1 - Palha Cana (Básico)")
        palha2 = st.number_input("Parâmetro 2 - Palha Cana (Básico)")
        palha3 = st.number_input("Parâmetro 3 - Palha Cana (Básico)")
        palha4 = st.selectbox("Parâmetro 4 - Palha Cana (Básico)", options=["Opção 1", "Opção 2", "Opção 3"])
        palha5 = st.selectbox("Parâmetro 5 - Palha Cana (Básico)", options=["Opção A", "Opção B", "Opção C"])
        palha6 = st.selectbox("Parâmetro 6 - Palha Cana (Básico)", options=["Sim", "Não"])
    
    # Se a biomassa escolhida foi Bagaço de Cana-de-Açúcar e o pré-tratamento é Organossolve
    if biomassa == "Bagaço de Cana-de-Açúcar" and pretratamento == "Organossolve":
        bagaco1 = st.number_input("Parâmetro 1 - Bagaço (Organossolve)")
        bagaco2 = st.number_input("Parâmetro 2 - Bagaço (Organossolve)")
        bagaco3 = st.number_input("Parâmetro 3 - Bagaço (Organossolve)")
        bagaco4 = st.selectbox("Parâmetro 4 - Bagaço (Organossolve)", options=["Opção 1", "Opção 2", "Opção 3"])
        bagaco5 = st.selectbox("Parâmetro 5 - Bagaço (Organossolve)", options=["Opção A", "Opção B", "Opção C"])
        bagaco6 = st.checkbox("Presença do reagente x (Organossolve)")

    # Se a biomassa escolhida foi Bagaço de Cana-de-Açúcar e o pré-tratamento é Hidrotérmico
    if biomassa == "Bagaço de Cana-de-Açúcar" and pretratamento == "Hidrotérmico":
        bagaco1 = st.number_input("Parâmetro 1 - Bagaço (Hidrotérmico)")
        bagaco2 = st.number_input("Parâmetro 2 - Bagaço (Hidrotérmico)")
        bagaco3 = st.number_input("Parâmetro 3 - Bagaço (Hidrotérmico)")
        bagaco4 = st.selectbox("Parâmetro 4 - Bagaço (Hidrotérmico)", options=["Opção 1", "Opção 2", "Opção 3"])
        bagaco5 = st.selectbox("Parâmetro 5 - Bagaço (Hidrotérmico)", options=["Opção A", "Opção B", "Opção C"])
        bagaco6 = st.checkbox("Presença do reagente x (Hidrotérmico)")

    # Se a biomassa escolhida foi Palha da Cana-de-Açúcar e o pré-tratamento é Organossolve
    if biomassa == "Palha da Cana-de-Açúcar" and pretratamento == "Organossolve":
        palha1 = st.number_input("Parâmetro 1 - Palha Cana (Organossolve)")
        palha2 = st.number_input("Parâmetro 2 - Palha Cana (Organossolve)")
        palha3 = st.number_input("Parâmetro 3 - Palha Cana (Organossolve)")
        palha4 = st.selectbox("Parâmetro 4 - Palha Cana (Organossolve)", options=["Opção 1", "Opção 2", "Opção 3"])
        palha5 = st.selectbox("Parâmetro 5 - Palha Cana (Organossolve)", options=["Opção A", "Opção B", "Opção C"])
        palha6 = st.selectbox("Parâmetro 6 - Palha Cana (Organossolve)", options=["Sim", "Não"])

    # Se a biomassa escolhida foi Palha da Cana-de-Açúcar e o pré-tratamento é Hidrotérmico
    if biomassa == "Palha da Cana-de-Açúcar" and pretratamento == "Hidrotérmico":
        palha1 = st.number_input("Parâmetro 1 - Palha Cana (Hidrotérmico)")
        palha2 = st.number_input("Parâmetro 2 - Palha Cana (Hidrotérmico)")
        palha3 = st.number_input("Parâmetro 3 - Palha Cana (Hidrotérmico)")
        palha4 = st.selectbox("Parâmetro 4 - Palha Cana (Hidrotérmico)", options=["Opção 1", "Opção 2", "Opção 3"])
        palha5 = st.selectbox("Parâmetro 5 - Palha Cana (Hidrotérmico)", options=["Opção A", "Opção B", "Opção C"])
        palha6 = st.selectbox("Parâmetro 6 - Palha Cana (Hidrotérmico)", options=["Sim", "Não"])

# Personalizando a coluna Resultados do Pré-Tratamento (col3)

with col3:
    st.header("Resultados do Pré-Tratamento")
    st.write(f"Aqui é possível ver os resultados obtidos para a etapa de Pré-Tratamento de {biomassa}. Alterne a disposição do gráfico para visualizar mais relações entre as variáveis.")
    st.button("Calcular Rendimento", key="pretratamento_resultados")
    if st.session_state.get("pretratamento_resultados"):
        if biomassa == "Bagaço de Cana-de-Açúcar" and pretratamento == "Ácido/Básico":
            try:
                with open('modelo_pretratamento.pkl', 'rb') as file:
                    modelo = pickle.load(file)
                dados_entrada = [[bagaco1, bagaco2, bagaco3, int(bagaco4[-1]), int(bagaco5[-1]), int(bagaco6)]]
                rendimento_previsto = modelo.predict(dados_entrada)[0]
                st.success(f"Rendimento previsto pelo modelo: {rendimento_previsto:.2f}%")
            except FileNotFoundError:
                st.error("O arquivo do modelo 'modelo_pretratamento.pkl' não foi encontrado.")
            except Exception as e:
                st.error(f"Ocorreu um erro ao carregar o modelo: {e}")

        if biomassa == "Palha da Cana-de-Açúcar" and pretratamento == "Ácido/Básico":
            try:
                with open('modelo_pretratamento.pkl', 'rb') as file:
                    modelo = pickle.load(file)
                dados_entrada = [[palha1, palha2, palha3, int(palha4[-1]), int(palha5[-1]), int(palha6)]]
                rendimento_previsto = modelo.predict(dados_entrada)[0]
                st.success(f"Rendimento previsto pelo modelo: {rendimento_previsto:.2f}%")
            except FileNotFoundError:
                st.error("O arquivo do modelo 'modelo_pretratamento.pkl' não foi encontrado.")
            except Exception as e:
                st.error(f"Ocorreu um erro ao carregar o modelo: {e}")

        if biomassa == "Bagaço de Cana-de-Açúcar" and pretratamento == "Explosão a Vapor":
            try:
                with open('modelo_pretratamento.pkl', 'rb') as file:
                    modelo = pickle.load(file)
                dados_entrada = [[bagaco1, bagaco2, bagaco3, int(bagaco4[-1]), int(bagaco5[-1]), int(bagaco6)]]
                rendimento_previsto = modelo.predict(dados_entrada)[0]
                st.success(f"Rendimento previsto pelo modelo: {rendimento_previsto:.2f}%")
            except FileNotFoundError:
                st.error("O arquivo do modelo 'modelo_pretratamento.pkl' não foi encontrado.")
            except Exception as e:
                st.error(f"Ocorreu um erro ao carregar o modelo: {e}")

        if biomassa == "Palha da Cana-de-Açúcar" and pretratamento == "Explosão a Vapor":
            try:
                with open('modelo_pretratamento.pkl', 'rb') as file:
                    modelo = pickle.load(file)
                dados_entrada = [[palha1, palha2, palha3, int(palha4[-1]), int(palha5[-1]), int(palha6)]]
                rendimento_previsto = modelo.predict(dados_entrada)[0]
                st.success(f"Rendimento previsto pelo modelo: {rendimento_previsto:.2f}%")
            except FileNotFoundError:
                st.error("O arquivo do modelo 'modelo_pretratamento.pkl' não foi encontrado.")
            except Exception as e:
                st.error(f"Ocorreu um erro ao carregar o modelo: {e}")

        if biomassa == "Bagaço de Cana-de-Açúcar" and pretratamento == "Organossolve":
            try:
                with open('modelo_pretratamento.pkl', 'rb') as file:
                    modelo = pickle.load(file)
                dados_entrada = [[bagaco1, bagaco2, bagaco3, int(bagaco4[-1]), int(bagaco5[-1]), int(bagaco6)]]
                rendimento_previsto = modelo.predict(dados_entrada)[0]
                st.success(f"Rendimento previsto pelo modelo: {rendimento_previsto:.2f}%")
            except FileNotFoundError:
                st.error("O arquivo do modelo 'modelo_pretratamento.pkl' não foi encontrado.")
            except Exception as e:
                st.error(f"Ocorreu um erro ao carregar o modelo: {e}")

        if biomassa == "Palha da Cana-de-Açúcar" and pretratamento == "Organossolve":
            try:
                with open('modelo_pretratamento.pkl', 'rb') as file:
                    modelo = pickle.load(file)
                dados_entrada = [[palha1, palha2, palha3, int(palha4[-1]), int(palha5[-1]), int(palha6)]]
                rendimento_previsto = modelo.predict(dados_entrada)[0]
                st.success(f"Rendimento previsto pelo modelo: {rendimento_previsto:.2f}%")
            except FileNotFoundError:
                st.error("O arquivo do modelo 'modelo_pretratamento.pkl' não foi encontrado.")
            except Exception as e:
                st.error(f"Ocorreu um erro ao carregar o modelo: {e}")
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
    st.write(f"Informe os parâmetros da hidrólise para definir a sua condição de operação de {biomassa} com o catalizador.")
    catalizador = st.selectbox("Catalizador", ['Tipo 1', 'Tipo 2', 'Tipo 3'])

    # Se o catalizador utilizado foi Tipo 1 e a biomassa é Bagaço de Cana-de-Açúcar
    if catalizador == "Tipo 1" and biomassa == "Bagaço de Cana-de-Açúcar":
        bagaco11 = st.number_input("Parâmetro 11 - Bagaço")
        bagaco21 = st.number_input("Parâmetro 21 - Bagaço")
        bagaco31 = st.number_input("Parâmetro 31 - Bagaço")
        bagaco41 = st.selectbox("Parâmetro 41 - Bagaço", options=["Opção 1", "Opção 2", "Opção 3"])
        bagaco51 = st.selectbox("Parâmetro 51 - Bagaço", options=["Opção A", "Opção B", "Opção C"])
        bagaco61 = st.selectbox("Parâmetro 61 - Bagaço", options=["Sim", "Não"])

    # Se o catalizador utilizado foi Tipo 1 e a biomassa é Palha da Cana-de-Açúcar
    if catalizador == "Tipo 1" and biomassa == "Palha da Cana-de-Açúcar":
        palha11 = st.number_input("Parâmetro 11 - Palha Cana")
        palha21 = st.number_input("Parâmetro 21 - Palha Cana")
        palha31 = st.number_input("Parâmetro 31 - Palha Cana")
        palha41 = st.selectbox("Parâmetro 41 - Palha Cana", options=["Opção 1", "Opção 2", "Opção 3"])
        palha51 = st.selectbox("Parâmetro 51 - Palha Cana", options=["Opção A", "Opção B", "Opção C"])
        palha61 = st.selectbox("Parâmetro 61 - Palha Cana", options=["Sim", "Não"])

    # Se o catalizador utilizado foi Tipo 2 e a biomassa é Bagaço de Cana-de-Açúcar
    if catalizador == "Tipo 2" and biomassa == "Bagaço de Cana-de-Açúcar":
        bagaco11 = st.number_input("Parâmetro 11 - Bagaço")
        bagaco21 = st.number_input("Parâmetro 21 - Bagaço")
        bagaco31 = st.number_input("Parâmetro 31 - Bagaço")
        bagaco41 = st.selectbox("Parâmetro 41 - Bagaço", options=["Opção 1", "Opção 2", "Opção 3"])
        bagaco51 = st.selectbox("Parâmetro 51 - Bagaço", options=["Opção A", "Opção B", "Opção C"])
        bagaco61 = st.selectbox("Parâmetro 61 - Bagaço", options=["Sim", "Não"])

    # Se o catalizador utilizado foi Tipo 2 e a biomassa é Palha da Cana-de-Açúcar
    if catalizador == "Tipo 2" and biomassa == "Palha da Cana-de-Açúcar":
        palha11 = st.number_input("Parâmetro 11 - Palha Cana")
        palha21 = st.number_input("Parâmetro 21 - Palha Cana")
        palha31 = st.number_input("Parâmetro 31 - Palha Cana")
        palha41 = st.selectbox("Parâmetro 41 - Palha Cana", options=["Opção 1", "Opção 2", "Opção 3"])
        palha51 = st.selectbox("Parâmetro 51 - Palha Cana", options=["Opção A", "Opção B", "Opção C"])
        palha61 = st.selectbox("Parâmetro 61 - Palha Cana", options=["Sim", "Não"])

    # Se o catalizador utilizado foi Tipo 3 e a biomassa é Bagaço de Cana-de-Açúcar
    if catalizador == "Tipo 3" and biomassa == "Bagaço de Cana-de-Açúcar":
        bagaco11 = st.number_input("Parâmetro 11 - Bagaço")
        bagaco21 = st.number_input("Parâmetro 21 - Bagaço")
        bagaco31 = st.number_input("Parâmetro 31 - Bagaço")
        bagaco41 = st.selectbox("Parâmetro 41 - Bagaço", options=["Opção 1", "Opção 2", "Opção 3"])
        bagaco51 = st.selectbox("Parâmetro 51 - Bagaço", options=["Opção A", "Opção B", "Opção C"])
        bagaco61 = st.selectbox("Parâmetro 61 - Bagaço", options=["Sim", "Não"])

    # Se o catalizador utilizado foi Tipo 3 e a biomassa é Palha da Cana-de-Açúcar
    if catalizador == "Tipo 3" and biomassa == "Palha da Cana-de-Açúcar":
        palha11 = st.number_input("Parâmetro 11 - Palha Cana")
        palha21 = st.number_input("Parâmetro 21 - Palha Cana")
        palha31 = st.number_input("Parâmetro 31 - Palha Cana")
        palha41 = st.selectbox("Parâmetro 41 - Palha Cana", options=["Opção 1", "Opção 2", "Opção 3"])
        palha51 = st.selectbox("Parâmetro 51 - Palha Cana", options=["Opção A", "Opção B", "Opção C"])
        palha61 = st.selectbox("Parâmetro 61 - Palha Cana", options=["Sim", "Não"])

# Alteração 3: Mapeando opções de seleção para valores numéricos antes de usá-los
opcoes_bagaco41 = {"Opção 1": 1, "Opção 2": 2, "Opção 3": 3}
opcoes_bagaco51 = {"Opção A": 1, "Opção B": 2, "Opção C": 3}

# Personalizando a coluna Resultados da Hidrólise (col6)
with col6:
    st.header("Resultados da Hidrólise")
    st.write(f"Aqui é possível ver os resultados obtidos para a etapa de Hidrólise de {biomassa}. Alterne a disposição do gráfico para visualizar mais relações entre as variáveis.")
    st.button("Calcular Rendimento", key="hidrolise_resultados")
    if st.session_state.get("hidrolise_resultados"):
        if catalizador == "Tipo 1" and biomassa == "Bagaço de Cana-de-Açúcar":
            try:
                with open('modelo_hidrolise.pkl', 'rb') as file:
                    modelo = pickle.load(file)
                dados_entrada = [[bagaco11, bagaco21, bagaco31, int(bagaco41[-1]), int(bagaco51[-1]), int(bagaco61 == "Sim")]]
                rendimento_previsto = modelo.predict(dados_entrada)[0]
                st.success(f"Rendimento previsto pelo modelo: {rendimento_previsto:.2f}%")
            except FileNotFoundError:
                st.error("O arquivo do modelo 'modelo_hidrolise.pkl' não foi encontrado.")
            except Exception as e:
                st.error(f"Ocorreu um erro ao carregar o modelo: {e}")

        if catalizador == "Tipo 1" and biomassa == "Palha da Cana-de-Açúcar":
            try:
                with open('modelo_hidrolise.pkl', 'rb') as file:
                    modelo = pickle.load(file)
                dados_entrada = [[palha11, palha21, palha31, int(palha41[-1]), int(palha51[-1]), int(palha61 == "Sim")]]
                rendimento_previsto = modelo.predict(dados_entrada)[0]
                st.success(f"Rendimento previsto pelo modelo: {rendimento_previsto:.2f}%")
            except FileNotFoundError:
                st.error("O arquivo do modelo 'modelo_hidrolise.pkl' não foi encontrado.")
            except Exception as e:
                st.error(f"Ocorreu um erro ao carregar o modelo: {e}")

        if catalizador == "Tipo 2" and biomassa == "Bagaço de Cana-de-Açúcar":
            try:
                with open('modelo_hidrolise.pkl', 'rb') as file:
                    modelo = pickle.load(file)
                bagaco41_valor = opcoes_bagaco41.get(bagaco41, 0)
                bagaco51_valor = opcoes_bagaco51.get(bagaco51, 0)
                dados_entrada = [[bagaco11, bagaco21, bagaco31, bagaco41_valor, bagaco51_valor, int(bagaco61 == "Sim")]]
                rendimento_previsto = modelo.predict(dados_entrada)[0]
                st.success(f"Rendimento previsto pelo modelo: {rendimento_previsto:.2f}%")
            except FileNotFoundError:
                st.error("O arquivo do modelo 'modelo_hidrolise.pkl' não foi encontrado.")
            except Exception as e:
                st.error(f"Ocorreu um erro ao carregar o modelo: {e}")

        if catalizador == "Tipo 2" and biomassa == "Palha da Cana-de-Açúcar":
            try:
                with open('modelo_hidrolise.pkl', 'rb') as file:
                    modelo = pickle.load(file)
                dados_entrada = [[palha11, palha21, palha31, int(palha41[-1]), int(palha51[-1]), int(palha61 == "Sim")]]
                rendimento_previsto = modelo.predict(dados_entrada)[0]
                st.success(f"Rendimento previsto pelo modelo: {rendimento_previsto:.2f}%")
            except FileNotFoundError:
                st.error("O arquivo do modelo 'modelo_hidrolise.pkl' não foi encontrado.")
            except Exception as e:
                st.error(f"Ocorreu um erro ao carregar o modelo: {e}")

        if catalizador == "Tipo 3" and biomassa == "Bagaço de Cana-de-Açúcar":
            try:
                with open('modelo_hidrolise.pkl', 'rb') as file:
                    modelo = pickle.load(file)
                dados_entrada = [[bagaco11, bagaco21, bagaco31, int(bagaco41[-1]), int(bagaco51[-1]), int(bagaco61 == "Sim")]]
                rendimento_previsto = modelo.predict(dados_entrada)[0]
                st.success(f"Rendimento previsto pelo modelo: {rendimento_previsto:.2f}%")
            except FileNotFoundError:
                st.error("O arquivo do modelo 'modelo_hidrolise.pkl' não foi encontrado.")
            except Exception as e:
                st.error(f"Ocorreu um erro ao carregar o modelo: {e}")
                
        if catalizador == "Tipo 3" and biomassa == "Palha da Cana-de-Açúcar":
            try:
                with open('modelo_hidrolise.pkl', 'rb') as file:
                    modelo = pickle.load(file)
                dados_entrada = [[palha11, palha21, palha31, int(palha41[-1]), int(palha51[-1]), int(palha61 == "Sim")]]
                rendimento_previsto = modelo.predict(dados_entrada)[0]
                st.success(f"Rendimento previsto pelo modelo: {rendimento_previsto:.2f}%")
            except FileNotFoundError:
                st.error("O arquivo do modelo 'modelo_hidrolise.pkl' não foi encontrado.")
            except Exception as e:
                st.error(f"Ocorreu um erro ao carregar o modelo: {e}")
    st.metric(label="🔍 **Rendimento Previsto (%)**", value="91%", delta="+5%", help="Este é o rendimento previsto para as condições selecionadas.")
    
    # Alteração 6: Substituindo gráficos fixos por gráficos baseados em dados reais
    # Exemplo de gráfico atualizado com dados reais
    if rendimento_previsto:
        fig1 = go.Figure(data=[go.Bar(x=['Celulose', 'Lignina', 'Hemicelulose'], y=[celulose1, lignina1, hemicelulose1])])
        fig1.update_layout(title="Composição após Hidrólise", xaxis_title="Componentes", yaxis_title="Porcentagem (%)")
        st.plotly_chart(fig1, key="hidrolise_grafico")