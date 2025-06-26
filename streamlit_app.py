import streamlit as st
import pandas as pd
import pickle
from plotly import graph_objs as go

# Configurando o layout para modo "wide"
st.set_page_config(layout="wide")

# Título do app
st.title('⚗️Ethanol AI (Software under development)')

# Informação principal
st.write("Ethanol AI is a tool created within a research program called scientific initiation by researchers from UFSCar and DTU with funding from FAPESP. It is particularly useful for studying the behavior of different second-generation ethanol production processes when subjected to various operating conditions. This software implements hybrid machine learning models, previously trained using knowledge generated from previous research works at UFSCar and abroad, to predict the outcomes. Here, you can test different combinations of initial conditions, essentially finding the maximum possible yield for each situation.")
st.markdown("<hr style='border: 1px solid #ccc;' />", unsafe_allow_html=True)

# Etapa de Pré-Tratamento
st.markdown(
    "<h1 style='font-size:50px;'>Pre-Treatment</h1>",
    unsafe_allow_html=True
)
st.write("In this section, introduce the relevant data for calculating the yield of the Pre-Treatment.")

# Criando colunas "Parâmetros" e "Resultados"
col1, spacer, col2, spacer2, col3 = st.columns([10, 2, 10, 2, 10])

# Customizing the Biomass column (col1)
with col1:
    # Model characteristics selection
    st.header("Initial Data")
    st.write("Enter the main data for your simulation.")
    biomassa = st.selectbox("Select a biomass type", ['Sugarcane Bagasse', 'Sugarcane Straw'])
    pretratamento = st.selectbox("Select a Pre-Treatment type", ['Acid', 'Basic', 'Organosolv', 'Hydrothermal'])
    celulose = st.number_input("Cellulose Percentage (0.00 - 100.00) (%)", min_value=0.0, max_value=100.0, format="%.2f")
    lignina = st.number_input("Lignin Percentage (0.00 - 100.00) (%)", min_value=0.0, max_value=100.0, format="%.2f")
    hemicelulose = st.number_input("Hemicellulose Percentage (0.00 - 100.00) (%)", min_value=0.0, max_value=100.0, format="%.2f")
    cinzas = st.number_input("Ash Percentage (0.00 - 100.00) (%)", min_value=0.0, max_value=100.0, format="%.2f")

# Customizing the Pre-Treatment Parameters column (col2)
with col2:
    
    st.header("Pre-Treatment Parameters")
    st.write(f"Enter the pre-treatment parameters for {biomassa}.")
    
    # If the chosen biomass was Sugarcane Bagasse and the pre-treatment is Acid
    if biomassa == "Sugarcane Bagasse" and pretratamento == "Acid":
        bagaco1 = st.number_input("Parameter 1 - Bagasse (Acid)")
        bagaco2 = st.number_input("Parameter 2 - Bagasse (Acid)")
        bagaco3 = st.number_input("Parameter 3 - Bagasse (Acid)")
        bagaco4 = st.selectbox("Parameter 4 - Bagasse (Acid)", options=["Option 1", "Option 2", "Option 3"])
        bagaco5 = st.selectbox("Parameter 5 - Bagasse (Acid)", options=["Option A", "Option B", "Option C"])
        bagaco6 = st.checkbox("Presence of reagent x (Acid)")

    # If the chosen biomass was Sugarcane Bagasse and the pre-treatment is Basic
    if biomassa == "Sugarcane Bagasse" and pretratamento == "Basic":
        bagaco1 = st.number_input("Parameter 1 - Bagasse (Basic)")
        bagaco2 = st.number_input("Parameter 2 - Bagasse (Basic)")
        bagaco3 = st.number_input("Parameter 3 - Bagasse (Basic)")
        bagaco4 = st.selectbox("Parameter 4 - Bagasse (Basic)", options=["Option 1", "Option 2", "Option 3"])
        bagaco5 = st.selectbox("Parameter 5 - Bagasse (Basic)", options=["Option A", "Option B", "Option C"])
        bagaco6 = st.checkbox("Presence of reagent x (Basic)")

    # If the chosen biomass was Sugarcane Straw and the pre-treatment is Acid
    if biomassa == "Sugarcane Straw" and pretratamento == "Acid":
        palha1 = st.number_input("Parameter 1 - Straw (Acid)")
        palha2 = st.number_input("Parameter 2 - Straw (Acid)")
        palha3 = st.number_input("Parameter 3 - Straw (Acid)")
        palha4 = st.selectbox("Parameter 4 - Straw (Acid)", options=["Option 1", "Option 2", "Option 3"])
        palha5 = st.selectbox("Parameter 5 - Straw (Acid)", options=["Option A", "Option B", "Option C"])
        palha6 = st.selectbox("Parameter 6 - Straw (Acid)", options=["Yes", "No"])

    # If the chosen biomass was Sugarcane Straw and the pre-treatment is Basic
    if biomassa == "Sugarcane Straw" and pretratamento == "Basic":
        palha1 = st.number_input("Parameter 1 - Straw (Basic)")
        palha2 = st.number_input("Parameter 2 - Straw (Basic)")
        palha3 = st.number_input("Parameter 3 - Straw (Basic)")
        palha4 = st.selectbox("Parameter 4 - Straw (Basic)", options=["Option 1", "Option 2", "Option 3"])
        palha5 = st.selectbox("Parameter 5 - Straw (Basic)", options=["Option A", "Option B", "Option C"])
        palha6 = st.selectbox("Parameter 6 - Straw (Basic)", options=["Yes", "No"])
    
    # If the chosen biomass was Sugarcane Bagasse and the pre-treatment is Organosolv
    if biomassa == "Sugarcane Bagasse" and pretratamento == "Organosolv":
        bagaco1 = st.number_input("Parameter 1 - Bagasse (Organosolv)")
        bagaco2 = st.number_input("Parameter 2 - Bagasse (Organosolv)")
        bagaco3 = st.number_input("Parameter 3 - Bagasse (Organosolv)")
        bagaco4 = st.selectbox("Parameter 4 - Bagasse (Organosolv)", options=["Option 1", "Option 2", "Option 3"])
        bagaco5 = st.selectbox("Parameter 5 - Bagasse (Organosolv)", options=["Option A", "Option B", "Option C"])
        bagaco6 = st.checkbox("Presence of reagent x (Organosolv)")

    # If the chosen biomass was Sugarcane Bagasse and the pre-treatment is Hydrothermal
    if biomassa == "Sugarcane Bagasse" and pretratamento == "Hydrothermal":
        bagaco1 = st.number_input("Parameter 1 - Bagasse (Hydrothermal)")
        bagaco2 = st.number_input("Parameter 2 - Bagasse (Hydrothermal)")
        bagaco3 = st.number_input("Parameter 3 - Bagasse (Hydrothermal)")
        bagaco4 = st.selectbox("Parameter 4 - Bagasse (Hydrothermal)", options=["Option 1", "Option 2", "Option 3"])
        bagaco5 = st.selectbox("Parameter 5 - Bagasse (Hydrothermal)", options=["Option A", "Option B", "Option C"])
        bagaco6 = st.checkbox("Presence of reagent x (Hydrothermal)")

    # If the chosen biomass was Sugarcane Straw and the pre-treatment is Organosolv
    if biomassa == "Sugarcane Straw" and pretratamento == "Organosolv":
        palha1 = st.number_input("Parameter 1 - Straw (Organosolv)")
        palha2 = st.number_input("Parameter 2 - Straw (Organosolv)")
        palha3 = st.number_input("Parameter 3 - Straw (Organosolv)")
        palha4 = st.selectbox("Parameter 4 - Straw (Organosolv)", options=["Option 1", "Option 2", "Option 3"])
        palha5 = st.selectbox("Parameter 5 - Straw (Organosolv)", options=["Option A", "Option B", "Option C"])
        palha6 = st.selectbox("Parameter 6 - Straw (Organosolv)", options=["Yes", "No"])

    # If the chosen biomass was Sugarcane Straw and the pre-treatment is Hydrothermal
    if biomassa == "Sugarcane Straw" and pretratamento == "Hydrothermal":
        palha1 = st.number_input("Parameter 1 - Straw (Hydrothermal)")
        palha2 = st.number_input("Parameter 2 - Straw (Hydrothermal)")
        palha3 = st.number_input("Parameter 3 - Straw (Hydrothermal)")
        palha4 = st.selectbox("Parameter 4 - Straw (Hydrothermal)", options=["Option 1", "Option 2", "Option 3"])
        palha5 = st.selectbox("Parameter 5 - Straw (Hydrothermal)", options=["Option A", "Option B", "Option C"])
        palha6 = st.selectbox("Parameter 6 - Straw (Hydrothermal)", options=["Yes", "No"])

# Customizing the Pre-Treatment Results column (col3)

with col3:
    st.header("Pre-Treatment Results")
    st.write(f"Here you can see the results obtained for the Pre-Treatment stage of {biomassa}. Change the chart layout to visualize more relationships between the variables.")
    st.button("Calculate Yield", key="pretratamento_resultados")
    if st.session_state.get("pretratamento_resultados"):
        if biomassa == "Sugarcane Bagasse" and (pretratamento == "Acid" or pretratamento == "Basic"):
            try:
                with open('modelo_pretratamento.pkl', 'rb') as file:
                    modelo = pickle.load(file)
                dados_entrada = [[bagaco1, bagaco2, bagaco3, int(bagaco4[-1]), int(bagaco5[-1]), int(bagaco6)]]
                rendimento_previsto = modelo.predict(dados_entrada)[0]
                st.success(f"Yield predicted by the model: {rendimento_previsto:.2f}%")
            except FileNotFoundError:
                st.error("The model file 'modelo_pretratamento.pkl' was not found.")
            except Exception as e:
                st.error(f"An error occurred while loading the model: {e}")

        if biomassa == "Sugarcane Straw" and (pretratamento == "Acid" or pretratamento == "Basic"):
            try:
                with open('modelo_pretratamento.pkl', 'rb') as file:
                    modelo = pickle.load(file)
                dados_entrada = [[palha1, palha2, palha3, int(palha4[-1]), int(palha5[-1]), int(palha6 == "Yes")]]
                rendimento_previsto = modelo.predict(dados_entrada)[0]
                st.success(f"Yield predicted by the model: {rendimento_previsto:.2f}%")
            except FileNotFoundError:
                st.error("The model file 'modelo_pretratamento.pkl' was not found.")
            except Exception as e:
                st.error(f"An error occurred while loading the model: {e}")

        # Removing conditions for "Steam Explosion" that doesn't exist in the options

        if biomassa == "Sugarcane Bagasse" and pretratamento == "Organosolv":
            try:
                with open('modelo_pretratamento.pkl', 'rb') as file:
                    modelo = pickle.load(file)
                dados_entrada = [[bagaco1, bagaco2, bagaco3, int(bagaco4[-1]), int(bagaco5[-1]), int(bagaco6)]]
                rendimento_previsto = modelo.predict(dados_entrada)[0]
                st.success(f"Yield predicted by the model: {rendimento_previsto:.2f}%")
            except FileNotFoundError:
                st.error("The model file 'modelo_pretratamento.pkl' was not found.")
            except Exception as e:
                st.error(f"An error occurred while loading the model: {e}")

        elif biomassa == "Sugarcane Straw" and pretratamento == "Organosolv":
            try:
                with open('modelo_pretratamento.pkl', 'rb') as file:
                    modelo = pickle.load(file)
                dados_entrada = [[palha1, palha2, palha3, int(palha4[-1]), int(palha5[-1]), int(palha6 == "Yes")]]
                rendimento_previsto = modelo.predict(dados_entrada)[0]
                st.success(f"Yield predicted by the model: {rendimento_previsto:.2f}%")
            except FileNotFoundError:
                st.error("The model file 'modelo_pretratamento.pkl' was not found.")
            except Exception as e:
                st.error(f"An error occurred while loading the model: {e}")

        elif biomassa == "Sugarcane Bagasse" and pretratamento == "Hydrothermal":
            try:
                with open('modelo_pretratamento.pkl', 'rb') as file:
                    modelo = pickle.load(file)
                dados_entrada = [[bagaco1, bagaco2, bagaco3, int(bagaco4[-1]), int(bagaco5[-1]), int(bagaco6)]]
                rendimento_previsto = modelo.predict(dados_entrada)[0]
                st.success(f"Yield predicted by the model: {rendimento_previsto:.2f}%")
            except FileNotFoundError:
                st.error("The model file 'modelo_pretratamento.pkl' was not found.")
            except Exception as e:
                st.error(f"An error occurred while loading the model: {e}")

        elif biomassa == "Sugarcane Straw" and pretratamento == "Hydrothermal":
            try:
                with open('modelo_pretratamento.pkl', 'rb') as file:
                    modelo = pickle.load(file)
                dados_entrada = [[palha1, palha2, palha3, int(palha4[-1]), int(palha5[-1]), int(palha6 == "Yes")]]
                rendimento_previsto = modelo.predict(dados_entrada)[0]
                st.success(f"Yield predicted by the model: {rendimento_previsto:.2f}%")
            except FileNotFoundError:
                st.error("The model file 'modelo_pretratamento.pkl' was not found.")
            except Exception as e:
                st.error(f"An error occurred while loading the model: {e}")
    
    # Initialize rendimento_previsto to avoid undefined variable error
    rendimento_previsto = None
    st.metric(label="🔍 **Predicted Yield (%)**", value= "86%", delta="+6%", help="This is the predicted yield for the selected conditions.")
    # Creating an example chart
    fig = go.Figure(data=[go.Bar(x=['Category 1', 'Category 2', 'Category 3'], y=[10, 20, 30])])
    fig.update_layout(title="Example Chart", xaxis_title="Categories", yaxis_title="Values")
    
    # Displaying the chart
    st.plotly_chart(fig, key = "pretratamento_grafico")

st.markdown("<hr style='border: 1px solid #ccc;' />", unsafe_allow_html=True)

# Next Stage: Hydrolysis

st.markdown(
    "<h1 style='font-size:50px;'>Hydrolysis</h1>",
    unsafe_allow_html=True
)
st.write("In this section, introduce the relevant data for calculating the yield of Hydrolysis.")
# Creating "Parameters" and "Results" columns
col4, spacer3, col5, spacer4, col6 = st.columns([10, 2, 10, 2, 10])

# Customizing the Pre-Treatment Data column (col4)
with col4:
    st.header("Pre-Treatment Data")
    st.write("Enter the data obtained from pre-treatment.")
    celulose1 = st.number_input("Cellulose Percentage", min_value=0.0, max_value=100.0, format="%.2f")
    lignina1 = st.number_input("Lignin Percentage", min_value=0.0, max_value=100.0, format="%.2f")
    hemicelulose1 = st.number_input("Hemicellulose Percentage", min_value=0.0, max_value=100.0, format="%.2f")
    cinzas1 = st.number_input("Ash Percentage", min_value=0.0, max_value=100.0, format="%.2f")

# Customizing the Hydrolysis Parameters column (col5)
with col5:
    st.header("Hydrolysis Parameters")
    st.write(f"Enter the hydrolysis parameters to define your operating condition for {biomassa} with the enzyme.")
    enzyme = st.selectbox("Enzyme", ['Type 1', 'Type 2', 'Type 3'])
    
    # Simplified parameters for all conditions
    solid_loading = st.number_input("Initial Solids Loading (g/L)", min_value=0.0, format="%.2f")
    enzyme_loading = st.number_input("Initial Enzyme Loading (g/L)", min_value=0.0, format="%.2f")

# Alteration 3: Mapping selection options to numerical values before using them (no longer needed for simplified version)
enzyme_types = {"Type 1": 1, "Type 2": 2, "Type 3": 3}

# Customizing the Hydrolysis Results column (col6)
with col6:
    st.header("Hydrolysis Results")
    st.write(f"Here you can see the results obtained for the Hydrolysis stage of {biomassa}. Change the chart layout to visualize more relationships between the variables.")
    st.button("Calculate Yield", key="hidrolise_resultados")
    
    # Initialize rendimento_previsto to avoid undefined variable error
    rendimento_previsto = None
    
    if st.session_state.get("hidrolise_resultados"):
        try:
            with open('modelo_hidrolise.pkl', 'rb') as file:
                modelo = pickle.load(file)
            
            # Convert enzyme type to numerical value
            enzyme_value = enzyme_types.get(enzyme, 1)
            
            # Convert biomass to numerical value (1 for Bagasse, 2 for Straw)
            biomass_value = 1 if biomassa == "Sugarcane Bagasse" else 2
            
            # Create input data with simplified parameters
            dados_entrada = [[celulose1, lignina1, hemicelulose1, cinzas1, solid_loading, enzyme_loading, enzyme_value, biomass_value]]
            rendimento_previsto = modelo.predict(dados_entrada)[0]
            st.success(f"Yield predicted by the model: {rendimento_previsto:.2f}%")
        except FileNotFoundError:
            st.error("The model file 'modelo_hidrolise.pkl' was not found.")
        except Exception as e:
            st.error(f"An error occurred while loading the model: {e}")
    st.metric(label="🔍 **Predicted Yield (%)**", value="91%", delta="+5%", help="This is the predicted yield for the selected conditions.")
    
    # Alteration 6: Replacing fixed charts with real data-based charts
    # Example of updated chart with real data
    if rendimento_previsto:
        fig1 = go.Figure(data=[go.Bar(x=['Cellulose', 'Lignin', 'Hemicellulose'], y=[celulose1, lignina1, hemicelulose1])])
        fig1.update_layout(title="Composition after Hydrolysis", xaxis_title="Components", yaxis_title="Percentage (%)")
        st.plotly_chart(fig1, key="hidrolise_grafico")