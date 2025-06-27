import streamlit as st
import pandas as pd
import pickle
from plotly import graph_objs as go

# Helper function to load model and make predictions
def load_model_and_predict(model_path, input_data):
    """
    Load a pickle model and make predictions with error handling.
    
    Args:
        model_path (str): Path to the pickle model file
        input_data (list): Input data for prediction
    
    Returns:
        tuple: (success, result) where success is bool and result is either prediction or error message
    """
    try:
        with open(model_path, 'rb') as file:
            modelo = pickle.load(file)
        rendimento_previsto = modelo.predict(input_data)[0]
        return True, rendimento_previsto
    except FileNotFoundError:
        return False, f"The model file '{model_path}' was not found."
    except Exception as e:
        return False, f"An error occurred while loading the model: {e}"

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
    
    # Validation function for pre-treatment inputs
    def validate_pretreatment_inputs():
        errors = []
        
        # Check if composition percentages sum to reasonable values
        total_composition = celulose + lignina + hemicelulose + cinzas
        if total_composition > 100:
            errors.append("Total composition cannot exceed 100%")
        elif total_composition < 50:
            errors.append("Total composition seems too low (< 50%)")
        
        # Check if any composition is zero
        if celulose == 0:
            errors.append("Cellulose percentage must be greater than 0")
        if lignina == 0:
            errors.append("Lignin percentage must be greater than 0")
        if hemicelulose == 0:
            errors.append("Hemicellulose percentage must be greater than 0")
        
        return errors
    
    # Validation function for parameter inputs
    def validate_parameter_inputs():
        errors = []
        
        try:
            if biomassa == "Sugarcane Bagasse":
                if 'bagaco1' not in locals() and 'bagaco1' not in globals():
                    errors.append("Please fill in all pre-treatment parameters")
                elif globals().get('bagaco1', 0) <= 0 or globals().get('bagaco2', 0) <= 0 or globals().get('bagaco3', 0) <= 0:
                    errors.append("All numeric parameters must be greater than 0")
            elif biomassa == "Sugarcane Straw":
                if 'palha1' not in locals() and 'palha1' not in globals():
                    errors.append("Please fill in all pre-treatment parameters")
                elif globals().get('palha1', 0) <= 0 or globals().get('palha2', 0) <= 0 or globals().get('palha3', 0) <= 0:
                    errors.append("All numeric parameters must be greater than 0")
        except:
            errors.append("Please fill in all pre-treatment parameters")
        
        return errors
    
    st.button("Calculate Yield", key="pretratamento_resultados")
    if st.session_state.get("pretratamento_resultados"):
        # Validate inputs before processing
        composition_errors = validate_pretreatment_inputs()
        parameter_errors = validate_parameter_inputs()
        validation_errors = composition_errors + parameter_errors
        
        if validation_errors:
            for error in validation_errors:
                st.error(f"❌ {error}")
            st.warning("Please correct the errors above before calculating the yield.")
        else:
            if biomassa == "Sugarcane Bagasse" and (pretratamento == "Acid" or pretratamento == "Basic"):
                try:
                    dados_entrada = [[bagaco1, bagaco2, bagaco3, int(bagaco4[-1]), int(bagaco5[-1]), int(bagaco6)]]
                    sucesso, resultado = load_model_and_predict('modelo_pretratamento.pkl', dados_entrada)
                    if sucesso:
                        st.success(f"Yield predicted by the model: {resultado:.2f}%")
                    else:
                        st.error(resultado)
                except Exception as e:
                    st.error(f"An error occurred while processing: {e}")

            elif biomassa == "Sugarcane Straw" and (pretratamento == "Acid" or pretratamento == "Basic"):
                try:
                    dados_entrada = [[palha1, palha2, palha3, int(palha4[-1]), int(palha5[-1]), int(palha6 == "Yes")]]
                    sucesso, resultado = load_model_and_predict('modelo_pretratamento.pkl', dados_entrada)
                    if sucesso:
                        st.success(f"Yield predicted by the model: {resultado:.2f}%")
                    else:
                        st.error(resultado)
                except Exception as e:
                    st.error(f"An error occurred while processing: {e}")

            elif biomassa == "Sugarcane Bagasse" and pretratamento == "Organosolv":
                try:
                    dados_entrada = [[bagaco1, bagaco2, bagaco3, int(bagaco4[-1]), int(bagaco5[-1]), int(bagaco6)]]
                    sucesso, resultado = load_model_and_predict('modelo_pretratamento.pkl', dados_entrada)
                    if sucesso:
                        st.success(f"Yield predicted by the model: {resultado:.2f}%")
                    else:
                        st.error(resultado)
                except Exception as e:
                    st.error(f"An error occurred while processing: {e}")

            elif biomassa == "Sugarcane Straw" and pretratamento == "Organosolv":
                try:
                    dados_entrada = [[palha1, palha2, palha3, int(palha4[-1]), int(palha5[-1]), int(palha6 == "Yes")]]
                    sucesso, resultado = load_model_and_predict('modelo_pretratamento.pkl', dados_entrada)
                    if sucesso:
                        st.success(f"Yield predicted by the model: {resultado:.2f}%")
                    else:
                        st.error(resultado)
                except Exception as e:
                    st.error(f"An error occurred while processing: {e}")

            elif biomassa == "Sugarcane Bagasse" and pretratamento == "Hydrothermal":
                try:
                    dados_entrada = [[bagaco1, bagaco2, bagaco3, int(bagaco4[-1]), int(bagaco5[-1]), int(bagaco6)]]
                    sucesso, resultado = load_model_and_predict('modelo_pretratamento.pkl', dados_entrada)
                    if sucesso:
                        st.success(f"Yield predicted by the model: {resultado:.2f}%")
                    else:
                        st.error(resultado)
                except Exception as e:
                    st.error(f"An error occurred while processing: {e}")

            elif biomassa == "Sugarcane Straw" and pretratamento == "Hydrothermal":
                try:
                    dados_entrada = [[palha1, palha2, palha3, int(palha4[-1]), int(palha5[-1]), int(palha6 == "Yes")]]
                    sucesso, resultado = load_model_and_predict('modelo_pretratamento.pkl', dados_entrada)
                    if sucesso:
                        st.success(f"Yield predicted by the model: {resultado:.2f}%")
                    else:
                        st.error(resultado)
                except Exception as e:
                    st.error(f"An error occurred while processing: {e}")
    
    # Initialize rendimento_previsto to avoid undefined variable error
    rendimento_previsto = None
    st.metric(label="🔍 **Predicted Yield (%)**", value= "86%", delta="+6%", help="This is the predicted yield for the selected conditions.")
    # Creating an example chart
    fig = go.Figure(data=[go.Bar(x=['Category 1', 'Category 2', 'Category 3'], y=[10, 20, 30])])
    fig.update_layout(title="Example Chart", xaxis_title="Categories", yaxis_title="Values")
    
    # Displaying the chart
    st.plotly_chart(fig, key = "pretratamento_grafico")

st.markdown("<hr style='border: 1px solid #ccc;' />", unsafe_allow_html=True)

# Next Stage: Enzymatic Hydrolysis

st.markdown(
    "<h1 style='font-size:50px;'>Enzymatic Hydrolysis</h1>",
    unsafe_allow_html=True
)
st.write("In this section, introduce the relevant data for calculating the yield of Enzymatic Hydrolysis.")
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

# Customizing the Enzymatic Hydrolysis Parameters column (col5)
with col5:
    st.header("Enzymatic Hydrolysis Parameters")
    st.write(f"Enter the enzymatic hydrolysis parameters to define your operating condition for {biomassa} with the enzyme.")
    enzyme = st.selectbox("Enzyme", ['Type 1', 'Type 2', 'Type 3'])
    
    # Simplified parameters for all conditions
    solid_loading = st.number_input("Initial Solids Loading (g/L)", min_value=0.0, format="%.2f")
    enzyme_loading = st.number_input("Initial Enzyme Loading (g/L)", min_value=0.0, format="%.2f")
    reaction_time = st.slider("Reaction Time (h)", min_value=0.0, max_value=96.0, value=96.0, step=0.1, format="%.2f")

# Alteration 3: Mapping selection options to numerical values before using them (no longer needed for simplified version)
enzyme_types = {"Type 1": 1, "Type 2": 2, "Type 3": 3}

# Customizing the Enzymatic Hydrolysis Results column (col6)
with col6:
    st.header("Enzymatic Hydrolysis Results")
    st.write(f"Here you can see the results obtained for the Enzymatic Hydrolysis stage of {biomassa}. Change the chart layout to visualize more relationships between the variables.")
    
    # Validation function for enzymatic hydrolysis inputs
    def validate_hydrolysis_inputs():
        errors = []
        
        # Check if composition percentages sum to reasonable values
        total_composition = celulose1 + lignina1 + hemicelulose1 + cinzas1
        if total_composition > 100:
            errors.append("Total composition cannot exceed 100%")
        elif total_composition < 50:
            errors.append("Total composition seems too low (< 50%)")
        
        # Check if any composition is zero (except ash which can be zero)
        if celulose1 == 0:
            errors.append("Cellulose percentage must be greater than 0")
        if lignina1 == 0:
            errors.append("Lignin percentage must be greater than 0")
        if hemicelulose1 == 0:
            errors.append("Hemicellulose percentage must be greater than 0")
        
        # Check if parameters are positive
        if solid_loading <= 0:
            errors.append("Initial Solids Loading must be greater than 0")
        if enzyme_loading <= 0:
            errors.append("Initial Enzyme Loading must be greater than 0")
        
        # Check for reasonable ranges
        if solid_loading > 500:
            errors.append("Initial Solids Loading seems too high (> 500 g/L)")
        if enzyme_loading > 100:
            errors.append("Initial Enzyme Loading seems too high (> 100 g/L)")
        
        return errors
    
    st.button("Calculate Yield", key="hidrolise_resultados")
    
    # Initialize rendimento_previsto to avoid undefined variable error
    rendimento_previsto = None
    
    if st.session_state.get("hidrolise_resultados"):
        # Validate inputs before processing
        validation_errors = validate_hydrolysis_inputs()
        
        if validation_errors:
            for error in validation_errors:
                st.error(f"❌ {error}")
            st.warning("Please correct the errors above before calculating the yield.")
        else:
            try:
                with open('modelo_hidrolise.pkl', 'rb') as file:
                    modelo = pickle.load(file)
                
                # Convert enzyme type to numerical value
                enzyme_value = enzyme_types.get(enzyme, 1)
                
                # Convert biomass to numerical value (1 for Bagasse, 2 for Straw)
                biomass_value = 1 if biomassa == "Sugarcane Bagasse" else 2
                
                # Create input data with simplified parameters
                dados_entrada = [[celulose1, lignina1, hemicelulose1, cinzas1, solid_loading, enzyme_loading, enzyme_value, biomass_value]]
                sucesso, resultado = load_model_and_predict('modelo_hidrolise.pkl', dados_entrada)
                if sucesso:
                    st.success(f"Yield predicted by the model: {resultado:.2f}%")
                else:
                    st.error(resultado)
            except Exception as e:
                st.error(f"An error occurred while processing: {e}")
    st.metric(label="🔍 **Predicted Yield (%)**", value="91%", delta="+5%", help="This is the predicted yield for the selected conditions.")
    
    # Alteration 6: Replacing fixed charts with real data-based charts
    # Example of updated chart with real data
    if rendimento_previsto:
        fig1 = go.Figure(data=[go.Bar(x=['Cellulose', 'Lignin', 'Hemicellulose'], y=[celulose1, lignina1, hemicelulose1])])
        fig1.update_layout(title="Composition after Enzymatic Hydrolysis", xaxis_title="Components", yaxis_title="Percentage (%)")
        st.plotly_chart(fig1, key="hidrolise_grafico")