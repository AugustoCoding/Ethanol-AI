import streamlit as st
import pandas as pd
from plotly import graph_objs as go
from Hydrothermal_Pretreatment import simulate_hydrothermal_degradation

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
    extrativos = st.number_input("Extractives Percentage (0.00 - 100.00) (%)", min_value=0.0, max_value=100.0, format="%.2f")
    cinzas = st.number_input("Ash Percentage (0.00 - 100.00) (%)", min_value=0.0, max_value=100.0, format="%.2f")

# Customizing the Pre-Treatment Parameters column (col2)
with col2:
    st.header("Pre-Treatment Parameters")
    st.write(f"Enter the pre-treatment parameters for {biomassa}.")
    
    # Configuration dictionary for different combinations
    parameter_configs = {
        ("Sugarcane Bagasse", "Acid"): {
            "params": [
                {"name": "Temperature (°C)", "type": "number", "min": 120, "max": 200, "value": 160},
                {"name": "Acid Concentration (%)", "type": "number", "min": 0.5, "max": 5.0, "value": 2.0},
                {"name": "Time (min)", "type": "number", "min": 10, "max": 120, "value": 60},
                {"name": "Acid Type", "type": "selectbox", "options": ["H2SO4", "HCl", "HNO3"]},
                {"name": "Pressure (bar)", "type": "selectbox", "options": ["Atmospheric", "2 bar", "5 bar"]},
                {"name": "Catalyst Present", "type": "checkbox"}
            ]
        },
        ("Sugarcane Bagasse", "Basic"): {
            "params": [
                {"name": "Temperature (°C)", "type": "number", "min": 80, "max": 180, "value": 120},
                {"name": "Base Concentration (%)", "type": "number", "min": 1.0, "max": 10.0, "value": 4.0},
                {"name": "Time (min)", "type": "number", "min": 30, "max": 180, "value": 90},
                {"name": "Base Type", "type": "selectbox", "options": ["NaOH", "KOH", "Ca(OH)2"]},
                {"name": "Mixing Speed", "type": "selectbox", "options": ["Low", "Medium", "High"]},
                {"name": "Oxygen Present", "type": "checkbox"}
            ]
        },
        ("Sugarcane Bagasse", "Organosolv"): {
            "params": [
                {"name": "Temperature (°C)", "type": "number", "min": 150, "max": 220, "value": 180},
                {"name": "Ethanol Concentration (%)", "type": "number", "min": 40, "max": 80, "value": 60},
                {"name": "Time (min)", "type": "number", "min": 30, "max": 150, "value": 75},
                {"name": "Catalyst", "type": "selectbox", "options": ["H2SO4", "HCl", "Formic Acid"]},
                {"name": "Liquid/Solid Ratio", "type": "selectbox", "options": ["5:1", "10:1", "15:1"]},
                {"name": "Acid Added", "type": "checkbox"}
            ]
        },
        ("Sugarcane Bagasse", "Hydrothermal"): {
            "params": [
                {"name": "Temperature (°C)", "type": "selectbox", "options": [180, 195, 210]},
                {"name": "Solid Loading (g/L)", "type": "number", "min": 50, "max": 200, "value": 100},
                {"name": "Time (min)", "type": "slider", "min": 10, "max": 120, "value": 40},
                {"name": "pH", "type": "selectbox", "options": ["Natural", "Acidic", "Basic"]},
                {"name": "Pressure", "type": "selectbox", "options": ["Autogenous", "Controlled"]},
                {"name": "Stirring", "type": "checkbox"}
            ]
        },
        ("Sugarcane Straw", "Hydrothermal"): {
            "params": [
                {"name": "Solid Loading (g/L)", "type": "number", "min": 1.0, "max": 500.0, "value": 100.0},
                {"name": "Temperature (°C)", "type": "selectbox", "options": [180, 195, 210]},
                {"name": "Time (min)", "type": "slider", "min": 1.0, "max": 120.0, "value": 15.0, "step": 0.1}
            ]
        }
    }
    
    # Default configuration for combinations not specifically defined
    default_config = {
        "params": [
            {"name": "Temperature (°C)", "type": "number", "min": 100, "max": 250, "value": 150},
            {"name": "Concentration (%)", "type": "number", "min": 1.0, "max": 10.0, "value": 5.0},
            {"name": "Time (min)", "type": "number", "min": 30, "max": 180, "value": 60},
            {"name": "Reagent Type", "type": "selectbox", "options": ["Type A", "Type B", "Type C"]},
            {"name": "Process Mode", "type": "selectbox", "options": ["Batch", "Continuous", "Semi-batch"]},
            {"name": "Catalyst Present", "type": "checkbox"}
        ]
    }
    
    # Get configuration for current combination
    current_config = parameter_configs.get((biomassa, pretratamento), default_config)
    
    # Store parameters in session state to access later
    if 'pretreatment_params' not in st.session_state:
        st.session_state.pretreatment_params = {}
    
    # Generate UI elements based on configuration
    for i, param in enumerate(current_config["params"]):
        param_key = f"{biomassa}_{pretratamento}_{param['name']}"
        
        if param["type"] == "number":
            value = st.number_input(
                param["name"],
                min_value=param.get("min", 0.0),
                max_value=param.get("max", 1000.0),
                value=param.get("value", 0.0),
                format="%.2f",
                key=param_key
            )
        elif param["type"] == "selectbox":
            value = st.selectbox(
                param["name"],
                options=param["options"],
                key=param_key
            )
        elif param["type"] == "slider":
            value = st.slider(
                param["name"],
                min_value=param.get("min", 0.0),
                max_value=param.get("max", 100.0),
                value=param.get("value", 50.0),
                step=param.get("step", 1.0),
                format="%.1f",
                key=param_key
            )
        elif param["type"] == "checkbox":
            value = st.checkbox(param["name"], key=param_key)
        
        # Store in session state
        st.session_state.pretreatment_params[param["name"]] = value
    
    # Special handling for Hydrothermal (backward compatibility)
    if pretratamento == "Hydrothermal" and biomassa == "Sugarcane Straw":
        solid_loading_hydro = st.session_state.pretreatment_params.get("Solid Loading (g/L)", 100.0)
        temperature_hydro = st.session_state.pretreatment_params.get("Temperature (°C)", 195)
        time_hydro = st.session_state.pretreatment_params.get("Time (min)", 15.0)

# Customizing the Pre-Treatment Results column (col3)

with col3:
    st.header("Pre-Treatment Results")
    st.write(f"Here you can see the results obtained for the {pretratamento} Pretreatment stage of {biomassa}. Change the chart layout to visualize more relationships between the variables.")
    
    # Special handling for Hydrothermal pretreatment
    if pretratamento == "Hydrothermal" and biomassa == "Sugarcane Straw":
        if st.button("Calculate Hydrothermal Degradation", key="hydrothermal_calc"):
            try:
                # Convert percentages to fractions
                cellulose_frac = celulose / 100.0
                hemicellulose_frac = hemicelulose / 100.0
                
                # Run simulation
                results = simulate_hydrothermal_degradation(
                    temperature=temperature_hydro,
                    solid_loading=solid_loading_hydro,
                    cellulose_fraction=cellulose_frac,
                    hemicellulose_fraction=hemicellulose_frac,
                    time_final=time_hydro
                )
                
                # Display results
                st.success("Simulation completed successfully!")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric(
                        label="Cellulose Degradation",
                        value=f"{results['cellulose_degraded_percent']:.1f}%",
                        help="Percentage of cellulose degraded during pretreatment"
                    )
                    st.metric(
                        label="Final Cellulose",
                        value=f"{results['final_cellulose']:.1f} g/L",
                        help="Remaining cellulose concentration"
                    )
                
                with col_b:
                    st.metric(
                        label="Hemicellulose Degradation",
                        value=f"{results['hemicellulose_degraded_percent']:.1f}%",
                        help="Percentage of hemicellulose degraded during pretreatment"
                    )
                    st.metric(
                        label="Final Hemicellulose",
                        value=f"{results['final_hemicellulose']:.1f} g/L",
                        help="Remaining hemicellulose concentration"
                    )
                
                # Create and display plot
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=results["time"],
                    y=results["cellulose"],
                    mode='lines+markers',
                    name='Cellulose',
                    line=dict(color='blue', width=3),
                    marker=dict(size=4)
                ))
                
                fig.add_trace(go.Scatter(
                    x=results["time"],
                    y=results["hemicellulose"],
                    mode='lines+markers',
                    name='Hemicellulose',
                    line=dict(color='green', width=3),
                    marker=dict(size=4)
                ))
                
                fig.update_layout(
                    title=f'Hydrothermal Degradation at {temperature_hydro}°C',
                    xaxis_title='Time (min)',
                    yaxis_title='Concentration (g/L)',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error in simulation: {str(e)}")
                st.info("Please check your input parameters and try again.")
    
    else:
        # For other pretreatment types, keep the original placeholder
        def select_model():
            return 0
    
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
    
    st.button("Calculate Yield", key="hidrolise_resultados")
    
    # Placeholder for enzymatic hydrolysis model
    # TODO: Replace with actual model function import
    def predict_enzymatic_hydrolysis(cellulose_pct, lignin_pct, hemicellulose_pct, ash_pct, 
                                   solid_loading, enzyme_loading, enzyme_type, biomass_type):
        """
        Placeholder function for enzymatic hydrolysis prediction.
        This should be replaced with your actual model function.
        """
        # Simple placeholder calculation - replace with real model
        base_yield = 75.0
        cellulose_factor = cellulose_pct * 0.8
        enzyme_factor = enzyme_loading * 2.0
        solid_factor = max(0, 100 - solid_loading * 0.1)
        
        predicted_yield = min(95.0, base_yield + cellulose_factor * 0.2 + enzyme_factor * 0.1 + solid_factor * 0.05)
        return predicted_yield
    
    if st.session_state.get("hidrolise_resultados"):
        try:
            # Convert enzyme type to numerical value
            enzyme_value = enzyme_types.get(enzyme, 1)
            
            # Convert biomass to numerical value (1 for Bagasse, 2 for Straw)
            biomass_value = 1 if biomassa == "Sugarcane Bagasse" else 2
            
            # Use the placeholder function (replace with actual model import)
            predicted_yield = predict_enzymatic_hydrolysis(
                celulose1, lignina1, hemicelulose1, cinzas1, 
                solid_loading, enzyme_loading, enzyme_value, biomass_value
            )
            
            st.success(f"Yield predicted by the model: {predicted_yield:.2f}%")
            
            # Store result for chart
            rendimento_previsto = predicted_yield
            
        except Exception as e:
            st.error(f"An error occurred while processing: {e}")
    else:
        rendimento_previsto = None

