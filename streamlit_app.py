import numpy as np
import streamlit as st
import pandas as pd
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from Hydrothermal_Pretreatment import simulate_hydrothermal_degradation

# Configurando o layout para modo "wide" - DEVE SER O PRIMEIRO COMANDO STREAMLIT
st.set_page_config(layout="wide")


page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(rgba(255, 255, 255, 0), rgba(255, 255, 255, 0)),
                url("https://images.unsplash.com/photo-1675251171768-5d49233cc410?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&q=80&w=764");
    background-size: cover;
    background-attachment: fixed;
}

[data-testid="stHeader"] {
    background-color: rgba(0, 0, 0, 0);
}

/* Garantir que o conteudo aparece sobre o fundo */
[data-testid="stToolbar"] {
    z-index: 1;
}

/* Bordas arredondadas nos gráficos Plotly */
.js-plotly-plot .plotly {
    border-radius: 15px;
    overflow: hidden;
}

[data-testid="stPlotlyChart"] > div {
    border-radius: 15px;
    overflow: hidden;
}
</style>
"""

# ============================================================================
# CARREGAMENTO DO MODELO ANN E SCALERS
# ============================================================================

@st.cache_resource
def load_ann_model_and_scalers():
    """
    Carrega modelos ANN (hidrólise e pré-tratamento) e configura scalers com dados de treinamento.
    Usa cache para carregar apenas uma vez.
    """
    import os
    try:
        # ===== HIDRÓLISE =====
        model_path = "champion_model.h5" if os.path.exists("champion_model.h5") else os.path.join("BEPE FAPESP", "Genetic ANNs", "Straw", "Hydrolysis", "champion_ann_strategy1_32_32_16.h5")
        data_path = "training_data.csv" if os.path.exists("training_data.csv") else os.path.join("BEPE FAPESP", "Enzymatic Hydrolysis", "Data Generation", "synthetic_hydrolysis_data_LHS.csv")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Hydrolysis model not found at '{model_path}'.")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Hydrolysis data not found at '{data_path}'.")
         
        # Carregar modelo de hidrólise
        champion_model = tf.keras.models.load_model(model_path, compile=False)
        champion_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Carregar dados de treinamento para ajustar scalers
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()
        
        INPUT_FEATURES = ['Cellulose', 'Hemicellulose', 'Lignin', 'Solids Loading [g/L]', 'Enzyme Loading [g/L]', 'Time [h]']
        OUTPUT_FEATURES = ['Glucose Concentration [g/L]', 'Xylose Concentration [g/L]', 'Cellobiose Concentration [g/L]']
        
        X_data = df[INPUT_FEATURES].values
        y_data = df[OUTPUT_FEATURES].values
        
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        scaler_X.fit(X_data)
        scaler_y.fit(y_data)
        
        # ===== PRÉ-TRATAMENTO =====
        pretreat_model_path = os.path.join("BEPE FAPESP", "Genetic ANNs", "Straw", "Pretreatment", "champion_ann_pretreatment_strategy1_32_64_16.h5")
        pretreat_data_path = os.path.join("BEPE FAPESP", "Pretreatment", "Data Generation", "synthetic_pretreatment_data_LHS.csv")
        
        if not os.path.exists(pretreat_model_path):
            raise FileNotFoundError(f"Pretreatment model not found at '{pretreat_model_path}'.")
        if not os.path.exists(pretreat_data_path):
            raise FileNotFoundError(f"Pretreatment data not found at '{pretreat_data_path}'.")
        
        # Carregar modelo de pré-tratamento
        pretreat_model = tf.keras.models.load_model(pretreat_model_path, compile=False)
        pretreat_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Carregar dados de treinamento para ajustar scalers
        df_pretreat = pd.read_csv(pretreat_data_path)
        df_pretreat.columns = df_pretreat.columns.str.strip()
        
        PRETREAT_INPUT_FEATURES = ['Temperature [°C]', 'Cellulose Fraction', 'Hemicellulose Fraction', 'Lignin Fraction', 'Solids Loading [g/L]', 'Time [min]']
        PRETREAT_OUTPUT_FEATURES = ['Cellulose Remaining [g/L]', 'Hemicellulose Remaining [g/L]']
        
        X_pretreat_data = df_pretreat[PRETREAT_INPUT_FEATURES].values
        y_pretreat_data = df_pretreat[PRETREAT_OUTPUT_FEATURES].values
        
        scaler_X_pretreat = MinMaxScaler(feature_range=(0, 1))
        scaler_y_pretreat = MinMaxScaler(feature_range=(0, 1))
        scaler_X_pretreat.fit(X_pretreat_data)
        scaler_y_pretreat.fit(y_pretreat_data)
        
        return champion_model, scaler_X, scaler_y, pretreat_model, scaler_X_pretreat, scaler_y_pretreat, True, None
        
    except Exception as e:
        return None, None, None, None, None, None, False, str(e)

# Carregar modelos ao iniciar app
champion_model, scaler_X, scaler_y, pretreat_model, scaler_X_pretreat, scaler_y_pretreat, model_loaded, load_error = load_ann_model_and_scalers()


def apply_physical_constraints(predictions: np.ndarray, time_inputs: np.ndarray) -> np.ndarray:
    """
    Aplica constraints físicos às predições da ANN.
    Regra: Se t=0h, então concentrações = [0, 0, 0]
    """
    constrained_predictions = predictions.copy()
    
    # Identificar amostras t=0h (tolerância para comparações float)
    t0_mask = np.abs(time_inputs) < 1e-6
    
    # Aplicar constraint: t=0h → concentrações = [0, 0, 0]
    if np.any(t0_mask):
        constrained_predictions[t0_mask] = 0.0
    
    # Garantir valores não-negativos para todas as amostras
    constrained_predictions = np.maximum(constrained_predictions, 0.0)
    
    return constrained_predictions


def apply_physical_constraints_pretreatment(
    predictions: np.ndarray,
    time_inputs: np.ndarray,
    cellulose_frac: float,
    hemi_frac: float,
    solids: float
) -> np.ndarray:
    """
    Aplica constraints físicos às predições da ANN de pré-tratamento.
    Regras: 
    - Se t=0min, então [Cellulose_Remaining, Hemicellulose_Remaining] = [C0, H0]
    - Concentrações não podem exceder valores iniciais (C0, H0)
    """
    constrained_predictions = predictions.copy()
    
    # Calcular concentrações iniciais
    C0 = solids * cellulose_frac
    H0 = solids * hemi_frac
    
    # Identificar amostras t=0min
    t0_mask = np.abs(time_inputs) < 1e-6
    
    # Aplicar constraint: t=0min → [C0, H0]
    if np.any(t0_mask):
        constrained_predictions[t0_mask, 0] = C0
        constrained_predictions[t0_mask, 1] = H0
    
    # Garantir que concentrações não excedem valores iniciais
    constrained_predictions[:, 0] = np.minimum(constrained_predictions[:, 0], C0)
    constrained_predictions[:, 1] = np.minimum(constrained_predictions[:, 1], H0)
    
    # Garantir valores não-negativos
    constrained_predictions = np.maximum(constrained_predictions, 0.0)
    
    return constrained_predictions


def simulate_pretreatment_ann(
    temperature: float,
    solid_loading: float,
    cellulose_percent: float,
    hemicellulose_percent: float,
    lignin_percent: float,
    time_final: float
) -> dict:
    """
    Modelo de pré-tratamento hidrotérmico usando Rede Neural ANN.
    Baseado em champion_ann_pretreatment_strategy1_32_64_16.h5
    """
    if time_final <= 0:
        raise ValueError("Time must be greater than zero.")
    
    if solid_loading <= 0:
        raise ValueError("Solids loading must be greater than zero.")
    
    if not model_loaded:
        raise RuntimeError(f"ANN model could not be loaded: {load_error}")
    
    # Converter percentagens para frações (0-1)
    cellulose_frac = cellulose_percent / 100.0
    hemicellulose_frac = hemicellulose_percent / 100.0
    lignin_frac = lignin_percent / 100.0
    
    # Gerar array de tempos de 0 até 60 min (sempre completo)
    time_array = np.linspace(0, 60.0, 61)
    
    # Preparar features de entrada para cada ponto de tempo
    # Formato: [Temperature, Cellulose Fraction, Hemicellulose Fraction, Lignin Fraction, Solids Loading, Time]
    X = np.array([
        [temperature, cellulose_frac, hemicellulose_frac, lignin_frac, solid_loading, t]
        for t in time_array
    ])
    
    # Normalizar entradas
    X_scaled = scaler_X_pretreat.transform(X)
    
    # Fazer predições com ANN
    y_pred_scaled = pretreat_model.predict(X_scaled, verbose=0)
    
    # Desnormalizar predições
    y_pred = scaler_y_pretreat.inverse_transform(y_pred_scaled)
    
    # Aplicar constraints físicos
    y_pred_constrained = apply_physical_constraints_pretreatment(
        y_pred, time_array, cellulose_frac, hemicellulose_frac, solid_loading
    )
    
    # Calcular concentrações iniciais e degradação
    C0 = solid_loading * cellulose_frac
    H0 = solid_loading * hemicellulose_frac
    
    final_cellulose = y_pred_constrained[-1, 0]
    final_hemicellulose = y_pred_constrained[-1, 1]
    
    cellulose_degraded_percent = ((C0 - final_cellulose) / C0 * 100) if C0 > 0 else 0
    hemicellulose_degraded_percent = ((H0 - final_hemicellulose) / H0 * 100) if H0 > 0 else 0
    
    return {
        'time': time_array,
        'cellulose': y_pred_constrained[:, 0],
        'hemicellulose': y_pred_constrained[:, 1],
        'final_cellulose': final_cellulose,
        'final_hemicellulose': final_hemicellulose,
        'cellulose_degraded_percent': cellulose_degraded_percent,
        'hemicellulose_degraded_percent': hemicellulose_degraded_percent,
        'time_final': time_final
    }


def simulate_enzymatic_hydrolysis(
    solid_loading: float,
    enzyme_loading: float,
    cellulose_percent: float,
    hemicellulose_percent: float,
    lignin_percent: float,
    reaction_time: float
) -> pd.DataFrame:
    """
    Modelo de hidrólise enzimática usando Rede Neural ANN.
    Baseado em champion_ann_strategy1_32_32_16.h5
    """
    
    if reaction_time <= 0:
        raise ValueError("Reaction time must be greater than zero.")
    
    if solid_loading <= 0:
        raise ValueError("Solids loading must be greater than zero.")
    
    if not model_loaded:
        raise RuntimeError(f"ANN model could not be loaded: {load_error}")
    
    # Converter percentagens para frações (0-1)
    cellulose_frac = cellulose_percent / 100.0
    hemicellulose_frac = hemicellulose_percent / 100.0
    lignin_frac = lignin_percent / 100.0
    
    # Gerar array de tempos de 0 a 96h
    t_final_simulation = 96.0
    time_array = np.linspace(0, t_final_simulation, int(t_final_simulation) + 1)
    
    # Preparar features de entrada para cada ponto de tempo
    # Formato: [Cellulose, Hemicellulose, Lignin, Solids Loading, Enzyme Loading, Time]
    X = np.array([
        [cellulose_frac, hemicellulose_frac, lignin_frac, solid_loading, enzyme_loading, t]
        for t in time_array
    ])
    
    # Normalizar entradas
    X_scaled = scaler_X.transform(X)
    
    # Fazer predições com ANN
    y_pred_scaled = champion_model.predict(X_scaled, verbose=0)
    
    # Desnormalizar predições
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    # Aplicar constraints físicos (t=0 → concentrações=0)
    y_pred_constrained = apply_physical_constraints(y_pred, time_array)
    
    # Extrair resultados
    return pd.DataFrame(
        {
            "Time (h)": time_array,
            "Glucose": y_pred_constrained[:, 0],
            "Xylose": y_pred_constrained[:, 1],
            "Cellobiose": y_pred_constrained[:, 2],
        }
    )

# Aplicar estilos CSS
st.markdown(page_bg_img, unsafe_allow_html=True)

# Título do app
st.markdown("""
<style>
/* Main title */
.main-title {
    background: linear-gradient(135deg, #1e3c72, #2c3e50, #27ae60, #3498db);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    font-size: 4em !important;
    font-weight: 800 !important;
    margin-bottom: 30px !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}
</style>

<h1 class='main-title'>Ethanol AI</h1>
""", unsafe_allow_html=True)

# Informação principal
st.write("Ethanol AI is a tool created within a research program called scientific initiation by researchers from UFSCar and DTU with funding from FAPESP. It is particularly useful for studying the behavior of different second-generation ethanol production processes when subjected to various operating conditions. This software implements hybrid machine learning models, previously trained using knowledge generated from previous research works at UFSCar and abroad, to predict the outcomes. Here, you can test different combinations of initial conditions, essentially finding the maximum possible yield for each situation.")
st.markdown("<hr style='border: 1px solid #ccc;' />", unsafe_allow_html=True)

# Etapa de Pré-Tratamento
st.markdown(
    "<h1 style='font-size:50px;'>♨️Pretreatment</h1>",
    unsafe_allow_html=True
)
st.write("In this section, introduce the relevant data for calculating the yield of the Pre-Treatment.")

# Criando colunas "Parâmetros" e "Resultados"
col1, spacer, col3 = st.columns([6, 1, 10])

# Customizing the unified Parameters column (col1)
with col1:
    st.header("📊Parameters")
    
    # Initial Data section
    biomassa = st.selectbox("Select a biomass type", ['Sugarcane Straw', 'Sugarcane Bagasse'], index=0, help="Note: Only Sugarcane Straw with Hydrothermal pretreatment is currently available")
    if biomassa == 'Sugarcane Bagasse':
        st.warning("⚠️ Sugarcane Bagasse models are not yet available for Pre-Treatment")
    pretratamento = st.selectbox("Select a Pre-Treatment type", ['Hydrothermal', 'Organosolv'], index=0, help="Note: Organosolv model is under development")
    if pretratamento == 'Organosolv':
        st.warning("⚠️ Organosolv pretreatment model is not yet available")
    celulose = st.number_input("Cellulose Percentage (0.00 - 100.00) (%)", min_value=0.0, max_value=100.0, value=40.0, format="%.2f")
    hemicelulose = st.number_input("Hemicellulose Percentage (0.00 - 100.00) (%)", min_value=0.0, max_value=100.0, value=30.0, format="%.2f")
    lignina = st.number_input("Lignin Percentage (0.00 - 100.00) (%)", min_value=0.0, max_value=100.0, value=20.0, format="%.2f")

    # Universal pretreatment parameters
    solid_loading_hydro = st.number_input(
        "Solid Loading (g/L)",
        min_value=1.0,
        max_value=500.0,
        value=100.0,
        format="%.2f"
    )
    temperature_hydro = st.number_input(
        "Temperature (°C)",
        min_value=180.0,
        max_value=210.0,
        value=195.0,
        format="%.1f",
        help="Temperature range: 180-210°C (ANN will extrapolate if needed)"
    )
    time_hydro = st.number_input(
        "Time (min)",
        min_value=1.0,
        max_value=60.0,
        value=15.0,
        format="%.1f",
        help="Maximum simulation time: 60 minutes"
    )

# Customizing the Pre-Treatment Results column (col3)

with col3:
    st.header("🎯Results")
    st.write(f"Here you can see the results obtained for the {pretratamento} Pretreatment stage of {biomassa}. Change the chart layout to visualize more relationships between the variables.")
    
    # Check if model is available for selected combination
    model_available = (pretratamento == "Hydrothermal" and biomassa == "Sugarcane Straw")
    
    # Special handling for Hydrothermal pretreatment
    if pretratamento == "Hydrothermal" and biomassa == "Sugarcane Straw":
        if st.button("Calculate Hydrothermal Degradation", key="hydrothermal_calc", disabled=not model_available, use_container_width=True):
            try:
                # Run simulation with ANN
                results = simulate_pretreatment_ann(
                    temperature=temperature_hydro,
                    solid_loading=solid_loading_hydro,
                    cellulose_percent=celulose,
                    hemicellulose_percent=hemicelulose,
                    lignin_percent=lignina,
                    time_final=time_hydro
                )
                
                # Display results
                st.success("Simulation completed successfully!")
                
                # Encontrar índice correspondente ao tempo escolhido
                time_idx = int(round(time_hydro))
                cellulose_at_time = results['cellulose'][time_idx]
                hemicellulose_at_time = results['hemicellulose'][time_idx]
                
                # Calcular concentrações iniciais
                C0 = solid_loading_hydro * (celulose / 100.0)
                H0 = solid_loading_hydro * (hemicelulose / 100.0)
                
                # Calcular degradação no tempo escolhido
                cellulose_degraded_at_time = ((C0 - cellulose_at_time) / C0 * 100) if C0 > 0 else 0
                hemicellulose_degraded_at_time = ((H0 - hemicellulose_at_time) / H0 * 100) if H0 > 0 else 0
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric(
                        label=f"Cellulose Degradation at {time_hydro:.1f} min",
                        value=f"{cellulose_degraded_at_time:.1f}%",
                        help="Percentage of cellulose degraded at selected time"
                    )
                    st.metric(
                        label=f"Cellulose at {time_hydro:.1f} min",
                        value=f"{cellulose_at_time:.1f} g/L",
                        help="Remaining cellulose concentration at selected time"
                    )
                
                with col_b:
                    st.metric(
                        label=f"Hemicellulose Degradation at {time_hydro:.1f} min",
                        value=f"{hemicellulose_degraded_at_time:.1f}%",
                        help="Percentage of hemicellulose degraded at selected time"
                    )
                    st.metric(
                        label=f"Hemicellulose at {time_hydro:.1f} min",
                        value=f"{hemicellulose_at_time:.1f} g/L",
                        help="Remaining hemicellulose concentration at selected time"
                    )
                
                # Create and display plot
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=results["time"],
                    y=results["cellulose"],
                    mode='lines',
                    name='Cellulose',
                    line=dict(color='blue', width=3)
                ))
                
                fig.add_trace(go.Scatter(
                    x=results["time"],
                    y=results["hemicellulose"],
                    mode='lines',
                    name='Hemicellulose',
                    line=dict(color='green', width=3)
                ))
                
                # Adicionar linha vertical no tempo escolhido pelo usuário
                fig.add_vline(
                    x=time_hydro,
                    line_dash="dash",
                    line_color="red",
                    line_width=2,
                    annotation_text=f"Selected Time: {time_hydro:.1f} min",
                    annotation_position="top"
                )
                
                fig.update_layout(
                    title={
                        'text': f'Cellulose and Hemicellulose Concentration at {temperature_hydro}°C',
                        'x': 0.5,
                        'xanchor': 'center'
                    },
                    xaxis_title='Time (min)',
                    yaxis_title='Concentration (g/L)',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Criar tabela com os dados
                with st.expander("📊 View Data Table"):
                    pretreat_df = pd.DataFrame({
                        'Time (min)': results['time'],
                        'Cellulose (g/L)': results['cellulose'],
                        'Hemicellulose (g/L)': results['hemicellulose']
                    })
                    st.dataframe(pretreat_df.round(3))
                
            except Exception as e:
                st.error(f"Error in simulation: {str(e)}")
                st.info("Please check your input parameters and try again.")
    
    else:
        # For other pretreatment types, show info message
        if biomassa == 'Sugarcane Bagasse':
            st.info("ℹ️ Models for Sugarcane Bagasse are under development.")
        elif pretratamento == "Organosolv":
            st.info(f"ℹ️ Organosolv pretreatment model for {biomassa} is under development.")
    
st.markdown("<hr style='border: 1px solid #ccc;' />", unsafe_allow_html=True)

# Next Stage: Enzymatic Hydrolysis

st.markdown(
    "<h1 style='font-size:50px;'>⚗️Enzymatic Hydrolysis</h1>",
    unsafe_allow_html=True
)
st.write("In this section, introduce the relevant data for calculating the yield of Enzymatic Hydrolysis.")
# Creating "Parameters" and "Results" columns
col4, spacer4, col6 = st.columns([6, 1, 10])

# Customizing the unified Parameters column (col4)
with col4:
    st.header("📊Parameters")
    
    biomassa_hydrolysis = st.selectbox("Select a biomass type", ['Sugarcane Straw', 'Sugarcane Bagasse'], index=0, key="biomassa_hydrolysis", disabled=False, help="Note: Only Sugarcane Straw model is currently available")
    if biomassa_hydrolysis == 'Sugarcane Bagasse':
        st.warning("⚠️ Sugarcane Bagasse model is not yet available for Enzymatic Hydrolysis")
    enzyme = st.selectbox("Enzyme", ['Saccharomyces cerevisiae'])
    celulose1 = st.number_input(
        "Cellulose Percentage",
        min_value=45.0,
        max_value=65.0,
        value=55.0,
        format="%.2f",
        placeholder="45.00 – 65.00"
    )
    lignina1 = st.number_input("Lignin Percentage", min_value=0.0, max_value=100.0, value=25.0, format="%.2f")
    hemicelulose1 = st.number_input(
        "Hemicellulose Percentage",
        min_value=5.0,
        max_value=15.0,
        value=8.0,
        format="%.2f",
        placeholder="05.00 – 15.00"
    )
    
    # Simplified parameters for all conditions
    solid_loading = st.number_input(
        "Initial Solids Loading (g/L)",
        min_value=50.0,
        max_value=250.0,
        value=175.0,
        format="%.2f",
        placeholder="50.00 – 250.00"
    )
    enzyme_loading = st.number_input(
        "Initial Enzyme Loading (g/L)",
        min_value=0.05,
        max_value=1.2,
        value=0.5,
        format="%.2f",
        placeholder="0.05 – 1.20"
    )
    reaction_time = st.number_input(
        "Reaction Time (h)",
        min_value=0.0,
        max_value=96.0,
        value=60.0,
        format="%.2f",
        placeholder="0.00 – 96.00"
    )

# Alteration 3: Mapping selection options to numerical values before using them (no longer needed for simplified version)
enzyme_types = {"Saccharomyces cerevisiae": 1}

# Customizing the Enzymatic Hydrolysis Results column (col6)
with col6:
    st.header("🎯Results")
    st.write(f"Here you can see the results obtained for the Enzymatic Hydrolysis stage of {biomassa_hydrolysis}. Change the chart layout to visualize more relationships between the variables.")
    
    # Check if model is available for selected biomass
    model_available = biomassa_hydrolysis == 'Sugarcane Straw'
    
    if st.button("Simulate Hydrolysis Profile", key="run_hydrolysis_profile", use_container_width=True, disabled=not model_available):
        if reaction_time <= 0:
            st.warning("Please set a reaction time greater than zero to generate the profile.")
        else:
            try:
                profile_df = simulate_enzymatic_hydrolysis(
                    solid_loading=solid_loading,
                    enzyme_loading=enzyme_loading,
                    cellulose_percent=celulose1,
                    hemicellulose_percent=hemicelulose1,
                    lignin_percent=lignina1,
                    reaction_time=reaction_time
                )

                glucose_final = profile_df["Glucose"].iloc[-1]
                xylose_final = profile_df["Xylose"].iloc[-1]
                cellobiose_final = profile_df["Cellobiose"].iloc[-1]
                
                # Calcular rendimento de glicose
                # Rendimento teórico: 1.111 g glicose por g de celulose (conversão estequiométrica)
                cellulose_initial = solid_loading * (celulose1 / 100.0)
                glucose_theoretical = cellulose_initial * 1.111
                glucose_yield_percent = (glucose_final / glucose_theoretical) * 100 if glucose_theoretical > 0 else 0
                
                # Exibir métricas em três colunas
                col_g, col_x, col_c = st.columns(3)
                with col_g:
                    st.metric("Glucose Produced", f"{glucose_final:.2f} g/L")
                    st.metric("Xylose Produced", f"{xylose_final:.2f} g/L")
                    
                with col_x:
                    st.metric("Theoretical Glucose", f"{glucose_theoretical:.2f} g/L", help="Maximum theoretical glucose from complete cellulose hydrolysis")
                    st.metric("Cellobiose Produced", f"{cellobiose_final:.2f} g/L")
                with col_c:
                    st.metric("Glucose Yield", f"{glucose_yield_percent:.1f}%", help="Percentage of theoretical maximum glucose production")

                fig = make_subplots(specs=[[{"secondary_y": True}]])

                fig.add_trace(
                    go.Scatter(
                        x=profile_df["Time (h)"],
                        y=profile_df["Glucose"],
                        mode="lines",
                        line=dict(color="#1f77b4", width=3),
                        name="Glucose"
                    ),
                    secondary_y=False
                )

                fig.add_trace(
                    go.Scatter(
                        x=profile_df["Time (h)"],
                        y=profile_df["Xylose"],
                        mode="lines",
                        line=dict(color="#2ca02c", width=3),
                        name="Xylose"
                    ),
                    secondary_y=True
                )

                fig.add_trace(
                    go.Scatter(
                        x=profile_df["Time (h)"],
                        y=profile_df["Cellobiose"],
                        mode="lines",
                        line=dict(color="#ff7f0e", width=3),
                        name="Cellobiose"
                    ),
                    secondary_y=True
                )

                # Adicionar linha vertical no tempo final selecionado
                fig.add_vline(
                    x=reaction_time,
                    line_dash="dash",
                    line_color="red",
                    line_width=2,
                    annotation_text=f"Final Time: {reaction_time:.1f}h",
                    annotation_position="top"
                )

                fig.update_layout(
                    title={
                        'text': "Enzymatic Hydrolysis Concentration Profiles",
                        'x': 0.5,
                        'xanchor': 'center'
                    },
                    hovermode="x unified"
                )
                fig.update_xaxes(title_text="Time (h)")
                fig.update_yaxes(title_text="Glucose (g/L)", secondary_y=False)
                fig.update_yaxes(title_text="Xylose / Cellobiose (g/L)", secondary_y=True, range=[0, 20])

                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("📊 View Data Table"):
                    st.dataframe(profile_df.round(3))
            except Exception as exc:
                st.error(f"Error while running hydrolysis simulation: {exc}")