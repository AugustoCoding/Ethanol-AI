import numpy as np
import streamlit as st
import pandas as pd
from plotly import graph_objs as go
from plotly.subplots import make_subplots
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from Hydrothermal_Pretreatment import simulate_hydrothermal_degradation


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


def calcular_enzima_equilibrio(E_T: float, S: float, E_max: float, k_ad: float) -> tuple[float, float]:
    """Calcula enzima livre e adsorvida em equilíbrio"""
    def equation(E_F):
        E_B = E_T - E_F
        return E_max * k_ad * E_F / (1 + k_ad * E_F) - E_B / S
    
    E_F = fsolve(equation, 0.1)[0]
    E_B = E_T - E_F
    return E_F, E_B


def simulate_enzymatic_hydrolysis(
    solid_loading: float,
    enzyme_loading: float,
    cellulose_percent: float,
    hemicellulose_percent: float,
    reaction_time: float
) -> pd.DataFrame:
    """
    Modelo de hidrólise enzimática - Angarita et al. 2015
    Implementação idêntica ao notebook Hydrolysis_SAGe.ipynb
    """

    if reaction_time <= 0:
        raise ValueError("Reaction time must be greater than zero.")

    if solid_loading <= 0:
        raise ValueError("Solids loading must be greater than zero.")

    # Condições operacionais
    S0 = solid_loading
    Cellulose = cellulose_percent / 100.0
    Hemicellulose = hemicellulose_percent / 100.0
    E_T = enzyme_loading
    alfa = 1.0

    # Parâmetros cinéticos (Angarita et al. 2015)
    k_1r = 0.177              # Taxa de reação r1 (h⁻¹)
    k_2r = 8.81               # Taxa de reação r2 (h⁻¹)
    k_3r = 201.0              # Taxa de reação r3 (h⁻¹)
    k_4r = 16.34              # Taxa de reação r4 (h⁻¹)

    # Constantes de inibição - Reação 1
    k_11G2 = 0.402            # Inibição por celobiose (g/L)
    k_11G = 2.71              # Inibição por glicose (g/L)
    k_11X = 2.15              # Inibição por xilose (g/L)

    # Constantes de inibição - Reação 2
    k_21G2 = 119.6            # Inibição por celobiose (g/L)
    k_21G = 4.69              # Inibição por glicose (g/L)
    k_21X = 0.095             # Inibição por xilose (g/L)

    # Constantes de Michaelis-Menten - Reação 3
    k_3M = 26.6               # Constante de Michaelis (g/L)
    k_31G = 11.06             # Inibição por glicose (g/L)
    k_31X = 1.023             # Inibição por xilose (g/L)

    # Constantes de inibição - Reação 4
    k_41G2 = 16.25            # Inibição por celobiose (g/L)
    k_41G = 4.0               # Inibição por glicose (g/L)
    k_41X = 154.0             # Inibição por xilose (g/L)

    # Parâmetros de adsorção enzimática
    k_ad = 7.16               # Constante de adsorção
    E_max = 8.32/1000         # Capacidade máxima de adsorção (g/L)

    # Calcular enzimas em equilíbrio inicial
    E_F_inicial, E_B_inicial = calcular_enzima_equilibrio(E_T, S0, E_max, k_ad)

    # Condições iniciais
    y0 = [
        S0 * Cellulose,      # Celulose inicial
        0.0,                 # Celobiose inicial
        0.0,                 # Glicose inicial
        S0 * Hemicellulose,  # Hemicelulose inicial
        0.0,                 # Xilose inicial
        S0,                  # Biomassa inicial
        E_B_inicial,         # Enzima adsorvida inicial
        E_F_inicial          # Enzima livre inicial
    ]

    def modelo_hidrolise(t, y):
        """
        Sistema de ODEs para hidrólise enzimática
        
        Variáveis de estado:
        y[0] = C    - Celulose (g/L)
        y[1] = G2   - Celobiose (g/L)
        y[2] = G    - Glicose (g/L)
        y[3] = H    - Hemicelulose (g/L)
        y[4] = X    - Xilose (g/L)
        y[5] = S    - Biomassa total (g/L)
        y[6] = E_B  - Enzima adsorvida (g/L)
        y[7] = E_F  - Enzima livre (g/L)
        """
        
        C, G2, G, H, X, S, E_B, E_F = y
        
        # Evitar divisão por zero
        S = max(S, 1e-6)
        
        # Fator de resistência
        R_S = alfa * S / S0
        
        # Concentrações de enzima específica
        Ebc = E_B * C / S     # Enzima específica para celulose
        Ebh = E_B * H / S     # Enzima específica para hemicelulose
        
        # Taxas de reação
        r1 = k_1r * Ebc * R_S * C / (1 + G2/k_11G2 + G/k_11G + X/k_11X)
        r2 = k_2r * Ebc * R_S * C / (1 + G2/k_21G2 + G/k_21G + X/k_21X)
        r3 = k_3r * E_F * G2 / (k_3M * (1 + G/k_31G + X/k_31X) + G2)
        r4 = k_4r * Ebh * R_S * H / (1 + G2/k_41G2 + G/k_41G + X/k_41X)
        
        # Equações diferenciais
        dCdt = -r1 - r2                 # Celulose
        dG2dt = 1.056 * r1 - r3         # Celobiose
        dGdt = 1.111 * r2 + 1.053 * r3  # Glicose
        dHdt = -r4                      # Hemicelulose
        dXdt = 1.136 * r4               # Xilose
        dSdt = -r1 - r2 - r4            # Biomassa
        
        # Enzimas (equações algébricas - derivadas = 0)
        dE_Bdt = 0
        dE_Fdt = 0
        
        return [dCdt, dG2dt, dGdt, dHdt, dXdt, dSdt, dE_Bdt, dE_Fdt]

    # Configuração da simulação - sempre simular até 96h
    t_final_simulation = 96.0
    t_pontos = np.linspace(0, t_final_simulation, int(t_final_simulation) + 1)

    # Resolver sistema de ODEs
    sol = solve_ivp(
        modelo_hidrolise, 
        [0, t_final_simulation], 
        y0, 
        t_eval=t_pontos, 
        method='LSODA', 
        rtol=1e-8
    )

    if not sol.success:
        raise RuntimeError(f"Hydrolysis simulation failed: {sol.message}")

    # Extrair resultados
    return pd.DataFrame(
        {
            "Time (h)": sol.t,
            "Glucose": sol.y[2],
            "Xylose": sol.y[4],
            "Cellobiose": sol.y[1],
        }
    )

# Configurando o layout para modo "wide"
st.set_page_config(layout="wide")

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
    "<h1 style='font-size:50px;'>Pre-Treatment</h1>",
    unsafe_allow_html=True
)
st.write("In this section, introduce the relevant data for calculating the yield of the Pre-Treatment.")

# Criando colunas "Parâmetros" e "Resultados"
col1, spacer, col3 = st.columns([6, 2, 10])

# Customizing the unified Parameters column (col1)
with col1:
    st.header("Parameters")
    
    # Initial Data section
    biomassa = st.selectbox("Select a biomass type", ['Sugarcane Straw', 'Sugarcane Bagasse'], index=0, help="Note: Only Sugarcane Straw with Hydrothermal pretreatment is currently available")
    if biomassa == 'Sugarcane Bagasse':
        st.warning("⚠️ Sugarcane Bagasse models are not yet available for Pre-Treatment")
    pretratamento = st.selectbox("Select a Pre-Treatment type", ['Hydrothermal', 'Organosolv'], index=0, help="Note: Organosolv model is under development")
    if pretratamento == 'Organosolv':
        st.warning("⚠️ Organosolv pretreatment model is not yet available")
    celulose = st.number_input("Cellulose Percentage (0.00 - 100.00) (%)", min_value=0.0, max_value=100.0, format="%.2f")
    lignina = st.number_input("Lignin Percentage (0.00 - 100.00) (%)", min_value=0.0, max_value=100.0, format="%.2f")
    hemicelulose = st.number_input("Hemicellulose Percentage (0.00 - 100.00) (%)", min_value=0.0, max_value=100.0, format="%.2f")
    extrativos = st.number_input("Extractives Percentage (0.00 - 100.00) (%)", min_value=0.0, max_value=100.0, format="%.2f")
    cinzas = st.number_input("Ash Percentage (0.00 - 100.00) (%)", min_value=0.0, max_value=100.0, format="%.2f")

    # Universal pretreatment parameters
    solid_loading_hydro = st.number_input(
        "Solid Loading (g/L)",
        min_value=1.0,
        max_value=500.0,
        value=100.0,
        format="%.2f"
    )
    temperature_hydro = st.selectbox(
        "Temperature (°C)",
        options=[180, 195, 210],
        index=1
    )
    time_hydro = st.number_input(
        "Time (min)",
        min_value=1.0,
        max_value=120.0,
        value=15.0,
        format="%.1f"
    )

# Customizing the Pre-Treatment Results column (col3)

with col3:
    st.header("Results")
    st.write(f"Here you can see the results obtained for the {pretratamento} Pretreatment stage of {biomassa}. Change the chart layout to visualize more relationships between the variables.")
    
    # Check if model is available for selected combination
    model_available = (pretratamento == "Hydrothermal" and biomassa == "Sugarcane Straw")
    
    # Special handling for Hydrothermal pretreatment
    if pretratamento == "Hydrothermal" and biomassa == "Sugarcane Straw":
        if st.button("Calculate Hydrothermal Degradation", key="hydrothermal_calc", disabled=not model_available):
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
        # For other pretreatment types, show info message
        if biomassa == 'Sugarcane Bagasse':
            st.info("ℹ️ Models for Sugarcane Bagasse are under development.")
        elif pretratamento == "Organosolv":
            st.info(f"ℹ️ Organosolv pretreatment model for {biomassa} is under development.")
    
st.markdown("<hr style='border: 1px solid #ccc;' />", unsafe_allow_html=True)

# Next Stage: Enzymatic Hydrolysis

st.markdown(
    "<h1 style='font-size:50px;'>Enzymatic Hydrolysis</h1>",
    unsafe_allow_html=True
)
st.write("In this section, introduce the relevant data for calculating the yield of Enzymatic Hydrolysis.")
# Creating "Parameters" and "Results" columns
col4, spacer4, col6 = st.columns([6, 1, 10])

# Customizing the unified Parameters column (col4)
with col4:
    st.header("Parameters")
    
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
    st.header("Results")
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
                st.dataframe(profile_df.round(3))
            except Exception as exc:
                st.error(f"Error while running hydrolysis simulation: {exc}")
    