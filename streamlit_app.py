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
    background: linear-gradient(rgba(255, 255, 255, 0.25), rgba(255, 255, 255, 0.25)),
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
</style>
"""


def calculate_enzyme_equilibrium(E_total: float, solids: float, E_max: float, k_ad: float) -> tuple[float, float]:
    """Compute free and adsorbed enzyme concentrations at equilibrium."""

    solids = max(solids, 1e-6)

    def equation(E_free: float) -> float:
        E_bound = max(E_total - E_free, 0.0)
        return (E_max * k_ad * E_free) / (1 + k_ad * E_free) - (E_bound / solids)

    try:
        initial_guess = min(E_total, 0.1)
        solution = fsolve(equation, x0=initial_guess, xtol=1e-10)
        E_free = float(np.clip(solution[0], 0.0, E_total))
    except Exception:
        # Fall back to assuming small free enzyme if convergence fails
        E_free = min(E_total, 0.01)

    E_bound = max(E_total - E_free, 0.0)
    return E_free, E_bound


def simulate_enzymatic_hydrolysis(
    solid_loading: float,
    enzyme_loading: float,
    cellulose_percent: float,
    hemicellulose_percent: float,
    reaction_time: float
) -> pd.DataFrame:
    """Run the Angarita et al. 2015 enzymatic hydrolysis model for UI inputs."""

    if reaction_time <= 0:
        raise ValueError("Reaction time must be greater than zero.")

    if solid_loading <= 0:
        raise ValueError("Solids loading must be greater than zero.")

    S0 = solid_loading
    Cellulose = max(cellulose_percent / 100.0, 0.0)
    Hemicellulose = max(hemicellulose_percent / 100.0, 0.0)
    E_T = max(enzyme_loading, 1e-6)

    if Cellulose + Hemicellulose <= 0:
        raise ValueError("Cellulose and hemicellulose fractions must be greater than zero.")

    alfa = 1.0

    # Kinetic parameters (Angarita et al. 2015)
    k_1r = 0.177
    k_2r = 8.81
    k_3r = 201.0
    k_4r = 16.34

    k_11G2 = 0.402
    k_11G = 2.71
    k_11X = 2.15

    k_21G2 = 119.6
    k_21G = 4.69
    k_21X = 0.095

    k_3M = 26.6
    k_31G = 11.06
    k_31X = 1.023

    k_41G2 = 16.25
    k_41G = 4.0
    k_41X = 154.0

    k_ad = 7.16
    E_max = 8.32 / 1000

    E_F_init, E_B_init = calculate_enzyme_equilibrium(E_T, S0, E_max, k_ad)

    y0 = [
        S0 * Cellulose,
        0.0,
        0.0,
        S0 * Hemicellulose,
        0.0,
        S0,
        E_B_init,
        E_F_init,
    ]

    def modelo_hidrolise(t: float, y: np.ndarray) -> list[float]:
        C, G2, G, H, X, S, E_B, E_F = y

        S = max(S, 1e-6)
        R_S = alfa * S / S0

        if S <= 0:
            Ebc = 0.0
            Ebh = 0.0
        else:
            Ebc = E_B * (C / S)
            Ebh = E_B * (H / S)

        r1 = k_1r * Ebc * R_S * C / (1 + G2 / k_11G2 + G / k_11G + X / k_11X)
        r2 = k_2r * Ebc * R_S * C / (1 + G2 / k_21G2 + G / k_21G + X / k_21X)
        r3 = k_3r * E_F * G2 / (k_3M * (1 + G / k_31G + X / k_31X) + G2)
        r4 = k_4r * Ebh * R_S * H / (1 + G2 / k_41G2 + G / k_41G + X / k_41X)

        dCdt = -r1 - r2
        dG2dt = 1.056 * r1 - r3
        dGdt = 1.111 * r2 + 1.053 * r3
        dHdt = -r4
        dXdt = 1.136 * r4
        dSdt = -r1 - r2 - r4

        return [dCdt, dG2dt, dGdt, dHdt, dXdt, dSdt, 0.0, 0.0]

    num_points = max(100, int(reaction_time * 2) + 1)
    t_eval = np.linspace(0.0, reaction_time, num_points)

    solution = solve_ivp(
        modelo_hidrolise,
        t_span=(0.0, reaction_time),
        y0=y0,
        t_eval=t_eval,
        method="LSODA",
        rtol=1e-6,
        atol=1e-8,
    )

    if not solution.success:
        raise RuntimeError(f"Hydrolysis simulation failed: {solution.message}")

    return pd.DataFrame(
        {
            "Time (h)": solution.t,
            "Glucose": solution.y[2],
            "Xylose": solution.y[4],
            "Cellobiose": solution.y[1],
        }
    )

# Configurando o layout para modo "wide"
st.set_page_config(layout="wide")

st.markdown(page_bg_img, unsafe_allow_html=True)

# Título do app
st.title('⚗️Ethanol AI (Beta)')

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
                {"name": "Temperature (°C)", "type": "number", "min": 120.0, "max": 200.0, "value": 160.0},
                {"name": "Acid Concentration (%)", "type": "number", "min": 0.5, "max": 5.0, "value": 2.0},
                {"name": "Time (min)", "type": "number", "min": 10.0, "max": 120.0, "value": 60.0},
                {"name": "Acid Type", "type": "selectbox", "options": ["H2SO4", "HCl", "HNO3"]},
                {"name": "Pressure (bar)", "type": "selectbox", "options": ["Atmospheric", "2 bar", "5 bar"]},
                {"name": "Catalyst Present", "type": "checkbox"}
            ]
        },
        ("Sugarcane Bagasse", "Basic"): {
            "params": [
                {"name": "Temperature (°C)", "type": "number", "min": 80.0, "max": 180.0, "value": 120.0},
                {"name": "Base Concentration (%)", "type": "number", "min": 1.0, "max": 10.0, "value": 4.0},
                {"name": "Time (min)", "type": "number", "min": 30.0, "max": 180.0, "value": 90.0},
                {"name": "Base Type", "type": "selectbox", "options": ["NaOH", "KOH", "Ca(OH)2"]},
                {"name": "Mixing Speed", "type": "selectbox", "options": ["Low", "Medium", "High"]},
                {"name": "Oxygen Present", "type": "checkbox"}
            ]
        },
        ("Sugarcane Bagasse", "Organosolv"): {
            "params": [
                {"name": "Temperature (°C)", "type": "number", "min": 150.0, "max": 220.0, "value": 180.0},
                {"name": "Ethanol Concentration (%)", "type": "number", "min": 40.0, "max": 80.0, "value": 60.0},
                {"name": "Time (min)", "type": "number", "min": 30.0, "max": 150.0, "value": 75.0},
                {"name": "Catalyst", "type": "selectbox", "options": ["H2SO4", "HCl", "Formic Acid"]},
                {"name": "Liquid/Solid Ratio", "type": "selectbox", "options": ["5:1", "10:1", "15:1"]},
                {"name": "Acid Added", "type": "checkbox"}
            ]
        },
        ("Sugarcane Bagasse", "Hydrothermal"): {
            "params": [
                {"name": "Temperature (°C)", "type": "selectbox", "options": [180, 195, 210]},
                {"name": "Solid Loading (g/L)", "type": "number", "min": 50.0, "max": 200.0, "value": 100.0},
                {"name": "Time (min)", "type": "slider", "min": 10.0, "max": 120.0, "value": 40.0},
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
            {"name": "Temperature (°C)", "type": "number", "min": 100.0, "max": 250.0, "value": 150.0},
            {"name": "Concentration (%)", "type": "number", "min": 1.0, "max": 10.0, "value": 5.0},
            {"name": "Time (min)", "type": "number", "min": 30.0, "max": 180.0, "value": 60.0},
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
            # Ensure all values are float for consistent display
            min_val = float(param.get("min", 0.0))
            max_val = float(param.get("max", 1000.0))
            default_val = float(param.get("value", 0.0))
            
            value = st.number_input(
                param["name"],
                min_value=min_val,
                max_value=max_val,
                value=default_val,
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
            # Ensure all values are float for sliders
            min_val = float(param.get("min", 0.0))
            max_val = float(param.get("max", 100.0))
            default_val = float(param.get("value", 50.0))
            step_val = float(param.get("step", 1.0))
            
            value = st.slider(
                param["name"],
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                step=step_val,
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
    celulose1 = st.number_input(
        "Cellulose Percentage",
        min_value=45.0,
        max_value=65.0,
        format="%.2f",
        placeholder="45.00 – 65.00"
    )
    lignina1 = st.number_input("Lignin Percentage", min_value=0.0, max_value=100.0, format="%.2f")
    hemicelulose1 = st.number_input(
        "Hemicellulose Percentage",
        min_value=5.0,
        max_value=15.0,
        format="%.2f",
        placeholder="05.00 – 15.00"
    )

# Customizing the Enzymatic Hydrolysis Parameters column (col5)
with col5:
    st.header("Enzymatic Hydrolysis Parameters")
    st.write(f"Enter the enzymatic hydrolysis parameters to define your operating condition for {biomassa} with the enzyme.")
    enzyme = st.selectbox("Enzyme", ['Saccharomyces cerevisiae'])
    
    # Simplified parameters for all conditions
    solid_loading = st.number_input(
        "Initial Solids Loading (g/L)",
        min_value=50.0,
        max_value=250.0,
        format="%.2f",
        placeholder="50.00 – 250.00"
    )
    enzyme_loading = st.number_input(
        "Initial Enzyme Loading (g/L)",
        min_value=0.05,
        max_value=1.2,
        format="%.2f",
        placeholder="0.05 – 1.20"
    )
    reaction_time = st.number_input(
        "Reaction Time (h)",
        min_value=0.0,
        max_value=96.0,
        format="%.2f",
        placeholder="0.00 – 96.00"
    )

# Alteration 3: Mapping selection options to numerical values before using them (no longer needed for simplified version)
enzyme_types = {"Saccharomyces cerevisiae": 1}

# Customizing the Enzymatic Hydrolysis Results column (col6)
with col6:
    st.header("Enzymatic Hydrolysis Results")
    st.write(f"Here you can see the results obtained for the Enzymatic Hydrolysis stage of {biomassa}. Change the chart layout to visualize more relationships between the variables.")
    if st.button("Simulate Hydrolysis Profile", key="run_hydrolysis_profile"):
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
                st.metric("Glucose Produced", f"{glucose_final:.2f} g/L")

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

                fig.update_layout(
                    title="Enzymatic Hydrolysis Concentration Profiles",
                    hovermode="x unified"
                )
                fig.update_xaxes(title_text="Time (h)")
                fig.update_yaxes(title_text="Glucose (g/L)", secondary_y=False)
                fig.update_yaxes(title_text="Xylose / Cellobiose (g/L)", secondary_y=True)

                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(profile_df.round(3))
            except Exception as exc:
                st.error(f"Error while running hydrolysis simulation: {exc}")
    