import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def simulate_hydrothermal_degradation(temperature, solid_loading, cellulose_fraction, hemicellulose_fraction, time_final):
    """
    Simula a degradação hidrotérmica de celulose e hemicelulose.
    
    Args:
        temperature (int): Temperatura em °C (180, 195, 210).
        solid_loading (float): Carga de sólidos em g/L.
        cellulose_fraction (float): Fração mássica de celulose.
        hemicellulose_fraction (float): Fração mássica de hemicelulose.
        time_final (int): Tempo final da simulação em minutos.
    
    Returns:
        dict: Resultados da simulação contendo os tempos e concentrações.
    """
    # Parâmetros cinéticos
    data_hemi = {
        'Temperature (°C)': [180, 195, 210],
        'k1 (1/min)': [0.0037, 0.0041, 0.0105],
        'k2 (1/min)': [0.0353, 0.0988, 0.2143],
        'k3 (1/min)': [0.0073, 0.0662, 0.2739],
        'k4 (1/min)': [0.0097, 0.0316, 0.0730],
        'k5 (1/min)': [0.0139, 0.0655, 0.1546],
        'k6 (1/min)': [0.0043, 0.0047, 0.0317]
    }

    data_cell = {
        'Temperature (°C)': [180, 195, 210],
        'k1 (1/min)': [0.0051, 0.0060, 0.0294],
        'k2 (1/min)': [0.0002, 0.0084, 0.0080],
        'k3 (1/min)': [0.0550, 0.2400, 0.3100],
        'k4 (1/min)': [0.0023, 0.0070, 0.0460],
        'k5 (1/min)': [0.0531, 0.1573, 0.3772],
        'k6 (1/min)': [0.0007, 0.0010, 0.0588]
    }

    df_kn_hemicellulose = pd.DataFrame(data_hemi)
    df_kn_cellulose = pd.DataFrame(data_cell)

    # Equações diferenciais
    def hemicellulose_kinetics(y, t, k):
        H, XOS, MH, F, D = y
        k1, k2, k3, k4, k5, k6 = k
        dH_dt = -(k1 + k2) * H
        dXOS_dt = k2 * H - k3 * XOS
        dMH_dt = k1 * H + k3 * XOS - (k4 + k5) * MH
        dF_dt = k4 * MH - k6 * F
        dD_dt = k5 * MH + k6 * F
        return [dH_dt, dXOS_dt, dMH_dt, dF_dt, dD_dt]

    def cellulose_kinetics(y, t, k):
        C, GOS, MC, HMF, D = y
        k1, k2, k3, k4, k5, k6 = k
        dC_dt = -(k1 + k2) * C
        dGOS_dt = k2 * C - k3 * GOS
        dMC_dt = k1 * C + k3 * GOS - (k4 + k5) * MC
        dHMF_dt = k4 * MC - k6 * HMF
        dD_dt = k5 * MC + k6 * HMF
        return [dC_dt, dGOS_dt, dMC_dt, dHMF_dt, dD_dt]

    # Configuração da simulação
    C0 = solid_loading * cellulose_fraction
    H0 = solid_loading * hemicellulose_fraction
    y0_hemi = [H0, 0.0, 0.0, 0.0, 0.0]
    y0_cell = [C0, 0.0, 0.0, 0.0, 0.0]
    t = np.linspace(0, time_final, 200)

    temp_idx = df_kn_hemicellulose[df_kn_hemicellulose['Temperature (°C)'] == temperature].index[0]
    k_hemi = df_kn_hemicellulose.iloc[temp_idx, 1:].values
    k_cell = df_kn_cellulose.iloc[temp_idx, 1:].values

    # Resolver EDOs
    sol_hemi = odeint(hemicellulose_kinetics, y0_hemi, t, args=(k_hemi,))
    sol_cell = odeint(cellulose_kinetics, y0_cell, t, args=(k_cell,))

    return {
        "time": t,
        "cellulose": sol_cell[:, 0],
        "hemicellulose": sol_hemi[:, 0]
    }