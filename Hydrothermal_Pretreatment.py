# MODELO DE DEGRADAÇÃO HIDROTÉRMICA
# Extraído do notebook Hydrothermal Pretreatment.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def simulate_hydrothermal_degradation(temperature, solid_loading, cellulose_fraction, hemicellulose_fraction, time_final):
    """
    Simula a degradação hidrotérmica de celulose e hemicelulose.
    
    Args:
        temperature (int): Temperatura em °C (180, 195, 210).
        solid_loading (float): Carga de sólidos em g/L.
        cellulose_fraction (float): Fração mássica de celulose (0-1).
        hemicellulose_fraction (float): Fração mássica de hemicelulose (0-1).
        time_final (int): Tempo final da simulação em minutos.
    
    Returns:
        dict: Resultados da simulação contendo os tempos e concentrações.
    """
    
    # =============================================================================
    # PARÂMETROS CINÉTICOS
    # =============================================================================
    
    # Parâmetros para degradação da hemicelulose
    data_hemi = {
        'Temperature (°C)': [180, 195, 210],
        'k1 (1/min)': [0.0037, 0.0041, 0.0105],
        'k2 (1/min)': [0.0353, 0.0988, 0.2143],
        'k3 (1/min)': [0.0073, 0.0662, 0.2739],
        'k4 (1/min)': [0.0097, 0.0316, 0.0730],
        'k5 (1/min)': [0.0139, 0.0655, 0.1546],
        'k6 (1/min)': [0.0043, 0.0047, 0.0317]
    }

    # Parâmetros para degradação da celulose
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

    # =============================================================================
    # EQUAÇÕES DIFERENCIAIS
    # =============================================================================

    def hemicellulose_kinetics(y, t, k):
        """
        Equações diferenciais para degradação da hemicelulose
        H: Hemicelulose, XOS: Xylo-oligossacarídeos, MH: Monossacarídeos hemicelulósicos, 
        F: Furfural, D: Produtos de degradação
        """
        H, XOS, MH, F, D = y
        k1, k2, k3, k4, k5, k6 = k

        dH_dt = -(k1 + k2) * H
        dXOS_dt = k2 * H - k3 * XOS
        dMH_dt = k1 * H + k3 * XOS - (k4 + k5) * MH
        dF_dt = k4 * MH - k6 * F
        dD_dt = k5 * MH + k6 * F

        return [dH_dt, dXOS_dt, dMH_dt, dF_dt, dD_dt]

    def cellulose_kinetics(y, t, k):
        """
        Equações diferenciais para degradação da celulose
        C: Celulose, GOS: Glicooligossacarídeos, MC: Monossacarídeos celulósicos, 
        HMF: Hidroximetilfurfural, D: Produtos de degradação
        """
        C, GOS, MC, HMF, D = y
        k1, k2, k3, k4, k5, k6 = k

        dC_dt = -(k1 + k2) * C
        dGOS_dt = k2 * C - k3 * GOS
        dMC_dt = k1 * C + k3 * GOS - (k4 + k5) * MC
        dHMF_dt = k4 * MC - k6 * HMF
        dD_dt = k5 * MC + k6 * HMF

        return [dC_dt, dGOS_dt, dMC_dt, dHMF_dt, dD_dt]

    # =============================================================================
    # CONFIGURAÇÃO DA SIMULAÇÃO
    # =============================================================================

    # Validação de temperatura
    if temperature not in [180, 195, 210]:
        raise ValueError("Temperatura deve ser 180, 195 ou 210°C")

    # Concentrações iniciais
    C0 = solid_loading * cellulose_fraction
    H0 = solid_loading * hemicellulose_fraction

    # Condições iniciais
    y0_hemi = [H0, 0.0, 0.0, 0.0, 0.0]  # H, XOS, MH, F, D
    y0_cell = [C0, 0.0, 0.0, 0.0, 0.0]  # C, GOS, MC, HMF, D

    # Vetor de tempo
    t = np.linspace(0, time_final, 200)

    # Obter parâmetros cinéticos para a temperatura escolhida
    temp_idx = df_kn_hemicellulose[df_kn_hemicellulose['Temperature (°C)'] == temperature].index[0]
    k_hemi = df_kn_hemicellulose.iloc[temp_idx, 1:].values
    k_cell = df_kn_cellulose.iloc[temp_idx, 1:].values

    # =============================================================================
    # SIMULAÇÃO
    # =============================================================================

    # Resolver as EDOs
    sol_hemi = odeint(hemicellulose_kinetics, y0_hemi, t, args=(k_hemi,))
    sol_cell = odeint(cellulose_kinetics, y0_cell, t, args=(k_cell,))

    # Concentrações de celulose e hemicelulose ao longo do tempo
    cellulose_conc = sol_cell[:, 0]
    hemicellulose_conc = sol_hemi[:, 0]

    # Calcular porcentagens de degradação
    cellulose_final = cellulose_conc[-1]
    hemicellulose_final = hemicellulose_conc[-1]
    cellulose_degraded = (1 - cellulose_final/C0) * 100 if C0 > 0 else 0
    hemicellulose_degraded = (1 - hemicellulose_final/H0) * 100 if H0 > 0 else 0

    return {
        "time": t,
        "cellulose": cellulose_conc,
        "hemicellulose": hemicellulose_conc,
        "initial_cellulose": C0,
        "initial_hemicellulose": H0,
        "final_cellulose": cellulose_final,
        "final_hemicellulose": hemicellulose_final,
        "cellulose_degraded_percent": cellulose_degraded,
        "hemicellulose_degraded_percent": hemicellulose_degraded,
        "temperature": temperature,
        "solid_loading": solid_loading,
        "time_final": time_final
    }

def create_hydrothermal_plot(results, show_plot=True):
    """
    Cria um gráfico da degradação hidrotérmica.
    
    Args:
        results (dict): Resultados da simulação
        show_plot (bool): Se deve mostrar o gráfico ou apenas retornar a figura
    
    Returns:
        matplotlib.figure.Figure: Figura do gráfico
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plotar celulose e hemicelulose
    ax.plot(results["time"], results["cellulose"], 'b-', linewidth=3, 
            label='Celulose', marker='o', markersize=4, alpha=0.7)
    ax.plot(results["time"], results["hemicellulose"], 'g-', linewidth=3, 
            label='Hemicelulose', marker='s', markersize=4, alpha=0.7)

    # Configurações do gráfico
    ax.set_title(f'Degradação Hidrotérmica a {results["temperature"]}°C\n'
                f'Carga de sólidos: {results["solid_loading"]} g/L', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Tempo (min)', fontsize=12)
    ax.set_ylabel('Concentração (g/L)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Adicionar anotações com valores finais
    ax.text(0.02, 0.98, f'Degradação em {results["time_final"]} min:\n'
                        f'Celulose: {results["cellulose_degraded_percent"]:.1f}%\n'
                        f'Hemicelulose: {results["hemicellulose_degraded_percent"]:.1f}%', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    
    if show_plot:
        plt.show()
    
    return fig

# Função auxiliar para converter string de temperatura para número
def parse_temperature(temp_str):
    """
    Converte string de temperatura para número.
    
    Args:
        temp_str (str): String da temperatura (ex: "195°C")
    
    Returns:
        int: Temperatura como número
    """
    if isinstance(temp_str, str):
        return int(temp_str.replace("°C", ""))
    return int(temp_str)

# Exemplo de uso (pode ser removido se não for necessário)
if __name__ == "__main__":
    # Teste da função
    results = simulate_hydrothermal_degradation(
        temperature=195,
        solid_loading=100,
        cellulose_fraction=0.348,
        hemicellulose_fraction=0.230,
        time_final=40
    )
    
    print("Teste da função realizado com sucesso!")
    print(f"Degradação da celulose: {results['cellulose_degraded_percent']:.1f}%")
    print(f"Degradação da hemicelulose: {results['hemicellulose_degraded_percent']:.1f}%")
