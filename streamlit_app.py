import json
import math
import os
import numpy as np
import streamlit as st
import pandas as pd
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Caminhos dos artefatos usados pelo app (relativos a este arquivo)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# Faixas da amostragem LHS usadas no treino das ANNs.
# Fonte: research/01_pretreatment/data_generation/LHS_pre.ipynb e
#        research/02_enzymatic_hydrolysis/data_generation/LHS.ipynb
# Nos dois conjuntos a lignina é calculada por diferença (100% − celulose − hemicelulose).
PRETREATMENT_RANGES = {
    "temperature": (180.0, 210.0),  # °C
    "solids": (50.0, 150.0),        # g/L
    "cellulose": (30.0, 50.0),      # %
    "hemicellulose": (15.0, 30.0),  # %
    "lignin": (20.0, 40.0),         # %
    "time": (0.0, 40.0),            # min
}
HYDROLYSIS_RANGES = {
    "solids": (50.0, 250.0),        # g/L
    "enzyme": (0.05, 1.2),          # g/L
    "cellulose": (45.0, 65.0),      # %
    "hemicellulose": (5.0, 15.0),   # %
    "lignin": (20.0, 30.0),         # %
    "time": (0.0, 96.0),            # h
}

# Os sliders vão além da faixa de treino (30% da largura da faixa para cada lado);
# essas pontas aparecem marcadas como zona de extrapolação.
EXTRAPOLATION_MARGIN = 0.30


def extended_limits(train: tuple[float, float], step: float, floor: float | None = None) -> tuple[float, float]:
    """Limites do slider: faixa de treino ampliada pela margem de extrapolação, alinhada ao passo."""
    lo, hi = train
    pad = (hi - lo) * EXTRAPOLATION_MARGIN
    lo_ext = math.floor((lo - pad) / step + 1e-9) * step
    hi_ext = math.ceil((hi + pad) / step - 1e-9) * step
    if floor is not None:
        lo_ext = max(lo_ext, floor)
    return round(lo_ext, 6), round(hi_ext, 6)


# Horizonte das simulações = maior tempo selecionável (inclui a margem extrapolada)
PRE_TIME_MAX = extended_limits(PRETREATMENT_RANGES["time"], 0.5, floor=1.0)[1]
HYD_TIME_MAX = extended_limits(HYDROLYSIS_RANGES["time"], 0.5, floor=1.0)[1]

st.set_page_config(page_title="Ethanol AI", page_icon="⚗️", layout="wide")

# ============================================================================
# CARREGAMENTO DO MODELO ANN E SCALERS
# ============================================================================

@st.cache_resource
def load_ann_model_and_scalers():
    """
    Carrega modelos ANN (hidrólise e pré-tratamento) e configura scalers com dados de treinamento.
    Usa cache para carregar apenas uma vez.
    """
    try:
        # ===== HIDRÓLISE =====
        model_path = os.path.join(MODELS_DIR, "champion_ann_strategy1_32_32_16.h5")
        data_path = os.path.join(DATA_DIR, "synthetic_hydrolysis_data_LHS.csv")
        
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
        pretreat_model_path = os.path.join(MODELS_DIR, "champion_ann_pretreatment_strategy1_32_64_16.h5")
        pretreat_data_path = os.path.join(DATA_DIR, "synthetic_pretreatment_data_LHS.csv")
        
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


@st.cache_data(show_spinner=False)
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
    
    # Gerar array de tempos de 0 até o maior tempo do slider (acima de 40 min é extrapolação)
    time_array = np.linspace(0, PRE_TIME_MAX, int(PRE_TIME_MAX) + 1)
    
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


@st.cache_data(show_spinner=False)
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
    
    # Gerar array de tempos de 0 até o maior tempo do slider (acima de 96 h é extrapolação)
    t_final_simulation = HYD_TIME_MAX
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


# ============================================================================
# ESTILO DOS GRÁFICOS
# ============================================================================

# Paleta categórica validada (contraste e daltonismo) sobre fundo branco.
# A cor acompanha a substância: celulose → glicose (azul), hemicelulose → xilose (laranja).
SERIES_COLORS = {
    "Cellulose": "#2a78d6",
    "Hemicellulose": "#eb6834",
    "Glucose": "#2a78d6",
    "Xylose": "#eb6834",
    "Cellobiose": "#1baf7a",
}
INK = "#10231D"
INK_SECONDARY = "#47605A"
MUTED = "#7A8C86"
GRID = "#E6ECE9"
AXIS = "#C9D5D0"
CHART_FONT = "Inter, system-ui, sans-serif"
PLOTLY_CONFIG = {"displaylogo": False, "displayModeBar": False}


def style_figure(fig: go.Figure, height: int) -> go.Figure:
    """Aplica o visual padrão do app a um gráfico Plotly."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=8, r=8, t=72, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=CHART_FONT, size=13, color=INK_SECONDARY),
        legend=dict(orientation="h", yanchor="top", y=0.99, yref="container", xanchor="left", x=0, font=dict(color=INK)),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", bordercolor=AXIS, font=dict(family=CHART_FONT, color=INK)),
    )
    fig.update_xaxes(
        showgrid=False, showline=True, linecolor=AXIS, ticks="outside", tickcolor=AXIS,
        zeroline=False, title_font=dict(color=MUTED, size=12),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=GRID, gridwidth=1, showline=False, zeroline=False,
        rangemode="tozero", title_font=dict(color=MUTED, size=12),
    )
    return fig


def add_series(fig: go.Figure, x, y, name: str, unit: str, **subplot) -> None:
    """Linha de uma substância, com a cor fixa da paleta."""
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="lines", name=name,
            line=dict(color=SERIES_COLORS[name], width=2),
            hovertemplate=f"%{{y:.2f}} {unit}",
        ),
        **subplot,
    )


def add_point(fig: go.Figure, x: float, y: float, name: str, **subplot) -> None:
    """Destaca o valor de uma série no tempo selecionado."""
    fig.add_trace(
        go.Scatter(
            x=[x], y=[y], mode="markers", showlegend=False, hoverinfo="skip",
            marker=dict(size=10, color=SERIES_COLORS[name], line=dict(color="white", width=2)),
        ),
        **subplot,
    )


def add_time_marker(fig: go.Figure, x: float, label: str | None = None, **subplot) -> None:
    """Linha vertical de referência no tempo selecionado."""
    kwargs = dict(x=x, line_dash="dot", line_color=INK_SECONDARY, line_width=1, **subplot)
    if label:
        kwargs.update(annotation_text=label, annotation_position="top", annotation_font=dict(color=INK_SECONDARY, size=12))
    fig.add_vline(**kwargs)


def add_extrapolation_zone(
    fig: go.Figure, x0: float, x1: float, label: bool = True, label_position: str = "inside top right", **subplot,
) -> None:
    """Sombreia o trecho do eixo de tempo fora da faixa de treino."""
    kwargs = dict(x0=x0, x1=x1, fillcolor="rgba(245, 158, 11, 0.10)", line_width=0, layer="below", **subplot)
    if label:
        kwargs.update(
            annotation_text="Extrapolated", annotation_position=label_position,
            annotation_font=dict(color="#92400E", size=11),
        )
    fig.add_vrect(**kwargs)


# ============================================================================
# COMPONENTES DE INTERFACE
# ============================================================================

def section_head(number: str, title: str, text: str) -> None:
    st.html(f'<div class="section-head"><span class="section-num">{number}</span><div><h2>{title}</h2><p>{text}</p></div></div>')


def card_header(title: str, subtitle: str) -> None:
    st.html(f'<p class="card-title">{title}</p><p class="card-subtitle">{subtitle}</p>')


def group_label(text: str, first: bool = False) -> None:
    st.html(f'<p class="group-label{" first" if first else ""}">{text}</p>')


def empty_state(title: str, text: str) -> None:
    st.html(f'<div class="empty-state"><strong>{title}</strong>{text}</div>')


def cellulose_bounds(ranges: dict) -> tuple[float, float]:
    """Faixa de celulose que permite lignina (por diferença) dentro da faixa de treino."""
    lo = max(ranges["cellulose"][0], 100.0 - ranges["lignin"][1] - ranges["hemicellulose"][1])
    hi = min(ranges["cellulose"][1], 100.0 - ranges["lignin"][0] - ranges["hemicellulose"][0])
    return lo, hi


def model_slider(
    label: str, key: str, train: tuple[float, float], value: float, step: float, fmt: str,
    flags: list[str], floor: float | None = None, help: str | None = None,
) -> float:
    """
    Slider que cobre a faixa de treino do modelo mais a margem de extrapolação.
    A trilha é verde na faixa de treino e âmbar nas pontas extrapoladas. Se o valor cair nelas, o marcador
    fica âmbar, a etiqueta "Extrapolating" aparece no rótulo e o nome da variável entra em `flags`.
    """
    lo, hi = extended_limits(train, step, floor)
    value = st.slider(label, min_value=lo, max_value=hi, value=value, step=step, format=fmt, key=key, help=help)

    # Onde começa e termina a faixa de treino na trilha (em % da largura); o CSS pinta o resto de âmbar
    span = hi - lo
    left = max(0.0, (train[0] - lo) / span * 100)
    right = min(100.0, (train[1] - lo) / span * 100)
    css = f".st-key-{key} {{ --zone-left: {left:.3f}%; --zone-right: {right:.3f}%; }}"
    if not (train[0] <= value <= train[1]):
        flags.append(label.split(" (")[0])
        css += f' .st-key-{key} {{ --slider-accent: var(--extrapolation); --slider-badge: "Extrapolating"; }}'
    st.html(f"<style>{css}</style>")
    return value


def lignin_by_difference(cellulose: float, hemicellulose: float, ranges: dict, flags: list[str]) -> float:
    """Mostra a lignina calculada por diferença, marcando quando ela sai da faixa de treino."""
    lignin = 100.0 - cellulose - hemicellulose
    lo, hi = ranges["lignin"]
    if lo <= lignin <= hi:
        st.html(f'<div class="computed-field"><span>Lignin (by difference)</span><strong>{lignin:.1f}%</strong></div>')
    else:
        flags.append("Lignin")
        st.html(
            '<div class="computed-field extrapolated"><span>Lignin (by difference)<em>Extrapolating</em></span>'
            f'<strong>{lignin:.1f}%</strong></div>'
            f'<p class="computed-hint">Model trained on {lo:.0f}–{hi:.0f}% lignin. Adjust cellulose or hemicellulose.</p>'
        )
    return lignin


def extrapolation_banner(flags: list[str]) -> None:
    """Resumo, no cartão de resultados, das entradas fora da faixa de treino."""
    if flags:
        st.html(
            f'<div class="extrap-banner"><span class="extrap-tag">Extrapolating</span>'
            f'{", ".join(flags)} outside the training range. Results are less reliable.</div>'
        )


ICON_DOWNLOAD = '<span class="icon icon-download" aria-hidden="true"></span>'
ICON_COPY = '<span class="icon icon-copy" aria-hidden="true"></span>'

# Roda no navegador: gera o PNG com o Plotly da própria página e usa a área de transferência.
EXPORT_SCRIPT = """
(() => {
  const bar = document.getElementById("__BAR_ID__");
  if (!bar || bar.dataset.bound) return;
  bar.dataset.bound = "1";
  const data = __DATA__;
  const status = bar.querySelector(".export-status");

  const say = (msg) => {
    status.textContent = msg;
    clearTimeout(bar._timer);
    bar._timer = setTimeout(() => { status.textContent = ""; }, 2500);
  };

  const download = (url, name) => {
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  // PNG em alta resolução, fundo branco e título com as condições simuladas
  const chartPng = () => {
    const gd = document.querySelector(".st-key-__CHART_KEY__ .js-plotly-plot");
    if (!gd || !window.Plotly) return Promise.reject(new Error("chart not ready"));
    // Largura mínima para a imagem sair igual em qualquer tela; altura extra para título e subtítulo
    const width = Math.max(gd._fullLayout.width, 960);
    const H = gd._fullLayout.height + 60;
    const bottom = (gd.layout.margin && gd.layout.margin.b) || 8;
    const layout = Object.assign({}, gd.layout, {
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
      margin: Object.assign({}, gd.layout.margin, { t: 132, l: 28, r: 28 }),
      // Sem tags HTML aqui: o st.html descarta scripts que contenham o sinal de menor seguido de letra
      title: {
        text: data.title, x: 0.02, xanchor: "left", y: 1 - 16 / H, yanchor: "top", yref: "container",
        font: { size: 16, color: "#10231D", weight: 700 },
        subtitle: { text: data.subtitle, font: { size: 12, color: "#47605A" } },
      },
      legend: Object.assign({}, gd.layout.legend, { y: 1 - 72 / H, x: 0.02, xref: "container" }),
      annotations: (gd.layout.annotations || []).concat([{
        text: "Ethanol AI", showarrow: false, xref: "paper", yref: "paper",
        x: 1, xanchor: "right", y: 1 + (132 - 18) / (H - 132 - bottom), yanchor: "top",
        font: { size: 11, color: "#7A8C86" },
      }]),
    });
    return window.Plotly.toImage({ data: gd.data, layout }, { format: "png", width, height: H, scale: 3 });
  };

  bar.addEventListener("click", async (event) => {
    const button = event.target.closest("button[data-action]");
    if (!button) return;
    try {
      switch (button.dataset.action) {
        case "png":
          download(await chartPng(), data.file + ".png");
          say("PNG downloaded");
          break;
        case "copy-img": {
          const blob = chartPng().then((url) => fetch(url)).then((r) => r.blob());
          let item;
          try { item = new ClipboardItem({ "image/png": blob }); }
          catch (e) { item = new ClipboardItem({ "image/png": await blob }); }
          await navigator.clipboard.write([item]);
          say("Image copied");
          break;
        }
        case "csv": {
          const url = URL.createObjectURL(new Blob([data.csv], { type: "text/csv" }));
          download(url, data.file + ".csv");
          setTimeout(() => URL.revokeObjectURL(url), 2000);
          say("CSV downloaded");
          break;
        }
        case "copy-table":
          await navigator.clipboard.writeText(data.tsv);
          say("Table copied");
          break;
      }
    } catch (err) {
      say("Could not complete: " + err.message);
    }
  });
})();
"""


def chart_exports(df: pd.DataFrame, chart_key: str, file_base: str, title: str, subtitle: str) -> None:
    """Botões para baixar/copiar a imagem do gráfico e a tabela, e a tabela em um expander."""
    table = df.round(4)
    payload = json.dumps({
        "file": file_base,
        "title": title,
        "subtitle": subtitle,
        "csv": table.to_csv(index=False),
        "tsv": table.to_csv(index=False, sep="\t"),  # cola em colunas no Excel/Sheets
    }).replace("</", "<\\/")
    bar_id = f"export-{chart_key}"
    script = (
        EXPORT_SCRIPT.replace("__BAR_ID__", bar_id)
        .replace("__CHART_KEY__", chart_key)
        .replace("__DATA__", payload)
    )
    st.html(
        f"""
<div class="export-bar" id="{bar_id}">
  <div class="export-group">
    <span class="export-label">Chart</span>
    <button type="button" data-action="png" title="Download the chart as a PNG image">{ICON_DOWNLOAD}PNG</button>
    <button type="button" data-action="copy-img" title="Copy the chart image to the clipboard">{ICON_COPY}Copy</button>
  </div>
  <div class="export-group">
    <span class="export-label">Table</span>
    <button type="button" data-action="csv" title="Download the data as CSV">{ICON_DOWNLOAD}CSV</button>
    <button type="button" data-action="copy-table" title="Copy the data (paste into Excel or Google Sheets)">{ICON_COPY}Copy</button>
  </div>
  <span class="export-status" role="status" aria-live="polite"></span>
</div>
<script>{script}</script>
""",
        unsafe_allow_javascript=True,
    )
    with st.expander("Data table", icon=":material/table_chart:"):
        st.dataframe(df.round(3), hide_index=True)


# ============================================================================
# PÁGINA
# ============================================================================

with open(os.path.join(ASSETS_DIR, "style.css"), encoding="utf-8") as css_file:
    st.html(f"<style>{css_file.read()}</style>")

ABOUT_TEXT = (
    "Ethanol AI is a tool created within a research program called scientific initiation by researchers "
    "from UFSCar and DTU with funding from FAPESP. It is particularly useful for studying the behavior of "
    "different second-generation ethanol production processes when subjected to various operating conditions. "
    "This software implements hybrid machine learning models, previously trained using knowledge generated "
    "from previous research works at UFSCar and abroad, to predict the outcomes. Here, you can test different "
    "combinations of initial conditions, essentially finding the maximum possible yield for each situation."
)

st.html(f"""
<section class="hero">
  <div class="hero-badges">
    <span class="hero-badge">UFSCar</span><span class="hero-badge">DTU</span><span class="hero-badge">FAPESP</span>
  </div>
  <h1 class="hero-title">Ethanol <span>AI</span></h1>
  <p class="hero-tagline">Simulate second-generation ethanol production from sugarcane residues with hybrid machine-learning models.</p>
  <p class="hero-text">{ABOUT_TEXT}</p>
  <div class="pipeline">
    <span class="pipeline-step"><span class="num">01</span>Hydrothermal pretreatment</span>
    <span class="pipeline-arrow">&rarr;</span>
    <span class="pipeline-step"><span class="num">02</span>Enzymatic hydrolysis</span>
    <span class="pipeline-arrow">&rarr;</span>
    <span class="pipeline-step soon"><span class="num">03</span>Fermentation<em>In development</em></span>
  </div>
</section>
""")

if not model_loaded:
    st.error(f"The ANN models could not be loaded: {load_error}", icon=":material/error:")

tab_pre, tab_hyd = st.tabs([
    ":material/local_fire_department: Pretreatment",
    ":material/science: Enzymatic hydrolysis",
])

# ----------------------------------------------------------------------------
# Etapa 1: Pré-tratamento
# ----------------------------------------------------------------------------
with tab_pre:
    section_head(
        "01", "Hydrothermal pretreatment",
        "Predict how much cellulose and hemicellulose remain in the biomass over the course of the pretreatment.",
    )
    col_params, col_results = st.columns([1, 2], gap="large")

    with col_params:
        with st.container(key="card-pre-params"):
            card_header("Parameters", "Feedstock, composition and operating conditions")

            group_label("Feedstock", first=True)
            biomassa = st.selectbox(
                "Biomass", ['Sugarcane Straw', 'Sugarcane Bagasse'], key="pre_biomass",
                help="Note: Only Sugarcane Straw with Hydrothermal pretreatment is currently available",
            )
            pretratamento = st.selectbox(
                "Pretreatment", ['Hydrothermal', 'Organosolv'], key="pre_type",
                help="Note: Organosolv model is under development",
            )

            group_label("Composition (% w/w)")
            R = PRETREATMENT_RANGES
            flags_pre: list[str] = []
            celulose = model_slider("Cellulose (%)", "pre_cellulose", cellulose_bounds(R), 40.0, 0.5, "%.1f", flags_pre)
            hemicelulose = model_slider("Hemicellulose (%)", "pre_hemicellulose", R["hemicellulose"], 30.0, 0.5, "%.1f", flags_pre)
            lignina = lignin_by_difference(celulose, hemicelulose, R, flags_pre)

            group_label("Operating conditions")
            solid_loading_hydro = model_slider("Solids loading (g/L)", "pre_solids", R["solids"], 100.0, 1.0, "%.0f", flags_pre)
            temperature_hydro = model_slider("Temperature (°C)", "pre_temperature", R["temperature"], 195.0, 0.5, "%.1f", flags_pre)
            time_hydro = model_slider("Time (min)", "pre_time", R["time"], 15.0, 0.5, "%.1f", flags_pre, floor=1.0)

    with col_results:
        with st.container(key="card-pre-results"):
            card_header(
                "Results",
                f"{biomassa} · {pretratamento} · {temperature_hydro:.1f} °C · {solid_loading_hydro:.0f} g/L · t = {time_hydro:.1f} min",
            )

            if not (pretratamento == "Hydrothermal" and biomassa == "Sugarcane Straw"):
                if biomassa == 'Sugarcane Bagasse':
                    empty_state("Model under development", "Models for Sugarcane Bagasse are not available yet.")
                else:
                    empty_state("Model under development", f"The Organosolv pretreatment model for {biomassa} is not available yet.")
            else:
                extrapolation_banner(flags_pre)
                try:
                    results = simulate_pretreatment_ann(
                        temperature=temperature_hydro,
                        solid_loading=solid_loading_hydro,
                        cellulose_percent=celulose,
                        hemicellulose_percent=hemicelulose,
                        lignin_percent=lignina,
                        time_final=time_hydro,
                    )
                except Exception as e:
                    st.error(f"Error in simulation: {e}", icon=":material/error:")
                else:
                    # Valores no tempo escolhido
                    cellulose_at_time = float(np.interp(time_hydro, results['time'], results['cellulose']))
                    hemicellulose_at_time = float(np.interp(time_hydro, results['time'], results['hemicellulose']))

                    C0 = solid_loading_hydro * (celulose / 100.0)
                    H0 = solid_loading_hydro * (hemicelulose / 100.0)
                    cellulose_degraded = ((C0 - cellulose_at_time) / C0 * 100) if C0 > 0 else 0
                    hemicellulose_degraded = ((H0 - hemicellulose_at_time) / H0 * 100) if H0 > 0 else 0

                    with st.container(horizontal=True, gap="small", key="metrics-pre"):
                        st.metric("Cellulose degraded", f"{cellulose_degraded:.1f}%")
                        st.metric("Hemicellulose degraded", f"{hemicellulose_degraded:.1f}%")
                        st.metric("Cellulose remaining", f"{cellulose_at_time:.1f} g/L")
                        st.metric("Hemicellulose remaining", f"{hemicellulose_at_time:.1f} g/L")

                    fig = go.Figure()
                    add_series(fig, results["time"], results["cellulose"], "Cellulose", "g/L")
                    add_series(fig, results["time"], results["hemicellulose"], "Hemicellulose", "g/L")
                    add_point(fig, time_hydro, cellulose_at_time, "Cellulose")
                    add_point(fig, time_hydro, hemicellulose_at_time, "Hemicellulose")
                    add_time_marker(fig, time_hydro, f"t = {time_hydro:.1f} min")
                    add_extrapolation_zone(fig, PRETREATMENT_RANGES["time"][1], PRE_TIME_MAX)
                    style_figure(fig, height=420)
                    fig.update_xaxes(title_text="Time (min)")
                    fig.update_yaxes(title_text="Concentration (g/L)")
                    st.plotly_chart(fig, width="stretch", theme=None, config=PLOTLY_CONFIG, key="chart-pre")

                    chart_exports(
                        pd.DataFrame({
                            'Time (min)': results['time'],
                            'Cellulose (g/L)': results['cellulose'],
                            'Hemicellulose (g/L)': results['hemicellulose'],
                        }),
                        "chart-pre", "pretreatment_profile",
                        f"Hydrothermal pretreatment · {biomassa}",
                        f"{temperature_hydro:.1f} °C · {solid_loading_hydro:.0f} g/L solids · {celulose:.1f}% cellulose · "
                        f"{hemicelulose:.1f}% hemicellulose · {lignina:.1f}% lignin",
                    )

# ----------------------------------------------------------------------------
# Etapa 2: Hidrólise enzimática
# ----------------------------------------------------------------------------
with tab_hyd:
    section_head(
        "02", "Enzymatic hydrolysis",
        "Predict the release of glucose, xylose and cellobiose during the enzymatic hydrolysis of the pretreated biomass.",
    )
    col_params, col_results = st.columns([1, 2], gap="large")

    with col_params:
        with st.container(key="card-hyd-params"):
            card_header("Parameters", "Feedstock, composition and operating conditions")

            group_label("Feedstock", first=True)
            biomassa_hydrolysis = st.selectbox(
                "Biomass", ['Sugarcane Straw', 'Sugarcane Bagasse'], key="hyd_biomass",
                help="Note: Only Sugarcane Straw model is currently available",
            )
            enzyme = st.selectbox("Enzyme", ['Cellic CTEC-2 (Novozymes)'], key="hyd_enzyme")

            group_label("Composition (% w/w)")
            R = HYDROLYSIS_RANGES
            flags_hyd: list[str] = []
            celulose1 = model_slider("Cellulose (%)", "hyd_cellulose", cellulose_bounds(R), 62.0, 0.1, "%.1f", flags_hyd)
            hemicelulose1 = model_slider("Hemicellulose (%)", "hyd_hemicellulose", R["hemicellulose"], 12.0, 0.1, "%.1f", flags_hyd)
            lignina1 = lignin_by_difference(celulose1, hemicelulose1, R, flags_hyd)

            group_label("Operating conditions")
            solid_loading = model_slider("Solids loading (g/L)", "hyd_solids", R["solids"], 175.0, 1.0, "%.0f", flags_hyd, floor=10.0)
            enzyme_loading = model_slider("Enzyme loading (g/L)", "hyd_enzyme_loading", R["enzyme"], 0.5, 0.01, "%.2f", flags_hyd, floor=0.01)
            reaction_time = model_slider(
                "Reaction time (h)", "hyd_time", R["time"], 60.0, 0.5, "%.1f", flags_hyd, floor=1.0,
                help=f"The profile is always simulated up to {HYD_TIME_MAX:.0f} h; results are read at this time.",
            )

    with col_results:
        with st.container(key="card-hyd-results"):
            card_header(
                "Results",
                f"{biomassa_hydrolysis} · {enzyme} · {solid_loading:.0f} g/L solids · {enzyme_loading:.2f} g/L enzyme · t = {reaction_time:.1f} h",
            )

            if biomassa_hydrolysis != 'Sugarcane Straw':
                empty_state("Model under development", "The Sugarcane Bagasse model for Enzymatic Hydrolysis is not available yet.")
            else:
                extrapolation_banner(flags_hyd)
                try:
                    profile_df = simulate_enzymatic_hydrolysis(
                        solid_loading=solid_loading,
                        enzyme_loading=enzyme_loading,
                        cellulose_percent=celulose1,
                        hemicellulose_percent=hemicelulose1,
                        lignin_percent=lignina1,
                        reaction_time=reaction_time,
                    )
                except Exception as exc:
                    st.error(f"Error while running hydrolysis simulation: {exc}", icon=":material/error:")
                else:
                    # Valores no tempo de reação escolhido
                    time_h = profile_df["Time (h)"]
                    glucose_at_time = float(np.interp(reaction_time, time_h, profile_df["Glucose"]))
                    xylose_at_time = float(np.interp(reaction_time, time_h, profile_df["Xylose"]))
                    cellobiose_at_time = float(np.interp(reaction_time, time_h, profile_df["Cellobiose"]))

                    # Rendimento teórico: 1.111 g glicose por g de celulose (conversão estequiométrica)
                    cellulose_initial = solid_loading * (celulose1 / 100.0)
                    glucose_theoretical = cellulose_initial * 1.111
                    glucose_yield_percent = (glucose_at_time / glucose_theoretical) * 100 if glucose_theoretical > 0 else 0

                    with st.container(horizontal=True, gap="small", key="metrics-hyd"):
                        st.metric("Glucose", f"{glucose_at_time:.2f} g/L")
                        st.metric("Glucose yield", f"{glucose_yield_percent:.1f}%",
                                  help=f"Percentage of the theoretical maximum glucose ({glucose_theoretical:.2f} g/L, complete cellulose hydrolysis)")
                        st.metric("Xylose", f"{xylose_at_time:.2f} g/L")
                        st.metric("Cellobiose", f"{cellobiose_at_time:.2f} g/L")

                    # Glicose tem escala muito maior que xilose e celobiose: dois painéis com o mesmo eixo de tempo
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.55, 0.45])
                    add_series(fig, time_h, profile_df["Glucose"], "Glucose", "g/L", row=1, col=1)
                    add_series(fig, time_h, profile_df["Xylose"], "Xylose", "g/L", row=2, col=1)
                    add_series(fig, time_h, profile_df["Cellobiose"], "Cellobiose", "g/L", row=2, col=1)
                    add_point(fig, reaction_time, glucose_at_time, "Glucose", row=1, col=1)
                    add_point(fig, reaction_time, xylose_at_time, "Xylose", row=2, col=1)
                    add_point(fig, reaction_time, cellobiose_at_time, "Cellobiose", row=2, col=1)
                    add_time_marker(fig, reaction_time, f"t = {reaction_time:.1f} h", row=1, col=1)
                    add_time_marker(fig, reaction_time, row=2, col=1)
                    add_extrapolation_zone(
                        fig, HYDROLYSIS_RANGES["time"][1], HYD_TIME_MAX, label_position="inside bottom right", row=1, col=1,
                    )
                    add_extrapolation_zone(fig, HYDROLYSIS_RANGES["time"][1], HYD_TIME_MAX, label=False, row=2, col=1)
                    style_figure(fig, height=520)
                    fig.update_xaxes(title_text="Time (h)", row=2, col=1)
                    fig.update_yaxes(title_text="Glucose (g/L)", row=1, col=1)
                    fig.update_yaxes(title_text="Xylose, cellobiose (g/L)", row=2, col=1)
                    st.plotly_chart(fig, width="stretch", theme=None, config=PLOTLY_CONFIG, key="chart-hyd")

                    chart_exports(
                        profile_df, "chart-hyd", "enzymatic_hydrolysis_profile",
                        f"Enzymatic hydrolysis · {biomassa_hydrolysis} · {enzyme}",
                        f"{solid_loading:.0f} g/L solids · {enzyme_loading:.2f} g/L enzyme · {celulose1:.1f}% cellulose · "
                        f"{hemicelulose1:.1f}% hemicellulose · {lignina1:.1f}% lignin",
                    )

st.html("""
<footer class="footer">
  <span>Ethanol AI · Scientific initiation research at UFSCar in collaboration with DTU, funded by FAPESP.</span>
  <a href="https://github.com/AugustoCoding/Ethanol-AI" target="_blank">Source code on GitHub ↗</a>
</footer>
""")
