import json

import numpy as np
import pandas as pd
import streamlit as st
from plotly import graph_objs as go
from plotly.subplots import make_subplots

from ethanol_ai.paths import ROOT, MODELS, EXPERIMENTAL
from ethanol_ai import pretreatment as pre_kinetics, hydrolysis as hyd_kinetics
from ethanol_ai.surrogate import PretreatmentSurrogate, HydrolysisSurrogate
from ethanol_ai.domain import PRETREATMENT, HYDROLYSIS, Variable, fpu_per_g_cellulose
from ethanol_ai.optimization import enzyme_frontier, pretreatment_map
from ethanol_ai.constants import GLUCAN_TO_GLUCOSE, XYLAN_TO_XYLOSE

st.set_page_config(page_title="Ethanol AI", page_icon="⚗️", layout="wide")

BAND = (5, 95)  # percentis da faixa de incerteza (ensemble das redes)
PRE_T_MAX = PRETREATMENT["time"].slider_range()[1]
HYD_T_MAX = HYDROLYSIS["time"].slider_range()[1]


# ============================================================================
# MODELOS
# ============================================================================
# As previsões vêm das redes (inferência em NumPy, ver ethanol_ai/surrogate.py).
# Os modelos cinéticos calibrados só conferem as recomendações da aba de otimização.

@st.cache_resource(show_spinner=False)
def load_models() -> dict:
    return {
        "pre": PretreatmentSurrogate.load(MODELS / "pretreatment_surrogate.npz"),
        "hyd": HydrolysisSurrogate.load(MODELS / "hydrolysis_surrogate.npz"),
        "pre_mech": pre_kinetics.load_kinetics(MODELS / "pretreatment_kinetics.json")[0],
        "hyd_mech": hyd_kinetics.load_kinetics(MODELS / "hydrolysis_kinetics.json")[0],
    }


@st.cache_data(show_spinner=False)
def load_experiments() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(EXPERIMENTAL / "pretreatment.csv"), pd.read_csv(EXPERIMENTAL / "hydrolysis.csv")


@st.cache_data(show_spinner=False)
def run_pretreatment(temperature: float, solids: float, cellulose_pct: float, hemicellulose_pct: float):
    t = np.arange(0.0, PRE_T_MAX + 0.25, 0.5)
    central, ens = load_models()["pre"].simulate_ensemble(temperature, t, solids, cellulose_pct / 100, hemicellulose_pct / 100)
    return t, central, ens


@st.cache_data(show_spinner=False)
def run_hydrolysis(cellulose_pct: float, hemicellulose_pct: float, solids: float, enzyme: float):
    t = np.arange(0.0, HYD_T_MAX + 0.25, 0.5)
    central, ens = load_models()["hyd"].simulate_ensemble(solids, enzyme, cellulose_pct / 100, hemicellulose_pct / 100, t)
    return t, central, ens


def at_time(t: np.ndarray, series: np.ndarray, t_sel: float):
    """Valor no tempo escolhido; aceita uma série (tempos) ou um ensemble (membros x tempos)."""
    if series.ndim == 1:
        return float(np.interp(t_sel, t, series))
    return np.array([np.interp(t_sel, t, s) for s in series])


def band(values: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(values, BAND[0])), float(np.percentile(values, BAND[1]))


def band_series(ens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.percentile(ens, BAND[0], axis=0), np.percentile(ens, BAND[1], axis=0)


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
    "Xylo-oligomers": "#1baf7a",
    "Furfural": "#eda100",
    "Cellobiose": "#1baf7a",
}
INK = "#10231D"
INK_SECONDARY = "#47605A"
MUTED = "#7A8C86"
GRID = "#E6ECE9"
AXIS = "#C9D5D0"
CHART_FONT = "Inter, system-ui, sans-serif"
PLOTLY_CONFIG = {"displaylogo": False, "displayModeBar": False}


def hex_to_rgba(color: str, alpha: float) -> str:
    c = color.lstrip("#")
    return f"rgba({int(c[0:2], 16)}, {int(c[2:4], 16)}, {int(c[4:6], 16)}, {alpha})"


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




def add_band(fig: go.Figure, x, lo, hi, name: str, **subplot) -> None:
    """Faixa de incerteza (percentis 5–95 do ensemble) na cor da série."""
    color = SERIES_COLORS[name]
    fig.add_trace(go.Scatter(x=x, y=hi, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), **subplot)
    fig.add_trace(go.Scatter(x=x, y=lo, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor=hex_to_rgba(color, 0.14), showlegend=False, hoverinfo="skip"), **subplot)


def add_experiment(fig: go.Figure, x, y, name: str, error=None, show_legend: bool = False, **subplot) -> None:
    """Pontos experimentais (marcador vazado na cor da série, com barra de erro se houver)."""
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="markers", name="Experiment", legendgroup="experiment", showlegend=show_legend,
            marker=dict(size=8, color="white", line=dict(color=SERIES_COLORS[name], width=2)),
            error_y=dict(type="data", array=error, color=SERIES_COLORS[name], thickness=1, width=3) if error is not None else None,
            hovertemplate=f"{name} (experiment): %{{y:.2f}} g/L<extra></extra>",
        ),
        **subplot,
    )


def plot_series(fig, t, central, ens, key, name, unit="g/L", t_sel=None, **subplot):
    lo, hi = band_series(ens[key])
    add_band(fig, t, lo, hi, name, **subplot)
    add_series(fig, t, central[key], name, unit, **subplot)
    if t_sel is not None:
        add_point(fig, t_sel, float(np.interp(t_sel, t, central[key])), name, **subplot)


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



def model_slider(var: Variable, key: str, flags: list[str], default: float | None = None) -> float:
    """
    Slider na faixa do app (ethanol_ai/domain.py). A trilha é verde onde há suporte experimental
    e âmbar fora dele. Se o valor cair na parte âmbar, o marcador fica âmbar, a etiqueta
    "Extrapolating" aparece no rótulo e o nome da variável entra em `flags`.
    """
    lo, hi = var.slider_range()
    value = st.slider(f"{var.label} ({var.unit})", min_value=lo, max_value=hi,
                      value=var.default if default is None else default, step=var.step,
                      format=var.fmt, key=key, help=var.help)
    span = hi - lo
    left = max(0.0, (var.validated[0] - lo) / span * 100)
    right = min(100.0, (var.validated[1] - lo) / span * 100)
    css = f".st-key-{key} {{ --zone-left: {left:.3f}%; --zone-right: {right:.3f}%; }}"
    if not var.is_validated(value):
        flags.append(var.label)
        css += f' .st-key-{key} {{ --slider-accent: var(--extrapolation); --slider-badge: "Extrapolating"; }}'
    st.html(f"<style>{css}</style>")
    return value


def composition_rest(cellulose_pct: float, hemicellulose_pct: float, label: str) -> None:
    """Mostra o restante da composição (não entra nos modelos)."""
    rest = 100.0 - cellulose_pct - hemicellulose_pct
    st.html(f'<div class="computed-field"><span>{label}</span><strong>{rest:.1f}%</strong></div>')


def metric_band(label: str, value: float, lo: float, hi: float, fmt: str, unit: str, help: str | None = None) -> None:
    st.metric(label, f"{value:{fmt}}{unit}", delta=f"{lo:{fmt}}–{hi:{fmt}}{unit}", delta_color="off",
              delta_arrow="off", help=help)


def extrapolation_banner(flags: list[str]) -> None:
    """Resumo, no cartão de resultados, das entradas fora da faixa com suporte experimental."""
    if flags:
        st.html(
            f'<div class="extrap-banner"><span class="extrap-tag">Extrapolating</span>'
            f'{", ".join(flags)} outside the experimentally validated range. Results are less reliable.</div>'
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

st.html(f"<style>{(ROOT / 'assets' / 'style.css').read_text(encoding='utf-8')}</style>")

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

tab_pre, tab_hyd, tab_opt = st.tabs([
    ":material/local_fire_department: Pretreatment",
    ":material/science: Enzymatic hydrolysis",
    ":material/tune: Optimization",
])
exp_pre, exp_hyd = load_experiments()

# ----------------------------------------------------------------------------
# Etapa 1: Pré-tratamento
# ----------------------------------------------------------------------------
with tab_pre:
    section_head(
        "01", "Hydrothermal pretreatment",
        "Predict how the biomass is solubilized over time: what remains in the solid and which sugars and "
        "inhibitors reach the liquor.",
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

            V = PRETREATMENT
            flags_pre: list[str] = []
            group_label("Composition (% w/w, dry basis)")
            celulose = model_slider(V["cellulose"], "pre_cellulose", flags_pre)
            hemicelulose = model_slider(V["hemicellulose"], "pre_hemicellulose", flags_pre)
            composition_rest(celulose, hemicelulose, "Lignin, extractives and ash")

            group_label("Operating conditions")
            temperature = model_slider(V["temperature"], "pre_temperature", flags_pre)
            time_pre = model_slider(V["time"], "pre_time", flags_pre)
            solids_pre = model_slider(V["solids"], "pre_solids", flags_pre)

    with col_results:
        with st.container(key="card-pre-results"):
            card_header(
                "Results",
                f"{biomassa} · {pretratamento} · {temperature:.1f} °C · {solids_pre:.0f} g/L · t = {time_pre:.1f} min",
            )
            if not (pretratamento == "Hydrothermal" and biomassa == "Sugarcane Straw"):
                if biomassa == 'Sugarcane Bagasse':
                    empty_state("Model under development", "Models for Sugarcane Bagasse are not available yet.")
                else:
                    empty_state("Model under development", f"The Organosolv pretreatment model for {biomassa} is not available yet.")
            else:
                extrapolation_banner(flags_pre)
                t, c, ens = run_pretreatment(temperature, solids_pre, celulose, hemicelulose)
                H0, C0 = solids_pre * hemicelulose / 100, solids_pre * celulose / 100

                hemi_sol = 100 * (1 - at_time(t, c["hemicellulose"], time_pre) / H0)
                hemi_sol_m = 100 * (1 - at_time(t, ens["hemicellulose"], time_pre) / H0)
                cel_loss = 100 * (1 - at_time(t, c["cellulose"], time_pre) / C0)
                cel_loss_m = 100 * (1 - at_time(t, ens["cellulose"], time_pre) / C0)
                c5 = at_time(t, c["xylose"] + c["xylooligomers"], time_pre)
                c5_m = at_time(t, ens["xylose"] + ens["xylooligomers"], time_pre)
                fur = at_time(t, c["furfural"], time_pre)
                fur_m = at_time(t, ens["furfural"], time_pre)

                with st.container(horizontal=True, gap="small", key="metrics-pre"):
                    metric_band("Hemicellulose solubilized", hemi_sol, *band(hemi_sol_m), ".1f", "%")
                    metric_band("Cellulose lost", cel_loss, *band(cel_loss_m), ".1f", "%",
                                help="Cellulose converted to soluble products or degraded. Inferred from the liquor "
                                     "composition (the solid was not measured): likely an upper estimate.")
                    metric_band("C5 sugars in liquor", c5, *band(c5_m), ".1f", " g/L",
                                help="Xylose + arabinose and their oligomers, as monomer equivalents")
                    metric_band("Furfural", fur, *band(fur_m), ".2f", " g/L",
                                help="Fermentation inhibitor formed from pentoses")

                st.caption("Below each value: 90% range across the network ensemble (calibration uncertainty).")
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.45, 0.55])
                plot_series(fig, t, c, ens, "cellulose", "Cellulose", t_sel=time_pre, row=1, col=1)
                plot_series(fig, t, c, ens, "hemicellulose", "Hemicellulose", t_sel=time_pre, row=1, col=1)
                plot_series(fig, t, c, ens, "glucose", "Glucose", t_sel=time_pre, row=2, col=1)
                plot_series(fig, t, c, ens, "xylose", "Xylose", t_sel=time_pre, row=2, col=1)
                plot_series(fig, t, c, ens, "xylooligomers", "Xylo-oligomers", t_sel=time_pre, row=2, col=1)
                plot_series(fig, t, c, ens, "furfural", "Furfural", t_sel=time_pre, row=2, col=1)

                # Dados experimentais quando as condições coincidem com um ensaio de calibração
                g = exp_pre[(exp_pre.temperature_C - temperature).abs() <= 0.25]
                matches = (len(g) and abs(solids_pre - 100) <= 0.5 and abs(celulose - 34.8) <= 0.05
                           and abs(hemicelulose - 23.0) <= 0.05)
                if matches:
                    add_experiment(fig, g.time_min, g.glucose_g_L, "Glucose", show_legend=True, row=2, col=1)
                    add_experiment(fig, g.time_min, g.xylose_g_L + g.arabinose_g_L, "Xylose", row=2, col=1)
                    add_experiment(fig, g.time_min, g.xylooligomers_g_L + g.arabinooligomers_g_L, "Xylo-oligomers", row=2, col=1)
                    add_experiment(fig, g.time_min, g.furfural_g_L, "Furfural", row=2, col=1)

                add_time_marker(fig, time_pre, f"t = {time_pre:.1f} min", row=1, col=1)
                add_time_marker(fig, time_pre, row=2, col=1)
                t_val = PRETREATMENT["time"].validated[1]
                add_extrapolation_zone(fig, t_val, PRE_T_MAX, row=1, col=1)
                add_extrapolation_zone(fig, t_val, PRE_T_MAX, label=False, row=2, col=1)
                style_figure(fig, height=560)
                fig.update_xaxes(title_text="Time after heat-up (min)", row=2, col=1)
                fig.update_yaxes(title_text="Solid (g/L)", row=1, col=1)
                fig.update_yaxes(title_text="Liquor (g/L)", row=2, col=1)
                st.plotly_chart(fig, width="stretch", theme=None, config=PLOTLY_CONFIG, key="chart-pre")
                if matches:
                    st.caption("Points: liquor measured at these conditions (Rocha et al., 2017). Shaded bands: 90% range from the calibration uncertainty.")
                else:
                    st.caption("Shaded bands: 90% range from the calibration uncertainty. Experimental points appear for "
                               "180, 195 or 210 °C with the calibration straw (34.8% cellulose, 23.0% hemicellulose, 100 g/L).")

                table = pd.DataFrame({"Time (min)": t})
                for key, label in [("cellulose", "Cellulose (g/L)"), ("hemicellulose", "Hemicellulose (g/L)"),
                                   ("glucose", "Glucose (g/L)"), ("glucooligomers", "Gluco-oligomers (g/L)"),
                                   ("hmf", "HMF (g/L)"), ("xylose", "Xylose (g/L)"),
                                   ("xylooligomers", "Xylo-oligomers (g/L)"), ("furfural", "Furfural (g/L)")]:
                    table[label] = c[key]
                chart_exports(
                    table, "chart-pre", "pretreatment_profile",
                    f"Hydrothermal pretreatment · {biomassa}",
                    f"{temperature:.1f} °C · {solids_pre:.0f} g/L solids · {celulose:.1f}% cellulose · "
                    f"{hemicelulose:.1f}% hemicellulose",
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

            V = HYDROLYSIS
            flags_hyd: list[str] = []
            group_label("Pretreated solid composition (% w/w)")
            celulose1 = model_slider(V["cellulose"], "hyd_cellulose", flags_hyd)
            hemicelulose1 = model_slider(V["hemicellulose"], "hyd_hemicellulose", flags_hyd)
            composition_rest(celulose1, hemicelulose1, "Lignin and others")

            group_label("Operating conditions")
            solid_loading = model_slider(V["solids"], "hyd_solids", flags_hyd)
            enzyme_loading = model_slider(V["enzyme"], "hyd_enzyme_loading", flags_hyd)
            st.caption(f"≈ {fpu_per_g_cellulose(enzyme_loading, solid_loading, celulose1 / 100):.1f} FPU/g cellulose")
            reaction_time = model_slider(V["time"], "hyd_time", flags_hyd)

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
                t, c, ens = run_hydrolysis(celulose1, hemicelulose1, solid_loading, enzyme_loading)
                theo = GLUCAN_TO_GLUCOSE * solid_loading * celulose1 / 100
                g_t, g_m = at_time(t, c["glucose"], reaction_time), at_time(t, ens["glucose"], reaction_time)
                x_t, x_m = at_time(t, c["xylose"], reaction_time), at_time(t, ens["xylose"], reaction_time)
                b_t, b_m = at_time(t, c["cellobiose"], reaction_time), at_time(t, ens["cellobiose"], reaction_time)

                with st.container(horizontal=True, gap="small", key="metrics-hyd"):
                    metric_band("Glucose", g_t, *band(g_m), ".1f", " g/L")
                    metric_band("Glucose yield", 100 * g_t / theo, *band(100 * g_m / theo), ".1f", "%",
                                help=f"Percentage of the theoretical maximum glucose ({theo:.1f} g/L, complete cellulose hydrolysis)")
                    metric_band("Xylose", x_t, *band(x_m), ".1f", " g/L")
                    metric_band("Cellobiose", b_t, *band(b_m), ".2f", " g/L")

                st.caption("Below each value: 90% range across the network ensemble (calibration uncertainty).")
                # Glicose tem escala muito maior que xilose e celobiose: dois painéis com o mesmo eixo de tempo
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.55, 0.45])
                plot_series(fig, t, c, ens, "glucose", "Glucose", t_sel=reaction_time, row=1, col=1)
                plot_series(fig, t, c, ens, "xylose", "Xylose", t_sel=reaction_time, row=2, col=1)
                plot_series(fig, t, c, ens, "cellobiose", "Cellobiose", t_sel=reaction_time, row=2, col=1)

                e = exp_hyd[(exp_hyd.solids_g_L - solid_loading).abs().le(0.5)
                            & (exp_hyd.enzyme_g_L - enzyme_loading).abs().le(0.006)]
                matches = (len(e) and abs(celulose1 - 100 * e.cellulose_frac.iloc[0]) <= 0.1
                           and abs(hemicelulose1 - 100 * e.hemicellulose_frac.iloc[0]) <= 0.05)
                if matches:
                    add_experiment(fig, e.time_h, e.glucose_g_L, "Glucose", e.glucose_sd, show_legend=True, row=1, col=1)
                    add_experiment(fig, e.time_h, e.xylose_g_L, "Xylose", e.xylose_sd, row=2, col=1)
                    ok = ~(e.cellobiose_censored | e.replicates_identical)
                    add_experiment(fig, e.time_h[ok], e.cellobiose_g_L[ok], "Cellobiose", e.cellobiose_sd[ok], row=2, col=1)

                add_time_marker(fig, reaction_time, f"t = {reaction_time:.1f} h", row=1, col=1)
                add_time_marker(fig, reaction_time, row=2, col=1)
                t_val = HYDROLYSIS["time"].validated[1]
                add_extrapolation_zone(fig, t_val, HYD_T_MAX, label_position="inside bottom right", row=1, col=1)
                add_extrapolation_zone(fig, t_val, HYD_T_MAX, label=False, row=2, col=1)
                style_figure(fig, height=520)
                fig.update_xaxes(title_text="Time (h)", row=2, col=1)
                fig.update_yaxes(title_text="Glucose (g/L)", row=1, col=1)
                fig.update_yaxes(title_text="Xylose, cellobiose (g/L)", row=2, col=1)
                st.plotly_chart(fig, width="stretch", theme=None, config=PLOTLY_CONFIG, key="chart-hyd")
                if matches:
                    st.caption("Points: experiment at these conditions (mean ± SD of duplicates). Shaded bands: 90% range from the calibration uncertainty.")
                else:
                    st.caption("Shaded bands: 90% range from the calibration uncertainty. Experimental points appear when the "
                               "conditions match one of the 8 calibration runs (66.6% cellulose, 8.2% xylan).")

                chart_exports(
                    pd.DataFrame({"Time (h)": t, "Glucose (g/L)": c["glucose"], "Xylose (g/L)": c["xylose"],
                                  "Cellobiose (g/L)": c["cellobiose"]}),
                    "chart-hyd", "enzymatic_hydrolysis_profile",
                    f"Enzymatic hydrolysis · {biomassa_hydrolysis} · {enzyme}",
                    f"{solid_loading:.0f} g/L solids · {enzyme_loading:.2f} g/L enzyme · {celulose1:.1f}% cellulose · "
                    f"{hemicelulose1:.1f}% xylan",
                )

# ----------------------------------------------------------------------------
# Etapa 3: Otimização
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_frontier(cel_pct, hemi_pct, solids, target_pct, t_max):
    return enzyme_frontier(load_models()["hyd"], cel_pct / 100, hemi_pct / 100, solids, target_pct / 100, t_max,
                           HYDROLYSIS["enzyme"].slider_range())


@st.cache_data(show_spinner=False)
def cached_pre_map(solids, cel_pct, hemi_pct, furfural_max, cel_loss_max):
    return pretreatment_map(load_models()["pre"], solids, cel_pct / 100, hemi_pct / 100,
                            PRETREATMENT["temperature"].slider_range(), (0.0, PRE_T_MAX),
                            furfural_max, cel_loss_max / 100)


with tab_opt:
    section_head(
        "03", "Process optimization",
        "Search the operating conditions that meet a target, using the neural networks over the whole decision space. "
        "Each recommendation is checked against the calibrated kinetic model.",
    )

    # --- 3a. Dose de enzima
    col_params, col_results = st.columns([1, 2], gap="large")
    with col_params:
        with st.container(key="card-opt-hyd-params"):
            card_header("Enzyme dosage", "Lowest enzyme loading that reaches a glucose yield in time")
            flags_oh: list[str] = []
            group_label("Pretreated solid", first=True)
            oh_cel = model_slider(HYDROLYSIS["cellulose"], "oh_cellulose", flags_oh)
            oh_hemi = model_slider(HYDROLYSIS["hemicellulose"], "oh_hemicellulose", flags_oh)
            oh_solids = model_slider(HYDROLYSIS["solids"], "oh_solids", flags_oh)
            group_label("Target")
            oh_target = st.slider("Glucose yield target (%)", 20.0, 75.0, 55.0, 1.0, format="%.0f", key="oh_target",
                                  help="The calibrated model has a recalcitrant cellulose fraction: yields above ~70% are not reachable")
            oh_tmax = st.slider("Maximum reaction time (h)", 6.0, 96.0, 48.0, 1.0, format="%.0f", key="oh_tmax")

    with col_results:
        with st.container(key="card-opt-hyd-results"):
            card_header("Recommendation", f"{oh_solids:.0f} g/L solids · {oh_cel:.1f}% cellulose · target {oh_target:.0f}% in {oh_tmax:.0f} h")
            extrapolation_banner(flags_oh)
            fr = cached_frontier(oh_cel, oh_hemi, oh_solids, oh_target, oh_tmax)
            if fr.best_central is None:
                empty_state("Target not reachable", f"No enzyme loading in {HYDROLYSIS['enzyme'].slider_range()[0]:.2f}–"
                            f"{HYDROLYSIS['enzyme'].slider_range()[1]:.2f} g/L reaches {oh_target:.0f}% within {oh_tmax:.0f} h. "
                            "Lower the target or allow more time.")
            else:
                mech = load_models()["hyd_mech"]
                theo = GLUCAN_TO_GLUCOSE * oh_solids * oh_cel / 100
                check = mech.simulate(oh_solids, fr.best_central, oh_cel / 100, oh_hemi / 100, [oh_tmax])["glucose"][0] / theo
                with st.container(horizontal=True, gap="small", key="metrics-opt-hyd"):
                    st.metric("Minimum enzyme", f"{fr.best_central:.2f} g/L",
                              delta=f"{fpu_per_g_cellulose(fr.best_central, oh_solids, oh_cel / 100):.1f} FPU/g cellulose",
                              delta_color="off", delta_arrow="off")
                    if fr.best_conservative is not None:
                        st.metric("Conservative", f"{fr.best_conservative:.2f} g/L",
                                  delta=f"{fpu_per_g_cellulose(fr.best_conservative, oh_solids, oh_cel / 100):.1f} FPU/g cellulose",
                                  delta_color="off", delta_arrow="off",
                                  help="Reaches the target even at the 10th percentile of the calibration uncertainty")
                    else:
                        st.metric("Conservative", "Not reached", help="The target is not reached at the 10th percentile of the uncertainty")
                    st.metric("Kinetic model check", f"{100 * check:.1f}%",
                              delta=f"yield at {fr.best_central:.2f} g/L, {oh_tmax:.0f} h", delta_color="off", delta_arrow="off",
                              help="Calibrated kinetic model evaluated at the recommended loading and maximum time")

                fig = go.Figure()
                ok = ~np.isnan(fr.t_reach_central)
                fig.add_trace(go.Scatter(x=fr.enzyme_g_L[ok], y=fr.t_reach_central[ok], mode="lines", name="Central estimate",
                                         line=dict(color=SERIES_COLORS["Glucose"], width=2),
                                         hovertemplate="%{x:.3f} g/L → %{y:.1f} h<extra></extra>"))
                okc = ~np.isnan(fr.t_reach_conservative)
                fig.add_trace(go.Scatter(x=fr.enzyme_g_L[okc], y=fr.t_reach_conservative[okc], mode="lines", name="Conservative (10th percentile)",
                                         line=dict(color=SERIES_COLORS["Glucose"], width=2, dash="dash"),
                                         hovertemplate="%{x:.3f} g/L → %{y:.1f} h<extra></extra>"))
                fig.add_hline(y=oh_tmax, line_dash="dot", line_color=INK_SECONDARY, line_width=1,
                              annotation_text=f"time limit {oh_tmax:.0f} h", annotation_position="top right",
                              annotation_font=dict(color=INK_SECONDARY, size=12))
                fig.add_trace(go.Scatter(x=[fr.best_central], y=[float(np.interp(fr.best_central, fr.enzyme_g_L[ok], fr.t_reach_central[ok]))],
                                         mode="markers", name="Recommended", showlegend=False, hoverinfo="skip",
                                         marker=dict(size=12, color=SERIES_COLORS["Glucose"], line=dict(color="white", width=2))))
                style_figure(fig, height=400)
                fig.update_layout(hovermode="closest")
                fig.update_xaxes(type="log", title_text="Enzyme loading (g/L, log scale)")
                fig.update_yaxes(title_text=f"Time to reach {oh_target:.0f}% yield (h)")
                st.plotly_chart(fig, width="stretch", theme=None, config=PLOTLY_CONFIG, key="chart-opt-hyd")
                st.caption("Each point on the curve is a combination of enzyme loading and time that reaches the target: "
                           "more enzyme, less time. The recommendation is the lowest loading that meets the time limit.")
                chart_exports(
                    pd.DataFrame({"Enzyme loading (g/L)": fr.enzyme_g_L, "Time to target, central (h)": fr.t_reach_central,
                                  "Time to target, conservative (h)": fr.t_reach_conservative}),
                    "chart-opt-hyd", "enzyme_frontier",
                    f"Enzyme loading to reach {oh_target:.0f}% glucose yield",
                    f"{oh_solids:.0f} g/L solids · {oh_cel:.1f}% cellulose · {oh_hemi:.1f}% xylan",
                )

    st.html('<div style="height: 18px"></div>')

    # --- 3b. Condições do pré-tratamento
    col_params, col_results = st.columns([1, 2], gap="large")
    with col_params:
        with st.container(key="card-opt-pre-params"):
            card_header("Pretreatment conditions", "Recover hemicellulose sugars while limiting inhibitors and cellulose loss")
            flags_op: list[str] = []
            group_label("Feedstock", first=True)
            op_cel = model_slider(PRETREATMENT["cellulose"], "op_cellulose", flags_op)
            op_hemi = model_slider(PRETREATMENT["hemicellulose"], "op_hemicellulose", flags_op)
            op_solids = model_slider(PRETREATMENT["solids"], "op_solids", flags_op)
            group_label("Constraints")
            op_fur = st.slider("Maximum furfural (g/L)", 0.2, 5.0, 1.0, 0.1, format="%.1f", key="op_furfural",
                               help="Furfural inhibits the fermentation of the hydrolysate")
            op_closs = st.slider("Maximum cellulose loss (%)", 2.0, 40.0, 10.0, 1.0, format="%.0f", key="op_closs",
                                 help="Cellulose lost in the pretreatment is not available for the enzymatic hydrolysis")

    with col_results:
        with st.container(key="card-opt-pre-results"):
            card_header("Recommendation", f"{op_solids:.0f} g/L solids · furfural ≤ {op_fur:.1f} g/L · cellulose loss ≤ {op_closs:.0f}%")
            extrapolation_banner(flags_op)
            mp = cached_pre_map(op_solids, op_cel, op_hemi, op_fur, op_closs)
            if mp.best is None:
                empty_state("No feasible condition", "No temperature and time meet both limits. Relax the furfural or cellulose loss limit.")
            else:
                b = mp.best
                mech = load_models()["pre_mech"].simulate(b["temperature_C"], [b["time_min"]], op_solids, op_cel / 100, op_hemi / 100)
                mech_rec = 100 * (mech["xylose"][0] + mech["xylooligomers"][0]) / XYLAN_TO_XYLOSE / (op_solids * op_hemi / 100)
                out_of_range = [v.label for v, x in ((PRETREATMENT["temperature"], b["temperature_C"]), (PRETREATMENT["time"], b["time_min"]))
                                if not v.is_validated(x)]
                with st.container(horizontal=True, gap="small", key="metrics-opt-pre"):
                    st.metric("Temperature", f"{b['temperature_C']:.1f} °C")
                    st.metric("Time", f"{b['time_min']:.1f} min")
                    st.metric("C5 sugars recovered", f"{100 * b['c5_recovery']:.1f}%",
                              delta=f"kinetic model: {mech_rec:.1f}%", delta_color="off", delta_arrow="off",
                              help="Xylan recovered in the liquor as xylose and xylo-oligomers")
                    st.metric("Furfural", f"{b['furfural_g_L']:.2f} g/L", delta=f"cellulose loss {100 * b['cellulose_loss']:.1f}%",
                              delta_color="off", delta_arrow="off")
                if out_of_range:
                    extrapolation_banner([f"Recommended {x.lower()}" for x in out_of_range])

                fig = go.Figure()
                fig.add_trace(go.Heatmap(
                    x=mp.times_min, y=mp.temperature_C, z=100 * mp.c5_recovery, colorscale=[
                        [0.0, "#f3f7f5"], [0.25, "#b7d3f6"], [0.5, "#6da7ec"], [0.75, "#2a78d6"], [1.0, "#104281"]],
                    colorbar=dict(title=dict(text="C5 recovered (%)", side="right"), thickness=12, outlinewidth=0),
                    hovertemplate="%{y:.1f} °C, %{x:.1f} min<br>C5 recovered %{z:.1f}%<extra></extra>",
                ))
                fig.add_trace(go.Heatmap(
                    x=mp.times_min, y=mp.temperature_C, z=np.where(mp.feasible, np.nan, 1.0),
                    colorscale=[[0, "rgba(255,255,255,0.72)"], [1, "rgba(255,255,255,0.72)"]], showscale=False, hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter(x=[b["time_min"]], y=[b["temperature_C"]], mode="markers", name="Recommended",
                                         marker=dict(size=14, symbol="star", color="#eb6834", line=dict(color="white", width=1.5)),
                                         hovertemplate="Recommended: %{y:.1f} °C, %{x:.1f} min<extra></extra>"))
                style_figure(fig, height=430)
                fig.update_layout(hovermode="closest")
                fig.update_yaxes(title_text="Temperature (°C)", rangemode="normal", showgrid=False)
                fig.update_xaxes(title_text="Time after heat-up (min)")
                st.plotly_chart(fig, width="stretch", theme=None, config=PLOTLY_CONFIG, key="chart-opt-pre")
                st.caption("Color: share of the initial xylan recovered as fermentable C5 sugars. Faded area: conditions that break "
                           "the furfural or cellulose-loss limit. The star is the best feasible condition.")
                tt, TT = np.meshgrid(mp.times_min, mp.temperature_C)
                chart_exports(
                    pd.DataFrame({"Temperature (°C)": TT.ravel(), "Time (min)": tt.ravel(),
                                  "C5 recovered (%)": 100 * mp.c5_recovery.ravel(), "Furfural (g/L)": mp.furfural_g_L.ravel(),
                                  "Cellulose loss (%)": 100 * mp.cellulose_loss.ravel(), "Feasible": mp.feasible.ravel()}),
                    "chart-opt-pre", "pretreatment_map",
                    "Hemicellulose sugar recovery in the pretreatment",
                    f"{op_solids:.0f} g/L solids · {op_cel:.1f}% cellulose · {op_hemi:.1f}% hemicellulose · "
                    f"furfural ≤ {op_fur:.1f} g/L · cellulose loss ≤ {op_closs:.0f}%",
                )

st.html("""
<footer class="footer">
  <span>Ethanol AI · Scientific initiation research at UFSCar in collaboration with DTU, funded by FAPESP.</span>
  <span class="footer-links">
    <a href="https://github.com/AugustoCoding/Ethanol-AI/blob/master/docs/METODOLOGIA.md" target="_blank">Methodology ↗</a>
    <a href="https://github.com/AugustoCoding/Ethanol-AI" target="_blank">Source code on GitHub ↗</a>
  </span>
</footer>
""")
