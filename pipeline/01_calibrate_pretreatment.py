"""
Etapa 01: calibra o modelo cinético do pré-tratamento com os dados de Rocha et al. (2017).

Ajuste global (180, 195 e 210 °C ao mesmo tempo) de 29 parâmetros:
    12 k_ref (k a 195 °C) + 12 energias de ativação + taxa de aquecimento + 4 açúcares livres
contra 6 espécies medidas no licor (glicose, glico-oligômeros, HMF, xilose + arabinose,
xilo + arabino-oligômeros, furfural) e o ácido fórmico como marcador da degradação da celulose.

Com só 3 temperaturas, algumas energias de ativação não ficam bem determinadas e tendem a valores
extremos, que distorcem a extrapolação em temperatura. Por isso há uma priori fraca nas Ea
(normal, média 120 kJ/mol e desvio 50 kJ/mol, faixa típica de hidrólise e desidratação de açúcares).

Incerteza: bootstrap dos resíduos (o ajuste é refeito com dados reamostrados); os conjuntos de
parâmetros resultantes alimentam a geração de dados sintéticos e o ensemble das redes.

Saídas:
    models/pretreatment_kinetics.json
    reports/pretreatment_calibration.md e reports/figures/pretreatment_*.png
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ethanol_ai.paths import EXPERIMENTAL, MODELS, REPORTS  # noqa: E402
from ethanol_ai.constants import R_GAS  # noqa: E402
from ethanol_ai.pretreatment import (  # noqa: E402
    PretreatmentKinetics, rate_matrix, HEXOSAN_TO_FORMIC,
)

SEED = 42
N_STARTS = 12
N_BOOTSTRAP = 30
EA_PRIOR_MEAN, EA_PRIOR_SD = 120.0, 50.0  # kJ/mol

# Espécies medidas: coluna do modelo -> função que extrai a medida do DataFrame experimental
MEASURED = {
    "glucose": lambda d: d.glucose_g_L,
    "glucooligomers": lambda d: d.glucooligomers_g_L,
    "hmf": lambda d: d.hmf_g_L,
    "xylose": lambda d: d.xylose_g_L + d.arabinose_g_L,
    "xylooligomers": lambda d: d.xylooligomers_g_L + d.arabinooligomers_g_L,
    "furfural": lambda d: d.furfural_g_L,
    "formic_acid": lambda d: d.formic_acid_g_L,
}
# Peso relativo: o ácido fórmico é só um marcador aproximado (também vem do furfural)
WEIGHT = {"formic_acid": 0.5}

# Constantes ajustadas por temperatura no artigo original (Tabela 4 e equivalente para celulose)
PAPER_K = {
    "hemi": {180: [0.0037, 0.0353, 0.0073, 0.0097, 0.0139, 0.0043],
             195: [0.0041, 0.0988, 0.0662, 0.0316, 0.0655, 0.0047],
             210: [0.0105, 0.2143, 0.2739, 0.0730, 0.1546, 0.0317]},
    "cell": {180: [0.0051, 0.0002, 0.0550, 0.0023, 0.0531, 0.0007],
             195: [0.0060, 0.0084, 0.2400, 0.0070, 0.1573, 0.0010],
             210: [0.0294, 0.0080, 0.3100, 0.0460, 0.3772, 0.0588]},
}
T_REF = 195.0


def load_data():
    d = pd.read_csv(EXPERIMENTAL / "pretreatment.csv")
    obs = pd.DataFrame({k: f(d) for k, f in MEASURED.items()})
    obs.insert(0, "time_min", d.time_min)
    obs.insert(0, "temperature_C", d.temperature_C)
    comp = d.iloc[0]
    return obs, dict(solids_g_L=comp.solids_g_L, cellulose_frac=comp.cellulose_frac,
                     hemicellulose_frac=comp.hemicellulose_frac)


def predict(params: PretreatmentKinetics, obs: pd.DataFrame, comp: dict) -> pd.DataFrame:
    parts = []
    for T, g in obs.groupby("temperature_C", sort=False):
        sim = params.simulate(T, g.time_min.values, **comp)
        p = pd.DataFrame({k: sim[k] for k in MEASURED if k != "formic_acid"}, index=g.index)
        p["formic_acid"] = sim["cellulose_degradation"] * HEXOSAN_TO_FORMIC
        parts.append(p)
    return pd.concat(parts).loc[obs.index]


def sigma(obs: pd.DataFrame) -> pd.Series:
    """Incerteza assumida por espécie: 0,1 g/L + 5% do máximo observado (não há réplicas)."""
    s = 0.1 + 0.05 * obs[list(MEASURED)].max()
    return s / pd.Series({k: WEIGHT.get(k, 1.0) for k in MEASURED})


def initial_guess() -> np.ndarray:
    """Arrhenius ajustado às constantes do artigo (ponto de partida)."""
    T = np.array([180, 195, 210]) + 273.15
    vec = []
    for branch in ("hemi", "cell"):
        lk, ea = [], []
        for i in range(6):
            k = np.array([PAPER_K[branch][t][i] for t in (180, 195, 210)])
            slope, inter = np.polyfit(1 / T, np.log(k), 1)
            ea.append(float(np.clip(-slope * R_GAS, 40, 280)))
            lk.append(float(inter + slope / (T_REF + 273.15)))
        vec += lk + ea
    return np.array(vec + [np.log(3.0), 0.005, 0.004, 0.005, 0.005])


LOWER = np.array([-14] * 6 + [30] * 6 + [-14] * 6 + [30] * 6 + [np.log(0.5)] + [0] * 4)
UPPER = np.array([3] * 6 + [300] * 6 + [3] * 6 + [300] * 6 + [np.log(20)] + [0.05] * 4)
EA_IDX = np.r_[6:12, 18:24]


def residuals(v, obs, comp, sig):
    p = PretreatmentKinetics.from_vector(v)
    pred = predict(p, obs, comp)
    data = ((pred[list(MEASURED)] - obs[list(MEASURED)]) / sig).values.ravel()
    prior = (v[EA_IDX] - EA_PRIOR_MEAN) / EA_PRIOR_SD
    return np.concatenate([data, prior])


def fit(obs, comp, sig, x0, rng=None, n_starts=1):
    best = None
    for s in range(n_starts):
        start = x0.copy()
        if s > 0:
            start[0:6] += rng.normal(0, 1.0, 6); start[12:18] += rng.normal(0, 1.0, 6)
            start[6:12] *= rng.uniform(0.7, 1.3, 6); start[18:24] *= rng.uniform(0.7, 1.3, 6)
            start[24] += rng.normal(0, 0.4)
        start = np.clip(start, LOWER + 1e-6, UPPER - 1e-6)
        res = least_squares(residuals, start, bounds=(LOWER, UPPER), args=(obs, comp, sig),
                            loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=4000)
        if best is None or res.cost < best.cost:
            best = res
    return best


def _bootstrap_fit(args):
    new, comp, sig, x0 = args
    return fit(new, comp, sig, x0).x


def paper_model_predictions(obs: pd.DataFrame, comp: dict) -> pd.DataFrame:
    """
    Reproduz a abordagem original: k por temperatura e condição inicial = medidas em t = 0
    (unidades medidas, sem fatores estequiométricos). Serve só para comparação em t > 0.
    """
    from scipy.linalg import expm
    rows = []
    for T, g in obs.groupby("temperature_C", sort=False):
        g0 = g[g.time_min == 0].iloc[0]
        S = comp["solids_g_L"]
        dissolved_h = g0.xylose + g0.xylooligomers + g0.furfural
        dissolved_c = g0.glucose + g0.glucooligomers + g0.hmf
        y0_h = np.array([S * comp["hemicellulose_frac"] - dissolved_h, g0.xylooligomers, g0.xylose, g0.furfural, 0])
        y0_c = np.array([S * comp["cellulose_frac"] - dissolved_c, g0.glucooligomers, g0.glucose, g0.hmf, 0])
        Kh, Kc = rate_matrix(PAPER_K["hemi"][int(T)]), rate_matrix(PAPER_K["cell"][int(T)])
        for idx, t in zip(g.index, g.time_min):
            yh, yc = expm(Kh * t) @ y0_h, expm(Kc * t) @ y0_c
            rows.append(pd.Series({"glucose": yc[2], "glucooligomers": yc[1], "hmf": yc[3],
                                   "xylose": yh[2], "xylooligomers": yh[1], "furfural": yh[3]}, name=idx))
    return pd.DataFrame(rows).loc[obs.index]


def metrics(obs, pred, species, mask=None):
    out = {}
    for s in species:
        y, p = obs[s].values, pred[s].values
        if mask is not None:
            y, p = y[mask], p[mask]
        rmse = float(np.sqrt(np.mean((y - p) ** 2)))
        r2 = float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
        out[s] = {"rmse": rmse, "r2": r2}
    return out


def make_figure(obs, comp, best, boot, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    species = ["glucose", "glucooligomers", "hmf", "xylose", "xylooligomers", "furfural"]
    titles = ["Glucose (+cellobiose)", "Gluco-oligomers", "HMF", "Xylose (+arabinose)", "Xylo-oligomers", "Furfural"]
    colors = {180: "#2a78d6", 195: "#1baf7a", 210: "#eb6834"}
    t = np.linspace(0, 40, 81)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, s, title in zip(axes.ravel(), species, titles):
        for T in (180, 195, 210):
            sims = np.array([p.simulate(T, t, **comp)[s] for p in boot])
            ax.fill_between(t, np.percentile(sims, 5, 0), np.percentile(sims, 95, 0), color=colors[T], alpha=0.15, lw=0)
            ax.plot(t, best.simulate(T, t, **comp)[s], color=colors[T], lw=2, label=f"{T} °C")
            g = obs[obs.temperature_C == T]
            ax.plot(g.time_min, g[s], "o", color=colors[T], ms=5, mec="white", mew=1)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Time after heat-up (min)")
        ax.set_ylabel("g/L")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Hydrothermal pretreatment: global Arrhenius fit (lines), 5–95% bootstrap band, data (points)")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(SEED)
    obs, comp = load_data()
    sig = sigma(obs)

    t0 = time.time()
    res = fit(obs, comp, sig, initial_guess(), rng, N_STARTS)
    best = PretreatmentKinetics.from_vector(res.x)
    print(f"ajuste global: custo {res.cost:.2f}, {time.time() - t0:.0f} s")

    pred = predict(best, obs, comp)
    species = [s for s in MEASURED if s != "formic_acid"]
    m_all = metrics(obs, pred, species)
    m_late = metrics(obs, pred, species, mask=(obs.time_min > 0).values)
    paper = paper_model_predictions(obs, comp)
    m_paper = metrics(obs, paper, species, mask=(obs.time_min > 0).values)

    # Bootstrap de resíduos (por espécie)
    resid = obs[list(MEASURED)] - pred[list(MEASURED)]
    t0 = time.time()
    jobs = []
    for b in range(N_BOOTSTRAP):
        new = obs.copy()
        for s in MEASURED:
            new[s] = np.clip(pred[s].values + rng.choice(resid[s].values, len(obs), replace=True), 0, None)
        jobs.append((new, comp, sig, res.x))
    with ProcessPoolExecutor(8) as ex:
        boot = [PretreatmentKinetics.from_vector(x) for x in ex.map(_bootstrap_fit, jobs)]
    print(f"bootstrap: {N_BOOTSTRAP} ajustes em {time.time() - t0:.0f} s")

    # Grandezas derivadas para conferência física (40 min)
    def derived(p, T):
        s = p.simulate(T, [40.0], **comp)
        return (100 * (1 - s["hemicellulose"][0] / (comp["solids_g_L"] * comp["hemicellulose_frac"])),
                100 * (1 - s["cellulose"][0] / (comp["solids_g_L"] * comp["cellulose_frac"])))
    summary = {}
    for T in (180, 195, 210):
        vals = np.array([derived(p, T) for p in boot])
        summary[T] = {"hemicellulose_solubilized_pct": derived(best, T)[0],
                      "hemicellulose_band": np.percentile(vals[:, 0], [5, 95]).tolist(),
                      "cellulose_loss_pct": derived(best, T)[1],
                      "cellulose_band": np.percentile(vals[:, 1], [5, 95]).tolist()}

    MODELS.mkdir(exist_ok=True)
    out = {
        "meta": {
            "source": "Rocha et al. (2017) Bioresour. Technol. 228:176-185; data/experimental/pretreatment.csv",
            "fit": "global Arrhenius, soft-L1 least squares, weak prior on Ea (120 ± 50 kJ/mol), residual bootstrap",
            "experimental_domain": {"temperature_C": [180, 210], "time_min": [0, 40], "solids_g_L": [100, 100],
                                    "cellulose_frac": [comp["cellulose_frac"]] * 2,
                                    "hemicellulose_frac": [comp["hemicellulose_frac"]] * 2},
            "metrics_all_points": m_all, "metrics_t_gt_0": m_late, "paper_model_metrics_t_gt_0": m_paper,
            "derived_40min": summary,
        },
        "best": best.to_dict(),
        "bootstrap": [p.to_dict() for p in boot],
    }
    (MODELS / "pretreatment_kinetics.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    # Relatório
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    make_figure(obs, comp, best, boot, REPORTS / "figures" / "pretreatment_fit.png")
    lines = [
        "# Calibração do pré-tratamento hidrotérmico",
        "",
        "Gerado por `pipeline/01_calibrate_pretreatment.py`.",
        "",
        "Ajuste global (Arrhenius + rampa de aquecimento) às 21 amostras de licor de Rocha et al. (2017),",
        "com priori fraca nas energias de ativação (120 ± 50 kJ/mol).",
        "",
        "![ajuste](figures/pretreatment_fit.png)",
        "",
        "## Qualidade do ajuste",
        "",
        "| Espécie | RMSE (g/L), todos os pontos | R², todos | RMSE t > 0 | RMSE t > 0, ajuste original por T |",
        "|---|---|---|---|---|",
    ]
    for s in species:
        lines.append(f"| {s} | {m_all[s]['rmse']:.3f} | {m_all[s]['r2']:.3f} | {m_late[s]['rmse']:.3f} | {m_paper[s]['rmse']:.3f} |")
    lines += [
        "",
        "O ajuste original usa as medidas em t = 0 como condição inicial em cada temperatura; o modelo",
        "global prevê também t = 0 (fim do aquecimento) e é contínuo em temperatura.",
        "",
        f"Taxa de aquecimento ajustada: {best.heating_rate:.2f} °C/min (de {best.ramp_start_C:.0f} °C até a temperatura de operação).",
        "",
        "## Parâmetros (k a 195 °C, 1/min; Ea, kJ/mol)",
        "",
        "| Reação | k_ref hemicelulose | Ea hemicelulose | k_ref celulose | Ea celulose |",
        "|---|---|---|---|---|",
    ]
    names = ["k1 polímero→monômero", "k2 polímero→oligômero", "k3 oligômero→monômero",
             "k4 monômero→furano", "k5 monômero→degradação", "k6 furano→degradação"]
    for i, n in enumerate(names):
        bh = np.exp([p.log_k_ref_hemi[i] for p in boot]); bc = np.exp([p.log_k_ref_cell[i] for p in boot])
        lines.append(
            f"| {n} | {np.exp(best.log_k_ref_hemi[i]):.4f} ({np.percentile(bh, 5):.4f}–{np.percentile(bh, 95):.4f}) "
            f"| {best.ea_hemi[i]:.0f} | {np.exp(best.log_k_ref_cell[i]):.4f} ({np.percentile(bc, 5):.4f}–{np.percentile(bc, 95):.4f}) "
            f"| {best.ea_cell[i]:.0f} |")
    lines += ["", "Entre parênteses: faixa 5–95% do bootstrap.", "",
              "## Conferência física (40 min)", "",
              "| T (°C) | Hemicelulose solubilizada (%) | Perda de celulose (%) |", "|---|---|---|"]
    for T, v in summary.items():
        lines.append(f"| {T} | {v['hemicellulose_solubilized_pct']:.1f} ({v['hemicellulose_band'][0]:.1f}–{v['hemicellulose_band'][1]:.1f}) "
                     f"| {v['cellulose_loss_pct']:.1f} ({v['cellulose_band'][0]:.1f}–{v['cellulose_band'][1]:.1f}) |")
    lines += ["", "Açúcares livres no licor no início do aquecimento (g por g de sólido, base anidra): "
              + ", ".join(f"{k} {v:.4f}" for k, v in best.free_sugars.items()), ""]
    (REPORTS / "pretreatment_calibration.md").write_text("\n".join(lines), encoding="utf-8")

    for s in species:
        print(f"  {s:16s} RMSE {m_all[s]['rmse']:.3f}  R² {m_all[s]['r2']:.3f} | t>0: {m_late[s]['rmse']:.3f} vs original {m_paper[s]['rmse']:.3f}")
    print(f"  taxa de aquecimento: {best.heating_rate:.2f} °C/min")
    for T, v in summary.items():
        print(f"  {T} °C, 40 min: hemicelulose solubilizada {v['hemicellulose_solubilized_pct']:.1f}%, perda de celulose {v['cellulose_loss_pct']:.1f}% {np.round(v['cellulose_band'], 1)}")


if __name__ == "__main__":
    main()
