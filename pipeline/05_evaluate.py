"""
Etapa 05: avalia as redes contra o modelo mecanístico e contra os experimentos.

    1. rede x modelo mecanístico em condições fora do treino, com tempo denso
    2. modelo mecanístico e rede x experimentos
    3. garantias físicas (balanço de massa, não negatividade, monotonicidade, limites teóricos)

Saída: reports/surrogate_evaluation.md e reports/figures/surrogate_*.png
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ethanol_ai.paths import MODELS, EXPERIMENTAL, REPORTS, SYNTHETIC  # noqa: E402
from ethanol_ai import pretreatment as pre, hydrolysis as hyd, surrogate as sg  # noqa: E402
from ethanol_ai.constants import GLUCAN_TO_GLUCOSE, XYLAN_TO_XYLOSE, PENTOSAN_TO_FURFURAL  # noqa: E402
from ethanol_ai.domain import PRETREATMENT  # noqa: E402

RNG = np.random.default_rng(2024)
PRE_SPECIES = ["glucose", "glucooligomers", "hmf", "xylose", "xylooligomers", "furfural", "cellulose", "hemicellulose"]
HYD_SPECIES = ["glucose", "xylose", "cellobiose"]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def evaluate_pretreatment(lines, figs):
    mech, _, meta = pre.load_kinetics(MODELS / "pretreatment_kinetics.json")
    surr = sg.PretreatmentSurrogate.load(MODELS / "pretreatment_surrogate.npz")

    # 1. Rede x mecanístico: temperaturas e composições sorteadas na faixa do slider
    t = np.linspace(0, 52, 105)
    errs = {s: [] for s in PRE_SPECIES}
    mass_err, neg, nonmono = [], 0, 0
    for _ in range(200):
        T = RNG.uniform(*PRETREATMENT["temperature"].slider_range())
        S = RNG.uniform(*PRETREATMENT["solids"].slider_range())
        cel = RNG.uniform(*PRETREATMENT["cellulose"].slider_range()) / 100
        hem = RNG.uniform(*PRETREATMENT["hemicellulose"].slider_range()) / 100
        a = mech.simulate(T, t, S, cel, hem)
        b = surr.simulate(T, t, S, cel, hem)
        for s in PRE_SPECIES:
            errs[s].append(b[s] - a[s])
        # balanço de massa da rede (estados anidros somados = inicial + açúcares livres)
        h = surr.heads(T)
        total0 = S * hem + S * h["g_hemi"].sum()
        total_t = (b["hemicellulose"] + b["xylooligomers"] / XYLAN_TO_XYLOSE + b["xylose"] / XYLAN_TO_XYLOSE
                   + b["furfural"] / PENTOSAN_TO_FURFURAL + b["hemicellulose_degradation"])
        mass_err.append(np.abs(total_t - total0).max() / total0)
        neg += int(min(v.min() for v in b.values()) < -1e-9)
        nonmono += int((np.diff(b["cellulose"]) > 1e-9).any() or (np.diff(b["hemicellulose"]) > 1e-9).any())
    lines += ["## Pré-tratamento", "", "### Rede × modelo mecanístico (200 condições sorteadas na faixa do app, 0–52 min)", "",
              "| Saída | RMSE (g/L) | Erro máximo (g/L) |", "|---|---|---|"]
    for s in PRE_SPECIES:
        e = np.concatenate(errs[s])
        lines.append(f"| {s} | {rmse(e, 0):.4f} | {np.abs(e).max():.4f} |")
    lines += ["", f"Garantias: erro máximo de balanço de massa {max(mass_err):.1e} (relativo); "
              f"condições com valor negativo: {neg}/200; com polímero aumentando no tempo: {nonmono}/200.", ""]

    # 2. x experimentos
    d = pd.read_csv(EXPERIMENTAL / "pretreatment.csv")
    obs = {"glucose": d.glucose_g_L, "glucooligomers": d.glucooligomers_g_L, "hmf": d.hmf_g_L,
           "xylose": d.xylose_g_L + d.arabinose_g_L, "xylooligomers": d.xylooligomers_g_L + d.arabinooligomers_g_L,
           "furfural": d.furfural_g_L}
    pm, ps = {s: [] for s in obs}, {s: [] for s in obs}
    for T, g in d.groupby("temperature_C"):
        a = mech.simulate(T, g.time_min.values)
        b = surr.simulate(T, g.time_min.values, 100.0, 0.348, 0.230)
        for s in obs:
            pm[s].append(a[s]); ps[s].append(b[s])
    paper = meta.get("paper_model_metrics_t_gt_0", {})
    late = (d.time_min > 0).values
    lines += ["### × experimentos (21 amostras de licor; RMSE em g/L)", "",
              "O ajuste original usa as medidas em t = 0 como condição inicial, então a comparação justa é em t > 0.", "",
              "| Espécie | Mecanístico, todos os pontos | Mecanístico, t > 0 | Rede, t > 0 | Ajuste original por T, t > 0 |",
              "|---|---|---|---|---|"]
    for s in obs:
        m_all, s_all, y = np.concatenate(pm[s]), np.concatenate(ps[s]), obs[s].values
        lines.append(f"| {s} | {rmse(m_all, y):.3f} | {rmse(m_all[late], y[late]):.3f} | {rmse(s_all[late], y[late]):.3f} | "
                     f"{paper.get(s, {}).get('rmse', float('nan')):.3f} |")
    lines.append("")


def evaluate_hydrolysis(lines, figs):
    mech, boot, meta = hyd.load_kinetics(MODELS / "hydrolysis_kinetics.json")
    surr = sg.HydrolysisSurrogate.load(MODELS / "hydrolysis_surrogate.npz")
    data = np.load(SYNTHETIC / "hydrolysis.npz")
    X, n_train = data["X"], int(data["n_train"])
    Xv = X[n_train:]

    # 1. Rede x mecanístico em condições de validação, tempo denso (inclui tempos fora do grid de treino)
    t = np.arange(0, 125.5, 0.5)
    P = surr.simulate(Xv[:, 2], Xv[:, 3], Xv[:, 0], Xv[:, 1], t)
    idx = RNG.choice(len(Xv), 150, replace=False)
    errs = {s: [] for s in HYD_SPECIES}
    yield_err = []
    for i in idx:
        cel, hem, S, E = Xv[i]
        m = mech.simulate(S, E, cel, hem, t)
        for s in HYD_SPECIES:
            errs[s].append(P[s][i] - m[s])
        yield_err.append((P["glucose"][i] - m["glucose"]) / (GLUCAN_TO_GLUCOSE * S * cel))
    # garantias da rede em todas as condições de validação
    dG = np.diff(P["glucose"], axis=1)
    dX = np.diff(P["xylose"], axis=1)
    yG = P["glucose"] / (GLUCAN_TO_GLUCOSE * Xv[:, 2:3] * Xv[:, 0:1])
    lines += ["## Hidrólise enzimática", "",
              "### Rede × modelo mecanístico (150 condições de validação, 0–125 h a cada 0,5 h)", "",
              "| Saída | RMSE (g/L) | p95 do erro absoluto (g/L) | Erro máximo (g/L) |", "|---|---|---|---|"]
    for s in HYD_SPECIES:
        e = np.concatenate(errs[s])
        lines.append(f"| {s} | {rmse(e, 0):.3f} | {np.percentile(np.abs(e), 95):.3f} | {np.abs(e).max():.3f} |")
    # Figura: rede (linha cheia, faixa do ensemble) x modelo cinético (tracejado) em 6 condições de validação
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"glucose": "#2a78d6", "xylose": "#eb6834", "cellobiose": "#1baf7a"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for ax, i in zip(axes.ravel(), idx[:6]):
        cel, hem, S, E = Xv[i]
        m = mech.simulate(S, E, cel, hem, t)
        central, ens = surr.simulate_ensemble(S, E, cel, hem, t)
        for s in HYD_SPECIES:
            ax.fill_between(t, np.percentile(ens[s], 5, 0), np.percentile(ens[s], 95, 0), color=colors[s], alpha=0.15, lw=0)
            ax.plot(t, central[s], color=colors[s], lw=2, label=f"{s} (network)")
            ax.plot(t, m[s], color="#10231D", lw=1, ls="--")
        ax.set_title(f"{100 * cel:.0f}% cellulose, {S:.0f} g/L, {E:.2f} g/L enzyme", fontsize=10)
        ax.set_xlabel("Time (h)"); ax.set_ylabel("g/L"); ax.grid(alpha=0.25)
    axes[0, 0].plot([], [], color="#10231D", lw=1, ls="--", label="kinetic model")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Hydrolysis network vs calibrated kinetic model on validation conditions (band: 5–95% of the ensemble)")
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTS / "figures" / "surrogate_hydrolysis.png", dpi=110)
    plt.close(fig)
    lines += ["![rede x modelo](figures/surrogate_hydrolysis.png)", ""]

    ye = np.concatenate(yield_err)
    lines += ["", f"Rendimento de glicose: erro médio absoluto {100 * np.mean(np.abs(ye)):.2f} p.p., "
              f"máximo {100 * np.abs(ye).max():.2f} p.p.", "",
              f"Garantias (512 condições): t = 0 → glicose {np.abs(P['glucose'][:, 0]).max():.1e} g/L; "
              f"xilose decrescente em {(dX < -1e-9).any(1).sum()} condições; "
              f"glicose decrescente (> 0,01 g/L) em {(dG < -1e-2).any(1).sum()}; "
              f"rendimento de glicose máximo {yG.max():.3f} (limite 1).", ""]

    # 2. x experimentos
    d = pd.read_csv(EXPERIMENTAL / "hydrolysis.csv")
    rows = []
    allm, alls, ally = {s: [] for s in HYD_SPECIES}, {s: [] for s in HYD_SPECIES}, {s: [] for s in HYD_SPECIES}
    cover = []
    for c, g in d.groupby("condition", sort=False):
        g = g[g.time_h > 0]
        r = g.iloc[0]
        m = mech.simulate(r.solids_g_L, r.enzyme_g_L, r.cellulose_frac, r.hemicellulose_frac, g.time_h.values)
        central, ens = surr.simulate_ensemble(r.solids_g_L, r.enzyme_g_L, r.cellulose_frac, r.hemicellulose_frac, g.time_h.values)
        mask = {"glucose": np.ones(len(g), bool), "xylose": np.ones(len(g), bool),
                "cellobiose": ~(g.cellobiose_censored.values | g.replicates_identical.values)}
        for s in HYD_SPECIES:
            k = mask[s]
            allm[s].append(m[s][k]); alls[s].append(central[s][k]); ally[s].append(g[f"{s}_g_L"].values[k])
        lo, hi = np.percentile(ens["glucose"], 5, 0), np.percentile(ens["glucose"], 95, 0)
        cover.append(((g.glucose_g_L.values >= lo - g.glucose_sd.values) & (g.glucose_g_L.values <= hi + g.glucose_sd.values)).mean())
        rows.append((c, g.time_h.values[-1], g.glucose_g_L.values[-1], m["glucose"][-1], central["glucose"][-1]))
    loco = meta.get("loco_rmse", {}).get(meta.get("variant", ""), {})
    lit = meta.get("literature_rmse", {}).get("original", {})
    lines += ["### × experimentos (8 ensaios)", "",
              "| Espécie | Literatura (sem ajuste) | Mecanístico calibrado | Validação cruzada (LOCO) | Rede |", "|---|---|---|---|---|"]
    for s in HYD_SPECIES:
        y = np.concatenate(ally[s])
        lines.append(f"| {s} | {lit.get(s, float('nan')):.2f} | {rmse(np.concatenate(allm[s]), y):.2f} | "
                     f"{loco.get(s, float('nan')):.2f} | {rmse(np.concatenate(alls[s]), y):.2f} |")
    lines += ["", f"Pontos de glicose dentro da faixa 5–95% do ensemble (± desvio das réplicas): {100 * np.mean(cover):.0f}%", "",
              "| Ensaio | t (h) | Glicose exp. | Mecanístico | Rede |", "|---|---|---|---|---|"]
    for c, tl, ye_, ym, ys in rows:
        lines.append(f"| {c} | {tl:.0f} | {ye_:.1f} | {ym:.1f} | {ys:.1f} |")
    lines.append("")


def main() -> None:
    lines = ["# Avaliação das redes substitutas", "", "Gerado por `pipeline/05_evaluate.py`.", ""]
    figs = []
    evaluate_pretreatment(lines, figs)
    evaluate_hydrolysis(lines, figs)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "surrogate_evaluation.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
