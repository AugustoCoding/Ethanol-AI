"""
Etapa 02: calibra o modelo de hidrólise enzimática com os 8 ensaios do repositório.

Método:
    - estimativa MAP: mínimos quadrados ponderados pelo desvio das réplicas + priori log-normal
      (desvio 1 em ln) em torno dos valores publicados por Angarita et al. (2015), que regulariza
      os parâmetros pouco identificáveis com só 8 condições
    - 4 variantes estruturais: adsorção fixada em t = 0 (original) ou quase-estacionária, e
      reatividade do substrato com expoente gamma = 1 (original) ou ajustado
    - escolha da variante por validação cruzada deixando uma condição de fora (LOCO)
    - incerteza por bootstrap de condições (reamostra os 8 ensaios com reposição)

Saídas:
    models/hydrolysis_kinetics.json
    reports/hydrolysis_calibration.md e reports/figures/hydrolysis_*.png
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
from ethanol_ai.hydrolysis import HydrolysisKinetics, LITERATURE  # noqa: E402

SEED = 42
N_BOOTSTRAP = 40
WORKERS = 10
PRIOR_SD = 1.0  # em ln: um fator e (~2,7x) de desvio em relação ao valor publicado
SPECIES = ("glucose", "xylose", "cellobiose")
FLOOR = {"glucose": 1.0, "xylose": 0.3, "cellobiose": 0.1}  # g/L, somado ao desvio das réplicas
REL = 0.05

VARIANTS = {
    "original": {"adsorption": "initial", "fit_gamma": False, "fit_reactive": False},
    "quasi_steady": {"adsorption": "quasi_steady", "fit_gamma": False, "fit_reactive": False},
    "initial_gamma": {"adsorption": "initial", "fit_gamma": True, "fit_reactive": False},
    "quasi_steady_gamma": {"adsorption": "quasi_steady", "fit_gamma": True, "fit_reactive": False},
    "initial_reactive": {"adsorption": "initial", "fit_gamma": False, "fit_reactive": True},
    "quasi_steady_reactive": {"adsorption": "quasi_steady", "fit_gamma": False, "fit_reactive": True},
    "quasi_steady_reactive_gamma": {"adsorption": "quasi_steady", "fit_gamma": True, "fit_reactive": True},
    "quasi_steady_reactive_twopool": {"adsorption": "quasi_steady", "fit_gamma": False, "fit_reactive": True,
                                      "fit_twopool": True},
}
BASE_KEYS = [k for k in LITERATURE if k not in ("alfa", "gamma", "f_C", "f_H", "f_fast", "m_fast")]
# Valores iniciais / centro da priori para os parâmetros que não existem na literatura
PRIOR_CENTER = {"f_fast": 0.3, "m_fast": 5.0}


def load_conditions() -> list[dict]:
    d = pd.read_csv(EXPERIMENTAL / "hydrolysis.csv")
    conds = []
    for c, g in d.groupby("condition", sort=False):
        g = g[g.time_h > 0]
        r = g.iloc[0]
        obs = {s: g[f"{s}_g_L"].values.astype(float) for s in SPECIES}
        sd = {s: g[f"{s}_sd"].values.astype(float) for s in SPECIES}
        mask = {s: np.ones(len(g), bool) for s in SPECIES}
        mask["cellobiose"] = ~(g.cellobiose_censored.values | g.replicates_identical.values)
        sig = {s: np.sqrt(sd[s] ** 2 + (REL * obs[s]) ** 2 + FLOOR[s] ** 2) for s in SPECIES}
        conds.append(dict(name=c, solids=r.solids_g_L, enzyme=r.enzyme_g_L, cel=r.cellulose_frac,
                          hemi=r.hemicellulose_frac, t=g.time_h.values, obs=obs, sig=sig, mask=mask, sd=sd))
    return conds


def keys_for(variant: str) -> list[str]:
    cfg = VARIANTS[variant]
    return (BASE_KEYS + (["gamma"] if cfg["fit_gamma"] else []) + (["f_C", "f_H"] if cfg["fit_reactive"] else [])
            + (["f_fast", "m_fast"] if cfg.get("fit_twopool") else []))


def center(k: str) -> float:
    return PRIOR_CENTER.get(k, LITERATURE[k])


def bounds_for(variant: str) -> tuple[np.ndarray, np.ndarray]:
    """Limites em ln: ±5 em torno da literatura; frações reativas entre 0,2 e 1."""
    lit = np.log([center(k) for k in keys_for(variant)])
    lo, hi = lit - 5, lit + 5
    for i, k in enumerate(keys_for(variant)):
        if k in ("f_C", "f_H"):
            lo[i], hi[i] = np.log(0.2), 0.0
        if k == "f_fast":
            lo[i], hi[i] = np.log(0.01), np.log(0.95)
        if k == "m_fast":
            lo[i], hi[i] = 0.0, np.log(200.0)
    return lo, hi


def make_model(theta: np.ndarray, variant: str) -> HydrolysisKinetics:
    params = dict(LITERATURE)
    params.update({k: float(np.exp(v)) for k, v in zip(keys_for(variant), theta)})
    return HydrolysisKinetics(params=params, adsorption=VARIANTS[variant]["adsorption"])


def simulate(model, c, times=None):
    return model.simulate(c["solids"], c["enzyme"], c["cel"], c["hemi"], c["t"] if times is None else times)


def residuals(theta, variant, conds):
    model = make_model(theta, variant)
    res = []
    for c in conds:
        try:
            s = simulate(model, c)
        except RuntimeError:
            return np.full(sum(int(c2["mask"][sp].sum()) for c2 in conds for sp in SPECIES) + len(theta), 1e3)
        for sp in SPECIES:
            m = c["mask"][sp]
            res.append(((s[sp] - c["obs"][sp]) / c["sig"][sp])[m])
    prior = (theta - np.log([center(k) for k in keys_for(variant)])) / PRIOR_SD
    return np.concatenate(res + [prior])


def fit(variant, conds, x0=None):
    lo, hi = bounds_for(variant)
    lit = np.log([center(k) for k in keys_for(variant)])
    x0 = np.clip(lit if x0 is None else x0, lo + 1e-6, hi - 1e-6)
    res = least_squares(residuals, x0, args=(variant, conds), bounds=(lo, hi),
                        x_scale=1.0, max_nfev=400)
    return res.x


def rmse_by_species(model, conds):
    err = {s: [] for s in SPECIES}
    for c in conds:
        s = simulate(model, c)
        for sp in SPECIES:
            m = c["mask"][sp]
            err[sp].append((s[sp] - c["obs"][sp])[m])
    return {sp: float(np.sqrt(np.mean(np.concatenate(v) ** 2))) for sp, v in err.items()}


def loco_fold(args):
    variant, i, conds = args
    train = [c for j, c in enumerate(conds) if j != i]
    model = make_model(fit(variant, train), variant)
    return variant, i, rmse_by_species(model, [conds[i]])


def bootstrap_fit(args):
    variant, seed, conds, x0 = args
    rng = np.random.default_rng(seed)
    sample = [conds[j] for j in rng.integers(0, len(conds), len(conds))]
    return fit(variant, sample, x0)


def make_figure(conds, best, boot, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"glucose": "#2a78d6", "xylose": "#eb6834", "cellobiose": "#1baf7a"}
    lit = HydrolysisKinetics()
    t = np.linspace(0, 96, 97)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), constrained_layout=True, sharey=False)
    order = sorted(conds, key=lambda c: (c["solids"], c["enzyme"]))
    for ax, c in zip(axes.ravel(), order):
        sims = [simulate(m, c, t) for m in boot]
        b = simulate(best, c, t)
        l = simulate(lit, c, t)
        for sp in SPECIES:
            arr = np.array([s[sp] for s in sims])
            ax.fill_between(t, np.percentile(arr, 5, 0), np.percentile(arr, 95, 0), color=colors[sp], alpha=0.15, lw=0)
            ax.plot(t, b[sp], color=colors[sp], lw=2, label=sp.capitalize())
            m = c["mask"][sp]
            ax.errorbar(c["t"][m], c["obs"][sp][m], yerr=c["sd"][sp][m], fmt="o", color=colors[sp], ms=4, mec="white", mew=0.8, elinewidth=1)
        ax.plot(t, l["glucose"], color=colors["glucose"], lw=1, ls="--", label="Glucose (literature params)")
        ax.set_title(f"{c['solids']:.0f} g/L solids, {c['enzyme']:.3f} g/L enzyme", fontsize=10)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("g/L")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Enzymatic hydrolysis: calibrated model (lines), 5–95% bootstrap band, experiments ± SD (points)")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    conds = load_conditions()
    print(f"{len(conds)} condições, {sum(int(c['mask'][s].sum()) for c in conds for s in SPECIES)} observações")

    # Parâmetros da literatura (sem ajuste), para referência
    lit_rmse = {v: rmse_by_species(HydrolysisKinetics(adsorption=VARIANTS[v]["adsorption"]), conds)
                for v in ("original", "quasi_steady")}

    t0 = time.time()
    jobs = [(v, i, conds) for v in VARIANTS for i in range(len(conds))]
    loco = {v: {} for v in VARIANTS}
    with ProcessPoolExecutor(WORKERS) as ex:
        for v, i, r in ex.map(loco_fold, jobs):
            loco[v][conds[i]["name"]] = r
    print(f"validação cruzada (LOCO): {time.time() - t0:.0f} s")
    loco_mean = {v: {sp: float(np.sqrt(np.mean([loco[v][c][sp] ** 2 for c in loco[v]]))) for sp in SPECIES} for v in VARIANTS}
    for v in VARIANTS:
        print(f"  {v:20s} " + "  ".join(f"{sp} {loco_mean[v][sp]:.2f}" for sp in SPECIES))

    # Critério: RMSE de glicose na validação cruzada (saída principal do processo)
    chosen = min(VARIANTS, key=lambda v: loco_mean[v]["glucose"])
    print(f"variante escolhida: {chosen}")

    t0 = time.time()
    theta = fit(chosen, conds)
    best = make_model(theta, chosen)
    fit_rmse = rmse_by_species(best, conds)
    with ProcessPoolExecutor(WORKERS) as ex:
        thetas = list(ex.map(bootstrap_fit, [(chosen, SEED + b, conds, theta) for b in range(N_BOOTSTRAP)]))
    boot = [make_model(th, chosen) for th in thetas]
    print(f"ajuste final + {N_BOOTSTRAP} bootstraps: {time.time() - t0:.0f} s")

    # Por condição: glicose final, experimental vs modelos
    per_cond = []
    for c in conds:
        tl = c["t"][-1:]
        per_cond.append({
            "condition": c["name"], "time_h": float(tl[0]), "glucose_exp": float(c["obs"]["glucose"][-1]),
            "glucose_literature": float(simulate(HydrolysisKinetics(), c, tl)["glucose"][0]),
            "glucose_calibrated": float(simulate(best, c, tl)["glucose"][0]),
            "glucose_loco": None,
        })

    MODELS.mkdir(exist_ok=True)
    keys = keys_for(chosen)
    out = {
        "meta": {
            "source": "data/experimental/hydrolysis.csv (8 ensaios, palha pré-tratada 66,55% celulose / 8,2% xilana)",
            "fit": "MAP (priori log-normal sd=1 em torno de Angarita et al. 2015), bootstrap de condições",
            "variant": chosen, "variants": VARIANTS,
            "loco_rmse": loco_mean, "loco_by_condition": loco,
            "literature_rmse": lit_rmse, "calibrated_rmse": fit_rmse,
            "experimental_domain": {"solids_g_L": [150, 200], "enzyme_g_L": [0.0875, 1.05], "time_h": [0, 96],
                                    "cellulose_frac": [0.6655, 0.6655], "hemicellulose_frac": [0.082, 0.082]},
            "fitted_parameters": keys,
            "parameter_ratio_to_prior_center": {k: best.params[k] / center(k) for k in keys},
        },
        "best": best.to_dict(),
        "bootstrap": [m.to_dict() for m in boot],
    }
    (MODELS / "hydrolysis_kinetics.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    make_figure(conds, best, boot, REPORTS / "figures" / "hydrolysis_fit.png")
    lines = [
        "# Calibração da hidrólise enzimática",
        "",
        "Gerado por `pipeline/02_calibrate_hydrolysis.py`.",
        "",
        "![ajuste](figures/hydrolysis_fit.png)",
        "",
        "## Escolha da estrutura (validação cruzada deixando uma condição de fora)",
        "",
        "RMSE (g/L) ao prever cada ensaio com o modelo calibrado nos outros 7:",
        "",
        "| Variante | Adsorção | gamma | Fração reativa | Glicose | Xilose | Celobiose |",
        "|---|---|---|---|---|---|---|",
    ]
    for v, cfg in VARIANTS.items():
        mark = " **(escolhida)**" if v == chosen else ""
        lines.append(f"| {v}{mark} | {cfg['adsorption']} | {'ajustado' if cfg['fit_gamma'] else '1'} | "
                     f"{'ajustada' if cfg['fit_reactive'] else '1'} | "
                     + " | ".join(f"{loco_mean[v][sp]:.2f}" for sp in SPECIES) + " |")
    lines += [
        "",
        "Referência, parâmetros publicados sem ajuste: "
        + "; ".join(f"{v}: " + ", ".join(f"{sp} {lit_rmse[v][sp]:.2f}" for sp in SPECIES) for v in lit_rmse),
        "",
        "## Ajuste final (todos os 8 ensaios)",
        "",
        "RMSE: " + ", ".join(f"{sp} {fit_rmse[sp]:.2f} g/L" for sp in SPECIES),
        "",
        "| Ensaio | t (h) | Glicose exp. | Literatura | Calibrado |",
        "|---|---|---|---|---|",
    ]
    for r in per_cond:
        lines.append(f"| {r['condition']} | {r['time_h']:.0f} | {r['glucose_exp']:.1f} | {r['glucose_literature']:.1f} | {r['glucose_calibrated']:.1f} |")
    lines += ["", "## Parâmetros (razão em relação ao valor publicado; faixa 5–95% do bootstrap)", "",
              "| Parâmetro | Calibrado | Publicado | Razão | Faixa bootstrap |", "|---|---|---|---|---|"]
    for k in keys:
        vals = np.array([m.params[k] for m in boot])
        lines.append(f"| {k} | {best.params[k]:.4g} | {center(k):.4g}{' (priori)' if k in PRIOR_CENTER else ''} | {best.params[k] / center(k):.2f} | "
                     f"{np.percentile(vals, 5):.4g}–{np.percentile(vals, 95):.4g} |")
    lines += ["", "Observações:",
              "- celobiose censurada (abaixo da celobiose do extrato enzimático) e réplicas idênticas ficam fora do ajuste;",
              "- todos os ensaios têm a mesma composição de sólido; outras composições dependem da estrutura do modelo.", ""]
    (REPORTS / "hydrolysis_calibration.md").write_text("\n".join(lines), encoding="utf-8")

    print("RMSE ajuste final:", {k: round(v, 2) for k, v in fit_rmse.items()})
    for r in per_cond:
        print(f"  {r['condition']}: exp {r['glucose_exp']:.1f} | literatura {r['glucose_literature']:.1f} | calibrado {r['glucose_calibrated']:.1f}")


if __name__ == "__main__":
    main()
