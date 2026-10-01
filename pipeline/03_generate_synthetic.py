"""
Etapa 03: gera os dados sintéticos para o treino das redes, a partir dos modelos calibrados.

Um conjunto de dados por membro do ensemble: membro 0 = parâmetros do ajuste central,
membros 1..N = conjuntos do bootstrap. Cada rede do ensemble aprende um deles, e a dispersão
entre as redes reproduz a incerteza da calibração.

Pré-tratamento: a rede só precisa da temperatura (carga e composição entram de forma linear
e exata), então os alvos são perfis unitários por temperatura:
    1 g/L de polímero sem açúcares livres, e 1 g/L de sólido só com os açúcares livres.

Hidrólise: amostragem Sobol em (celulose, xilana, sólidos, log enzima) na faixa de treino
de ethanol_ai/domain.py, com 28 tempos de 0 a 130 h.

Saídas: data/synthetic/*.npz (todos os membros) e *_central.csv.gz (membro 0, legível).
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ethanol_ai.paths import MODELS, SYNTHETIC  # noqa: E402
from ethanol_ai import pretreatment as pre, hydrolysis as hyd  # noqa: E402
from ethanol_ai.domain import PRETREATMENT, HYDROLYSIS  # noqa: E402

N_MEMBERS = 11
SEED = 7
WORKERS = 10

PRE_N_T = 141                                   # temperaturas (passo de 0,5 °C)
PRE_TIMES = np.arange(0.0, 61.0, 2.0)           # min
HYD_N_TRAIN, HYD_N_VAL = 4096, 512
HYD_TIMES = np.array([0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 21, 24,
                      30, 36, 42, 48, 60, 72, 84, 96, 108, 120, 130], dtype=float)


def members(load):
    best, boot, _ = load
    return [best] + boot[: N_MEMBERS - 1]


# ------------------------------------------------------------ pré-tratamento
def pretreatment_member(params: pre.PretreatmentKinetics, temps: np.ndarray) -> dict[str, np.ndarray]:
    out = {k: np.zeros((len(temps), len(PRE_TIMES), 5)) for k in ("poly_hemi", "poly_cell", "free_hemi", "free_cell")}
    for i, T in enumerate(temps):
        for branch in ("hemi", "cell"):
            out[f"poly_{branch}"][i] = params.simulate_branch(T, PRE_TIMES, branch, polymer_g_L=1.0, solids_g_L=0.0)
            out[f"free_{branch}"][i] = params.simulate_branch(T, PRE_TIMES, branch, polymer_g_L=0.0, solids_g_L=1.0)
    return out


def _pre_job(args):
    d, temps = args
    return pretreatment_member(pre.PretreatmentKinetics.from_dict(d), temps)


# ------------------------------------------------------------ hidrólise
def hydrolysis_conditions(n: int, seed: int) -> np.ndarray:
    keys = ("cellulose", "hemicellulose", "solids", "enzyme")
    lo = np.array([HYDROLYSIS[k].training_range()[0] for k in keys], float)
    hi = np.array([HYDROLYSIS[k].training_range()[1] for k in keys], float)
    lo[:2] /= 100; hi[:2] /= 100                     # % -> fração
    lo[3], hi[3] = np.log(lo[3]), np.log(hi[3])      # enzima em escala log
    u = qmc.Sobol(4, scramble=True, seed=seed).random(n)
    x = lo + u * (hi - lo)
    x[:, 3] = np.exp(x[:, 3])
    return x  # celulose, xilana, sólidos, enzima


def _hyd_job(args):
    d, X = args
    model = hyd.HydrolysisKinetics.from_dict(d)
    Y = np.zeros((len(X), len(HYD_TIMES), 3))
    for i, (cel, hemi, S, E) in enumerate(X):
        s = model.simulate(S, E, cel, hemi, HYD_TIMES)
        Y[i] = np.column_stack([s["glucose"], s["xylose"], s["cellobiose"]])
    return Y


def main() -> None:
    SYNTHETIC.mkdir(parents=True, exist_ok=True)
    pre_members = members(pre.load_kinetics(MODELS / "pretreatment_kinetics.json"))
    hyd_members = members(hyd.load_kinetics(MODELS / "hydrolysis_kinetics.json"))

    # --- pré-tratamento
    t0 = time.time()
    lo, hi = PRETREATMENT["temperature"].training_range()
    temps = np.linspace(lo, hi, PRE_N_T)
    with ProcessPoolExecutor(WORKERS) as ex:
        results = list(ex.map(_pre_job, [(p.to_dict(), temps) for p in pre_members]))
    arrays = {"temperature_C": temps, "times_min": PRE_TIMES}
    for k in results[0]:
        arrays[k] = np.stack([r[k] for r in results])  # membros x T x tempos x estados
    np.savez_compressed(SYNTHETIC / "pretreatment.npz", **arrays)
    # Versão legível do membro central, para a composição da palha de calibração
    rows = []
    best = pre_members[0]
    for T in temps[::4]:
        s = best.simulate(T, PRE_TIMES)
        rows.append(pd.DataFrame({"temperature_C": T, "time_min": PRE_TIMES, **{k: s[k] for k in pre.OUTPUTS}}))
    pd.concat(rows).round(5).to_csv(SYNTHETIC / "pretreatment_central.csv.gz", index=False)
    print(f"pré-tratamento: {len(pre_members)} membros x {len(temps)} temperaturas x {len(PRE_TIMES)} tempos "
          f"({time.time() - t0:.0f} s)")

    # --- hidrólise
    t0 = time.time()
    X_train = hydrolysis_conditions(HYD_N_TRAIN, SEED)
    X_val = hydrolysis_conditions(HYD_N_VAL, SEED + 1000)
    X = np.vstack([X_train, X_val])
    chunks = np.array_split(np.arange(len(X)), WORKERS)
    Y_all = []
    with ProcessPoolExecutor(WORKERS) as ex:
        for p in hyd_members:
            parts = list(ex.map(_hyd_job, [(p.to_dict(), X[c]) for c in chunks]))
            Y_all.append(np.concatenate(parts))
    Y_all = np.stack(Y_all)  # membros x condições x tempos x (glicose, xilose, celobiose)
    np.savez_compressed(SYNTHETIC / "hydrolysis.npz", X=X, times_h=HYD_TIMES, Y=Y_all,
                        n_train=HYD_N_TRAIN, columns_X=np.array(["cellulose_frac", "hemicellulose_frac", "solids_g_L", "enzyme_g_L"]),
                        columns_Y=np.array(["glucose_g_L", "xylose_g_L", "cellobiose_g_L"]))
    n, nt = len(X), len(HYD_TIMES)
    df = pd.DataFrame({
        "condition": np.repeat(np.arange(n), nt), "split": np.repeat(np.where(np.arange(n) < HYD_N_TRAIN, "train", "val"), nt),
        "cellulose_frac": np.repeat(X[:, 0], nt), "hemicellulose_frac": np.repeat(X[:, 1], nt),
        "solids_g_L": np.repeat(X[:, 2], nt), "enzyme_g_L": np.repeat(X[:, 3], nt),
        "time_h": np.tile(HYD_TIMES, n),
        "glucose_g_L": Y_all[0, :, :, 0].ravel(), "xylose_g_L": Y_all[0, :, :, 1].ravel(),
        "cellobiose_g_L": Y_all[0, :, :, 2].ravel(),
    })
    df.round(5).to_csv(SYNTHETIC / "hydrolysis_central.csv.gz", index=False)
    print(f"hidrólise: {len(hyd_members)} membros x {n} condições x {nt} tempos ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
