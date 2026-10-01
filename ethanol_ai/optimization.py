"""
Busca de condições de operação a partir das redes (antiga "otimização reversa").

Em vez de uma soma ponderada de objetivos com pesos arbitrários, cada pergunta é formulada
como objetivo + restrições explícitas, e o espaço de decisão (2 variáveis) é avaliado por
inteiro numa grade. Isso é instantâneo com as redes e mostra o compromisso todo, não só um ponto.

    hidrólise:       menor carga de enzima que atinge um rendimento de glicose dentro de um tempo
    pré-tratamento:  maior recuperação dos açúcares da hemicelulose no licor (xilose + oligômeros),
                     com limites de furfural e de perda de celulose

A versão "conservadora" usa o percentil 10 do ensemble (incerteza da calibração).
O ponto recomendado deve ser conferido com o modelo mecanístico (o app faz isso).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import GLUCAN_TO_GLUCOSE, XYLAN_TO_XYLOSE
from .surrogate import HydrolysisSurrogate, PretreatmentSurrogate

CONSERVATIVE_PERCENTILE = 10


# ------------------------------------------------------------------ hidrólise
@dataclass
class EnzymeFrontier:
    enzyme_g_L: np.ndarray        # grade de cargas de enzima
    times_h: np.ndarray           # grade de tempos
    yield_central: np.ndarray     # rendimento de glicose (enzima x tempo)
    yield_conservative: np.ndarray
    t_reach_central: np.ndarray   # tempo para atingir o alvo, por carga de enzima (nan = não atinge)
    t_reach_conservative: np.ndarray
    best_central: float | None    # menor enzima que atinge o alvo até t_max
    best_conservative: float | None


def _first_time(yields: np.ndarray, times: np.ndarray, target: float) -> np.ndarray:
    """Primeiro tempo (interpolado) em que o rendimento atinge o alvo, para cada linha."""
    out = np.full(yields.shape[0], np.nan)
    for i, y in enumerate(yields):
        j = np.argmax(y >= target)
        if y[j] >= target:
            if j == 0:
                out[i] = times[0]
            else:
                out[i] = np.interp(target, [y[j - 1], y[j]], [times[j - 1], times[j]])
    return out


def enzyme_frontier(surr: HydrolysisSurrogate, cellulose_frac: float, hemicellulose_frac: float,
                    solids_g_L: float, target_yield: float, t_max_h: float,
                    enzyme_range: tuple[float, float], n_enzyme: int = 120) -> EnzymeFrontier:
    E = np.geomspace(*enzyme_range, n_enzyme)
    t = np.arange(0.0, t_max_h + 1e-9, 0.25)
    n = len(E)
    args = (np.full(n, solids_g_L), E, np.full(n, cellulose_frac), np.full(n, hemicellulose_frac), t)
    theo = GLUCAN_TO_GLUCOSE * solids_g_L * cellulose_frac
    central = surr.simulate(*args, member=0)["glucose"] / theo
    members = np.array([surr.simulate(*args, member=m)["glucose"] / theo for m in range(1, len(surr.members))])
    conservative = np.percentile(members, CONSERVATIVE_PERCENTILE, axis=0) if len(members) else central
    tc = _first_time(central, t, target_yield)
    tk = _first_time(conservative, t, target_yield)
    ok_c, ok_k = ~np.isnan(tc), ~np.isnan(tk)
    return EnzymeFrontier(
        enzyme_g_L=E, times_h=t, yield_central=central, yield_conservative=conservative,
        t_reach_central=tc, t_reach_conservative=tk,
        best_central=float(E[ok_c][0]) if ok_c.any() else None,
        best_conservative=float(E[ok_k][0]) if ok_k.any() else None,
    )


# ------------------------------------------------------------------ pré-tratamento
@dataclass
class PretreatmentMap:
    temperature_C: np.ndarray
    times_min: np.ndarray
    c5_recovery: np.ndarray      # fração da xilana inicial recuperada como xilose + xilo-oligômeros (T x t)
    furfural_g_L: np.ndarray
    cellulose_loss: np.ndarray   # fração da celulose inicial perdida
    feasible: np.ndarray
    best: dict | None


def pretreatment_map(surr: PretreatmentSurrogate, solids_g_L: float, cellulose_frac: float,
                     hemicellulose_frac: float, temperature_range: tuple[float, float],
                     time_range: tuple[float, float], furfural_max_g_L: float, cellulose_loss_max: float,
                     n_T: int = 97, n_t: int = 105) -> PretreatmentMap:
    T = np.linspace(*temperature_range, n_T)
    t = np.linspace(*time_range, n_t)
    rec = np.zeros((n_T, n_t)); fur = np.zeros_like(rec); closs = np.zeros_like(rec)
    H0, C0 = solids_g_L * hemicellulose_frac, solids_g_L * cellulose_frac
    for i, Ti in enumerate(T):
        s = surr.simulate(Ti, t, solids_g_L, cellulose_frac, hemicellulose_frac)
        rec[i] = (s["xylose"] + s["xylooligomers"]) / XYLAN_TO_XYLOSE / H0
        fur[i] = s["furfural"]
        closs[i] = 1 - s["cellulose"] / C0
    feasible = (fur <= furfural_max_g_L) & (closs <= cellulose_loss_max)
    best = None
    if feasible.any():
        masked = np.where(feasible, rec, -np.inf)
        i, j = np.unravel_index(np.argmax(masked), masked.shape)
        best = {"temperature_C": float(T[i]), "time_min": float(t[j]), "c5_recovery": float(rec[i, j]),
                "furfural_g_L": float(fur[i, j]), "cellulose_loss": float(closs[i, j])}
    return PretreatmentMap(T, t, rec, fur, closs, feasible, best)
