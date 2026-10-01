"""Redes substitutas: fidelidade ao modelo mecanístico e garantias físicas."""
import numpy as np

from ethanol_ai.domain import PRETREATMENT, HYDROLYSIS
from ethanol_ai.constants import GLUCAN_TO_GLUCOSE, XYLAN_TO_XYLOSE
from ethanol_ai.optimization import enzyme_frontier, pretreatment_map

RNG = np.random.default_rng(0)


def _sample(var, n, scale=1.0):
    lo, hi = var.slider_range()
    return RNG.uniform(lo, hi, n) * scale


def test_pretreatment_surrogate_matches_kinetic_model(pre_kinetics, pre_surrogate):
    best, _, _ = pre_kinetics
    t = np.linspace(0, PRETREATMENT["time"].slider_range()[1], 53)
    errors = []
    for T, S, c, h in zip(_sample(PRETREATMENT["temperature"], 15), _sample(PRETREATMENT["solids"], 15),
                          _sample(PRETREATMENT["cellulose"], 15, 0.01), _sample(PRETREATMENT["hemicellulose"], 15, 0.01)):
        a = best.simulate(T, t, S, c, h)
        b = pre_surrogate.simulate(T, t, S, c, h)
        for k in ("cellulose", "hemicellulose", "xylose", "xylooligomers", "furfural", "glucose"):
            errors.append(a[k] - b[k])
    e = np.concatenate(errors)
    # a rede deve errar bem menos que o próprio modelo cinético contra os experimentos (~0,4-1,5 g/L)
    assert np.sqrt(np.mean(e ** 2)) < 0.1, f"RMSE {np.sqrt(np.mean(e ** 2)):.3f} g/L"
    assert np.abs(e).max() < 0.6, f"erro máximo {np.abs(e).max():.3f} g/L"


def test_pretreatment_surrogate_guarantees(pre_surrogate):
    t = np.linspace(0, 52, 105)
    for T in (171.0, 195.0, 219.0):
        s = pre_surrogate.simulate(T, t, 100.0, 0.348, 0.230)
        assert min(v.min() for v in s.values()) >= -1e-9
        assert (np.diff(s["cellulose"]) <= 1e-9).all() and (np.diff(s["hemicellulose"]) <= 1e-9).all()


def test_hydrolysis_surrogate_matches_kinetic_model(hyd_kinetics, hyd_surrogate):
    best, _, _ = hyd_kinetics
    t = np.arange(0, 121, 2.0)
    n = 20
    cel = _sample(HYDROLYSIS["cellulose"], n, 0.01)
    hem = _sample(HYDROLYSIS["hemicellulose"], n, 0.01)
    S = _sample(HYDROLYSIS["solids"], n)
    E = np.exp(RNG.uniform(*np.log(HYDROLYSIS["enzyme"].slider_range()), n))
    P = hyd_surrogate.simulate(S, E, cel, hem, t)
    err = []
    for i in range(n):
        m = best.simulate(S[i], E[i], cel[i], hem[i], t)
        err.append(P["glucose"][i] - m["glucose"])
    rmse = np.sqrt(np.mean(np.concatenate(err) ** 2))
    assert rmse < 1.5, f"RMSE de glicose {rmse:.2f} g/L"


def test_hydrolysis_surrogate_guarantees(hyd_surrogate):
    n = 200
    cel = _sample(HYDROLYSIS["cellulose"], n, 0.01)
    hem = _sample(HYDROLYSIS["hemicellulose"], n, 0.01)
    S = _sample(HYDROLYSIS["solids"], n)
    E = _sample(HYDROLYSIS["enzyme"], n)
    t = np.linspace(0, 125, 251)
    P = hyd_surrogate.simulate(S, E, cel, hem, t)
    assert np.allclose(P["glucose"][:, 0], 0) and np.allclose(P["xylose"][:, 0], 0)
    assert (np.diff(P["xylose"], axis=1) >= -1e-9).all()
    assert (P["glucose"] / (GLUCAN_TO_GLUCOSE * (S * cel)[:, None]) <= 1 + 1e-9).all()
    assert (P["xylose"] / (XYLAN_TO_XYLOSE * (S * hem)[:, None]) <= 1 + 1e-9).all()
    assert (P["cellobiose"] >= 0).all()


def test_enzyme_frontier_is_consistent(hyd_kinetics, hyd_surrogate):
    best, _, _ = hyd_kinetics
    fr = enzyme_frontier(hyd_surrogate, 0.6655, 0.082, 150.0, 0.55, 48.0, HYDROLYSIS["enzyme"].slider_range())
    assert fr.best_central is not None
    # mais enzima nunca deve exigir mais tempo
    ok = ~np.isnan(fr.t_reach_central)
    assert (np.diff(fr.t_reach_central[ok]) <= 0.25 + 1e-9).all()
    # o modelo mecanístico confirma o alvo no ponto recomendado (tolerância de 5 p.p.)
    y = best.simulate(150.0, fr.best_central, 0.6655, 0.082, [48.0])["glucose"][0] / (GLUCAN_TO_GLUCOSE * 150 * 0.6655)
    assert y > 0.50
    if fr.best_conservative is not None:
        assert fr.best_conservative >= fr.best_central


def test_pretreatment_map_respects_constraints(pre_surrogate):
    mp = pretreatment_map(pre_surrogate, 100.0, 0.348, 0.230, (171.0, 219.0), (0.0, 52.0), 1.0, 0.10, n_T=25, n_t=27)
    assert mp.best is not None
    assert mp.best["furfural_g_L"] <= 1.0 and mp.best["cellulose_loss"] <= 0.10
    assert mp.best["c5_recovery"] == mp.c5_recovery[mp.feasible].max()
