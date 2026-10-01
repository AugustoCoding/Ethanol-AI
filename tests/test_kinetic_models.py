"""Modelos cinéticos: propriedades físicas e qualidade da calibração."""
import numpy as np
import pandas as pd

from ethanol_ai.paths import EXPERIMENTAL
from ethanol_ai.pretreatment import rate_matrix
from ethanol_ai.hydrolysis import HydrolysisKinetics, free_enzyme
from ethanol_ai.constants import GLUCAN_TO_GLUCOSE, GLUCAN_TO_CELLOBIOSE


# ------------------------------------------------------------- pré-tratamento
def test_rate_matrix_conserves_mass():
    K = rate_matrix(np.array([0.01, 0.1, 0.05, 0.02, 0.03, 0.004]))
    assert np.allclose(K.sum(axis=0), 0.0)


def test_pretreatment_mass_balance(pre_kinetics):
    best, _, _ = pre_kinetics
    for branch in ("hemi", "cell"):
        y = best.simulate_branch(205.0, np.linspace(0, 60, 31), branch, polymer_g_L=25.0, solids_g_L=100.0)
        assert np.allclose(y.sum(axis=1), y[0].sum(), rtol=1e-8)
        assert (y >= -1e-10).all()
        assert (np.diff(y[:, 0]) <= 1e-12).all(), "o polímero só pode diminuir"


def test_rates_increase_with_temperature(pre_kinetics):
    best, _, _ = pre_kinetics
    for branch in ("hemi", "cell"):
        assert (best.k(210.0, branch) > best.k(180.0, branch)).all()


def test_pretreatment_fit_quality(pre_kinetics):
    _, _, meta = pre_kinetics
    m = meta["metrics_all_points"]
    assert m["xylooligomers"]["rmse"] < 2.0
    assert m["furfural"]["rmse"] < 1.0
    assert m["xylose"]["r2"] > 0.8


# ------------------------------------------------------------- hidrólise
def test_langmuir_equilibrium():
    E_T, S, E_max, k_ad = 0.35, 150.0, 0.03, 20.0
    E_F = free_enzyme(E_T, S, E_max, k_ad)
    E_B = E_T - E_F
    assert 0 < E_F < E_T
    assert np.isclose(E_B / S, E_max * k_ad * E_F / (1 + k_ad * E_F))


def test_hydrolysis_mass_balance(hyd_kinetics):
    best, _, _ = hyd_kinetics
    S, cel, hemi = 150.0, 0.6655, 0.082
    s = best.simulate(S, 0.35, cel, hemi, np.array([0, 6, 24, 72, 120.0]))
    consumed = S * cel - s["cellulose"]
    produced = s["glucose"] / GLUCAN_TO_GLUCOSE + s["cellobiose"] / GLUCAN_TO_CELLOBIOSE
    assert np.allclose(consumed, produced, rtol=5e-3, atol=1e-3)
    assert (np.diff(s["glucose"]) >= -1e-9).all()


def test_calibration_beats_literature_on_experiments(hyd_kinetics):
    best, _, meta = hyd_kinetics
    assert meta["calibrated_rmse"]["glucose"] < meta["literature_rmse"]["original"]["glucose"]
    d = pd.read_csv(EXPERIMENTAL / "hydrolysis.csv")
    lit = HydrolysisKinetics()
    err_cal, err_lit = [], []
    for _, g in d.groupby("condition"):
        r = g.iloc[0]
        args = (r.solids_g_L, r.enzyme_g_L, r.cellulose_frac, r.hemicellulose_frac, g.time_h.values)
        err_cal.append(best.simulate(*args)["glucose"] - g.glucose_g_L.values)
        err_lit.append(lit.simulate(*args)["glucose"] - g.glucose_g_L.values)
    rmse = lambda e: np.sqrt(np.mean(np.concatenate(e) ** 2))
    assert rmse(err_cal) < rmse(err_lit)


def test_experimental_dataset_corrections():
    d = pd.read_csv(EXPERIMENTAL / "hydrolysis.csv")
    s200 = d[d.condition == "S200_E10FPU"].enzyme_g_L.iloc[0]
    assert abs(s200 - 0.2333) < 1e-3, "carga de enzima do ensaio 20% sólidos deve ser 0,233 g/L"
    assert d.cellobiose_censored.sum() > 0
