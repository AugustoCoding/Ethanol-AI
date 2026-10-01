"""
Modelo cinético da hidrólise enzimática da palha pré-tratada.

Estrutura de Angarita et al. (2015), Biochem. Eng. J. 104:10-19, da família de Kadam et al. (2004):

    r1: celulose    -> celobiose  (enzima adsorvida, inibida por celobiose, glicose e xilose)
    r2: celulose    -> glicose    (enzima adsorvida, idem)
    r3: celobiose   -> glicose    (enzima livre, Michaelis-Menten com inibição competitiva)
    r4: hemicelulose -> xilose    (enzima adsorvida, idem)

    adsorção de Langmuir: E_B / S = E_max k_ad E_F / (1 + k_ad E_F), com E_T = E_F + E_B
    reatividade do substrato: R_S = alfa (S / S0)^gamma  (gamma = 1 no original)
    fração reativa: só f_C da celulose e f_H da hemicelulose são hidrolisáveis; o restante fica no
        sólido (conta em S e "sequestra" enzima adsorvida). f_C = f_H = 1 no original
    dois pools de celulose: uma fração f_fast da celulose reativa é hidrolisada m_fast vezes mais
        rápido (celulose de fácil acesso). f_fast = 0 no original

Diferenças em relação ao código original (research/02_enzymatic_hydrolysis):
    - equilíbrio de adsorção resolvido analiticamente (raiz positiva de uma quadrática)
    - opção adsorption="quasi_steady": o equilíbrio é recalculado com o substrato atual a cada
      instante (Kadam et al.); "initial" reproduz o original (adsorção fixada em t = 0)
    - parâmetros calibrados com os 8 ensaios do repositório (pipeline/02_calibrate_hydrolysis.py)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from .constants import GLUCAN_TO_GLUCOSE, GLUCAN_TO_CELLOBIOSE, CELLOBIOSE_TO_GLUCOSE, XYLAN_TO_XYLOSE

# Valores publicados (Angarita et al., 2015), usados como ponto de partida e priori da calibração
LITERATURE = {
    "k_1r": 0.177, "k_2r": 8.81, "k_3r": 201.0, "k_4r": 16.34,
    "k_11G2": 0.402, "k_11G": 2.71, "k_11X": 2.15,
    "k_21G2": 119.6, "k_21G": 4.69, "k_21X": 0.095,
    "k_3M": 26.6, "k_31G": 11.06, "k_31X": 1.023,
    "k_41G2": 16.25, "k_41G": 4.0, "k_41X": 154.0,
    "k_ad": 7.16, "E_max": 8.32e-3, "alfa": 1.0, "gamma": 1.0,
    "f_C": 1.0, "f_H": 1.0, "f_fast": 0.0, "m_fast": 1.0,
}

OUTPUTS = ("glucose", "xylose", "cellobiose", "cellulose", "hemicellulose")


def free_enzyme(E_T: float, S: float, E_max: float, k_ad: float) -> float:
    """Enzima livre no equilíbrio de Langmuir (raiz positiva de k_ad E_F² + b E_F - E_T = 0)."""
    b = 1.0 + k_ad * (S * E_max - E_T)
    return (-b + np.sqrt(b * b + 4.0 * k_ad * E_T)) / (2.0 * k_ad)


@dataclass
class HydrolysisKinetics:
    params: dict = field(default_factory=lambda: dict(LITERATURE))
    adsorption: str = "initial"  # "initial" (original) ou "quasi_steady"

    def rhs(self, t, y, S0, E_T, E_F0):
        p = self.params
        Cf, Cs, G2, G, H, X, S = np.maximum(y, 0.0)
        S = max(S, 1e-9)
        if self.adsorption == "quasi_steady":
            E_F = free_enzyme(E_T, S, p["E_max"], p["k_ad"])
        else:
            E_F = E_F0
        E_B = E_T - E_F
        R_S = p["alfa"] * (S / S0) ** p.get("gamma", 1.0)
        # Enzima adsorvida dividida entre os pools na proporção da massa de cada um
        inh1 = 1 + G2 / p["k_11G2"] + G / p["k_11G"] + X / p["k_11X"]
        inh2 = 1 + G2 / p["k_21G2"] + G / p["k_21G"] + X / p["k_21X"]
        m = p.get("m_fast", 1.0)
        r1f = p["k_1r"] * m * (E_B * Cf / S) * R_S * Cf / inh1
        r2f = p["k_2r"] * m * (E_B * Cf / S) * R_S * Cf / inh2
        r1s = p["k_1r"] * (E_B * Cs / S) * R_S * Cs / inh1
        r2s = p["k_2r"] * (E_B * Cs / S) * R_S * Cs / inh2
        r1, r2 = r1f + r1s, r2f + r2s
        Ebh = E_B * H / S
        r3 = p["k_3r"] * E_F * G2 / (p["k_3M"] * (1 + G / p["k_31G"] + X / p["k_31X"]) + G2)
        r4 = p["k_4r"] * Ebh * R_S * H / (1 + G2 / p["k_41G2"] + G / p["k_41G"] + X / p["k_41X"])
        return [
            -r1f - r2f,
            -r1s - r2s,
            GLUCAN_TO_CELLOBIOSE * r1 - r3,
            GLUCAN_TO_GLUCOSE * r2 + CELLOBIOSE_TO_GLUCOSE * r3,
            -r4,
            XYLAN_TO_XYLOSE * r4,
            -r1 - r2 - r4,
        ]

    def simulate(self, solids_g_L: float, enzyme_g_L: float, cellulose_frac: float,
                 hemicellulose_frac: float, times_h) -> dict[str, np.ndarray]:
        """Perfis em g/L. Celulose e hemicelulose são o sólido remanescente (massa anidra)."""
        times = np.atleast_1d(np.asarray(times_h, dtype=float))
        S0 = float(solids_g_L)
        E_F0 = free_enzyme(enzyme_g_L, S0, self.params["E_max"], self.params["k_ad"])
        f_C, f_H = self.params.get("f_C", 1.0), self.params.get("f_H", 1.0)
        inert_C, inert_H = S0 * cellulose_frac * (1 - f_C), S0 * hemicellulose_frac * (1 - f_H)
        f_fast = self.params.get("f_fast", 0.0)
        C_reactive = S0 * cellulose_frac * f_C
        y0 = [C_reactive * f_fast, C_reactive * (1 - f_fast), 0.0, 0.0, S0 * hemicellulose_frac * f_H, 0.0, S0]
        sol = solve_ivp(self.rhs, (0.0, float(times.max())), y0, t_eval=np.sort(np.unique(times)),
                        method="LSODA", rtol=1e-6, atol=1e-9, args=(S0, enzyme_g_L, E_F0))
        if not sol.success:
            raise RuntimeError(f"integração falhou: {sol.message}")
        idx = np.searchsorted(sol.t, times)
        Y = np.maximum(sol.y[:, idx], 0.0)
        return {"cellulose": Y[0] + Y[1] + inert_C, "cellobiose": Y[2], "glucose": Y[3],
                "hemicellulose": Y[4] + inert_H, "xylose": Y[5]}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HydrolysisKinetics":
        return cls(**d)


def load_kinetics(path: Path) -> tuple[HydrolysisKinetics, list[HydrolysisKinetics], dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    best = HydrolysisKinetics.from_dict(data["best"])
    ensemble = [HydrolysisKinetics.from_dict(p) for p in data.get("bootstrap", [])]
    return best, ensemble, data.get("meta", {})
