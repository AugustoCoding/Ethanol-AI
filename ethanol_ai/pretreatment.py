"""
Modelo cinético do pré-tratamento hidrotérmico da palha de cana.

Esquema de reações (Rocha et al., 2017, Bioresource Technology 228:176-185), aplicado a
cada fração polimérica, todas de primeira ordem:

    polímero --k2--> oligômeros --k3--> monômeros --k4--> furano --k6--> degradação
    polímero --k1--> monômeros  --k5--> degradação

    hemicelulose: H -> XOS -> xilose (+ arabinose) -> furfural
    celulose:     C -> GOS -> glicose (+ celobiose) -> HMF

Diferenças em relação ao modelo original:
    - todos os estados em massa anidra do polímero (g/L de glucana ou xilana); as saídas são
      convertidas para as unidades medidas com os fatores estequiométricos
    - dependência de Arrhenius global, k(T) = k_ref * exp(-Ea/R (1/T - 1/T_ref)), ajustada às
      três temperaturas de uma vez (antes: k ajustado por temperatura e interpolado linearmente)
    - aquecimento do reator modelado como rampa linear até a temperatura de operação; isso explica
      os produtos já presentes em t = 0 (fim do aquecimento) e torna o modelo contínuo em T
    - açúcares livres solúveis (dos extrativos) como concentração inicial no licor

Como o sistema é linear, a solução é exata: y(t) = expm(K t) y0.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from .constants import (
    R_GAS, GLUCAN_TO_GLUCOSE, XYLAN_TO_XYLOSE, HEXOSAN_TO_HMF, PENTOSAN_TO_FURFURAL,
)

# Ordem dos estados em cada fração
STATES = ("polymer", "oligomers", "monomers", "furan", "degradation")

# Saídas do modelo (unidades medidas, g/L)
OUTPUTS = (
    "cellulose", "hemicellulose",                      # sólido remanescente (massa anidra)
    "glucose", "glucooligomers", "hmf",                # licor, fração celulósica
    "xylose", "xylooligomers", "furfural",             # licor, fração hemicelulósica
    "cellulose_degradation", "hemicellulose_degradation",  # produtos de degradação (massa anidra)
)

# Massa de ácido fórmico por massa de hexosana degradada (1 mol de fórmico por hexose,
# rota HMF -> ácido levulínico + ácido fórmico); usado como marcador da degradação da celulose
HEXOSAN_TO_FORMIC = 46.025 / 162.141


def rate_matrix(k: np.ndarray) -> np.ndarray:
    """Matriz K (5x5) do sistema dy/dt = K y para k = [k1..k6]."""
    k1, k2, k3, k4, k5, k6 = k
    return np.array([
        [-(k1 + k2), 0.0, 0.0, 0.0, 0.0],
        [k2, -k3, 0.0, 0.0, 0.0],
        [k1, k3, -(k4 + k5), 0.0, 0.0],
        [0.0, 0.0, k4, -k6, 0.0],
        [0.0, 0.0, k5, k6, 0.0],
    ])


@dataclass
class PretreatmentKinetics:
    """Parâmetros do modelo. k em 1/min, Ea em kJ/mol, frações de açúcar livre em g/g de sólido."""
    log_k_ref_hemi: list[float]
    ea_hemi: list[float]
    log_k_ref_cell: list[float]
    ea_cell: list[float]
    heating_rate: float = 3.0              # °C/min durante a rampa de aquecimento
    free_sugars: dict[str, float] = field(default_factory=lambda: {
        "glucose": 0.0, "glucooligomers": 0.0, "xylose": 0.0, "xylooligomers": 0.0,
    })                                     # massa anidra por massa de sólido seco
    t_ref_C: float = 195.0
    ramp_start_C: float = 100.0            # abaixo disso as reações são desprezíveis
    ramp_step_C: float = 2.0

    # --- parâmetros -> taxas
    def k(self, temperature_C: float, branch: str) -> np.ndarray:
        log_k_ref = np.asarray(self.log_k_ref_hemi if branch == "hemi" else self.log_k_ref_cell)
        ea = np.asarray(self.ea_hemi if branch == "hemi" else self.ea_cell)
        T, T_ref = temperature_C + 273.15, self.t_ref_C + 273.15
        return np.exp(log_k_ref - ea / R_GAS * (1.0 / T - 1.0 / T_ref))

    def end_of_heating(self, temperature_C: float, branch: str, y0: np.ndarray) -> np.ndarray:
        """Integra a rampa linear de aquecimento (ramp_start_C -> temperature_C)."""
        if temperature_C <= self.ramp_start_C:
            return y0
        edges = np.arange(self.ramp_start_C, temperature_C, self.ramp_step_C)
        edges = np.append(edges, temperature_C)
        mids, dts = 0.5 * (edges[:-1] + edges[1:]), np.diff(edges) / self.heating_rate
        # Um passo isotérmico por intervalo de temperatura (regra do ponto médio)
        steps = expm(np.stack([rate_matrix(self.k(m, branch)) * dt for m, dt in zip(mids, dts)]))
        y = y0.copy()
        for M in steps:
            y = M @ y
        return y

    # --- simulação
    def simulate_branch(self, temperature_C: float, times_min: np.ndarray, branch: str,
                        polymer_g_L: float, solids_g_L: float) -> np.ndarray:
        """Estados (n_tempos x 5) em massa anidra para uma fração."""
        fs = self.free_sugars
        if branch == "hemi":
            y0 = np.array([polymer_g_L, fs["xylooligomers"] * solids_g_L, fs["xylose"] * solids_g_L, 0.0, 0.0])
        else:
            y0 = np.array([polymer_g_L, fs["glucooligomers"] * solids_g_L, fs["glucose"] * solids_g_L, 0.0, 0.0])
        y_start = self.end_of_heating(temperature_C, branch, y0)
        K = rate_matrix(self.k(temperature_C, branch))
        times = np.atleast_1d(np.asarray(times_min, dtype=float))
        return np.einsum("tij,j->ti", expm(K[None, :, :] * times[:, None, None]), y_start)

    def simulate(self, temperature_C: float, times_min, solids_g_L: float = 100.0,
                 cellulose_frac: float = 0.348, hemicellulose_frac: float = 0.230) -> dict[str, np.ndarray]:
        """
        Perfis no tempo após o fim do aquecimento (t = 0 quando o reator atinge a temperatura).
        Retorna g/L nas unidades medidas (monômeros, oligômeros como monômero, furanos).
        """
        h = self.simulate_branch(temperature_C, times_min, "hemi", solids_g_L * hemicellulose_frac, solids_g_L)
        c = self.simulate_branch(temperature_C, times_min, "cell", solids_g_L * cellulose_frac, solids_g_L)
        return {
            "cellulose": c[:, 0],
            "glucooligomers": c[:, 1] * GLUCAN_TO_GLUCOSE,
            "glucose": c[:, 2] * GLUCAN_TO_GLUCOSE,
            "hmf": c[:, 3] * HEXOSAN_TO_HMF,
            "cellulose_degradation": c[:, 4],
            "hemicellulose": h[:, 0],
            "xylooligomers": h[:, 1] * XYLAN_TO_XYLOSE,
            "xylose": h[:, 2] * XYLAN_TO_XYLOSE,
            "furfural": h[:, 3] * PENTOSAN_TO_FURFURAL,
            "hemicellulose_degradation": h[:, 4],
        }

    # --- vetor de parâmetros (para calibração)
    FREE_KEYS = ("glucose", "glucooligomers", "xylose", "xylooligomers")

    def to_vector(self) -> np.ndarray:
        return np.concatenate([
            self.log_k_ref_hemi, self.ea_hemi, self.log_k_ref_cell, self.ea_cell,
            [np.log(self.heating_rate)], [self.free_sugars[k] for k in self.FREE_KEYS],
        ])

    @classmethod
    def from_vector(cls, v: np.ndarray, **kwargs) -> "PretreatmentKinetics":
        v = np.asarray(v, dtype=float)
        return cls(
            log_k_ref_hemi=v[0:6].tolist(), ea_hemi=v[6:12].tolist(),
            log_k_ref_cell=v[12:18].tolist(), ea_cell=v[18:24].tolist(),
            heating_rate=float(np.exp(v[24])),
            free_sugars={k: float(x) for k, x in zip(cls.FREE_KEYS, v[25:29])},
            **kwargs,
        )

    # --- persistência
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PretreatmentKinetics":
        return cls(**d)


def load_kinetics(path: Path) -> tuple[PretreatmentKinetics, list[PretreatmentKinetics], dict]:
    """Lê o arquivo de calibração: (melhor ajuste, conjunto bootstrap, metadados)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    best = PretreatmentKinetics.from_dict(data["best"])
    ensemble = [PretreatmentKinetics.from_dict(p) for p in data.get("bootstrap", [])]
    return best, ensemble, data.get("meta", {})
