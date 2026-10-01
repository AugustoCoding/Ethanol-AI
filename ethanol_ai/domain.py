"""
Faixas de operação: fonte única para a geração de dados, os sliders do app e os avisos.

Três faixas por variável:
    validated  onde há suporte experimental (ou, para composição e sólidos no pré-tratamento,
               onde a premissa do modelo é razoável; ver docs/METODOLOGIA.md). Fora dela o app
               marca a entrada como extrapolação.
    slider     faixa oferecida no app (por padrão, a validada ampliada em 30% da largura de cada lado)
    training   faixa dos dados sintéticos usados no treino das redes; contém a do slider com folga,
               para que a rede nunca extrapole dentro do app
"""
from __future__ import annotations

import math
from dataclasses import dataclass

EXTRAPOLATION_MARGIN = 0.30


@dataclass(frozen=True)
class Variable:
    key: str
    label: str
    unit: str
    validated: tuple[float, float]
    default: float
    step: float
    fmt: str
    slider: tuple[float, float] | None = None
    training: tuple[float, float] | None = None
    floor: float | None = None
    help: str | None = None

    def slider_range(self) -> tuple[float, float]:
        if self.slider is not None:
            return self.slider
        lo, hi = self.validated
        pad = (hi - lo) * EXTRAPOLATION_MARGIN
        lo_ext = math.floor((lo - pad) / self.step + 1e-9) * self.step
        hi_ext = math.ceil((hi + pad) / self.step - 1e-9) * self.step
        if self.floor is not None:
            lo_ext = max(lo_ext, self.floor)
        return round(lo_ext, 6), round(hi_ext, 6)

    def training_range(self) -> tuple[float, float]:
        if self.training is not None:
            return self.training
        lo, hi = self.slider_range()
        pad = 0.1 * (hi - lo)
        return lo - pad, hi + pad

    def is_validated(self, value: float) -> bool:
        lo, hi = self.validated
        return lo - 1e-9 <= value <= hi + 1e-9


# --- Pré-tratamento hidrotérmico (Rocha et al., 2017: 180/195/210 °C, 0-40 min, 100 g/L, uma palha)
PRETREATMENT = {
    "temperature": Variable("temperature", "Temperature", "°C", (180.0, 210.0), 195.0, 0.5, "%.1f",
                            training=(160.0, 230.0)),
    "time": Variable("time", "Time", "min", (0.0, 40.0), 15.0, 0.5, "%.1f", floor=1.0,
                     help="Time after the reactor reaches the set temperature (the heat-up ramp is included)."),
    "solids": Variable("solids", "Solids loading", "g/L", (80.0, 120.0), 100.0, 1.0, "%.0f", slider=(50.0, 150.0),
                       help="Experiments used 100 g/L (1:10 w/v). First-order kinetics assumed independent of loading."),
    "cellulose": Variable("cellulose", "Cellulose", "%", (31.0, 39.0), 34.8, 0.1, "%.1f", slider=(25.0, 45.0),
                          help="Calibration straw: 34.8% cellulose."),
    "hemicellulose": Variable("hemicellulose", "Hemicellulose", "%", (20.0, 26.0), 23.0, 0.1, "%.1f", slider=(15.0, 30.0),
                              help="Calibration straw: 23.0% hemicellulose."),
}

# --- Hidrólise enzimática (8 ensaios: 150/200 g/L, 0,0875-1,05 g/L de enzima, uma composição)
HYDROLYSIS = {
    "cellulose": Variable("cellulose", "Cellulose", "%", (62.0, 71.0), 66.6, 0.1, "%.1f", slider=(50.0, 75.0),
                          training=(45.0, 80.0), help="Calibration solid: 66.55% cellulose."),
    "hemicellulose": Variable("hemicellulose", "Hemicellulose (xylan)", "%", (7.0, 9.5), 8.2, 0.1, "%.1f",
                              slider=(3.0, 15.0), training=(1.0, 18.0), help="Calibration solid: 8.2% xylan."),
    "solids": Variable("solids", "Solids loading", "g/L", (150.0, 200.0), 150.0, 1.0, "%.0f", slider=(50.0, 250.0),
                       training=(30.0, 280.0)),
    "enzyme": Variable("enzyme", "Enzyme loading", "g/L", (0.0875, 1.05), 0.35, 0.01, "%.2f", floor=0.01,
                       training=(0.005, 1.8)),
    "time": Variable("time", "Reaction time", "h", (0.0, 96.0), 72.0, 0.5, "%.1f", floor=1.0, training=(0.0, 130.0)),
}

# Massa de proteína enzimática por FPU, obtida dos ensaios (para mostrar a dose em FPU/g celulose)
G_PROTEIN_PER_FPU = 0.175 / (10 * 150 * 0.6655)


def fpu_per_g_cellulose(enzyme_g_L: float, solids_g_L: float, cellulose_frac: float) -> float:
    return enzyme_g_L / (G_PROTEIN_PER_FPU * solids_g_L * cellulose_frac)
