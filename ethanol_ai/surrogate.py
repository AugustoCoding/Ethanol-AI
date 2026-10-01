"""
Redes neurais substitutas (surrogates) com a física na estrutura. Inferência em NumPy.

O treino (pipeline/04_train_surrogates.py) usa TensorFlow com exatamente as mesmas
equações; aqui só se reproduz a passagem direta, a partir dos pesos exportados (.npz).
Assim o app não depende do TensorFlow.

Pré-tratamento: "rede cinética"
    entrada: temperatura
    saída da rede: 12 constantes de velocidade (k1..k6 de cada fração), a distribuição dos
        estados ao fim do aquecimento para 1 g de polímero (a, soma 1) e para 1 g de sólido
        de açúcares livres (g)
    perfil: y(t) = expm(K t) (P a + S g), solução exata do sistema linear
    garantias: balanço de massa, não negatividade, t = 0 coerente, polímero só diminui

Hidrólise: "rede de base monotônica"
    entrada: celulose, xilana, sólidos, enzima
    conversão da celulose:     X_C(t) = soma_k w_k (1 - exp(-lambda_k t)), soma w <= 1
    conversão da hemicelulose: X_H(t) = soma_k v_k (1 - exp(-mu_k t)),     soma v <= 1
    fração convertida que está como celobiose: phi(t) = sigmoide(c0 + soma_j c_j exp(-nu_j t))
    glicose = 1,111 C0 X_C (1 - phi); celobiose = 1,056 C0 X_C phi; xilose = 1,136 H0 X_H
    garantias: zero em t = 0, xilose crescente, rendimentos <= 100%, balanço de massa da celulose
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from .constants import (
    GLUCAN_TO_GLUCOSE, GLUCAN_TO_CELLOBIOSE, XYLAN_TO_XYLOSE, HEXOSAN_TO_HMF, PENTOSAN_TO_FURFURAL,
)
from .pretreatment import rate_matrix

# ---------------------------------------------------------------- utilidades


def swish(x):
    return x / (1.0 + np.exp(-x))


def softplus(x):
    return np.logaddexp(0.0, x)


def softmax(x, axis=-1):
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def mlp(weights: list[tuple[np.ndarray, np.ndarray]], x: np.ndarray) -> np.ndarray:
    """MLP com ativação swish nas camadas ocultas e saída linear."""
    h = x
    for W, b in weights[:-1]:
        h = swish(h @ W + b)
    W, b = weights[-1]
    return h @ W + b


def load_members(path: Path) -> tuple[list[list[tuple[np.ndarray, np.ndarray]]], dict]:
    """Lê os pesos de um ensemble salvo em .npz (membro 0 = ajuste central)."""
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    members = []
    for m in range(meta["n_members"]):
        layers = []
        for i in range(meta["n_layers"]):
            layers.append((data[f"m{m}_W{i}"], data[f"m{m}_b{i}"]))
        members.append(layers)
    return members, meta


def save_members(path: Path, members: list[list[tuple[np.ndarray, np.ndarray]]], meta: dict) -> None:
    arrays = {}
    for m, layers in enumerate(members):
        for i, (W, b) in enumerate(layers):
            arrays[f"m{m}_W{i}"] = W.astype(np.float32)
            arrays[f"m{m}_b{i}"] = b.astype(np.float32)
    meta = dict(meta, n_members=len(members), n_layers=len(members[0]))
    np.savez_compressed(path, meta=json.dumps(meta), **arrays)


# ---------------------------------------------------------------- pré-tratamento

T_CENTER, T_SCALE = 195.0, 15.0
FREE_SCALE = 0.01  # g/g; escala dos açúcares livres na saída da rede


def pretreatment_heads(out: np.ndarray) -> dict[str, np.ndarray]:
    """Converte a saída bruta da rede (n, 32) nos parâmetros físicos."""
    return {
        "k_hemi": np.exp(np.clip(out[:, 0:6], -20, 5)),
        "k_cell": np.exp(np.clip(out[:, 6:12], -20, 5)),
        "a_hemi": softmax(out[:, 12:17]),
        "a_cell": softmax(out[:, 17:22]),
        "g_hemi": FREE_SCALE * softplus(out[:, 22:27]),
        "g_cell": FREE_SCALE * softplus(out[:, 27:32]),
    }


class PretreatmentSurrogate:
    N_OUT = 32

    def __init__(self, members, meta):
        self.members, self.meta = members, meta

    @classmethod
    def load(cls, path: Path) -> "PretreatmentSurrogate":
        return cls(*load_members(path))

    def heads(self, temperature_C: float, member: int = 0) -> dict[str, np.ndarray]:
        x = np.array([[(temperature_C - T_CENTER) / T_SCALE]])
        return {k: v[0] for k, v in pretreatment_heads(mlp(self.members[member], x)).items()}

    def simulate(self, temperature_C: float, times_min, solids_g_L: float, cellulose_frac: float,
                 hemicellulose_frac: float, member: int = 0) -> dict[str, np.ndarray]:
        h = self.heads(temperature_C, member)
        t = np.atleast_1d(np.asarray(times_min, dtype=float))
        out = {}
        for branch, frac in (("hemi", hemicellulose_frac), ("cell", cellulose_frac)):
            y0 = solids_g_L * frac * h[f"a_{branch}"] + solids_g_L * h[f"g_{branch}"]
            K = rate_matrix(h[f"k_{branch}"])
            out[branch] = np.einsum("tij,j->ti", expm(K[None] * t[:, None, None]), y0)
        c, hh = out["cell"], out["hemi"]
        return {
            "cellulose": c[:, 0], "glucooligomers": c[:, 1] * GLUCAN_TO_GLUCOSE, "glucose": c[:, 2] * GLUCAN_TO_GLUCOSE,
            "hmf": c[:, 3] * HEXOSAN_TO_HMF, "cellulose_degradation": c[:, 4],
            "hemicellulose": hh[:, 0], "xylooligomers": hh[:, 1] * XYLAN_TO_XYLOSE, "xylose": hh[:, 2] * XYLAN_TO_XYLOSE,
            "furfural": hh[:, 3] * PENTOSAN_TO_FURFURAL, "hemicellulose_degradation": hh[:, 4],
        }

    def simulate_ensemble(self, *args, **kwargs) -> tuple[dict, dict]:
        """(previsão central, {espécie: matriz membros x tempos}) usando os membros 1..n."""
        central = self.simulate(*args, member=0, **kwargs)
        sims = [self.simulate(*args, member=m, **kwargs) for m in range(1, len(self.members))]
        return central, {k: np.array([s[k] for s in sims]) for k in central}


# ---------------------------------------------------------------- hidrólise

K_TERMS, J_TERMS = 4, 3
LAMBDA_OFFSETS = np.log([1.0, 0.2, 0.04, 0.008])  # 1/h, espalha as escalas de tempo na inicialização
NU_OFFSETS = np.log([0.5, 0.05, 0.01])
HYD_CENTER = np.array([0.625, 0.08, 150.0, np.log(0.3)])
HYD_SCALE = np.array([0.10, 0.05, 70.0, 1.2])


def hydrolysis_inputs(cellulose_frac, hemicellulose_frac, solids_g_L, enzyme_g_L) -> np.ndarray:
    x = np.column_stack([np.atleast_1d(cellulose_frac), np.atleast_1d(hemicellulose_frac),
                         np.atleast_1d(solids_g_L), np.log(np.atleast_1d(enzyme_g_L))]).astype(float)
    return (x - HYD_CENTER) / HYD_SCALE


def hydrolysis_heads(out: np.ndarray) -> dict[str, np.ndarray]:
    K, J = K_TERMS, J_TERMS
    i = 0
    w = softmax(out[:, i:i + K + 1])[:, :K]; i += K + 1
    lam = np.exp(np.clip(out[:, i:i + K] + LAMBDA_OFFSETS, -12, 4)); i += K
    v = softmax(out[:, i:i + K + 1])[:, :K]; i += K + 1
    mu = np.exp(np.clip(out[:, i:i + K] + LAMBDA_OFFSETS, -12, 4)); i += K
    c0 = out[:, i]; i += 1
    c = out[:, i:i + J]; i += J
    nu = np.exp(np.clip(out[:, i:i + J] + NU_OFFSETS, -12, 4)); i += J
    return {"w": w, "lam": lam, "v": v, "mu": mu, "c0": c0, "c": c, "nu": nu}


HYD_N_OUT = 2 * (K_TERMS + 1) + 2 * K_TERMS + 1 + 2 * J_TERMS


def hydrolysis_profiles(heads: dict, t: np.ndarray) -> dict[str, np.ndarray]:
    """Conversões adimensionais (n_condições x n_tempos)."""
    t = np.atleast_1d(t)[None, :, None]
    X_C = np.sum(heads["w"][:, None, :] * (1 - np.exp(-heads["lam"][:, None, :] * t)), axis=-1)
    X_H = np.sum(heads["v"][:, None, :] * (1 - np.exp(-heads["mu"][:, None, :] * t)), axis=-1)
    phi = sigmoid(heads["c0"][:, None] + np.sum(heads["c"][:, None, :] * np.exp(-heads["nu"][:, None, :] * t), axis=-1))
    return {"X_C": X_C, "X_H": X_H, "phi": phi}


class HydrolysisSurrogate:
    def __init__(self, members, meta):
        self.members, self.meta = members, meta

    @classmethod
    def load(cls, path: Path) -> "HydrolysisSurrogate":
        return cls(*load_members(path))

    def simulate(self, solids_g_L, enzyme_g_L, cellulose_frac, hemicellulose_frac, times_h,
                 member: int = 0) -> dict[str, np.ndarray]:
        """Aceita condições vetoriais (n) e devolve matrizes n x tempos (ou vetores para n = 1)."""
        x = hydrolysis_inputs(cellulose_frac, hemicellulose_frac, solids_g_L, enzyme_g_L)
        p = hydrolysis_profiles(hydrolysis_heads(mlp(self.members[member], x)), np.asarray(times_h, float))
        S = np.atleast_1d(solids_g_L)[:, None]
        C0 = S * np.atleast_1d(cellulose_frac)[:, None]
        H0 = S * np.atleast_1d(hemicellulose_frac)[:, None]
        out = {
            "glucose": GLUCAN_TO_GLUCOSE * C0 * p["X_C"] * (1 - p["phi"]),
            "cellobiose": GLUCAN_TO_CELLOBIOSE * C0 * p["X_C"] * p["phi"],
            "xylose": XYLAN_TO_XYLOSE * H0 * p["X_H"],
            "cellulose": C0 * (1 - p["X_C"]),
            "hemicellulose": H0 * (1 - p["X_H"]),
        }
        if x.shape[0] == 1:
            out = {k: v[0] for k, v in out.items()}
        return out

    def simulate_ensemble(self, *args, **kwargs) -> tuple[dict, dict]:
        central = self.simulate(*args, member=0, **kwargs)
        sims = [self.simulate(*args, member=m, **kwargs) for m in range(1, len(self.members))]
        return central, {k: np.array([s[k] for s in sims]) for k in central}
