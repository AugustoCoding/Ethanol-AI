"""
Etapa 04: treina as redes substitutas (uma por membro do ensemble) e exporta os pesos.

As equações das "cabeças" físicas são as mesmas de ethanol_ai/surrogate.py, escritas aqui em
TensorFlow para o treino. Ao final, confere-se que a inferência em NumPy reproduz o TensorFlow.

Saídas: models/pretreatment_surrogate.npz, models/hydrolysis_surrogate.npz
        reports/training_history.json
Uso: python pipeline/04_train_surrogates.py [pretreatment|hydrolysis]   (padrão: os dois)
"""
from pathlib import Path
import json
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ethanol_ai.paths import MODELS, SYNTHETIC, REPORTS  # noqa: E402
from ethanol_ai import surrogate as sg  # noqa: E402
from ethanol_ai.constants import GLUCAN_TO_GLUCOSE, GLUCAN_TO_CELLOBIOSE, XYLAN_TO_XYLOSE  # noqa: E402

SEED = 123
PRE_HIDDEN = (32, 32, 32)
PRE_STEPS = 4000
HYD_HIDDEN = (64, 64, 64)
HYD_EPOCHS = 500
HYD_BATCH = 256
CELLOBIOSE_SCALE = 2.0  # g/L, normaliza o erro da celobiose no treino

tf.config.experimental.enable_op_determinism()


def build_mlp(n_in, hidden, n_out, seed, out_bias=None):
    tf.keras.utils.set_random_seed(seed)
    x = inp = tf.keras.Input((n_in,))
    for u in hidden:
        x = tf.keras.layers.Dense(u, activation="swish")(x)
    out = tf.keras.layers.Dense(n_out)(x)
    model = tf.keras.Model(inp, out)
    if out_bias is not None:
        model.layers[-1].bias.assign(out_bias.astype(np.float32))
    return model


def export_weights(model):
    return [(l.kernel.numpy(), l.bias.numpy()) for l in model.layers if isinstance(l, tf.keras.layers.Dense)]


# =========================================================== pré-tratamento
def tf_rate_matrix(k):
    k1, k2, k3, k4, k5, k6 = tf.unstack(k, axis=-1)
    z = tf.zeros_like(k1)
    rows = [[-(k1 + k2), z, z, z, z], [k2, -k3, z, z, z], [k1, k3, -(k4 + k5), z, z],
            [z, z, k4, -k6, z], [z, z, k5, k6, z]]
    return tf.stack([tf.stack(r, axis=-1) for r in rows], axis=-2)


def tf_pre_heads(out):
    return {
        "k_hemi": tf.exp(tf.clip_by_value(out[:, 0:6], -20.0, 5.0)),
        "k_cell": tf.exp(tf.clip_by_value(out[:, 6:12], -20.0, 5.0)),
        "a_hemi": tf.nn.softmax(out[:, 12:17]), "a_cell": tf.nn.softmax(out[:, 17:22]),
        "g_hemi": sg.FREE_SCALE * tf.nn.softplus(out[:, 22:27]),
        "g_cell": sg.FREE_SCALE * tf.nn.softplus(out[:, 27:32]),
    }


def tf_pre_profiles(heads, n_times, dt):
    """
    (n_T, n_t, 5) para polímero unitário e para açúcares livres por g de sólido, numa grade de
    tempo uniforme. Uma exponencial de matriz por temperatura (o passo dt) e propagação por
    multiplicação: expm(K dt)^n = expm(K n dt), idêntico à solução exata e muito mais rápido.
    """
    res = {}
    for b in ("hemi", "cell"):
        M = tf.linalg.expm(tf_rate_matrix(heads[f"k_{b}"]) * dt)             # n x 5 x 5
        for kind, y in (("poly", heads[f"a_{b}"]), ("free", heads[f"g_{b}"])):
            states = [y]
            for _ in range(n_times - 1):
                y = tf.einsum("nij,nj->ni", M, y)
                states.append(y)
            res[f"{kind}_{b}"] = tf.stack(states, axis=1)
    return res


PRE_KEYS = ("poly_hemi", "poly_cell", "free_hemi", "free_cell")


# Os perfis de treino são por 1 g/L de polímero e por 1 g/L de sólido. Multiplicar os erros pelas
# concentrações típicas põe os dois termos da perda em g/L, com o mesmo peso. (Sem isso, o termo do
# polímero, que é a maior parte da massa, ficava ~500 vezes menos pesado que o dos açúcares livres.)
G_PER_L = {"poly": 40.0, "free": 120.0}


def train_pretreatment_member(d, m, steps=None):
    steps = steps or PRE_STEPS
    temps, times = d["temperature_C"], d["times_min"]
    x_all = ((temps - sg.T_CENTER) / sg.T_SCALE)[:, None].astype(np.float32)
    val = np.arange(len(temps)) % 5 == 2           # 20% das temperaturas fora do treino
    dt = float(times[1] - times[0])
    assert np.allclose(np.diff(times), dt), "a grade de tempo do pré-tratamento deve ser uniforme"
    n_times = len(times)
    bias = np.zeros(sg.PretreatmentSurrogate.N_OUT); bias[:12] = np.log(0.02)
    t0 = time.time()
    Y = {k: tf.constant(d[k][m], tf.float32) for k in PRE_KEYS}

    model = build_mlp(1, PRE_HIDDEN, sg.PretreatmentSurrogate.N_OUT, SEED + m, bias)
    opt = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(5e-3, steps, alpha=0.01))

    def loss_fn(idx):
        p = tf_pre_profiles(tf_pre_heads(model(tf.gather(x_all, idx))), n_times, dt)
        return sum(tf.reduce_mean(tf.square((p[k] - tf.gather(Y[k], idx)) * G_PER_L[k.split("_")[0]])) for k in PRE_KEYS)

    tr_idx, va_idx = tf.constant(np.where(~val)[0]), tf.constant(np.where(val)[0])

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            l = loss_fn(tr_idx)
        opt.apply_gradients(zip(tape.gradient(l, model.trainable_variables), model.trainable_variables))
        return l

    best, best_w = np.inf, None
    for s_ in range(steps):
        l = step()
        if s_ % 100 == 99 or s_ == steps - 1:
            lv = float(loss_fn(va_idx))
            if lv < best:
                best, best_w = lv, model.get_weights()
    model.set_weights(best_w)
    info = {"member": m, "val_loss": best, "train_loss": float(l), "seconds": time.time() - t0}
    print(f"  pré-tratamento membro {m}: val {best:.2e} (g/L)² ({time.time() - t0:.0f} s)")
    return export_weights(model), info


def train_pretreatment():
    d = np.load(SYNTHETIC / "pretreatment.npz")
    temps = d["temperature_C"]
    members, history = [], []
    for m in range(d["poly_hemi"].shape[0]):
        w, info = train_pretreatment_member(d, m)
        members.append(w)
        history.append(info)

    meta = {"kind": "pretreatment", "hidden": PRE_HIDDEN, "activation": "swish",
            "t_center": sg.T_CENTER, "t_scale": sg.T_SCALE, "free_scale": sg.FREE_SCALE,
            "training_temperature_C": [float(temps.min()), float(temps.max())]}
    sg.save_members(MODELS / "pretreatment_surrogate.npz", members, meta)

    # Conferência NumPy x TensorFlow
    surr = sg.PretreatmentSurrogate.load(MODELS / "pretreatment_surrogate.npz")
    model_w = members[0]
    h_np = surr.heads(203.0)
    out_tf = sg.mlp(model_w, np.array([[(203.0 - sg.T_CENTER) / sg.T_SCALE]]))
    assert np.allclose(sg.pretreatment_heads(out_tf)["k_hemi"][0], h_np["k_hemi"], rtol=1e-5)
    return history


# =========================================================== hidrólise
LAMBDA_OFF = tf.constant(sg.LAMBDA_OFFSETS, tf.float32)
NU_OFF = tf.constant(sg.NU_OFFSETS, tf.float32)


def tf_hyd_heads(out):
    K, J = sg.K_TERMS, sg.J_TERMS
    i = 0
    w = tf.nn.softmax(out[:, i:i + K + 1])[:, :K]; i += K + 1
    lam = tf.exp(tf.clip_by_value(out[:, i:i + K] + LAMBDA_OFF, -12.0, 4.0)); i += K
    v = tf.nn.softmax(out[:, i:i + K + 1])[:, :K]; i += K + 1
    mu = tf.exp(tf.clip_by_value(out[:, i:i + K] + LAMBDA_OFF, -12.0, 4.0)); i += K
    c0 = out[:, i]; i += 1
    c = out[:, i:i + J]; i += J
    nu = tf.exp(tf.clip_by_value(out[:, i:i + J] + NU_OFF, -12.0, 4.0))
    return w, lam, v, mu, c0, c, nu


def tf_hyd_profiles(out, t):
    w, lam, v, mu, c0, c, nu = tf_hyd_heads(out)
    tt = t[None, :, None]
    X_C = tf.reduce_sum(w[:, None, :] * (1 - tf.exp(-lam[:, None, :] * tt)), -1)
    X_H = tf.reduce_sum(v[:, None, :] * (1 - tf.exp(-mu[:, None, :] * tt)), -1)
    phi = tf.sigmoid(c0[:, None] + tf.reduce_sum(c[:, None, :] * tf.exp(-nu[:, None, :] * tt), -1))
    return X_C, X_H, phi


def train_hydrolysis():
    d = np.load(SYNTHETIC / "hydrolysis.npz")
    X, times, Yall, n_train = d["X"], d["times_h"], d["Y"], int(d["n_train"])
    x_all = sg.hydrolysis_inputs(X[:, 0], X[:, 1], X[:, 2], X[:, 3]).astype(np.float32)
    C0 = (X[:, 2] * X[:, 0]).astype(np.float32)
    H0 = (X[:, 2] * X[:, 1]).astype(np.float32)
    t_tf = tf.constant(times, tf.float32)
    bias = np.zeros(sg.HYD_N_OUT); bias[18] = -3.0  # celobiose começa pequena
    members, history = [], []
    for m in range(Yall.shape[0]):
        t0 = time.time()
        Y = Yall[m].astype(np.float32)
        yG = Y[:, :, 0] / (GLUCAN_TO_GLUCOSE * C0[:, None])
        yX = Y[:, :, 1] / (XYLAN_TO_XYLOSE * H0[:, None])
        cb = Y[:, :, 2]
        targets = np.stack([yG, yX, cb], -1).astype(np.float32)
        model = build_mlp(4, HYD_HIDDEN, sg.HYD_N_OUT, SEED + 100 + m, bias)
        steps = HYD_EPOCHS * int(np.ceil(n_train / HYD_BATCH))
        opt = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(3e-3, steps, alpha=0.01))

        def loss_fn(xb, yb, c0b):
            X_C, X_H, phi = tf_hyd_profiles(model(xb), t_tf)
            pG = X_C * (1 - phi)
            pX = X_H
            pB = GLUCAN_TO_CELLOBIOSE * c0b[:, None] * X_C * phi
            return (tf.reduce_mean(tf.square(pG - yb[..., 0])) + tf.reduce_mean(tf.square(pX - yb[..., 1]))
                    + tf.reduce_mean(tf.square((pB - yb[..., 2]) / CELLOBIOSE_SCALE)))

        @tf.function
        def step(xb, yb, c0b):
            with tf.GradientTape() as tape:
                l = loss_fn(xb, yb, c0b)
            opt.apply_gradients(zip(tape.gradient(l, model.trainable_variables), model.trainable_variables))
            return l

        xv, yv, cv = x_all[n_train:], targets[n_train:], C0[n_train:]
        rng = np.random.default_rng(SEED + m)
        best, best_w = np.inf, None
        for ep in range(HYD_EPOCHS):
            perm = rng.permutation(n_train)
            for b in range(0, n_train, HYD_BATCH):
                i = perm[b:b + HYD_BATCH]
                step(x_all[i], targets[i], C0[i])
            if ep % 10 == 9 or ep == HYD_EPOCHS - 1:
                lv = float(loss_fn(xv, yv, cv))
                if lv < best:
                    best, best_w = lv, model.get_weights()
        model.set_weights(best_w)
        members.append(export_weights(model))

        # Erro de validação em g/L (membro contra os seus próprios dados)
        surr = sg.HydrolysisSurrogate(members[-1:], {})
        P = surr.simulate(X[n_train:, 2], X[n_train:, 3], X[n_train:, 0], X[n_train:, 1], times)
        rmse = {k: float(np.sqrt(np.mean((P[k] - Y[n_train:, :, j]) ** 2)))
                for j, k in enumerate(("glucose", "xylose", "cellobiose"))}
        history.append({"member": m, "val_loss": best, "val_rmse_g_L": rmse, "seconds": time.time() - t0})
        print(f"  hidrólise membro {m}: val {best:.2e}, RMSE " + ", ".join(f"{k} {v:.2f}" for k, v in rmse.items())
              + f" g/L ({time.time() - t0:.0f} s)")

    meta = {"kind": "hydrolysis", "hidden": HYD_HIDDEN, "activation": "swish", "K_terms": sg.K_TERMS,
            "J_terms": sg.J_TERMS, "input_center": sg.HYD_CENTER.tolist(), "input_scale": sg.HYD_SCALE.tolist()}
    sg.save_members(MODELS / "hydrolysis_surrogate.npz", members, meta)

    # Conferência NumPy x TensorFlow no membro 0
    model = build_mlp(4, HYD_HIDDEN, sg.HYD_N_OUT, 0)
    model.set_weights([a for pair in members[0] for a in pair])
    X_C, _, _ = tf_hyd_profiles(model(x_all[:5]), t_tf)
    p_np = sg.hydrolysis_profiles(sg.hydrolysis_heads(sg.mlp(members[0], x_all[:5].astype(float))), times)
    assert np.allclose(X_C.numpy(), p_np["X_C"], atol=1e-5), "NumPy e TensorFlow divergem"
    return history


def main() -> None:
    which = sys.argv[1:] or ["pretreatment", "hydrolysis"]
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / "training_history.json"
    for kind, train in (("pretreatment", train_pretreatment), ("hydrolysis", train_hydrolysis)):
        if kind in which:
            result = train()
            # relê antes de gravar: permite treinar as duas etapas em processos separados
            hist = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            hist[kind] = result
            path.write_text(json.dumps(hist, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
