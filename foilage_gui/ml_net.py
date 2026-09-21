"""A compact deep multilayer-perceptron regressor in pure numpy.

No deep-learning dependency: a fully connected network with ReLU hidden
layers and a linear output, trained with mini-batch Adam on a
standardized copy of the data, with a validation split, early stopping
and best-weights restore. The model is a plain dict of arrays, so it can
be pickled or stored alongside the sample database.
"""

import numpy as np


def make_model(n_in, n_out, hidden=(64, 64, 32), seed=0):
    """Initialize {'W': [..], 'b': [..]} with He-initialized layers."""
    rng = np.random.default_rng(seed)
    sizes = [n_in] + [int(h) for h in hidden] + [n_out]
    W, b = [], []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        W.append(rng.normal(0.0, np.sqrt(2.0 / max(fan_in, 1)),
                            size=(fan_in, fan_out)))
        b.append(np.zeros(fan_out))
    return {"W": W, "b": b}


def _forward(model, X):
    """Forward pass; returns (output, [input-to-each-layer activations])."""
    acts = [X]
    a = X
    last = len(model["W"]) - 1
    for i, (W, b) in enumerate(zip(model["W"], model["b"])):
        z = a @ W + b
        a = z if i == last else np.maximum(z, 0.0)      # ReLU hidden layers
        acts.append(a)
    return a, acts


def predict(model, X):
    """Model output for (n, n_in) X in *raw* units (de-standardized)."""
    out, _ = _forward(model, _apply(model, X, model["x_mu"], model["x_sig"]))
    return out * model["y_sig"] + model["y_mu"]


def _apply(model, X, mu, sig):
    return (np.asarray(X, dtype=float) - mu) / sig


def train_mlp(X, y, hidden=(64, 64, 32), epochs=400, lr=1e-3, batch=32,
              val_frac=0.2, seed=0, patience=60):
    """Fit a deep MLP; returns (model, info).

    X: (n, n_in), y: (n, n_out) raw arrays. Rows with non-finite values
    are dropped. The data is standardized, split into train/validation
    (shuffled, seeded), and trained with mini-batch Adam on MSE with
    early stopping on the validation loss; the best-validation weights
    are restored. `info` carries the per-epoch loss history (in raw y
    units) and the final validation predictions for parity plots.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y).all(axis=1)
    X, y = X[ok], y[ok]
    n = len(X)
    if n < 8:
        raise ValueError(f"need at least 8 finite samples to train (got {n})")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_val = max(2, int(round(val_frac * n)))
    val_idx, tr_idx = order[:n_val], order[n_val:]

    model = make_model(X.shape[1], y.shape[1], hidden, seed)
    model["x_mu"] = X[tr_idx].mean(axis=0)
    model["x_sig"] = X[tr_idx].std(axis=0) + 1e-12
    model["y_mu"] = y[tr_idx].mean(axis=0)
    model["y_sig"] = y[tr_idx].std(axis=0) + 1e-12

    Xtr = _apply(model, X[tr_idx], model["x_mu"], model["x_sig"])
    ytr = (y[tr_idx] - model["y_mu"]) / model["y_sig"]
    Xva = _apply(model, X[val_idx], model["x_mu"], model["x_sig"])
    yva = (y[val_idx] - model["y_mu"]) / model["y_sig"]

    adam = {"mW": [np.zeros_like(w) for w in model["W"]],
            "vW": [np.zeros_like(w) for w in model["W"]],
            "mb": [np.zeros_like(b) for b in model["b"]],
            "vb": [np.zeros_like(b) for b in model["b"]],
            "t": 0}

    history = []
    best_val, best_snapshot, since_best = np.inf, None, 0

    def val_loss(snap):
        pred, _ = _forward(snap, Xva)
        return float(np.mean((pred - yva) ** 2))

    best_val = val_loss(model)
    best_snapshot = _snapshot(model)

    for epoch in range(1, int(epochs) + 1):
        batches = rng.permutation(len(Xtr))
        for start in range(0, len(batches), int(batch)):
            idx = batches[start:start + int(batch)]
            grad = _gradients(model, Xtr[idx], ytr[idx])
            adam["t"] += 1
            _adam_step(model, grad, adam, lr)

        tr_loss = val_loss(model)  # cheap enough at these sizes
        va_loss = val_loss(model)
        # history keeps the losses in raw y units for plotting
        history.append([epoch,
                        _raw_mse(model, Xtr, ytr, model["y_mu"], model["y_sig"]),
                        _raw_mse(model, Xva, yva, model["y_mu"], model["y_sig"])])

        if va_loss < best_val - 1e-12:
            best_val, since_best = va_loss, 0
            best_snapshot = _snapshot(model)
        else:
            since_best += 1
            if since_best >= int(patience):
                break

    _restore(model, best_snapshot)
    pred_val = predict(model, X[val_idx])
    info = {"history": history,
            "val_index": val_idx,
            "val_actual": y[val_idx],
            "val_pred": pred_val,
            "val_rmse": float(np.sqrt(np.mean(
                (pred_val - y[val_idx]) ** 2))),
            "epochs_run": len(history),
            "n_train": len(tr_idx), "n_val": len(val_idx)}
    return model, info


def _raw_mse(model, Xs, ys, y_mu, y_sig):
    pred, _ = _forward(model, Xs)
    return float(np.mean((pred * y_sig + y_mu - (ys * y_sig + y_mu)) ** 2))


def _snapshot(model):
    return {"W": [w.copy() for w in model["W"]],
            "b": [b.copy() for b in model["b"]]}


def _restore(model, snap):
    model["W"] = [w.copy() for w in snap["W"]]
    model["b"] = [b.copy() for b in snap["b"]]


def _gradients(model, Xb, yb):
    """MSE gradients (dL/dW, dL/db) for one mini-batch."""
    pred, acts = _forward(model, Xb)
    diff = 2.0 * (pred - yb) / len(Xb)                # dL/doutput
    grads_W = [None] * len(model["W"])
    grads_b = [None] * len(model["b"])
    delta = diff
    for i in range(len(model["W"]) - 1, -1, -1):
        grads_W[i] = acts[i].T @ delta
        grads_b[i] = delta.sum(axis=0)
        if i > 0:
            delta = (delta @ model["W"][i].T) * (acts[i] > 0.0)
    return {"W": grads_W, "b": grads_b}


def _adam_step(model, grads, adam, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    for key, gk in (("W", "W"), ("b", "b")):
        for i, (p, g) in enumerate(zip(model[key], grads[gk])):
            adam[f"m{key}"][i] = beta1 * adam[f"m{key}"][i] + (1 - beta1) * g
            adam[f"v{key}"][i] = beta2 * adam[f"v{key}"][i] + (1 - beta2) * g * g
            m_hat = adam[f"m{key}"][i] / (1 - beta1 ** adam["t"])
            v_hat = adam[f"v{key}"][i] / (1 - beta2 ** adam["t"])
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)
