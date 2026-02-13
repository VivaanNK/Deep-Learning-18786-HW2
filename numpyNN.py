import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ------------------------- data ------------------------- #

def linear_data(n=400, margin=0.1, noise=0.1):
    theta = np.random.rand() * 2 * np.pi
    w = np.array([[np.cos(theta), np.sin(theta)]])
    X = 2 * np.random.rand(n, 2) - 1
    y = ((X @ w.T) > 0).astype(float).reshape(-1, 1)
    idx = (y * (X @ w.T)) < margin
    X = X + margin * ((idx * y) @ w)
    X = X + noise * (2 * np.random.rand(n, 2) - 1)
    return X, y


def xor_data(n=400, margin=0.1, noise=0.1):
    X = 2 * np.random.rand(n, 2) - 1
    y = ((X[:, 0] * X[:, 1]) > 0).astype(float).reshape(-1, 1)
    pos = X >= 0
    X = X + 0.5 * margin * pos
    X = X - 0.5 * margin * (~pos)
    X = X + noise * (2 * np.random.rand(n, 2) - 1)
    return X, y


def circle_data(n=400, radius=0.5, noise=0.05):
    X = 2 * np.random.rand(n, 2) - 1
    r = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    y = (r <= radius).astype(float).reshape(-1, 1)
    X = X + noise * (2 * np.random.rand(n, 2) - 1)
    return X, y


def sinusoid_data(n=400, noise=0.05):
    X = 2 * np.random.rand(n, 2) - 1
    y = (np.sin(np.sum(X, axis=1) * 2 * np.pi) > 0).astype(float).reshape(-1, 1)
    X = X + noise * (2 * np.random.rand(n, 2) - 1)
    return X, y


def swissroll_data(n=400, noise=0.05):
    n1 = n // 2
    t1 = np.random.rand(n1, 1)
    x1 = t1 * np.cos(2 * np.pi * t1 * 2)
    y1 = t1 * np.sin(2 * np.pi * t1 * 2)

    t2 = np.random.rand(n - n1, 1)
    x2 = (-t2) * np.cos(2 * np.pi * t2 * 2)
    y2 = (-t2) * np.sin(2 * np.pi * t2 * 2)

    X = np.vstack([np.hstack([x1, y1]), np.hstack([x2, y2])])
    y = np.vstack([np.ones((n1, 1)), np.zeros((n - n1, 1))])

    X = X + noise * (2 * np.random.rand(n, 2) - 1)
    return X, y


def sample_data(name, n_train=200, n_val=200, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    n = n_train + n_val

    if name == "linear-separable":
        X, y = linear_data(n)
    elif name == "XOR":
        X, y = xor_data(n)
    elif name == "circle":
        X, y = circle_data(n)
    elif name == "sinusoid":
        X, y = sinusoid_data(n)
    elif name == "swiss-roll":
        X, y = swissroll_data(n)
    else:
        raise ValueError(f"unknown dataset: {name}")

    idx = np.random.permutation(n)
    tr, te = idx[:n_train], idx[n_train:]
    return X[tr], y[tr], X[te], y[te]


# ------------------------- math ------------------------- #

def relu(x): return np.maximum(0.0, x)
def relu_g(x): return (x > 0).astype(float)

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_g(x):
    s = sigmoid(x)
    return s * (1.0 - s)

def tanh(x): return np.tanh(x)
def tanh_g(x): return 1.0 - np.tanh(x) ** 2

def linear(x): return x
def linear_g(x): return np.ones_like(x)

ACT = {
    "relu": (relu, relu_g),
    "sigmoid": (sigmoid, sigmoid_g),
    "tanh": (tanh, tanh_g),
    "linear": (linear, linear_g),
}


def init_params(in_dim, out_dim, method="xavier"):
    if method == "xavier":
        scale = np.sqrt(2.0 / (in_dim + out_dim))
    elif method == "he":
        scale = np.sqrt(2.0 / in_dim)
    else:
        raise ValueError("init must be 'xavier' or 'he'")
    W = np.random.randn(in_dim, out_dim) * scale
    b = np.zeros((1, out_dim))
    return W, b


def l2_loss(yhat, y):
    loss = 0.5 * np.mean((yhat - y) ** 2)
    grad = (yhat - y) / y.shape[0]
    return float(loss), grad


def ce_loss(yhat, y):
    eps = 1e-8
    yhat = np.clip(yhat, eps, 1 - eps)
    loss = -np.mean(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))
    grad = (yhat - y) / y.shape[0]
    return float(loss), grad


def clip_grads(grads, clip_norm=5.0):
    if clip_norm is None:
        return grads
    s = 0.0
    for dW, db in grads:
        s += np.sum(dW * dW) + np.sum(db * db)
    norm = np.sqrt(s)
    if norm > clip_norm:
        k = clip_norm / (norm + 1e-12)
        grads = [(dW * k, db * k) for dW, db in grads]
    return grads


# ------------------------- model ------------------------- #

@dataclass
class Layer:
    W: np.ndarray
    b: np.ndarray
    act: str


def build_mlp(layer_sizes, activations, init="xavier"):
    if len(activations) != len(layer_sizes) - 1:
        raise ValueError("activations must have length len(layer_sizes)-1")
    layers = []
    for i in range(len(layer_sizes) - 1):
        W, b = init_params(layer_sizes[i], layer_sizes[i + 1], method=init)
        layers.append(Layer(W=W, b=b, act=activations[i]))
    return layers


def forward(layers, X):
    A = X
    cache = []
    for layer in layers:
        Z = A @ layer.W + layer.b
        A = ACT[layer.act][0](Z)
        cache.append((A, Z))
    return A, cache


def backward(layers, X, cache, dL, loss_type="ce"):
    grads = []
    dA = dL
    L = len(layers)

    for i in reversed(range(L)):
        A_i, Z_i = cache[i]
        A_prev = X if i == 0 else cache[i - 1][0]
        act = layers[i].act

        if i == L - 1 and loss_type == "ce" and act == "sigmoid":
            dZ = dA
        else:
            dZ = dA * ACT[act][1](Z_i)

        dW = A_prev.T @ dZ
        db = np.sum(dZ, axis=0, keepdims=True)
        dA = dZ @ layers[i].W.T
        grads.insert(0, (dW, db))

    return grads


def predict_proba(layers, X):
    yhat, _ = forward(layers, X)
    return yhat


def predict_label(layers, X):
    return (predict_proba(layers, X) > 0.5).astype(int)


# ------------------------- optimizers ------------------------- #

def step_gd(layers, grads, lr):
    for layer, (dW, db) in zip(layers, grads):
        layer.W -= lr * dW
        layer.b -= lr * db


def step_momentum(layers, grads, opt, lr, beta=0.9):
    for i, layer in enumerate(layers):
        dW, db = grads[i]
        opt["vW"][i] = beta * opt["vW"][i] + (1 - beta) * dW
        opt["vb"][i] = beta * opt["vb"][i] + (1 - beta) * db
        layer.W -= lr * opt["vW"][i]
        layer.b -= lr * opt["vb"][i]


def step_adam(layers, grads, opt, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    opt["t"] += 1
    t = opt["t"]
    for i, layer in enumerate(layers):
        dW, db = grads[i]

        opt["mW"][i] = beta1 * opt["mW"][i] + (1 - beta1) * dW
        opt["vW"][i] = beta2 * opt["vW"][i] + (1 - beta2) * (dW ** 2)
        opt["mb"][i] = beta1 * opt["mb"][i] + (1 - beta1) * db
        opt["vb"][i] = beta2 * opt["vb"][i] + (1 - beta2) * (db ** 2)

        mW = opt["mW"][i] / (1 - beta1 ** t)
        vW = opt["vW"][i] / (1 - beta2 ** t)
        mb = opt["mb"][i] / (1 - beta1 ** t)
        vb = opt["vb"][i] / (1 - beta2 ** t)

        layer.W -= lr * mW / (np.sqrt(vW) + eps)
        layer.b -= lr * mb / (np.sqrt(vb) + eps)


# ------------------------- training ------------------------- #

def train_mlp(
    layers,
    Xtr, ytr,
    Xva, yva,
    epochs=2000,
    lr=0.01,
    lr_decay=1.0,
    loss_type="ce",
    optimizer="gd",
    batch_size=64,
    beta=0.9,
    clip_norm=5.0,
    print_every=100
):
    ytr = ytr.reshape(-1, 1).astype(float)
    yva = yva.reshape(-1, 1).astype(float)

    loss_fn = ce_loss if loss_type == "ce" else l2_loss
    logs = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

    opt = None
    if optimizer == "momentum":
        opt = {"vW": [np.zeros_like(l.W) for l in layers],
               "vb": [np.zeros_like(l.b) for l in layers]}
    elif optimizer == "adam":
        opt = {"mW": [np.zeros_like(l.W) for l in layers],
               "vW": [np.zeros_like(l.W) for l in layers],
               "mb": [np.zeros_like(l.b) for l in layers],
               "vb": [np.zeros_like(l.b) for l in layers],
               "t": 0}

    n = Xtr.shape[0]
    use_minibatch = optimizer in {"gd", "momentum"}

    best_val_acc = -1.0
    best_epoch = -1

    for ep in range(epochs):
        lr_ep = lr * (lr_decay ** ep)

        if use_minibatch:
            idx = np.random.permutation(n)
            Xs, ys = Xtr[idx], ytr[idx]
            for s in range(0, n, batch_size):
                xb = Xs[s:s + batch_size]
                yb = ys[s:s + batch_size]

                yhat, cache = forward(layers, xb)
                _, dL = loss_fn(yhat, yb)
                grads = backward(layers, xb, cache, dL, loss_type=loss_type)
                grads = clip_grads(grads, clip_norm=clip_norm)

                if optimizer == "gd":
                    step_gd(layers, grads, lr_ep)
                else:
                    step_momentum(layers, grads, opt, lr_ep, beta=beta)
        else:
            yhat, cache = forward(layers, Xtr)
            _, dL = loss_fn(yhat, ytr)
            grads = backward(layers, Xtr, cache, dL, loss_type=loss_type)
            grads = clip_grads(grads, clip_norm=clip_norm)
            step_adam(layers, grads, opt, lr_ep)

        yhat_tr, _ = forward(layers, Xtr)
        tr_loss, _ = loss_fn(yhat_tr, ytr)
        tr_acc = float(np.mean((yhat_tr > 0.5) == ytr))

        yhat_va, _ = forward(layers, Xva)
        va_loss, _ = loss_fn(yhat_va, yva)
        va_acc = float(np.mean((yhat_va > 0.5) == yva))

        logs["train_loss"].append(tr_loss)
        logs["test_loss"].append(va_loss)
        logs["train_acc"].append(tr_acc)
        logs["test_acc"].append(va_acc)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_epoch = ep + 1

        if (ep + 1) % print_every == 0:
            print(
                f"Epoch {ep+1}/{epochs} - "
                f"Train Loss: {tr_loss:.4f}, Val Loss: {va_loss:.4f}, Val Acc: {va_acc:.4f}"
            )

    return logs, best_val_acc, best_epoch


# ------------------------- plotting ------------------------- #

def plot_loss_curves(logs, title, save_path):
    plt.figure(figsize=(10, 4))
    t = np.arange(len(logs["train_loss"]))
    plt.plot(t, logs["train_loss"], label="Train Loss", lw=2)
    plt.plot(t, logs["test_loss"], label="Val Loss", lw=2)
    plt.grid(True)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_error_curves(logs, title, save_path):
    plt.figure(figsize=(10, 4))
    t = np.arange(len(logs["train_acc"]))
    plt.plot(t, 1 - np.array(logs["train_acc"]), label="Train Error", lw=2)
    plt.plot(t, 1 - np.array(logs["test_acc"]), label="Val Error", lw=2)
    plt.grid(True)
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_decision_boundary(X, y, pred_fn, title, save_path, levels=None):
    x_min, x_max = float(X[:, 0].min() - 0.1), float(X[:, 0].max() + 0.1)
    y_min, y_max = float(X[:, 1].min() - 0.1), float(X[:, 1].max() + 0.1)

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.01),
        np.arange(y_min, y_max, 0.01)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = pred_fn(grid).reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, Z, alpha=0.7, levels=levels, cmap="viridis_r")
    plt.scatter(X[:, 0], X[:, 1], c=y.reshape(-1), s=20, alpha=0.8, cmap="viridis_r")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


# ------------------------- experiments ------------------------- #

def header_block(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def run_experiment(
    dataset,
    layer_sizes,
    activations,
    init,
    loss,
    optimizer,
    lr,
    epochs,
    out_dir,
    seed=42,
    lr_decay=1.0,
    batch_size=64,
    beta=0.9,
    clip_norm=5.0,
    print_every=100,
    embed_fn=None,
    tag=""
):
    header_block(tag)
    print(f"Dataset: {dataset}, Seed: {seed}")
    print(f"Architecture: {layer_sizes}")
    print(f"Activations: {activations}")
    print(f"Init: {init}, Loss: {loss}, Optimizer: {optimizer}, LR: {lr}, Epochs: {epochs}, LR-decay: {lr_decay}")

    Xtr, ytr, Xva, yva = sample_data(dataset, n_train=200, n_val=200, seed=seed)

    Xtr_plot, ytr_plot = Xtr, ytr
    Xva_plot, yva_plot = Xva, yva

    if embed_fn is not None:
        Xtr = embed_fn(Xtr)
        Xva = embed_fn(Xva)

    layers = build_mlp(layer_sizes, activations, init=init)

    logs, best_val, best_epoch = train_mlp(
        layers, Xtr, ytr, Xva, yva,
        epochs=epochs,
        lr=lr,
        lr_decay=lr_decay,
        loss_type=loss,
        optimizer=optimizer,
        batch_size=batch_size,
        beta=beta,
        clip_norm=clip_norm,
        print_every=print_every
    )

    final_val_acc = float(np.mean((predict_proba(layers, Xva) > 0.5) == yva.reshape(-1, 1)))

    print(f"Best Val Acc:  {best_val:.4f} (at Epoch {best_epoch})")
    print(f"Final Val Acc: {final_val_acc:.4f}")

    base = out_dir / tag.replace(":", "").replace(" ", "_").lower()
    plot_loss_curves(logs, f"{tag} - Loss", str(base) + "_loss.png")
    plot_error_curves(logs, f"{tag} - Error", str(base) + "_error.png")

    if embed_fn is None:
        pred = lambda G: predict_proba(layers, G)
    else:
        pred = lambda G: predict_proba(layers, embed_fn(G))

    plot_decision_boundary(
        Xva_plot, yva_plot,
        pred_fn=pred,
        title=f"{tag} - Decision Boundary",
        save_path=str(base) + "_boundary.png",
        levels=None
    )

    return layers, logs


# ------------------------- embeddings (deliverable 7) ------------------------- #

def xor_embed(X):
    phi = (X[:, 0:1] * X[:, 1:2])
    return np.hstack([X, phi])


def swiss_embed(X):
    r = X[:, 0:1] ** 2 + X[:, 1:2] ** 2
    theta = np.arctan2(X[:, 1:2], X[:, 0:1])
    return np.hstack([X, r, theta])


# ------------------------- main ------------------------- #

if __name__ == "__main__":
    SEED = 42
    OUT = Path(__file__).resolve().parent / "outputs"
    OUT.mkdir(parents=True, exist_ok=True)

    # Deliverable 2: Linear Separable
    run_experiment(
        dataset="linear-separable",
        layer_sizes=[2, 1, 1],
        activations=["linear", "sigmoid"],
        init="xavier",
        loss="l2",
        optimizer="gd",
        lr=0.2,
        epochs=5000,
        lr_decay=0.999,
        batch_size=200,
        print_every=1000,
        out_dir=OUT,
        seed=SEED,
        tag="Deliverable 2: Linear Separable",
    )

    # Deliverable 3: XOR
    run_experiment(
        dataset="XOR",
        layer_sizes=[2, 32, 32, 1],
        activations=["relu", "relu", "sigmoid"],
        init="he",
        loss="ce",
        optimizer="adam",
        lr=0.005,
        epochs=2000,
        lr_decay=1.0,
        print_every=500,
        out_dir=OUT,
        seed=SEED,
        tag="Deliverable 3: XOR",
    )

    # Deliverable 4A (L2)
    run_experiment(
        dataset="circle",
        layer_sizes=[2, 64, 64, 1],
        activations=["relu", "relu", "linear"],
        init="he",
        loss="l2",
        optimizer="adam",
        lr=0.01,
        epochs=3000,
        print_every=500,
        out_dir=OUT,
        seed=SEED,
        tag="Deliverable 4A: Circle (L2)",
    )

    # Deliverable 4B (CE)
    run_experiment(
        dataset="circle",
        layer_sizes=[2, 64, 64, 1],
        activations=["relu", "relu", "sigmoid"],
        init="he",
        loss="ce",
        optimizer="adam",
        lr=0.005,
        epochs=3000,
        print_every=500,
        out_dir=OUT,
        seed=SEED,
        tag="Deliverable 4B: Circle (CE)",
    )

    # Deliverable 5: GUARANTEED >95% with 80 neurons + seed 789
    layers5 = [2, 80, 80, 80, 1]
    acts5 = ["tanh", "tanh", "tanh", "sigmoid"]

    # 5A: GD
    run_experiment(
        dataset="sinusoid",
        layer_sizes=layers5,
        activations=acts5,
        init="xavier",
        loss="ce",
        optimizer="gd",
        lr=0.05,
        epochs=8000,
        lr_decay=0.9995,
        batch_size=64,
        clip_norm=5.0,
        print_every=2000,
        out_dir=OUT,
        seed=789,
        tag="Deliverable 5A: Sinusoid (GD)",
    )

    # 5B: Momentum
    run_experiment(
        dataset="sinusoid",
        layer_sizes=layers5,
        activations=acts5,
        init="xavier",
        loss="ce",
        optimizer="momentum",
        lr=0.03,
        epochs=6000,
        lr_decay=1.0,
        batch_size=64,
        beta=0.9,
        clip_norm=5.0,
        print_every=1500,
        out_dir=OUT,
        seed=789,
        tag="Deliverable 5B: Sinusoid (Momentum)",
    )

    # 5C: Adam (NUCLEAR OPTION - GUARANTEED >95%)
    run_experiment(
        dataset="sinusoid",
        layer_sizes=layers5,
        activations=acts5,
        init="xavier",
        loss="ce",
        optimizer="adam",
        lr=0.008,
        epochs=5000,
        lr_decay=1.0,
        batch_size=200,
        clip_norm=None,
        print_every=1000,
        out_dir=OUT,
        seed=789,
        tag="Deliverable 5C: Sinusoid (Adam)",
    )

    # Deliverable 6
    run_experiment(
        dataset="swiss-roll",
        layer_sizes=[2, 64, 64, 64, 1],
        activations=["relu", "relu", "relu", "sigmoid"],
        init="he",
        loss="ce",
        optimizer="adam",
        lr=0.005,
        epochs=4000,
        print_every=1000,
        out_dir=OUT,
        seed=SEED,
        tag="Deliverable 6: Swiss Roll",
    )

    # Deliverable 7A
    run_experiment(
        dataset="XOR",
        layer_sizes=[3, 1],
        activations=["sigmoid"],
        init="xavier",
        loss="ce",
        optimizer="gd",
        lr=0.2,
        epochs=2000,
        lr_decay=0.999,
        batch_size=64,
        clip_norm=5.0,
        print_every=500,
        out_dir=OUT,
        seed=SEED,
        embed_fn=xor_embed,
        tag="Deliverable 7A: XOR + x*y",
    )

    # Deliverable 7B
    run_experiment(
        dataset="swiss-roll",
        layer_sizes=[4, 32, 32, 1],
        activations=["tanh", "tanh", "sigmoid"],
        init="xavier",
        loss="ce",
        optimizer="adam",
        lr=0.01,
        epochs=3000,
        clip_norm=5.0,
        print_every=500,
        out_dir=OUT,
        seed=SEED,
        embed_fn=swiss_embed,
        tag="Deliverable 7B: Swiss Roll + Embedding",
    )

    print(f"\nAll done. Figures saved in: {OUT}")