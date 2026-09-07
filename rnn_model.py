"""
Recurrent Neural Network (RNN) model
  - Architecture : SimpleRNN(64) → Dropout → Linear(32) → Linear(1)
  - Input        : Sliding-window sequences (SEQ_LEN=40 steps, HORIZON=50 steps ahead)
  - Training     : Adam + early stopping
  - Evaluation   : RMSE / MAE / R²
  - Plots        : Actual vs Predicted  |  Residuals  |  Training Loss
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from .metrics    import evaluate
from .torch_utils import DEVICE, train_torch, torch_predict

MODEL_NAME = "RNN"
COLOR      = "#C44E52"


# ──────────────────────────────────────────────────────────────────────────────
class _RNNModel(nn.Module):
    def __init__(self, n_feat: int, hidden: int = 64):
        super().__init__()
        self.rnn  = nn.RNN(n_feat, hidden, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.fc   = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(self.drop(out[:, -1, :]))


# ──────────────────────────────────────────────────────────────────────────────
def run(data: dict, results_dir: str) -> dict:
    """Train, evaluate, and plot RNN. Returns {name: metrics}."""
    os.makedirs(results_dir, exist_ok=True)

    # ── Validation split (10 % of train sequences) ────────────────────────────
    split  = int(0.9 * len(data["Xs_tr"]))
    Xs_tv  = torch.tensor(data["Xs_tr"][:split], dtype=torch.float32)
    ys_tv  = torch.tensor(data["ys_tr"][:split], dtype=torch.float32)
    Xs_vv  = torch.tensor(data["Xs_tr"][split:], dtype=torch.float32)
    ys_vv  = torch.tensor(data["ys_tr"][split:], dtype=torch.float32)
    Xs_te_t = torch.tensor(data["Xs_te"], dtype=torch.float32)

    # ── Train ──────────────────────────────────────────────────────────────────
    model, val_losses, n_epochs = train_torch(
        _RNNModel(data["n_features"]), Xs_tv, ys_tv, Xs_vv, ys_vv
    )

    # ── Predict + inverse-scale ────────────────────────────────────────────────
    y_pred = data["scaler_y"].inverse_transform(
        torch_predict(model, Xs_te_t).reshape(-1, 1)
    ).ravel()
    y_true = data["ys_te_raw"]   # original unscaled targets

    # ── Evaluate ───────────────────────────────────────────────────────────────
    metrics = evaluate(y_true, y_pred)
    _print_metrics(metrics, n_epochs)

    # ── Plot ───────────────────────────────────────────────────────────────────
    _plot(y_true, y_pred, val_losses, metrics, results_dir)

    return {MODEL_NAME: metrics}


# ──────────────────────────────────────────────────────────────────────────────
def _print_metrics(m: dict, n_epochs: int):
    print(f"  {MODEL_NAME:<22}  RMSE={m['RMSE']:.6f}  "
          f"MAE={m['MAE']:.6f}  R²={m['R2']:.4f}  (epochs: {n_epochs})")


def _plot(y_true, y_pred, val_losses, metrics, results_dir):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    fig.suptitle(
        f"{MODEL_NAME}  —  R²={metrics['R2']:.4f} | "
        f"RMSE={metrics['RMSE']:.6f} | MAE={metrics['MAE']:.6f}",
        fontsize=11, fontweight="bold"
    )

    idx = np.random.choice(len(y_true), min(2000, len(y_true)), replace=False)

    # Actual vs Predicted
    ax = axes[0]
    ax.scatter(y_true[idx], y_pred[idx], alpha=0.35, s=8, color=COLOR)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.3, label="Ideal")
    ax.set_xlabel("Actual Pref1");  ax.set_ylabel("Predicted Pref1")
    ax.set_title("Actual vs Predicted");  ax.legend(fontsize=8)

    # Residuals
    ax = axes[1]
    residuals = y_true - y_pred
    ax.scatter(y_pred[idx], residuals[idx], alpha=0.35, s=8, color=COLOR)
    ax.axhline(0, color="k", linestyle="--", lw=1.3)
    ax.set_xlabel("Predicted Pref1");  ax.set_ylabel("Residual (Actual − Predicted)")
    ax.set_title("Residual Plot")

    # Training loss
    ax = axes[2]
    ax.plot(val_losses, color=COLOR, lw=1.8)
    ax.set_xlabel("Epoch");  ax.set_ylabel("Validation MSE Loss")
    ax.set_title("Training Loss Curve")

    plt.tight_layout()
    path = os.path.join(results_dir, "rnn.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved → {path}")
