"""
Random Forest Regression model
  - Training  : RandomForestRegressor on full training set
  - Evaluation: RMSE / MAE / R²
  - Plots     : Actual vs Predicted  |  Residuals
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from .metrics import evaluate

MODEL_NAME = "Random Forest"
COLOR      = "#DD8452"


# ──────────────────────────────────────────────────────────────────────────────
def run(data: dict, results_dir: str) -> dict:
    """Train, evaluate, and plot Random Forest. Returns {name: metrics}."""
    os.makedirs(results_dir, exist_ok=True)

    # ── Train ──────────────────────────────────────────────────────────────────
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(data["X_tr_sc"], data["y_tr"])
    y_pred = model.predict(data["X_te_sc"])

    # ── Evaluate ───────────────────────────────────────────────────────────────
    metrics = evaluate(data["y_te"], y_pred)
    _print_metrics(metrics)

    # ── Plot ───────────────────────────────────────────────────────────────────
    _plot(data["y_te"], y_pred, metrics, results_dir)

    return {MODEL_NAME: metrics}


# ──────────────────────────────────────────────────────────────────────────────
def _print_metrics(m: dict):
    print(f"  {MODEL_NAME:<22}  RMSE={m['RMSE']:.6f}  "
          f"MAE={m['MAE']:.6f}  R²={m['R2']:.4f}")


def _plot(y_true, y_pred, metrics, results_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
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

    plt.tight_layout()
    path = os.path.join(results_dir, "random_forest.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved → {path}")
