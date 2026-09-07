"""
Central runner — trains all 5 models and saves results to results/
Usage:  python run_all.py
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

from models.data_prep import prepare_all
from models import (
    random_forest_model,
    dnn_model,
    rnn_model,
    lstm_model,
)

RESULTS_DIR = "results"

COLORS = {
    "Linear Regression": "#4C72B0",
    "Random Forest":     "#DD8452",
    "DNN":               "#55A868",
    "RNN":               "#C44E52",
    "LSTM":              "#8172B3",
}


# ──────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 62)
    print("  Pref1 Prediction  |  Area-1 Features  |  Run-based Split")
    print("=" * 62)

    # ── Load & preprocess once ────────────────────────────────────────────────
    data = prepare_all()

    # ── Train each model ──────────────────────────────────────────────────────
    all_results = {}
    models_to_run = [
        ("Random Forest",     random_forest_model),
        ("DNN",               dnn_model),
        ("RNN",               rnn_model),
        ("LSTM",              lstm_model),
    ]

    for label, module in models_to_run:
        print(f"\n[{label}]")
        result = module.run(data, RESULTS_DIR)
        all_results.update(result)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"  {'Model':<22} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
    print("=" * 62)
    for name, m in all_results.items():
        print(f"  {name:<22} {m['RMSE']:>10.6f} {m['MAE']:>10.6f} {m['R2']:>10.4f}")
    print("=" * 62)

    best = max(all_results, key=lambda k: all_results[k]["R2"])
    print(f"\n  Best model: {best}  (R² = {all_results[best]['R2']:.4f})")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df_res = pd.DataFrame(all_results).T.reset_index()
    df_res.columns = ["Model", "RMSE", "MAE", "R2"]
    csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\n  CSV  → {csv_path}")

    # ── Comparison plots ──────────────────────────────────────────────────────
    _plot_comparison(all_results)
    print(f"  Plot → {os.path.join(RESULTS_DIR, 'comparison.png')}")
    print("\nDone.")


# ──────────────────────────────────────────────────────────────────────────────
def _plot_comparison(results: dict):
    names  = list(results.keys())
    colors = [COLORS.get(n, "#aaaaaa") for n in names]
    r2_vals   = [results[n]["R2"]   for n in names]
    rmse_vals = [results[n]["RMSE"] for n in names]
    mae_vals  = [results[n]["MAE"]  for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Model Comparison — Pref1 Prediction (Area-1 Features)",
                 fontsize=13, fontweight="bold")

    spec = [
        (axes[0], r2_vals,   "R² Score",  "R² — higher is better",  ".4f", True),
        (axes[1], rmse_vals, "RMSE",       "RMSE — lower is better", ".6f", False),
        (axes[2], mae_vals,  "MAE",        "MAE — lower is better",  ".6f", False),
    ]

    for ax, vals, ylabel, title, fmt, is_r2 in spec:
        bars = ax.bar(names, vals, color=colors, width=0.55, edgecolor="white")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20, labelsize=9)

        offset = max(vals) * 0.012
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    f"{v:{fmt}}", ha="center", fontsize=8)

        if is_r2:
            ax.set_ylim(0, 1.15)
            ax.axhline(0.90, color="dimgray", linestyle="--", lw=1.2, label="90 %")
            ax.axhline(0.92, color="black",   linestyle="--", lw=1.2, label="92 %")
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
