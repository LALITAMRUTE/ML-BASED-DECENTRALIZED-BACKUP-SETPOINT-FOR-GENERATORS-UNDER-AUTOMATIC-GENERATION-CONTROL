"""
Pref1 Prediction using Area-1 parameters only (no *2 columns).
Models: Random Forest, DNN, RNN, LSTM
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
torch.manual_seed(42)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────
df = pd.read_csv("Data/AGC_AI_Training_Data_Final2.csv")
print(f"Dataset shape: {df.shape}")

# ─────────────────────────────────────────
# 2. Features 
# ─────────────────────────────────────────
FEATURES = [
    "df1",
    "RoCoF1",
    "df1_lag1",
    "df1_lag2",
    "RoCoF1_lag1",
    "GeneratorMW_lag5",
    "BreakerStatus",
]
TARGET   = "Pref1"
SEQ_LEN  = 40    # timesteps of past context (1.6 s at 0.04 s step)
HORIZON  = 50    # predict THIS many steps ahead (2.0 s into future)

print(f"\nFeatures ({len(FEATURES)}): {FEATURES}")
print(f"Target: {TARGET}")

# ─────────────────────────────────────────
# 2a. Feature engineering — lags & derived cols
# ─────────────────────────────────────────
df["GeneratorMW"] = df["Pm1"]
if "BreakerStatus" not in df.columns:
    df["BreakerStatus"] = 1
# Lag features computed within each run to prevent cross-run leakage
for _, grp in df.groupby("Run_ID", sort=False):
    idx = grp.index
    df.loc[idx, "df1_lag1"]         = grp["df1"].shift(1)
    df.loc[idx, "df1_lag2"]         = grp["df1"].shift(2)
    df.loc[idx, "RoCoF1_lag1"]      = grp["RoCoF1"].shift(1)
    df.loc[idx, "GeneratorMW_lag5"] = grp["GeneratorMW"].shift(5)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"Engineered: {df.shape[0]:,} rows after lag feature engineering")

# ─────────────────────────────────────────
# 3. Run-based train/test split
#    Hold out 6 entire runs for testing → forces generalization
#    to completely new disturbance scenarios (no temporal leakage)
# ─────────────────────────────────────────
all_runs    = sorted(df["Run_ID"].unique())
np.random.seed(42)
test_runs   = list(np.random.choice(all_runs, size=6, replace=False))
train_runs  = [r for r in all_runs if r not in test_runs]
print(f"\nRun-based split — Train runs: {len(train_runs)}  Test runs: {len(test_runs)}")
print(f"Test run IDs: {test_runs}")

df_train = df[df["Run_ID"].isin(train_runs)].reset_index(drop=True)
df_test  = df[df["Run_ID"].isin(test_runs)].reset_index(drop=True)

X_tr = df_train[FEATURES].values;  y_tr = df_train[TARGET].values
X_te = df_test[FEATURES].values;   y_te = df_test[TARGET].values
print(f"Tabular  — Train: {len(X_tr):,}  Test: {len(X_te):,}")

scaler_X = StandardScaler()
X_tr_sc  = scaler_X.fit_transform(X_tr)
X_te_sc  = scaler_X.transform(X_te)

scaler_y = StandardScaler()
y_tr_sc  = scaler_y.fit_transform(y_tr.reshape(-1, 1)).ravel()

# ─────────────────────────────────────────
# 4. Sequence dataset for RNN / LSTM
#    Sliding window within each run (time order preserved)
#    Train/test split is at the run level — no temporal leakage
# ─────────────────────────────────────────
def make_sequences(dataframe, feature_cols, target_col, seq_len, horizon):
    Xs, ys = [], []
    for _, grp in dataframe.groupby("Run_ID"):
        grp  = grp.sort_values("Time")
        feat = grp[feature_cols].values
        tgt  = grp[target_col].values
        for i in range(len(feat) - seq_len - horizon + 1):
            Xs.append(feat[i : i + seq_len])
            ys.append(tgt[i + seq_len + horizon - 1])   # predict HORIZON steps ahead
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

Xs_tr_raw, ys_tr_raw = make_sequences(df_train, FEATURES, TARGET, SEQ_LEN, HORIZON)
Xs_te_raw, ys_te_raw = make_sequences(df_test,  FEATURES, TARGET, SEQ_LEN, HORIZON)
n_tr, steps, n_feat = Xs_tr_raw.shape
print(f"Sequences — Train: {n_tr:,}  Test: {len(ys_te_raw):,}  Shape: {Xs_tr_raw.shape}")

# Scale using train statistics only
Xs_tr = scaler_X.transform(Xs_tr_raw.reshape(-1, n_feat)).reshape(n_tr, steps, n_feat)
Xs_te = scaler_X.transform(Xs_te_raw.reshape(-1, n_feat)).reshape(len(ys_te_raw), steps, n_feat)
ys_tr = scaler_y.transform(ys_tr_raw.reshape(-1, 1)).ravel()
ys_te = scaler_y.transform(ys_te_raw.reshape(-1, 1)).ravel()
# ─────────────────────────────────────────
# Helper: evaluate predictions
# ─────────────────────────────────────────
def evaluate(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE" : float(mean_absolute_error(y_true, y_pred)),
        "R²"  : float(r2_score(y_true, y_pred)),
    }

results     = {}   # {model_name: {RMSE, MAE, R²}}
predictions = {}   # {model_name: (y_true, y_pred)}

EPOCHS = 50
BATCH  = 256

# ─────────────────────────────────────────
# PyTorch training helper
# ─────────────────────────────────────────
def train_torch(model, X_train_t, y_train_t, X_val_t, y_val_t,
                epochs=EPOCHS, batch=BATCH, patience=6, lr=1e-3):
    model = model.to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    ds    = TensorDataset(X_train_t.to(DEVICE), y_train_t.to(DEVICE))
    loader = DataLoader(ds, batch_size=batch, shuffle=True)

    best_val, best_state, wait = float("inf"), None, 0
    val_losses = []
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb).squeeze(), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(X_val_t.to(DEVICE)).squeeze(), y_val_t.to(DEVICE)).item()
        val_losses.append(vl)
        if vl < best_val:
            best_val, best_state, wait = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, val_losses, epoch + 1

def torch_predict(model, X_t):
    model.eval()
    with torch.no_grad():
        return model(X_t.to(DEVICE)).squeeze().cpu().numpy()

# ─────────────────────────────────────────
# Tabular tensors (for DNN)
# ─────────────────────────────────────────
split_val = int(0.9 * len(X_tr_sc))
X_tv = torch.tensor(X_tr_sc[:split_val],  dtype=torch.float32)
y_tv = torch.tensor(y_tr_sc[:split_val],  dtype=torch.float32)
X_vv = torch.tensor(X_tr_sc[split_val:],  dtype=torch.float32)
y_vv = torch.tensor(y_tr_sc[split_val:],  dtype=torch.float32)
X_te_t = torch.tensor(X_te_sc, dtype=torch.float32)

# Sequence tensors (for RNN / LSTM)
split_seq = int(0.9 * len(Xs_tr))
Xs_tv = torch.tensor(Xs_tr[:split_seq],  dtype=torch.float32)
ys_tv = torch.tensor(ys_tr[:split_seq],  dtype=torch.float32)
Xs_vv = torch.tensor(Xs_tr[split_seq:],  dtype=torch.float32)
ys_vv = torch.tensor(ys_tr[split_seq:],  dtype=torch.float32)
Xs_te_t = torch.tensor(Xs_te, dtype=torch.float32)


# ─────────────────────────────────────────
# 7. DNN
# ─────────────────────────────────────────
print("\n[3/5] Training DNN...")

class DNNModel(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),   nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),    nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x): return self.net(x)

dnn_model, history_dnn, ep_dnn = train_torch(
    DNNModel(len(FEATURES)), X_tv, y_tv, X_vv, y_vv
)
y_pred_dnn = scaler_y.inverse_transform(
    torch_predict(dnn_model, X_te_t).reshape(-1, 1)
).ravel()
results["DNN"] = evaluate(y_te, y_pred_dnn)
predictions["DNN"] = (y_te, y_pred_dnn)
print(f"      R² = {results['DNN']['R²']:.4f}  (epochs: {ep_dnn})")

# ─────────────────────────────────────────
# 8. RNN
# ─────────────────────────────────────────
print("\n[4/5] Training RNN...")

class RNNModel(nn.Module):
    def __init__(self, n_feat, hidden=64):
        super().__init__()
        self.rnn  = nn.RNN(n_feat, hidden, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.fc   = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(self.drop(out[:, -1, :]))

rnn_model, history_rnn, ep_rnn = train_torch(
    RNNModel(len(FEATURES)), Xs_tv, ys_tv, Xs_vv, ys_vv
)
y_pred_rnn = scaler_y.inverse_transform(
    torch_predict(rnn_model, Xs_te_t).reshape(-1, 1)
).ravel()
y_te_seq = ys_te_raw   # original (unscaled) test targets for sequences
results["RNN"] = evaluate(y_te_seq, y_pred_rnn)
predictions["RNN"] = (y_te_seq, y_pred_rnn)
print(f"      R² = {results['RNN']['R²']:.4f}  (epochs: {ep_rnn})")

# ─────────────────────────────────────────
# 9. LSTM
# ─────────────────────────────────────────
print("\n[5/5] Training LSTM...")

class LSTMModel(nn.Module):
    def __init__(self, n_feat, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.fc   = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.drop(out[:, -1, :]))

lstm_model, history_lstm, ep_lstm = train_torch(
    LSTMModel(len(FEATURES)), Xs_tv, ys_tv, Xs_vv, ys_vv
)
y_pred_lstm = scaler_y.inverse_transform(
    torch_predict(lstm_model, Xs_te_t).reshape(-1, 1)
).ravel()
results["LSTM"] = evaluate(y_te_seq, y_pred_lstm)
predictions["LSTM"] = (y_te_seq, y_pred_lstm)
print(f"      R² = {results['LSTM']['R²']:.4f}  (epochs: {ep_lstm})")

# ─────────────────────────────────────────
# 10. Summary table
# ─────────────────────────────────────────
print("\n" + "=" * 58)
print(f"{'Model':<22} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
print("=" * 58)
for name, m in results.items():
    print(f"{name:<22} {m['RMSE']:>10.6f} {m['MAE']:>10.6f} {m['R²']:>10.4f}")
print("=" * 58)
best = max(results, key=lambda k: results[k]["R²"])
print(f"\nBest model: {best}  (R² = {results[best]['R²']:.4f})")

# ─────────────────────────────────────────
# 11. Plots
# ─────────────────────────────────────────
COLORS = {
    "DNN"              : "#55A868",
    "RNN"              : "#C44E52",
    "LSTM"             : "#8172B3",
}
MODEL_NAMES = list(results.keys())

fig = plt.figure(figsize=(20, 16))
gs  = gridspec.GridSpec(3, 5, figure=fig, hspace=0.55, wspace=0.4)

# ── Row 0: Actual vs Predicted (one per model) ──
for col, name in enumerate(MODEL_NAMES):
    ax = fig.add_subplot(gs[0, col])
    y_true, y_pred = predictions[name]
    idx_s = np.random.choice(len(y_true), min(1500, len(y_true)), replace=False)
    ax.scatter(y_true[idx_s], y_pred[idx_s],
               alpha=0.25, s=6, color=COLORS[name])
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2)
    ax.set_title(f"{name}\nR²={results[name]['R²']:.4f}", fontsize=9)
    ax.set_xlabel("Actual", fontsize=8)
    ax.set_ylabel("Predicted", fontsize=8)
    ax.tick_params(labelsize=7)

# ── Row 1: R² bar chart ──
ax_r2 = fig.add_subplot(gs[1, :2])
r2_vals = [results[m]["R²"] for m in MODEL_NAMES]
bars = ax_r2.bar(MODEL_NAMES, r2_vals,
                 color=[COLORS[m] for m in MODEL_NAMES], width=0.5)
ax_r2.axhline(0.90, color="gray", linestyle="--", lw=1, label="90% target")
ax_r2.axhline(0.92, color="black", linestyle="--", lw=1, label="92% target")
ax_r2.set_ylim(0, 1.05)
ax_r2.set_ylabel("R² Score")
ax_r2.set_title("Model Comparison — R²")
ax_r2.legend(fontsize=8)
for bar, v in zip(bars, r2_vals):
    ax_r2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f"{v:.4f}", ha="center", fontsize=8)
ax_r2.tick_params(axis="x", labelsize=8)

# ── Row 1: RMSE bar chart ──
ax_rmse = fig.add_subplot(gs[1, 2:4])
rmse_vals = [results[m]["RMSE"] for m in MODEL_NAMES]
ax_rmse.bar(MODEL_NAMES, rmse_vals,
            color=[COLORS[m] for m in MODEL_NAMES], width=0.5)
ax_rmse.set_ylabel("RMSE")
ax_rmse.set_title("Model Comparison — RMSE")
for i, (v, b) in enumerate(zip(rmse_vals, ax_rmse.patches)):
    ax_rmse.text(b.get_x() + b.get_width()/2, b.get_height() + 0.0002,
                 f"{v:.5f}", ha="center", fontsize=8)
ax_rmse.tick_params(axis="x", labelsize=8)

# ── Row 1: Training loss (DNN / RNN / LSTM) ──
ax_loss = fig.add_subplot(gs[1, 4])
for name, hist, c in [("DNN",  history_dnn,  COLORS["DNN"]),
                       ("RNN",  history_rnn,  COLORS["RNN"]),
                       ("LSTM", history_lstm, COLORS["LSTM"])]:
    ax_loss.plot(hist, label=name, color=c)
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Val MSE Loss")
ax_loss.set_title("Training History")
ax_loss.legend(fontsize=8)
ax_loss.tick_params(labelsize=7)

# ── Row 2: Residuals (best model) ──
ax_res = fig.add_subplot(gs[2, :])
y_true_b, y_pred_b = predictions[best]
residuals = y_true_b - y_pred_b
idx_s2 = np.random.choice(len(y_true_b), min(3000, len(y_true_b)), replace=False)
ax_res.scatter(y_pred_b[idx_s2], residuals[idx_s2],
               alpha=0.25, s=6, color=COLORS[best])
ax_res.axhline(0, color="k", linestyle="--", lw=1.2)
ax_res.set_xlabel(f"Predicted Pref1 ({best})")
ax_res.set_ylabel("Residual (Actual − Predicted)")
ax_res.set_title(f"Residual Plot — Best Model: {best}")

plt.suptitle("Pref1 Prediction | Area-1 Features | LR · SVR · DNN · RNN · LSTM",
             fontsize=13, fontweight="bold", y=1.01)
plt.savefig("Data/pref1_model_comparison.png", dpi=150, bbox_inches="tight")
print("\nPlot saved → Data/pref1_model_comparison.png")
plt.show()

import matplotlib.pyplot as plt

def plot_time_series_waveforms(predictions_dict, colors_dict, save_path="time_series_comparison.png", zoom_points=500):
    """
    Generates chronological line plots for Actual vs Predicted.
    zoom_points: How many timesteps to plot (500 steps * 0.04s = 20 seconds of grid response)
    """
    models = list(predictions_dict.keys())
    n_models = len(models)
    
    # Create a vertical stack of subplots (one for each model)
    fig, axes = plt.subplots(n_models, 1, figsize=(15, 3.5 * n_models), sharex=True)
    if n_models == 1: 
        axes = [axes]
        
    fig.suptitle(f"Time-Series Tracking: Actual vs Predicted Pref1 (First {zoom_points} steps)", 
                 fontsize=14, fontweight="bold", y=0.98)

    for i, model_name in enumerate(models):
        ax = axes[i]
        y_true, y_pred = predictions_dict[model_name]
        
        # Slice the arrays to zoom in on a specific fault/event
        y_true_zoomed = y_true[:zoom_points]
        y_pred_zoomed = y_pred[:zoom_points]
        time_axis = range(zoom_points)

        # Plot Actual (Thick Black Line)
        ax.plot(time_axis, y_true_zoomed, color="black", linewidth=2.5, label="Actual Pref1", alpha=0.7)
        
        # Plot Predicted (Thinner, Colored, Dashed Line so it overlays cleanly)
        ax.plot(time_axis, y_pred_zoomed, color=colors_dict.get(model_name, "blue"), 
                linewidth=1.8, linestyle="--", label=f"Predicted ({model_name})")

        ax.set_title(f"{model_name} Waveform Tracking", fontsize=11)
        ax.set_ylabel("Pref1 (MW)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right")

    # Only add the X-axis label to the very bottom plot
    axes[-1].set_xlabel("Time Steps (0.04s per step)", fontsize=11)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93) # Make room for suptitle
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nTime-Series Plot saved → {save_path}")
    plt.show()

# ==========================================
# Add this right at the very end of your predict_pref1.py script:
# ==========================================
plot_time_series_waveforms(predictions, COLORS, save_path="Data/time_series_all_models.png", zoom_points=400)

