"""
Shared data loading, splitting, scaling, and sequence building.
Used by all model modules and run_all.py.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_PATH = "Data/AGC_AI_Training_Data_Final2.csv"
FEATURES  = [
    "df1",
    "RoCoF1",
    "df1_lag1",
    "df1_lag2",
    "RoCoF1_lag1",
    "GeneratorMW_lag5",
    "BreakerStatus",
]
TARGET    = "Pref1"
SEQ_LEN   = 40   # timesteps of past context (1.6 s at 0.04 s/step)
HORIZON   = 50   # steps ahead to predict  (2.0 s into the future)


# ──────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add GeneratorMW, BreakerStatus, and lag features (computed per run)."""
    df = df.copy()
    df["GeneratorMW"] = df["Pm1"]
    if "BreakerStatus" not in df.columns:
        df["BreakerStatus"] = 1
    # Compute lags within each run to prevent cross-run leakage
    for _, grp in df.groupby("Run_ID", sort=False):
        idx = grp.index
        df.loc[idx, "df1_lag1"]         = grp["df1"].shift(1)
        df.loc[idx, "df1_lag2"]         = grp["df1"].shift(2)
        df.loc[idx, "RoCoF1_lag1"]      = grp["RoCoF1"].shift(1)
        df.loc[idx, "GeneratorMW_lag5"] = grp["GeneratorMW"].shift(5)
    df.dropna(inplace=True)
    return df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
def load_data(data_path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    print(f"Loaded  : {df.shape[0]:,} rows × {df.shape[1]} columns  ({data_path})")
    df = engineer_features(df)
    print(f"Engineered: {df.shape[0]:,} rows after lag feature engineering")
    return df


# ──────────────────────────────────────────────────────────────────────────────
def run_based_split(df: pd.DataFrame, n_test_runs: int = 6, seed: int = 42):
    """Hold out entire runs for testing — no temporal leakage between splits."""
    all_runs   = sorted(df["Run_ID"].unique())
    np.random.seed(seed)
    test_runs  = list(np.random.choice(all_runs, size=n_test_runs, replace=False))
    train_runs = [r for r in all_runs if r not in test_runs]
    df_train = df[df["Run_ID"].isin(train_runs)].reset_index(drop=True)
    df_test  = df[df["Run_ID"].isin(test_runs)].reset_index(drop=True)
    print(f"Split   : {len(train_runs)} train runs | {len(test_runs)} test runs {sorted(test_runs)}")
    return df_train, df_test


# ──────────────────────────────────────────────────────────────────────────────
def build_tabular(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """Return scaled tabular arrays for LR / RF / DNN."""
    X_tr = df_train[FEATURES].values;  y_tr = df_train[TARGET].values
    X_te = df_test[FEATURES].values;   y_te = df_test[TARGET].values

    scaler_X = StandardScaler()
    X_tr_sc  = scaler_X.fit_transform(X_tr)
    X_te_sc  = scaler_X.transform(X_te)

    scaler_y = StandardScaler()
    y_tr_sc  = scaler_y.fit_transform(y_tr.reshape(-1, 1)).ravel()

    print(f"Tabular : train {len(X_tr):,} | test {len(X_te):,}")
    return X_tr_sc, X_te_sc, y_tr, y_te, y_tr_sc, scaler_X, scaler_y


# ──────────────────────────────────────────────────────────────────────────────
def _sliding_windows(dataframe: pd.DataFrame, seq_len: int, horizon: int):
    Xs, ys = [], []
    for _, grp in dataframe.groupby("Run_ID"):
        grp  = grp.sort_values("Time")
        feat = grp[FEATURES].values
        tgt  = grp[TARGET].values
        for i in range(len(feat) - seq_len - horizon + 1):
            Xs.append(feat[i : i + seq_len])
            ys.append(tgt[i + seq_len + horizon - 1])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def build_sequences(df_train, df_test, scaler_X, scaler_y,
                    seq_len: int = SEQ_LEN, horizon: int = HORIZON):
    """Sliding-window sequences for RNN / LSTM. Scaled with tabular scalers."""
    Xs_tr_raw, ys_tr_raw = _sliding_windows(df_train, seq_len, horizon)
    Xs_te_raw, ys_te_raw = _sliding_windows(df_test,  seq_len, horizon)

    n_tr, steps, n_feat = Xs_tr_raw.shape
    n_te = len(ys_te_raw)

    Xs_tr = scaler_X.transform(Xs_tr_raw.reshape(-1, n_feat)).reshape(n_tr, steps, n_feat)
    Xs_te = scaler_X.transform(Xs_te_raw.reshape(-1, n_feat)).reshape(n_te, steps, n_feat)
    ys_tr = scaler_y.transform(ys_tr_raw.reshape(-1, 1)).ravel()
    ys_te = scaler_y.transform(ys_te_raw.reshape(-1, 1)).ravel()

    print(f"Sequences: train {n_tr:,} | test {n_te:,} | window {Xs_tr_raw.shape}")
    return Xs_tr, Xs_te, ys_tr, ys_te, ys_tr_raw, ys_te_raw


# ──────────────────────────────────────────────────────────────────────────────
def prepare_all(data_path: str = DATA_PATH) -> dict:
    """One-shot helper: load → split → tabular → sequences. Returns data dict."""
    df = load_data(data_path)
    df_train, df_test = run_based_split(df)
    X_tr_sc, X_te_sc, y_tr, y_te, y_tr_sc, scaler_X, scaler_y = build_tabular(df_train, df_test)
    Xs_tr, Xs_te, ys_tr, ys_te, ys_tr_raw, ys_te_raw = build_sequences(
        df_train, df_test, scaler_X, scaler_y
    )
    return {
        # tabular arrays
        "X_tr_sc":  X_tr_sc,  "X_te_sc":  X_te_sc,
        "y_tr":     y_tr,     "y_te":     y_te,
        "y_tr_sc":  y_tr_sc,
        # sequence arrays
        "Xs_tr":    Xs_tr,    "Xs_te":    Xs_te,
        "ys_tr":    ys_tr,    "ys_te":    ys_te,
        "ys_tr_raw":ys_tr_raw,"ys_te_raw":ys_te_raw,
        # scalers
        "scaler_X": scaler_X, "scaler_y": scaler_y,
        # dimensions
        "n_features": len(FEATURES),
        "seq_len":    SEQ_LEN,
    }
