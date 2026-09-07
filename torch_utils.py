"""PyTorch training loop and prediction helper shared by DNN / RNN / LSTM."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def train_torch(model, X_train_t, y_train_t, X_val_t, y_val_t,
                epochs: int = 50, batch: int = 256,
                patience: int = 6, lr: float = 1e-3):
    """
    Train a PyTorch model with early stopping.

    Returns
    -------
    model      : trained model (best weights restored)
    val_losses : list of per-epoch validation MSE losses
    n_epochs   : number of epochs actually run
    """
    model   = model.to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    ds      = TensorDataset(X_train_t.to(DEVICE), y_train_t.to(DEVICE))
    loader  = DataLoader(ds, batch_size=batch, shuffle=True)

    best_val, best_state, wait = float("inf"), None, 0
    val_losses = []
    epoch = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb).squeeze(-1), yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vl = loss_fn(
                model(X_val_t.to(DEVICE)).squeeze(-1),
                y_val_t.to(DEVICE)
            ).item()
        val_losses.append(vl)

        if vl < best_val:
            best_val   = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait       = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model, val_losses, epoch + 1


def torch_predict(model, X_t) -> "np.ndarray":
    """Run inference; returns a 1-D numpy array."""
    model.eval()
    with torch.no_grad():
        return model(X_t.to(DEVICE)).squeeze().cpu().numpy()
