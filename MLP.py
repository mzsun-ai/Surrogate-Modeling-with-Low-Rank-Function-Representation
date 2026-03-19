import os
import math
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
import optuna

# Basic settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load data
X = pd.read_csv('data/inputdata_6.txt', sep=r'\s+', comment='#').values
Y = pd.read_csv('data/outputdata_6.txt', sep=r'\s+', comment='#').iloc[:, 0].values

X = torch.from_numpy(X).float()
Y = torch.from_numpy(Y).float().unsqueeze(1)

# Split into training and test sets
n = len(X)
idx = np.random.permutation(n)
split = int(0.7 * n)
train_idx, test_idx = idx[:split], idx[split:]

X_train, Y_train = X[train_idx], Y[train_idx]
X_test,  Y_test  = X[test_idx],  Y[test_idx]

N_INPUTS = X.shape[1]

# Standardize inputs and outputs
X_mean = X_train.mean(dim=0, keepdim=True)
X_std  = X_train.std(dim=0, keepdim=True).clamp_min(1e-8)
Y_mean = Y_train.mean(dim=0, keepdim=True)
Y_std  = Y_train.std(dim=0, keepdim=True).clamp_min(1e-8)

X_train_n = (X_train - X_mean) / X_std
X_test_n  = (X_test  - X_mean) / X_std
Y_train_n = (Y_train - Y_mean) / Y_std
Y_test_n  = (Y_test  - Y_mean) / Y_std

X_train_n = X_train_n.to(device); X_test_n = X_test_n.to(device)
Y_train_n = Y_train_n.to(device); Y_test_n = Y_test_n.to(device)
Y_train   = Y_train.to(device);   Y_test   = Y_test.to(device)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, depth, act="relu"):
        super().__init__()
        act_layer = nn.ReLU if act == "relu" else nn.Tanh

        layers = [nn.Linear(in_dim, hidden_dim), act_layer()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act_layer()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Train the model and save the best checkpoint
def train_model(model, lr,
                max_epochs=10000, patience=80, min_delta=1e-7,
                clip_norm=1.0, wd=1e-4, log_every=5, ema_alpha=0.90,
                save_path="best_temp.pth"):

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=wd
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=25, threshold=min_delta, verbose=False
    )

    best_test = float("inf")
    no_improve = 0

    log_epochs, log_train, log_test = [], [], []
    log_train_ema, log_test_ema = [], []
    train_ema = None
    test_ema  = None

    for epoch in range(max_epochs):
        model.train()
        pred = model(X_train_n)
        loss = criterion(pred, Y_train_n)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_loss_val = criterion(model(X_train_n), Y_train_n).item()
            test_loss_val  = criterion(model(X_test_n),  Y_test_n).item()

        scheduler.step(test_loss_val)

        if test_loss_val + min_delta < best_test:
            best_test = test_loss_val
            no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1

        if epoch % log_every == 0:
            log_epochs.append(epoch)
            log_train.append(train_loss_val)
            log_test.append(test_loss_val)

            train_ema = train_loss_val if train_ema is None else (ema_alpha * train_ema + (1-ema_alpha) * train_loss_val)
            test_ema  = test_loss_val  if test_ema  is None else (ema_alpha * test_ema  + (1-ema_alpha) * test_loss_val)

            log_train_ema.append(train_ema)
            log_test_ema.append(test_ema)

        if epoch % 100 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:5d} | lr={lr_now:.2e} | Train={train_loss_val:.3e} | Test={test_loss_val:.3e}")

        if no_improve >= patience:
            print(f"⏸ Early stopping triggered at epoch={epoch}, best_test={best_test:.6e}")
            break

    model.load_state_dict(torch.load(save_path, map_location=device))

    return {
        "epochs": log_epochs,
        "train": log_train,
        "test": log_test,
        "train_ema": log_train_ema,
        "test_ema": log_test_ema,
        "best_test": best_test
    }

# Define the Optuna objective
def objective(trial):
    hidden_dim = trial.suggest_int("hidden_dim", 512, 2048)
    depth      = trial.suggest_int("depth", 10, 20)
    act        = trial.suggest_categorical("act", ["relu", "tanh"])
    lr         = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    model = MLP(N_INPUTS, hidden_dim, depth, act).to(device)

    RUN_ID = time.strftime("%Y%m%d_%H%M%S") + f"_pid{os.getpid()}"
    tmp_path = f"__optuna_mlp_best_{N_INPUTS}D_trial{trial.number}_{RUN_ID}.pth"

    hist = train_model(
        model, lr,
        max_epochs=1200, patience=80, min_delta=1e-7,
        clip_norm=1.0, wd=1e-4, log_every=20, ema_alpha=0.90,
        save_path=tmp_path
    )

    try:
        os.remove(tmp_path)
    except:
        pass

    return hist["best_test"]

# Run Bayesian optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=40, show_progress_bar=True)
best_params = study.best_params
print("Best params:", best_params)

# Retrain with the optimal hyperparameters
model = MLP(
    N_INPUTS,
    best_params["hidden_dim"],
    best_params["depth"],
    best_params["act"]
).to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_params:,d}")

best_model_path = f"MLP_{N_INPUTS}D_best_temp_shuaijian.pth"
hist = train_model(
    model,
    lr=best_params["lr"],
    max_epochs=10000,
    patience=100,
    min_delta=1e-7,
    clip_norm=1.0,
    wd=1e-4,
    log_every=5,
    ema_alpha=0.92,
    save_path=best_model_path
)

# Save the loss log
os.makedirs("logs_MLP", exist_ok=True)
df = pd.DataFrame({
    "epoch": hist["epochs"],
    "train_loss": hist["train"],
    "test_loss": hist["test"],
    "train_loss_ema": hist["train_ema"],
    "test_loss_ema": hist["test_ema"],
})
csv_path = f"logs_MLP/loss_curve_{N_INPUTS}_shuaijian.csv"
df.to_csv(csv_path, index=False)
print(f"Loss log saved to {csv_path}")

# Compute relative errors on the original scale
def rel_error(y_true, y_pred):
    eps = 1e-8
    return torch.abs(y_true - y_pred) / (torch.abs(y_true) + eps)

model.eval()
with torch.no_grad():
    y_train_pred_n = model(X_train_n)
    y_test_pred_n  = model(X_test_n)

    y_train_pred = y_train_pred_n * Y_std.to(device) + Y_mean.to(device)
    y_test_pred  = y_test_pred_n  * Y_std.to(device) + Y_mean.to(device)

    train_re = rel_error(Y_train, y_train_pred)
    test_re  = rel_error(Y_test,  y_test_pred)

print(f"Train mean relative error MRE = {train_re.mean().item():.6f}")
print(f"Train maximum relative error MaxRE = {train_re.max().item():.6f}")
print(f"Test mean relative error MRE = {test_re.mean().item():.6f}")
print(f"Test maximum relative error MaxRE = {test_re.max().item():.6f}")


# Save the final model and normalization statistics
os.makedirs("MLP_weight", exist_ok=True)
final_path = f"MLP_weight/MLP_{N_INPUTS}D_best_shuaijian.pth"
torch.save({
    "state_dict": model.state_dict(),
    "X_mean": X_mean,
    "X_std": X_std,
    "Y_mean": Y_mean,
    "Y_std": Y_std,
    "best_params": best_params,
}, final_path)
print(f"Final model saved to {final_path}")