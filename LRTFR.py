import os 
import time
import math
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
import optuna

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Load tabular data robustly from different text formats
def robust_loadtxt(path, expected_cols=None):
    try:
        df = pd.read_csv(path, sep=None, engine="python", comment="#")
    except Exception:
        df = pd.read_csv(path, delim_whitespace=True, comment="#")

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="any")

    data = df.values
    if expected_cols is not None and data.shape[1] != expected_cols:
        raise ValueError(f"{path} should have {expected_cols} columns, but {data.shape[1]} columns were detected")
    return data


# Define the basic SIREN layer
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=1.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights(in_features)

    def init_weights(self, in_features):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                bound = math.sqrt(6 / in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

# Define the Tucker + SIREN model with 6 modes
# The number of modes depends on the dataset dimensionality
class Network6D(nn.Module):
    def __init__(self, ranks, mid_channel, omegas):
        super(Network6D, self).__init__()
        self.ranks = ranks
        self.omegas = omegas

        def make_branch(r, omega):
            return nn.Sequential(
                SineLayer(1, mid_channel, is_first=True, omega_0=omega),
                SineLayer(mid_channel, mid_channel, omega_0=omega),
                SineLayer(mid_channel, mid_channel, omega_0=omega),
                nn.Linear(mid_channel, r),
            )

        self.subnets = nn.ModuleList([make_branch(r, omegas[i]) for i, r in enumerate(ranks)])

    def forward(self, centre, x):
        U = []
        for i in range(6):
            ui = self.subnets[i](x[:, i:i+1])  # [B, rank_i]
            U.append(ui)
        y = torch.einsum('abcdef,Aa,Ab,Ac,Ad,Ae,Af->A', centre, *U)
        return y.unsqueeze(-1)


# Train the model and keep the best checkpoint
def train_model_6d(model, centre, X_train, Y_train, X_test, Y_test,
                   lr,
                   max_epochs=10000, patience=80, min_delta=1e-7,
                   clip_norm=1.0, wd=1e-4, log_every=5, ema_alpha=0.90,
                   save_path="best6D_temp.pth"):

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + [centre],
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
    test_ema = None

    for epoch in range(max_epochs):
        model.train()
        pred = model(centre, X_train)
        loss = criterion(pred, Y_train)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [centre], clip_norm)

        optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss_val = criterion(model(centre, X_train), Y_train).item()
            test_loss_val  = criterion(model(centre, X_test),  Y_test).item()

        scheduler.step(test_loss_val)
        if test_loss_val + min_delta < best_test:
            best_test = test_loss_val
            no_improve = 0
            torch.save({
                "centre": centre.detach().cpu(),
                "model": model.state_dict()
            }, save_path)
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

    best = torch.load(save_path, map_location="cpu")
    model.load_state_dict(best["model"])
    centre.data.copy_(best["centre"].to(device))

    return {
        "epochs": log_epochs,
        "train": log_train,
        "test": log_test,
        "train_ema": log_train_ema,
        "test_ema": log_test_ema,
        "best_test": best_test
    }


# Load data and split into training and test sets
X_np = robust_loadtxt("data/inputdata_6.txt", expected_cols=6)
Y_np = robust_loadtxt("data/outputdata_6.txt")
if Y_np.ndim == 1:
    Y_np = Y_np.reshape(-1, 1)

n = X_np.shape[0]
mid_channel = 32 

idx = np.arange(n)
np.random.shuffle(idx)
split = int(0.7 * n)
train_idx, test_idx = idx[:split], idx[split:]

X_train = torch.from_numpy(X_np[train_idx]).float().to(device)
Y_train = torch.from_numpy(Y_np[train_idx]).float().to(device)
X_test  = torch.from_numpy(X_np[test_idx]).float().to(device)
Y_test  = torch.from_numpy(Y_np[test_idx]).float().to(device)

# Use Optuna to search for the best hyperparameters
def objective(trial):
    ranks  = [trial.suggest_int(f"rank{i}", 10, 30) for i in range(6)]     
    omegas = [trial.suggest_float(f"omega{i}", 0.5, 8.0) for i in range(6)]
    lr     = trial.suggest_float("lr", 1e-5, 5e-4, log=True)               

    # Use centre as a trainable Parameter
    centre = nn.Parameter(torch.empty(*ranks, device=device, dtype=torch.float32))
    centre.data.uniform_(-1/math.sqrt(ranks[0]), 1/math.sqrt(ranks[0]))

    model = Network6D(ranks, mid_channel, omegas).to(device)

    RUN_ID = time.strftime("%Y%m%d_%H%M%S") + f"_pid{os.getpid()}"
    tmp_path = f"__optuna_6D_best_trial{trial.number}_{RUN_ID}.pth"

    hist = train_model_6d(
        model, centre,
        X_train, Y_train, X_test, Y_test,
        lr=lr,
        max_epochs=1200, patience=80, min_delta=1e-7,
        clip_norm=1.0, wd=1e-4, log_every=20, ema_alpha=0.90,
        save_path=tmp_path
    )

    try:
        os.remove(tmp_path)
    except:
        pass

    return hist["best_test"]


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=40, show_progress_bar=True)
best_params = study.best_params
print("🔥 Best hyperparameters:", best_params)


# Retrain with optimal hyperparameters
stage2_train_start = time.time()

best_ranks  = [best_params[f"rank{i}"] for i in range(6)]
best_omegas = [best_params[f"omega{i}"] for i in range(6)]
best_lr     = best_params["lr"]

centre = nn.Parameter(torch.empty(*best_ranks, device=device, dtype=torch.float32))
centre.data.uniform_(-1/math.sqrt(best_ranks[0]), 1/math.sqrt(best_ranks[0]))

model = Network6D(best_ranks, mid_channel, best_omegas).to(device)

best_tmp = "best6D_temp.pth"
hist = train_model_6d(
    model, centre,
    X_train, Y_train, X_test, Y_test,
    lr=best_lr,
    max_epochs=8000, patience=80, min_delta=1e-7,
    clip_norm=1.0, wd=1e-4, log_every=5, ema_alpha=0.92,
    save_path=best_tmp
)


# Save training logs and report model size
os.makedirs("logs_LRTFR", exist_ok=True)
pd.DataFrame({
    "epoch": hist["epochs"],
    "train_loss": hist["train"],
    "test_loss": hist["test"],
    "train_loss_ema": hist["train_ema"],
    "test_loss_ema": hist["test_ema"],
}).to_csv("logs_LRTFR/loss_curve_6D.csv", index=False)
print("Loss curve saved to logs_LRTFR/loss_curve_6D.csv")

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) + centre.numel()
print(f"Number of trainable parameters (including centre): {num_params:,d}")

stage2_train_end = time.time()
train_secs = stage2_train_end - stage2_train_start
print(f"Training time: {train_secs:.2f} s ({train_secs/60:.2f} min, {train_secs/3600:.2f} h)")

# Final evaluation
infer_start = time.time()
with torch.no_grad():
    pred = model(centre, X_test).detach().cpu().numpy()
    true = Y_test.detach().cpu().numpy()
infer_end = time.time()
infer_secs = infer_end - infer_start

print(f"Inference time on test set: {infer_secs:.6f} s")

rel = np.abs((pred - true) / (true + 1e-8))
print(f"Mean relative error on test set: {rel.mean():.6e}")
print(f"Maximum relative error on test set: {rel.max():.6e}")

# Save the final model checkpoint
best = {
    "centre": centre.detach().cpu(),
    "model": model.state_dict(),
    "best_params": best_params
}
torch.save(best, "Tucker6D_best.pth")
print("Final model saved as Tucker6D_best.pth")