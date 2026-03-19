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
idx = np.arange(n)
np.random.shuffle(idx)
split = int(0.7 * n)
train_idx, test_idx = idx[:split], idx[split:]
X_train, Y_train = X[train_idx], Y[train_idx]
X_test,  Y_test  = X[test_idx],  Y[test_idx]
N_INPUTS = X.shape[1]

# Standardize using training-set statistics
X_mean = X_train.mean(dim=0, keepdim=True)
X_std  = X_train.std(dim=0, keepdim=True).clamp_min(1e-8)
Y_mean = Y_train.mean(dim=0, keepdim=True)
Y_std  = Y_train.std(dim=0, keepdim=True).clamp_min(1e-8)

X_train = (X_train - X_mean) / X_std
X_test  = (X_test  - X_mean) / X_std
Y_train_n = (Y_train - Y_mean) / Y_std
Y_test_n  = (Y_test  - Y_mean) / Y_std

X_train = X_train.to(device); X_test = X_test.to(device)
Y_train_n = Y_train_n.to(device); Y_test_n = Y_test_n.to(device)
Y_train = Y_train.to(device); Y_test = Y_test.to(device)

# Define the PLRNet model
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=1.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features,
                                            1 / self.linear.in_features)
            else:
                bound = math.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class MultiPairwiseTucker(nn.Module):
    def __init__(self, n_inputs, ranks, hidden_sizes, depths, omegas,
                 final_hidden, final_depth, final_omega):
        super().__init__()
        self.n_inputs = n_inputs
        self.ranks = ranks
        self.embed_nets = nn.ModuleList()

        for i in range(n_inputs):
            layers = [SineLayer(1, hidden_sizes[i], is_first=True, omega_0=omegas[i])]
            for _ in range(depths[i] - 1):
                layers.append(SineLayer(hidden_sizes[i], hidden_sizes[i], omega_0=omegas[i]))
            layers.append(nn.Linear(hidden_sizes[i], ranks[i]))
            self.embed_nets.append(nn.Sequential(*layers))

        self.cores = nn.ParameterDict()
        for i in range(n_inputs):
            for j in range(i + 1, n_inputs):
                self.cores[f"C_{i}_{j}"] = nn.Parameter(torch.randn(ranks[i], ranks[j]) * 0.05)

        in_dim = n_inputs * (n_inputs - 1) // 2
        final_layers = [SineLayer(in_dim, final_hidden, is_first=True, omega_0=final_omega)]
        for _ in range(final_depth - 1):
            final_layers.append(SineLayer(final_hidden, final_hidden, omega_0=final_omega))
        final_layers.append(nn.Linear(final_hidden, 1))
        self.final = nn.Sequential(*final_layers)

    def forward(self, x):
        embeds = [net(x[:, i:i + 1]) for i, net in enumerate(self.embed_nets)]
        pair_vals = []
        for i in range(self.n_inputs):
            for j in range(i + 1, self.n_inputs):
                C = self.cores[f"C_{i}_{j}"]
                val = torch.sum((embeds[i] @ C) * embeds[j], dim=1, keepdim=True)
                pair_vals.append(val)
        pair_vals = torch.cat(pair_vals, dim=1)
        return self.final(pair_vals)

# Train the model and save the best checkpoint
def train_model(model, lr, max_epochs=10000, patience=80, min_delta=1e-7,
                clip_norm=1.0, wd=1e-4, log_every=5, ema_alpha=0.90,
                save_path="best_temp_xiangsu.pth"):

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=lr,
                                  betas=(0.9, 0.99),
                                  eps=1e-8,
                                  weight_decay=wd)

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
        pred = model(X_train)
        loss = criterion(pred, Y_train_n)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_loss_val = criterion(model(X_train), Y_train_n).item()
            test_loss_val  = criterion(model(X_test),  Y_test_n).item()

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
    n_inputs = N_INPUTS

    ranks = [trial.suggest_int(f"r{i}", 4, 64) for i in range(n_inputs)]
    hidden_sizes = [trial.suggest_int(f"h{i}", 32, 512) for i in range(n_inputs)]
    depths = [trial.suggest_int(f"d{i}", 1, 4) for i in range(n_inputs)]
    omegas = [trial.suggest_float(f"omega{i}", 0.5, 5.0) for i in range(n_inputs)]

    final_hidden = trial.suggest_int("final_hidden", 32, 512)
    final_depth  = trial.suggest_int("final_depth", 1, 3)
    final_omega  = trial.suggest_float("final_omega", 0.5, 5.0)

    lr = trial.suggest_float("lr", 1e-6, 3e-4, log=True)

    model = MultiPairwiseTucker(
        n_inputs, ranks, hidden_sizes, depths, omegas,
        final_hidden, final_depth, final_omega
    ).to(device)

    RUN_ID = time.strftime("%Y%m%d_%H%M%S") + f"_pid{os.getpid()}"
    tmp_path = f"__optuna_best_{N_INPUTS}D_trial{trial.number}_{RUN_ID}.pth"
    hist = train_model(model, lr, max_epochs=1200, patience=80,
                       clip_norm=1.0, wd=1e-4, log_every=20,
                       save_path=tmp_path)
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
n_inputs = N_INPUTS
model = MultiPairwiseTucker(
    n_inputs,
    [best_params[f"r{i}"] for i in range(n_inputs)],
    [best_params[f"h{i}"] for i in range(n_inputs)],
    [best_params[f"d{i}"] for i in range(n_inputs)],
    [best_params[f"omega{i}"] for i in range(n_inputs)],
    best_params["final_hidden"],
    best_params["final_depth"],
    best_params["final_omega"]
).to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_params:,d}")

best_model_path = f"pairwise_tucker_{n_inputs}D_best_temp_xiangsu.pth"
hist = train_model(model,
                   lr=best_params["lr"],
                   max_epochs=10000,
                   patience=3000,
                   min_delta=1e-7,
                   clip_norm=1.0,
                   wd=1e-4,
                   log_every=5,
                   ema_alpha=0.92,
                   save_path=best_model_path)

# Save the loss log
os.makedirs("logs", exist_ok=True)
df = pd.DataFrame({
    "epoch": hist["epochs"],
    "train_loss": hist["train"],
    "test_loss": hist["test"],
    "train_loss_ema": hist["train_ema"],
    "test_loss_ema": hist["test_ema"],
})
csv_path = f"logs/loss_curve_{n_inputs}_xiangsu.csv"
df.to_csv(csv_path, index=False)
print(f"Loss log saved to {csv_path}")

# Compute relative errors on the original scale
def rel_error(y_true, y_pred):
    eps = 1e-8
    return torch.abs(y_true - y_pred) / (torch.abs(y_true) + eps)

model.eval()
with torch.no_grad():
    y_train_pred_n = model(X_train)
    y_test_pred_n  = model(X_test)

    y_train_pred = y_train_pred_n * Y_std.to(device) + Y_mean.to(device)
    y_test_pred  = y_test_pred_n  * Y_std.to(device) + Y_mean.to(device)

    train_re = rel_error(Y_train, y_train_pred)
    test_re  = rel_error(Y_test,  y_test_pred)


print(f"Train mean relative error MRE = {train_re.mean().item():.6f}")
print(f"Train maximum relative error MaxRE = {train_re.max().item():.6f}")
print(f"Test mean relative error MRE = {test_re.mean().item():.6f}")
print(f"Test maximum relative error MaxRE = {test_re.max().item():.6f}")


# Save the final model
final_path = f"PLRNet_weight/pairwise_tucker_{n_inputs}D_best_xiangsu.pth"
os.makedirs(os.path.dirname(final_path), exist_ok=True)
torch.save({
    "state_dict": model.state_dict(),
    "X_mean": X_mean,
    "X_std": X_std,
    "Y_mean": Y_mean,
    "Y_std": Y_std,
    "best_params": best_params,
}, final_path)
print(f"Model saved to {final_path}")