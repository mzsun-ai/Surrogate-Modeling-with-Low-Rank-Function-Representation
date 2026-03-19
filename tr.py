import os
import math
import time
import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
import optuna


# Utility functions
def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def robust_loadtxt(path: str):
    try:
        df = pd.read_csv(path, sep=None, engine="python", comment="#")
    except Exception:
        df = pd.read_csv(path, delim_whitespace=True, comment="#")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="any")
    return df.values


def train_test_split(X, Y, split=0.7, seed=42):
    n = X.shape[0]
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    s = int(split * n)
    tr, te = idx[:s], idx[s:]
    return X[tr], Y[tr], X[te], Y[te]


def standardize_train_only(X_tr, X_te, Y_tr, Y_te, eps=1e-8):
    X_mean = X_tr.mean(axis=0, keepdims=True)
    X_std = X_tr.std(axis=0, keepdims=True)
    X_std = np.clip(X_std, eps, None)

    Y_mean = Y_tr.mean(axis=0, keepdims=True)
    Y_std = Y_tr.std(axis=0, keepdims=True)
    Y_std = np.clip(Y_std, eps, None)

    X_tr_n = (X_tr - X_mean) / X_std
    X_te_n = (X_te - X_mean) / X_std
    Y_tr_n = (Y_tr - Y_mean) / Y_std
    Y_te_n = (Y_te - Y_mean) / Y_std
    return (X_tr_n, X_te_n, Y_tr_n, Y_te_n, X_mean, X_std, Y_mean, Y_std)


def compute_relative_error(
    model: nn.Module,
    X: torch.Tensor,
    Y_norm: torch.Tensor,
    Y_mean: np.ndarray,
    Y_std: np.ndarray,
    eps: float = 1e-12,
):
    """
    Relative error on the original scale:
        RE = |y_pred - y_true| / (|y_true| + eps)
    """
    device = X.device
    Y_mean_t = torch.from_numpy(Y_mean).float().to(device)
    Y_std_t  = torch.from_numpy(Y_std).float().to(device)

    model.eval()
    with torch.no_grad():
        pred_norm = model(X)
        pred = pred_norm * Y_std_t + Y_mean_t
        y_true = Y_norm * Y_std_t + Y_mean_t
        re = (pred - y_true).abs() / (y_true.abs() + eps)
        return re.view(-1)


# SIREN blocks
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
                self.linear.weight.uniform_(
                    -1.0 / self.linear.in_features, 1.0 / self.linear.in_features
                )
            else:
                bound = math.sqrt(6.0 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SirenBranch(nn.Module):
    def __init__(self, hidden_dim: int, depth: int, omega: float, out_dim: int):
        super().__init__()
        layers = [SineLayer(1, hidden_dim, is_first=True, omega_0=omega)]
        for _ in range(depth - 1):
            layers.append(SineLayer(hidden_dim, hidden_dim, omega_0=omega))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_1d):
        return self.net(x_1d)


# TR model
class TR_SIREN(nn.Module):
    """
    Tensor Ring:
      each dimension outputs Gi(x_i) in R^{r x r}
      y = trace( Π_i Gi(x_i) )
    """
    def __init__(self, d_inputs: int, ring_rank: int, hidden_dim: int, depth: int, omegas):
        super().__init__()
        assert len(omegas) == d_inputs
        self.d_inputs = d_inputs
        self.r = ring_rank

        self.branches = nn.ModuleList()
        for i in range(d_inputs):
            out_dim = ring_rank * ring_rank
            self.branches.append(SirenBranch(hidden_dim, depth, omegas[i], out_dim))

        self.final_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.final_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x):
        B = x.shape[0]
        mats = []
        for i in range(self.d_inputs):
            v = self.branches[i](x[:, i:i+1])
            m = v.view(B, self.r, self.r)
            mats.append(m)

        acc = mats[0]
        for i in range(1, self.d_inputs):
            acc = torch.bmm(acc, mats[i])

        y = torch.einsum("bii->b", acc).unsqueeze(1)
        return y * self.final_scale + self.final_bias


# Training loop
def train_model(
    model: nn.Module,
    X_tr: torch.Tensor,
    Y_tr: torch.Tensor,
    X_te: torch.Tensor,
    Y_te: torch.Tensor,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    min_delta: float,
    clip_norm: float,
    log_every: int,
    save_path: str,
):
    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=25,
        threshold=min_delta,
        verbose=False,
    )

    best_te = float("inf")
    no_improve = 0

    logs = {"epoch": [], "train_loss": [], "test_loss": []}

    for ep in range(max_epochs):
        model.train()
        pred = model(X_tr)
        loss = criterion(pred, Y_tr)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            tr_loss = criterion(model(X_tr), Y_tr).item()
            te_loss = criterion(model(X_te), Y_te).item()

        scheduler.step(te_loss)

        if ep % log_every == 0:
            logs["epoch"].append(ep)
            logs["train_loss"].append(tr_loss)
            logs["test_loss"].append(te_loss)

        if te_loss + min_delta < best_te:
            best_te = te_loss
            no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1

        if ep % 100 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {ep:5d} | lr={lr_now:.2e} | Train={tr_loss:.3e} | Test={te_loss:.3e}")

        if no_improve >= patience:
            print(f"Early stopping at epoch={ep}, best_test={best_te:.6e}")
            break

    model.load_state_dict(torch.load(save_path, map_location=X_tr.device))
    return best_te, logs


# Main entry and Optuna search
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x_path", type=str, required=True)
    ap.add_argument("--y_path", type=str, required=True)
    ap.add_argument("--y_col", type=int, default=1, help="0-based column index in Y file")
    ap.add_argument("--split", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--n_trials", type=int, default=40)

    ap.add_argument("--stage1_epochs", type=int, default=600)
    ap.add_argument("--stage1_patience", type=int, default=80)

    ap.add_argument("--stage2_epochs", type=int, default=8000)
    ap.add_argument("--stage2_patience", type=int, default=120)

    ap.add_argument("--min_delta", type=float, default=1e-7)
    ap.add_argument("--clip_norm", type=float, default=1.0)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--log_every", type=int, default=10)

    ap.add_argument("--out_dir", type=str, default="out_tr")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = robust_loadtxt(args.x_path)
    Y_all = robust_loadtxt(args.y_path)
    if Y_all.ndim == 1:
        Y_all = Y_all.reshape(-1, 1)

    if args.y_col < 0 or args.y_col >= Y_all.shape[1]:
        raise ValueError(f"--y_col={args.y_col} out of range, Y has {Y_all.shape[1]} cols")

    Y = Y_all[:, args.y_col:args.y_col+1]

    X_tr, Y_tr, X_te, Y_te = train_test_split(X, Y, split=args.split, seed=args.seed)
    X_tr_n, X_te_n, Y_tr_n, Y_te_n, X_mean, X_std, Y_mean, Y_std = standardize_train_only(
        X_tr, X_te, Y_tr, Y_te
    )

    X_tr_t = torch.from_numpy(X_tr_n).float().to(device)
    X_te_t = torch.from_numpy(X_te_n).float().to(device)
    Y_tr_t = torch.from_numpy(Y_tr_n).float().to(device)
    Y_te_t = torch.from_numpy(Y_te_n).float().to(device)

    D = X.shape[1]
    run_id = time.strftime("%Y%m%d_%H%M%S") + f"_pid{os.getpid()}"
    print(f"TR run_id={run_id} | D={D} | device={device} | y_col={args.y_col}")

    def objective(trial: optuna.Trial):
        ring_rank = trial.suggest_int("ring_rank", 4, 64)
        hidden_dim = trial.suggest_int("hidden_dim", 32, 512)
        depth = trial.suggest_int("depth", 4, 10)
        omegas = [trial.suggest_float(f"omega{i}", 0.5, 8.0) for i in range(D)]
        lr = trial.suggest_float("lr", 1e-6, 5e-4, log=True)

        model = TR_SIREN(D, ring_rank, hidden_dim, depth, omegas).to(device)

        tmp_ckpt = os.path.join(args.out_dir, f"__tr_trial{trial.number}_{run_id}.pth")
        best_te, _ = train_model(
            model=model,
            X_tr=X_tr_t, Y_tr=Y_tr_t,
            X_te=X_te_t, Y_te=Y_te_t,
            lr=lr,
            weight_decay=args.weight_decay,
            max_epochs=args.stage1_epochs,
            patience=args.stage1_patience,
            min_delta=args.min_delta,
            clip_norm=args.clip_norm,
            log_every=max(20, args.log_every),
            save_path=tmp_ckpt,
        )
        try:
            os.remove(tmp_ckpt)
        except Exception:
            pass
        return best_te

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best = study.best_params
    print("Best params:", best)

    ring_rank = best["ring_rank"]
    hidden_dim = best["hidden_dim"]
    depth = best["depth"]
    omegas = [best[f"omega{i}"] for i in range(D)]
    lr = best["lr"]

    model = TR_SIREN(D, ring_rank, hidden_dim, depth, omegas).to(device)

    best_ckpt = os.path.join(args.out_dir, f"TR_{D}D_best_{run_id}.pth")

    # Measure stage-2 training time only
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    best_te, logs = train_model(
        model=model,
        X_tr=X_tr_t, Y_tr=Y_tr_t,
        X_te=X_te_t, Y_te=Y_te_t,
        lr=lr,
        weight_decay=args.weight_decay,
        max_epochs=args.stage2_epochs,
        patience=args.stage2_patience,
        min_delta=args.min_delta,
        clip_norm=args.clip_norm,
        log_every=args.log_every,
        save_path=best_ckpt,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.time() - t0

    # Compute relative errors on the original scale
    train_re = compute_relative_error(model, X_tr_t, Y_tr_t, Y_mean=Y_mean, Y_std=Y_std, eps=1e-12)
    test_re  = compute_relative_error(model, X_te_t, Y_te_t, Y_mean=Y_mean, Y_std=Y_std, eps=1e-12)

  
    print(f"Train mean relative error MRE = {train_re.mean().item():.6f}")
    print(f"Train maximum relative error MaxRE = {train_re.max().item():.6f}")
    print(f"Test mean relative error MRE = {test_re.mean().item():.6f}")
    print(f"Test maximum relative error MaxRE = {test_re.max().item():.6f}")
    print(f"Stage-2 training time (excluding Optuna) = {train_seconds:.2f} s")

    # Save logs
    log_csv = os.path.join(args.out_dir, f"loss_curve_TR_{D}D_{run_id}.csv")
    pd.DataFrame(logs).to_csv(log_csv, index=False)
    print(f"Loss saved: {log_csv}")

    # Save model bundle
    bundle_path = os.path.join(args.out_dir, f"TR_{D}D_bundle_{run_id}.pth")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "X_mean": torch.from_numpy(X_mean).float(),
            "X_std": torch.from_numpy(X_std).float(),
            "Y_mean": torch.from_numpy(Y_mean).float(),
            "Y_std": torch.from_numpy(Y_std).float(),
            "best_params": best,
            "best_test_loss": best_te,
            "x_path": args.x_path,
            "y_path": args.y_path,
            "y_col": args.y_col,
            "D": D,
            "kind": "TR_SIREN",
            "stage2_train_seconds": train_seconds,
            "train_MRE": train_re.mean().item(),
            "train_MaxRE": train_re.max().item(),
            "test_MRE": test_re.mean().item(),
            "test_MaxRE": test_re.max().item(),
        },
        bundle_path,
    )
    print(f"Bundle saved: {bundle_path}")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {num_params:,d}")


if __name__ == "__main__":
    main()