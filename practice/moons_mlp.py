import os, random
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # (B,)


def make_data(n=2000, noise=0.25, test_size=0.2):
    X, y = make_moons(n_samples=n, noise=noise, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=0, stratify=y)
    scaler = StandardScaler().fit(X_tr)  # 仅用训练集拟合，避免泄漏
    X_tr = scaler.transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)
    return (torch.from_numpy(X_tr), torch.from_numpy(y_tr)), (torch.from_numpy(X_te), torch.from_numpy(y_te))


def evaluate(model, loader):
    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            all_logits.append(logits.cpu())
            all_targets.append(yb.cpu())
    logits = torch.cat(all_logits).numpy()
    y_true = torch.cat(all_targets).numpy()
    probs = 1 / (1 + np.exp(-logits))
    auc = roc_auc_score(y_true, probs)
    acc = accuracy_score(y_true, (probs >= 0.5).astype(np.float32))
    return auc, acc


def save_ckpt(path, model, optim, epoch, best_auc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "epoch": epoch,
        "best_auc": best_auc
    }, path)


def load_ckpt(path, model, optim):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optim.load_state_dict(ckpt["optim"])
    return ckpt["epoch"], ckpt["best_auc"]


def main():
    set_seed(42)
    (X_tr, y_tr), (X_te, y_te) = make_data(n=2000, noise=0.25, test_size=0.2)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=512, shuffle=False)

    model = MLP(in_dim=2, hidden=64)
    criterion = nn.BCEWithLogitsLoss()  # logits -> loss 更稳定
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)

    ckpt_path = "./../out/best.ckpt"
    start_epoch, best_auc = 0, -1.0
    if os.path.exists(ckpt_path):
        print(f"[Info] Resume from {ckpt_path}")
        start_epoch, best_auc = load_ckpt(ckpt_path, model, optimizer)
        print(f"[Info] start_epoch={start_epoch}, best_auc={best_auc:.4f}")

    max_epochs = 200
    patience, bad = 20, 0

    for epoch in range(start_epoch, max_epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
        tr_loss = running / len(train_loader.dataset)

        val_auc, val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} | val_auc={val_auc:.4f} | val_acc={val_acc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            save_ckpt(ckpt_path, model, optimizer, epoch + 1, best_auc)
            bad = 0
        else:
            bad += 1

        if best_auc >= 0.95:
            print("[OK] 达到目标：AUC ≥ 0.95")
            break
        if bad >= patience:
            print("[EarlyStop] 验证集 AUC 未提升，提前停止。")
            break

    print(f"[Result] best AUC = {best_auc:.4f} | ckpt={ckpt_path}")


if __name__ == "__main__":
    main()
