# linear_regression_train.py
import math, os, random, json
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


def set_seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)


def make_data(n=1000, noise_std=0.2):
    set_seed(0)
    x = torch.empty(n, 1).uniform_(-2.0, 2.0)
    y = 3.0 * x - 2.0 + noise_std * torch.randn_like(x)
    return x, y


def r2_score(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-12)


def save_ckpt(path, model, optim, epoch, best_r2):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "epoch": epoch,
        "best_r2": best_r2
    }, path)


def load_ckpt(path, model, optim):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optim.load_state_dict(ckpt["optim"])
    return ckpt["epoch"], ckpt["best_r2"]


def main():
    set_seed(42)
    # 1) 数据
    x, y = make_data(n=1000, noise_std=0.2)
    n_train = int(0.8 * len(x))
    x_tr, y_tr = x[:n_train], y[:n_train]
    x_te, y_te = x[n_train:], y[n_train:]

    # 2) DataLoader
    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_te, y_te), batch_size=256, shuffle=False)

    # 3) 模型/损失/优化器
    model = nn.Linear(1, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # 4) 可选：从 checkpoint 恢复
    start_epoch, best_r2 = 0, -1e9
    ckpt_path = "./../out/best.ckpt"
    if os.path.exists(ckpt_path):
        print(f"[Info] Resume from {ckpt_path}")
        start_epoch, best_r2 = load_ckpt(ckpt_path, model, optimizer)
        print(f"[Info] start_epoch={start_epoch}, best_r2={best_r2:.4f}")

    # 5) 训练循环
    max_epochs = 50
    for epoch in range(start_epoch, max_epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            # TODO: 前向、计算损失
            pred = model(xb)
            loss = criterion(pred, yb)
            # TODO: 反向与更新
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(xb)
        train_loss = running_loss / len(train_loader.dataset)

        # 6) 评估 R²
        model.eval()
        with torch.no_grad():
            y_pred = []
            y_true = []
            for xb, yb in test_loader:
                y_pred.append(model(xb))
                y_true.append(yb)
            y_pred = torch.cat(y_pred, dim=0)
            y_true = torch.cat(y_true, dim=0)
            r2 = r2_score(y_true, y_pred).item()

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | R2={r2:.4f}")

        # 7) 保存最优
        if r2 > best_r2:
            best_r2 = r2
            save_ckpt(ckpt_path, model, optimizer, epoch + 1, best_r2)

        # 8) 及格线：R² ≥ 0.99
        if r2 >= 0.99:
            print("[OK] 达到目标 R² ≥ 0.99，训练完成。")
            break

    print(f"[Result] best R² = {best_r2:.4f} | ckpt={ckpt_path}")


if __name__ == "__main__":
    main()
