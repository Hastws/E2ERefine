import os, argparse, random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def set_seed(seed=42):
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class CNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total, correct, running = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = ce(logits, yb)
        running += loss.item() * yb.size(0)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return running / total, correct / total


def build_optimizer(model, opt_name, lr, weight_decay):
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    else:
        raise ValueError("opt_name must be adam or sgd")


def build_scheduler(optimizer, sched_name, base_lr, max_lr, steps_per_epoch, epochs):
    if sched_name == "none":
        return None, "none"
    if sched_name == "steplr":
        # 每 epoch 调度
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 2), gamma=0.1), "epoch"
    if sched_name == "onecycle":
        # 每 step 调度（推荐写法：提供 steps_per_epoch 和 epochs）
        oc = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            pct_start=0.1,  # 前 30% 逐步升到 max_lr（相当于 warmup）
            div_factor=25.0,  # 初始 lr = max_lr / div_factor
            final_div_factor=1e4,  # 末尾降到 max_lr / (div_factor*final_div_factor)
            anneal_strategy="cos"
        )
        return oc, "step"
    raise ValueError("sched_name must be none/steplr/onecycle")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="./data")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--opt", type=str, default="sgd", choices=["sgd", "adam"])
    ap.add_argument("--sched", type=str, default="onecycle", choices=["none", "steplr", "onecycle"])
    ap.add_argument("--lr", type=float, default=0.1, help="base lr (for adam default try 1e-3)")
    ap.add_argument("--max_lr", type=float, default=0.2, help="OneCycle peak lr (sgd常设0.2~0.5；adam可设2e-3~3e-3)")
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.cuda.is_available()

    # 数据
    mean, std = (0.1307,), (0.3081,)
    train_tf = transforms.Compose([
        transforms.RandomCrop(28, padding=2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_ds = datasets.MNIST(args.data, train=True, download=True, transform=train_tf)
    test_ds = datasets.MNIST(args.data, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
                              persistent_workers=args.num_workers > 0)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                             num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
                             persistent_workers=args.num_workers > 0)

    model = CNN2().to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = build_optimizer(model, args.opt, args.lr, args.wd)
    scheduler, sched_mode = build_scheduler(
        optimizer, args.sched, base_lr=args.lr, max_lr=args.max_lr,
        steps_per_epoch=len(train_loader), epochs=args.epochs
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    ce = nn.CrossEntropyLoss()

    ckpt_path = f"./../out/{args.opt}_{args.sched}.ckpt"
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    best_acc, first98 = 0.0, None

    for epoch in range(args.epochs):
        model.train()
        run_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(xb)
                loss = ce(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and sched_mode == "step":
                scheduler.step()
            run_loss += loss.item() * yb.size(0)
        train_loss = run_loss / len(train_loader.dataset)

        if scheduler is not None and sched_mode == "epoch":
            scheduler.step()

        test_loss, test_acc = evaluate(model, test_loader, device)
        if first98 is None and test_acc >= 0.98:
            first98 = epoch
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_acc": best_acc
            }, ckpt_path)

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:02d} | lr={cur_lr:.5f} | train_loss={train_loss:.4f} | "
              f"test_loss={test_loss:.4f} | test_acc={test_acc:.4f}")

        if best_acc >= 0.98:
            # 不提前停，为了看 first98 在哪里；你也可以 break
            pass

    print(f"[Result] best_acc={best_acc:.4f} | first_epoch_>=98%={first98} | ckpt={ckpt_path}")


if __name__ == "__main__":
    main()

# Epoch 00 | lr=0.08466 | train_loss=0.3248 | test_loss=0.0627 | test_acc=0.9799
# Epoch 01 | lr=0.24154 | train_loss=0.1189 | test_loss=0.0706 | test_acc=0.9772
# Epoch 02 | lr=0.37443 | train_loss=0.1384 | test_loss=0.0510 | test_acc=0.9832
# Epoch 03 | lr=0.39774 | train_loss=nan | test_loss=1504035499.4176 | test_acc=0.0980
# Epoch 04 | lr=0.37313 | train_loss=nan | test_loss=1504035499.4176 | test_acc=0.0980
# Epoch 05 | lr=0.32457 | train_loss=nan | test_loss=1504035499.4176 | test_acc=0.0980
# Epoch 06 | lr=0.25880 | train_loss=nan | test_loss=1504035499.4176 | test_acc=0.0980
# Epoch 07 | lr=0.18490 | train_loss=nan | test_loss=1504035499.4176 | test_acc=0.0980
# Epoch 08 | lr=0.11308 | train_loss=nan | test_loss=1504035499.4176 | test_acc=0.0980
# Epoch 09 | lr=0.05328 | train_loss=nan | test_loss=1504035499.4176 | test_acc=0.0980
# Epoch 10 | lr=0.01377 | train_loss=nan | test_loss=1504035499.4176 | test_acc=0.0980
# Epoch 11 | lr=0.00000 | train_loss=nan | test_loss=1504035499.4176 | test_acc=0.0980
# 上面说的是典型的爆炸场景，成功率变为0.1左右， 说明结果已经是随机选择的了，loss已经变为nan，说明已经爆炸，
# 降低最大lr，同时降低初始lr，从0.3改为0.2

# Epoch 00 | lr=0.18736 | train_loss=0.3777 | test_loss=0.0567 | test_acc=0.9810
# Epoch 01 | lr=0.19729 | train_loss=0.0934 | test_loss=0.0368 | test_acc=0.9881
# Epoch 02 | lr=0.18657 | train_loss=0.0681 | test_loss=0.0327 | test_acc=0.9879
# Epoch 03 | lr=0.16858 | train_loss=0.0583 | test_loss=0.0435 | test_acc=0.9859
# Epoch 04 | lr=0.14482 | train_loss=0.0543 | test_loss=0.0235 | test_acc=0.9928
# Epoch 05 | lr=0.11730 | train_loss=0.0499 | test_loss=0.0347 | test_acc=0.9887
# Epoch 06 | lr=0.08833 | train_loss=0.0433 | test_loss=0.0295 | test_acc=0.9905
# Epoch 07 | lr=0.06034 | train_loss=0.0372 | test_loss=0.0204 | test_acc=0.9932
# Epoch 08 | lr=0.03567 | train_loss=0.0310 | test_loss=0.0320 | test_acc=0.9894
# Epoch 09 | lr=0.01642 | train_loss=0.0238 | test_loss=0.0177 | test_acc=0.9943
# Epoch 10 | lr=0.00418 | train_loss=0.0197 | test_loss=0.0149 | test_acc=0.9949
# Epoch 11 | lr=0.00000 | train_loss=0.0152 | test_loss=0.0143 | test_acc=0.9949
# 上面说的就是简单的正常训练的结果，loss先增大后减小，可大大提高训练速度
