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

# MNIST 是一个28*28的手写数字集合
class CNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # 改为32个特征，依然是28*28
            nn.Conv2d(1, 32, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            # 会变成之前的一半
            nn.MaxPool2d(2),
            # 还是初始的一半
            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            # 一半的一半
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    running_loss = 0.0
    ce = nn.CrossEntropyLoss()
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = ce(logits, yb)
        running_loss += loss.item() * yb.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = torch.cuda.is_available()

    # 数据
    mean, std = (0.1307,), (0.3081,)
    train_tf = transforms.Compose([
        transforms.RandomCrop(28, padding=2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = datasets.MNIST(args.data, train=True, download=True, transform=train_tf)
    test_ds = datasets.MNIST(args.data, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=args.num_workers > 0
    )
    test_loader = DataLoader(
        test_ds, batch_size=512, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=args.num_workers > 0
    )

    model = CNN2().to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # 断点续训
    ckpt_path = ("./../out"
                 "/best.ckpt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    start_epoch, best_acc = 0, 0.0
    if os.path.exists(ckpt_path):
        print(f"[Info] Resume from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        start_epoch = ckpt.get("epoch", 0)
        best_acc = ckpt.get("best_acc", 0.0)
        print(f"[Info] start_epoch={start_epoch}, best_acc={best_acc:.4f}")

    # 训练
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * yb.size(0)
        train_loss = running / len(train_loader.dataset)

        test_loss, test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | test_loss={test_loss:.4f} | test_acc={test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_acc": best_acc
            }, ckpt_path)

        if best_acc >= 0.99:
            print("[OK] 达到目标：测试集准确率 ≥ 99%")
            break

    print(f"[Result] best_acc={best_acc:.4f} | ckpt={ckpt_path}")


if __name__ == "__main__":
    main()
