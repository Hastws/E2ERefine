import os, argparse, random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def set_seed(seed=42):
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def build_loaders(data_dir, batch_size, num_workers, subset=None):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    if subset is not None and subset < len(train_ds):
        indices = np.random.RandomState(0).permutation(len(train_ds))[:subset]
        train_ds = torch.utils.data.Subset(train_ds, indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                              persistent_workers=num_workers > 0)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                             num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                             persistent_workers=num_workers > 0)
    return train_loader, test_loader


def one_hot(labels, num_classes, on_value=1.0, off_value=0.0):
    y = torch.full((labels.size(0), num_classes), off_value, dtype=torch.float, device=labels.device)
    y.scatter_(1, labels.view(-1, 1), on_value)
    return y


def soft_cross_entropy(logits, soft_targets):
    # logits: (B, C) raw scores; soft_targets: (B, C) probabilities
    log_prob = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_prob).sum(dim=1).mean()


def mixup_batch(x, y, num_classes, alpha=0.2):
    if alpha <= 0.0:
        return x, one_hot(y, num_classes), 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[index, :]
    y_one = one_hot(y, num_classes)
    y_mix = lam * y_one + (1 - lam) * y_one[index, :]
    return x_mix, y_mix, lam


def rand_bbox(W, H, lam):
    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx = np.random.randint(W);
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W);
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W);
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def cutmix_batch(x, y, num_classes, alpha=1.0):
    if alpha <= 0.0:
        return x, one_hot(y, num_classes), 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    B, C, H, W = x.size()
    x1, y1, x2, y2 = rand_bbox(W, H, lam)
    x_cut = x.clone()
    x_cut[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    # 以实际被替换面积修正 lam
    lam_eff = 1.0 - ((x2 - x1) * (y2 - y1) / (W * H))
    y_one = one_hot(y, num_classes)
    y_mix = lam_eff * y_one + (1 - lam_eff) * y_one[index, :]
    return x_cut, y_mix, lam_eff


def build_model(num_classes=10):
    # ResNet18（不加载预训练；CIFAR 输入 32x32 可直接用）
    m = models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, top1, top5, run_loss = 0, 0, 0, 0.0
    ce = nn.CrossEntropyLoss()
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = ce(logits, yb)
        run_loss += loss.item() * yb.size(0)
        total += yb.size(0)
        pred = logits.argmax(1)
        top1 += (pred == yb).sum().item()
        # top-5
        _, pred5 = logits.topk(5, dim=1)
        top5 += (pred5 == yb.view(-1, 1)).any(dim=1).sum().item()
    return run_loss / total, top1 / total, top5 / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="./data")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--opt", type=str, default="sgd", choices=["sgd", "adamw"])
    ap.add_argument("--aug", type=str, default="mixup", choices=["none", "mixup", "cutmix", "both"])
    ap.add_argument("--alpha_mixup", type=float, default=0.2)
    ap.add_argument("--alpha_cutmix", type=float, default=1.0)
    ap.add_argument("--prob", type=float, default=1.0, help="应用混合增强的概率")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--subset", type=int, default=None, help="快速模式：仅用前 N 条训练样本")
    args = ap.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.cuda.is_available()

    train_loader, test_loader = build_loaders(args.data, args.batch_size, args.num_workers, subset=args.subset)

    model = build_model(10).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    if args.opt == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.wd)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    ckpt_path = f"./../out/{args.aug}.ckpt"
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            use_aug = (args.aug != "none") and (random.random() < args.prob)
            if use_aug:
                if args.aug == "mixup":
                    xb_m, yb_soft, _ = mixup_batch(xb, yb, 10, alpha=args.alpha_mixup)
                elif args.aug == "cutmix":
                    xb_m, yb_soft, _ = cutmix_batch(xb, yb, 10, alpha=args.alpha_cutmix)
                else:  # both
                    if random.random() < 0.5:
                        xb_m, yb_soft, _ = mixup_batch(xb, yb, 10, alpha=args.alpha_mixup)
                    else:
                        xb_m, yb_soft, _ = cutmix_batch(xb, yb, 10, alpha=args.alpha_cutmix)
                x_in, soft_target, use_soft = xb_m, yb_soft, True
            else:
                x_in, soft_target, use_soft = xb, None, False

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(x_in)
                if use_soft:
                    loss = soft_cross_entropy(logits, soft_target)
                else:
                    loss = F.cross_entropy(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * yb.size(0)

        scheduler.step()
        train_loss = running / len(train_loader.dataset)

        test_loss, test_top1, test_top5 = evaluate(model, test_loader, device)
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:02d} | lr={cur_lr:.5f} | train_loss={train_loss:.4f} | "
              f"test_loss={test_loss:.4f} | top1={test_top1:.4f} | top5={test_top5:.4f}")

        if test_top1 > best_acc:
            best_acc = test_top1
            torch.save({
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_acc": best_acc,
                "args": vars(args),
            }, ckpt_path)

    print(f"[Result] best_top1={best_acc:.4f} | ckpt={ckpt_path}")


if __name__ == "__main__":
    main()

# # 基线（无混合）
# python cifar_mixup_cutmix.py --aug none   --epochs 30 --lr 0.1 --opt sgd
#
# # Mixup
# python cifar_mixup_cutmix.py --aug mixup  --alpha_mixup 0.2  --epochs 30 --lr 0.1 --opt sgd
#
# # CutMix
# python cifar_mixup_cutmix.py --aug cutmix --alpha_cutmix 1.0 --epochs 30 --lr 0.1 --opt sgd
#
# # 二择其一（both）
# python cifar_mixup_cutmix.py --aug both   --epochs 30 --lr 0.1 --opt sgd
