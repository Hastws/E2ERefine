import os, random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score


# ----------------------------
# 工具
# ----------------------------
def set_seed(seed=42):
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)


PAD_IDX = 0


# ----------------------------
# 数据集：可变长序列 + 是否包含模式 [7, 3]
# ----------------------------
class VarLenPatternDataset(Dataset):
    def __init__(self, n_samples=4000, vocab_size=20, min_len=10, max_len=50,
                 pos_ratio=0.5, pattern=(7, 3), seed=0):
        assert vocab_size > 3 and PAD_IDX == 0
        self.vocab_size = vocab_size
        self.min_len, self.max_len = min_len, max_len
        self.pattern = list(pattern)
        self.pos_ratio = pos_ratio
        self.rng = np.random.default_rng(seed)

        self.samples = []
        n_pos = int(n_samples * pos_ratio)
        n_neg = n_samples - n_pos

        # 生成正样本：强制注入模式
        for _ in range(n_pos):
            L = self.rng.integers(min_len, max_len + 1)
            # token 范围 1..vocab_size-1（0 留给 PAD）
            seq = self.rng.integers(1, vocab_size, size=L).tolist()
            # 随机位置插入模式（若 L<2 会被上面范围避免）
            i = self.rng.integers(0, L - len(self.pattern) + 1)
            seq[i:i + len(self.pattern)] = self.pattern
            self.samples.append((torch.tensor(seq, dtype=torch.long), torch.tensor(1.0)))

        # 生成负样本：尽量避免出现该模式
        for _ in range(n_neg):
            while True:
                L = self.rng.integers(min_len, max_len + 1)
                seq = self.rng.integers(1, vocab_size, size=L).tolist()
                if not self._contains_pattern(seq):
                    break
                # 若误撞模式，改一个位置
                j = self.rng.integers(0, L)
                # 改成与原值不同且不破坏 PAD 约定
                cand = self.rng.integers(1, vocab_size)
                seq[j] = cand if cand != seq[j] else (cand % (vocab_size - 1) + 1)
            self.samples.append((torch.tensor(seq, dtype=torch.long), torch.tensor(0.0)))

        self.rng.shuffle(self.samples)

    def _contains_pattern(self, seq):
        p = self.pattern
        for i in range(len(seq) - len(p) + 1):
            if seq[i:i + len(p)] == p:
                return True
        return False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ----------------------------
# collate_fn：padding + lengths + attention_mask
# ----------------------------
def collate_varlen(batch, pad_idx=PAD_IDX):
    # batch: List[(seq: LongTensor(L_i), label: FloatTensor)]
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    max_len = int(lengths.max().item())

    padded = torch.full((len(seqs), max_len), pad_idx, dtype=torch.long)
    for i, s in enumerate(seqs):
        L = len(s)
        padded[i, :L] = s

    # mask: 非 pad 位置为 1.0
    attn_mask = (padded != pad_idx).float()
    labels = torch.stack([l for l in labels]).float()  # (B,)
    return padded, lengths, attn_mask, labels


# ----------------------------
# 模型：Embedding + BiLSTM + Linear
# 使用 pack_padded_sequence 避免 pad 计算
# ----------------------------
class PatternBiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden=64, num_layers=1, pad_idx=PAD_IDX, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=0.0 if num_layers == 1 else dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden * 2, 1)

    def forward(self, x, lengths):
        # x: (B, T) LongTensor, lengths: (B,)
        emb = self.emb(x)  # (B, T, E)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)  # h_n: (num_layers*2, B, H)
        # 取最后层的双向 hidden，拼接
        # TODO(理解点)：bidirectional -> 前向与后向各一份
        h_last_fw = h_n[-2]  # (B, H)
        h_last_bw = h_n[-1]  # (B, H)
        h = torch.cat([h_last_fw, h_last_bw], dim=-1)  # (B, 2H)
        h = self.dropout(h)
        logits = self.classifier(h).squeeze(1)  # (B,)
        return logits


# ----------------------------
# 训练与评估
# ----------------------------
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_logits, all_labels = [], []
    for x, lengths, mask, y in loader:
        logits = model(x, lengths)
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())
    logits = torch.cat(all_logits).numpy()
    y_true = torch.cat(all_labels).numpy()
    probs = 1 / (1 + np.exp(-logits))
    acc = accuracy_score(y_true, (probs >= 0.5).astype(np.float32))
    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = float("nan")  # 若单类则 AUC 无法计算
    return acc, auc


def train_once():
    set_seed(42)
    # 数据集：8:2 切分
    full = VarLenPatternDataset(n_samples=4000, seed=123)
    n_train = int(0.8 * len(full))
    train_set, val_set = torch.utils.data.random_split(full, [n_train, len(full) - n_train],
                                                       generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True,
                              collate_fn=collate_varlen, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False,
                            collate_fn=collate_varlen, num_workers=0, pin_memory=False)

    model = PatternBiLSTM(vocab_size=20, emb_dim=64, hidden=64, num_layers=1, dropout=0.1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    ckpt_path = "./../out/best.ckpt"
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    best_acc, bad, patience = 0.0, 0, 10
    max_epochs = 50

    for epoch in range(max_epochs):
        model.train()
        running = 0.0
        for x, lengths, mask, y in train_loader:
            logits = model(x, lengths)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # 防爆
            optimizer.step()
            running += loss.item() * len(y)
        tr_loss = running / len(train_loader.dataset)

        val_acc, val_auc = evaluate(model, val_loader)
        print(f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} | val_acc={val_acc:.4f} | val_auc={val_auc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(),
                        "optim": optimizer.state_dict(),
                        "epoch": epoch + 1,
                        "best_acc": best_acc}, ckpt_path)
            bad = 0
        else:
            bad += 1

        if best_acc >= 0.95:
            print("[OK] 达标：val_acc ≥ 0.95")
            break
        if bad >= patience:
            print("[EarlyStop] 验证集未提升，提前停止。")
            break

    print(f"[Result] best_acc={best_acc:.4f} | ckpt={ckpt_path}")


if __name__ == "__main__":
    train_once()
