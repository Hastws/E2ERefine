import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BATCH = 1
LAYERS = 8
HIDDEN = 1024
NQ = 16
D_HEAD = HIDDEN // NQ
DTYPE_BYTES = 2
STEPS = 32

schemes = {
    "MHA n_kv = n_q": NQ,
    "GQA n_kv = 4": 4,
    "MQA n_kv = 1": 1
}

# 通俗理解就是 seq_len 就是 token 有多长
# n_kv 就是头有多少个，d_head 就是每个头的维度
# batch 就是同时操作的样本数
# dtype_bytes 就是每个运算单元的字节数
# layers 就是一共有多少层的 transformer
# 之所以乘以2就是因为 K 算一个，V 也算一个
def kv_memory_bytes(seq_len, n_kv, layers, batch=BATCH, d_head=D_HEAD, dtype_bytes=DTYPE_BYTES):
    return batch * seq_len * n_kv * layers * d_head * dtype_bytes * 2

# Build per-step tables
tables = {}
for name, n_kv in schemes.items():
    rows = []
    for t in range(1, STEPS + 1):
        bytes_total = kv_memory_bytes(t, n_kv, LAYERS)
        rows.append({
            "step": t,
            "seq_len": t,
            "n_kv_heads": n_kv,
            "per_layer_K_shape": f"[{BATCH}, {t}, {n_kv}, {D_HEAD}]",
            "per_layer_V_shape": f"[{BATCH}, {t}, {n_kv}, {D_HEAD}]",
            "total_KV_cache_MB": bytes_total / (1024**2)
        })
    tables[name] = pd.DataFrame(rows)

# Show combined comparison at final step
final_rows = []
baseline_bytes = kv_memory_bytes(STEPS, schemes["MHA n_kv = n_q"], LAYERS)
for name, df in tables.items():
    last = df.iloc[-1]
    ratio = (last["total_KV_cache_MB"] * (1024**2)) / baseline_bytes
    final_rows.append({
        "scheme": name,
        "n_kv_heads": last["n_kv_heads"],
        "per_layer_K_shape@T": last["per_layer_K_shape"],
        "per_layer_V_shape@T": last["per_layer_V_shape"],
        "KV_cache_MB@T": round(last["total_KV_cache_MB"], 3),
        "relative_to_MHA": round(ratio, 3)
    })
compare_df = pd.DataFrame(final_rows).sort_values("n_kv_heads", ascending=False).reset_index(drop=True)

# Plot memory vs step
plt.figure(figsize=(7, 5))
for name, df in tables.items():
    plt.plot(df["step"], df["total_KV_cache_MB"], label=name)
plt.title("KV Cache B=1, LAYERS=8, H=1024, d_head=64, fp16")
plt.xlabel("step (= seq_len)")
plt.ylabel("Total KV Cache (MB)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()