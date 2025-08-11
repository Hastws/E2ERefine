import torch, torch.nn.functional as F

def safe_log_softmax(x, dim: int = -1):
    orig_dtype = x.dtype
    # 统一在 fp32 上做稳定计算
    x32 = x.to(torch.float32) if x.dtype in (torch.float16, torch.bfloat16) else x

    # logsumexp 是稳定的，但当该维所有元素为 -inf 时会得到 -inf
    lse = torch.logsumexp(x32, dim=dim, keepdim=True)
    out32 = x32 - lse  # 若全为 -inf:  -inf - (-inf) => NaN（需额外处理）

    # 针对“该维全是 -inf”的切片，把结果回填为 -inf（与 log softmax 的极限一致）
    all_neg_inf = torch.isneginf(x32).all(dim=dim, keepdim=True)
    if all_neg_inf.any():
        out32 = torch.where(all_neg_inf, torch.full_like(out32, float('-inf')), out32)

    # 转回原始精度
    return out32.to(orig_dtype)


# 1) 随机大幅值
x = torch.randn(7, 5) * 50
assert torch.allclose(safe_log_softmax(x, dim=1), F.log_softmax(x, dim=1), atol=1e-3, rtol=0)

# 2) 半精度
x16 = (torch.randn(4, 6) * 30).half()
y = safe_log_softmax(x16, dim=-1)
y_ref = F.log_softmax(x16.float(), dim=-1).half()
assert torch.allclose(y, y_ref, atol=1e-3, rtol=0)  # half 允许更松一点

# 3) bfloat16
xb = (torch.randn(3, 7) * 20).to(torch.bfloat16)
yb = safe_log_softmax(xb, dim=-1)
yb_ref = F.log_softmax(xb.float(), dim=-1).to(torch.bfloat16)
assert torch.allclose(yb, yb_ref, atol=2e-3, rtol=0)

# 4) 全是 -inf 的切片
x_inf = torch.full((2, 4), float('-inf'))
y_inf = safe_log_softmax(x_inf, dim=1)
assert torch.isneginf(y_inf).all()