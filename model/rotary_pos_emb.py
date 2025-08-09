import torch


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def _pair_norm(x):
    """把最后一维按前半/后半成对求范数平方。"""
    D = x.size(-1)
    x1, x2 = x[..., : D // 2], x[..., D // 2:]
    return (x1 ** 2 + x2 ** 2)


def main():
    torch.manual_seed(0)

    # 形状 (B,S,H,D)，D 必须为偶数；全程 float32
    B, S, H, D = 2, 8, 4, 64
    assert D % 2 == 0

    device = "cuda" if torch.cuda.is_available() else "cpu"

    q = torch.randn(B, S, H, D, dtype=torch.float32, device=device)
    k = torch.randn(B, S, H, D, dtype=torch.float32, device=device)

    # 预计算 cos/sin
    cos_all, sin_all = precompute_freqs_cis(dim=D, end=1024, theta=1e6)
    cos_all = cos_all.to(device=device, dtype=torch.float32)
    sin_all = sin_all.to(device=device, dtype=torch.float32)
    cos, sin = cos_all[:S], sin_all[:S]  # (S,D)

    # 1) t=0 恒等
    cos_id = torch.ones_like(cos)
    sin_id = torch.zeros_like(sin)
    q_id, k_id = apply_rotary_pos_emb(q, k, cos_id, sin_id)
    id_err_q = (q_id - q).abs().max().item()
    id_err_k = (k_id - k).abs().max().item()

    # 2) 范数保持（逐对通道）
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    q_in, q_out = _pair_norm(q), _pair_norm(q_rot)
    k_in, k_out = _pair_norm(k), _pair_norm(k_rot)

    q_abs = (q_in - q_out).abs()
    k_abs = (k_in - k_out).abs()
    q_rel = q_abs / (q_in.abs() + 1e-12)
    k_rel = k_abs / (k_in.abs() + 1e-12)

    q_abs_max = q_abs.max().item()
    k_abs_max = k_abs.max().item()
    q_rel_max = q_rel.max().item()
    k_rel_max = k_rel.max().item()

    # 3) 角度可加性 R(t1)R(t2)=R(t1+t2)
    t1, t2 = 3, 7
    q1, _ = apply_rotary_pos_emb(q[:, :1], q[:, :1], cos_all[t1:t1 + 1], sin_all[t1:t1 + 1])
    q12, _ = apply_rotary_pos_emb(q1, q1, cos_all[t2:t2 + 1], sin_all[t2:t2 + 1])
    qsum, _ = apply_rotary_pos_emb(q[:, :1], q[:, :1], cos_all[t1 + t2:t1 + t2 + 1], sin_all[t1 + t2:t1 + t2 + 1])
    comp_err = (q12 - qsum).abs().max().item()

    print("=== RoPE 验证（float32）===")
    print(f"[恒等] max|q_rot - q| = {id_err_q:.3e}, max|k_rot - k| = {id_err_k:.3e}")
    print(f"[范数保持-绝对] q max|Δ| = {q_abs_max:.3e}, k max|Δ| = {k_abs_max:.3e}")
    print(f"[范数保持-相对] q max rel = {q_rel_max:.3e}, k max rel = {k_rel_max:.3e}")
    print(f"[角度可加] max|R(t1)R(t2) - R(t1+t2)| = {comp_err:.3e}")

    # 判定（对 float32 更合理的阈值）
    ok = (
            torch.allclose(q_id, q, rtol=0, atol=0) and
            torch.allclose(k_id, k, rtol=0, atol=0) and
            torch.allclose(q_in, q_out, rtol=1e-6, atol=2e-5) and  # 放宽到 2e-5
            torch.allclose(k_in, k_out, rtol=1e-6, atol=2e-5) and
            torch.allclose(q12, qsum, rtol=1e-6, atol=2e-6)
    )
    print("结果：", "✔ 正确" if ok else "✘ 异常")


if __name__ == "__main__":
    main()
