import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # γ，逐通道缩放

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # 1) 归一化用 FP32 计算更稳
        x_fp32 = x.float()
        # 2) 按元素平方
        x_sq = x_fp32.pow(2)
        # 3) 沿最后一维求均值（保持维度便于广播）
        mean_sq = x_sq.mean(dim=-1, keepdim=True)
        # 4) 计算缩放因子 1 / sqrt(mean(x^2) + eps)
        inv_rms = torch.rsqrt(mean_sq + self.eps)
        # 5) 应用缩放
        x_hat = x_fp32 * inv_rms
        return x_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 可选：形状校验
        if x.shape[-1] != self.weight.shape[0]:
            raise ValueError(f"RMSNorm: x.shape[-1]={x.shape[-1]} != weight.shape[0]={self.weight.shape[0]}")

        # 6) 先做归一化
        x_hat = self._norm(x)
        # 7) 乘以可学习权重（逐通道广播）
        y_fp32 = self.weight * x_hat
        # 8) 转回输入 dtype（支持半精度）
        y = y_fp32.type_as(x)
        return y


# class RMSNorm(torch.nn.Module):
#     def __init__(self, dim: int, eps: float = 1e-5):
#         super().__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(dim))
#
#     def _norm(self, x):
#         return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
#
#     def forward(self, x):
#         return (self.weight * self._norm(x.float())).type_as(x)


class RMSNormStep(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # γ，逐通道缩放

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # 1) 归一化用 FP32 计算更稳
        x_fp32 = x.float()
        # 2) 按元素平方
        x_sq = x_fp32.pow(2)
        # 3) 沿最后一维求均值（保持维度便于广播）
        mean_sq = x_sq.mean(dim=-1, keepdim=True)
        # 4) 计算缩放因子 1 / sqrt(mean(x^2) + eps)
        inv_rms = torch.rsqrt(mean_sq + self.eps)
        # 5) 应用缩放
        x_hat = x_fp32 * inv_rms
        return x_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 可选：形状校验
        if x.shape[-1] != self.weight.shape[0]:
            raise ValueError(f"RMSNorm: x.shape[-1]={x.shape[-1]} != weight.shape[0]={self.weight.shape[0]}")

        # 6) 先做归一化
        x_hat = self._norm(x)
        # 7) 乘以可学习权重（逐通道广播）
        y_fp32 = self.weight * x_hat
        # 8) 转回输入 dtype（支持半精度）
        y = y_fp32.type_as(x)
        return y


class RMSNormOneLine(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self._norm(x.float())).type_as(x)


def check_once(shape, eps=1e-5, device="cpu"):
    print(f"\n== shape={shape}, device={device} ==")
    x = torch.randn(*shape, dtype=torch.float32, device=device, requires_grad=True)
    x2 = x.clone().detach().requires_grad_(True)

    m1 = RMSNormStep(dim=shape[-1], eps=eps).to(device)
    m2 = RMSNormOneLine(dim=shape[-1], eps=eps).to(device)
    # 同步参数，避免初始化差异
    m2.load_state_dict(m1.state_dict())

    # 前向
    y1 = m1(x)
    y2 = m2(x2)

    # 前向一致性
    max_abs = (y1 - y2).abs().max().item()
    allclose = torch.allclose(y1, y2, rtol=0, atol=0)  # float32 下应能 bit 一致；若驱动等差异也至少 1e-7 内
    print(f"forward max|Δ| = {max_abs:.3e}, bitwise_equal={allclose}")

    # 反向（用同一个简单标量 loss）
    loss1 = y1.sum()
    loss2 = y2.sum()
    loss1.backward()
    loss2.backward()

    # 梯度一致性（输入 & 参数）
    grad_in_max = (x.grad - x2.grad).abs().max().item()
    grad_w_max = (m1.weight.grad - m2.weight.grad).abs().max().item()
    print(f"backward dX max|Δ| = {grad_in_max:.3e}, dW max|Δ| = {grad_w_max:.3e}")


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 依然用 float32
    shapes = [(2, 3, 8), (4, 16, 64), (1, 7, 128)]
    for s in shapes:
        check_once(s, eps=1e-5, device=device)


if __name__ == "__main__":
    main()
