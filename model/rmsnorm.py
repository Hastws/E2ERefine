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

