# Visualize Rotary Position Embeddings (RoPE) as 2D rotations for two channel pairs
import torch
import numpy as np
import matplotlib.pyplot as plt

# Parameters
dim = 8  # per-head dimension (must be even for this layout)
theta = 1e6  # RoPE base
steps = 200  # number of positions to visualize
dtype = torch.float32

# Compute inverse frequencies for the "pair indices" k = 0..dim/2-1
idx = torch.arange(0, dim, 2, dtype=dtype)  # (dim//2,)
power = idx / float(dim)
theta_t = torch.tensor(theta, dtype=dtype)
inv_freq = torch.exp(-torch.log(theta_t) * power)  # (dim//2,)

# Choose two pairs to visualize (skip k=0 because it spins very fast; pick k=1 & k=2)
pairs_to_show = [1, 2]

# Unit circle for reference
phi = np.linspace(0, 2 * np.pi, 400)
circle_x = np.cos(phi)
circle_y = np.sin(phi)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(circle_x, circle_y, linewidth=1, label="unit circle")

# Initial vector per pair: (1, 0) in its 2D plane
x0, y0 = 1.0, 0.0
t = torch.arange(steps, dtype=dtype)

for k in pairs_to_show:
    w = inv_freq[k].item()  # angular step per position
    angles = (t * w).numpy()  # shape (steps,)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    x = x0 * cos_a - y0 * sin_a  # rotate (x0, y0) by angle t*w
    y = x0 * sin_a + y0 * cos_a
    ax.plot(x, y, marker='o', markersize=2, linewidth=1, label=f"pair k={k}, ω≈{w:.4f}")

ax.set_aspect('equal', 'box')
ax.set_xlabel("even-dim component")
ax.set_ylabel("odd-dim component")
ax.set_title("RoPE: rotations in 2D planes for two channel pairs")
ax.grid(True)
ax.legend()

plt.show()
