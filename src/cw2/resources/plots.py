import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

# Define activation functions


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.02):
    return np.where(x >= 0, x, alpha * x)


def elu(x, alpha=1.0):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))


def tanh(x):
    return np.tanh(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# # Generate x values
# x = np.linspace(-5, 5, 400)

# # Create figure and axis with constrained layout
# fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

# # Set bigger fonts for all text elements
title_fontsize = 22
label_fontsize = 22
tick_fontsize = 22
legend_fontsize = 22


# # Plot each function
# ax.plot(x, relu(x), label='ReLU', color='#1f77b4', linewidth=2)
# ax.plot(x, leaky_relu(x), label='Leaky ReLU',
#         color='#ff7f0e', linewidth=2, linestyle='--')
# ax.plot(x, elu(x), label='ELU', color='#2ca02c', linewidth=2, linestyle='--')
# ax.plot(x, tanh(x), label='Tanh', color='#d62728', linewidth=2, linestyle='--')
# ax.plot(x, sigmoid(x), label='Sigmoid', color='#9467bd', linewidth=2)


# ax.set_xlabel('$x$', fontsize=label_fontsize)
# ax.set_ylabel('$f(x)$', fontsize=label_fontsize)
# ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
# ax.legend(loc='upper left', fontsize=legend_fontsize)


# # Grid and legend
# # ax.grid(True, linestyle='--', alpha=0.6)


# # Set limits and aspect ratio
# ax.set_xlim([-5, 5])
# ax.set_ylim([-2, 5])
# # ax.set_aspect('equal', adjustable='box')


# # Remove extra whitespace/margins exactly
# plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
# plt.savefig("activations.png", dpi=300)


# Define ParamTanh

class ParamTanh(nn.Module):
    def __init__(self, delta1=0.0, delta2=0.0, delta3=0.0, delta4=0.0):
        super(ParamTanh, self).__init__()
        self.delta1 = delta1
        self.delta2 = delta2
        self.delta3 = delta3
        self.delta4 = delta4

        self.C1 = 2.0
        self.C2 = 1.0
        self.C3 = 2.0
        self.C4 = 1.0

    def forward(self, x: torch.Tensor):
        c1 = self.C1 + self.delta1
        c2 = self.C2 + self.delta2
        c3 = self.C3 + self.delta3
        c4 = self.C4 + self.delta4
        return (c1 / (c2 + torch.exp(-c3 * x))) - c4


# Define parameter sets
param_sets = [
    (-0.33105, 0.04181, -0.35800, -0.11906),
    (0.20636, -0.37003, -0.00292, 0.35259),
    (1.0, -0.12230, -0.22106, 1.0),
    (1.0, 0.81879, -0.10271, 0.20318)
]

# Generate input values
x_vals = torch.linspace(-5, 5, 500)

# Prepare plot
plt.figure(figsize=(8, 8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Nice color palette

for idx, (d1, d2, d3, d4) in enumerate(param_sets):
    act_fn = ParamTanh(delta1=d1, delta2=d2, delta3=d3, delta4=d4)
    y_vals = act_fn(x_vals)
    plt.plot(x_vals.numpy(), y_vals.detach().numpy(),
             color=colors[idx], linewidth=2, label=f"ParamTanh({idx})")

# Plot baseline (no delta)
baseline = ParamTanh()
y_baseline = baseline(x_vals)
plt.plot(x_vals.numpy(), y_baseline.detach().numpy(),
         color='black', linewidth=2.5, linestyle='--', label='ParamTanh (no noise)')

# Labels and title
plt.xlabel("Input $x$", fontsize=label_fontsize)
plt.ylabel("Output $f(x)$", fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
plt.title("Variants of ParamTanh Activation Function", fontsize=label_fontsize)

# Legend
plt.legend(loc='lower right', fontsize=12)

# Add faint axis lines
# plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
# plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)

# Axes limits
plt.xlim(-4, 4)
plt.ylim(-3, 3)

# Save with no margins
# plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig("paramtanh-values.png", dpi=300)
plt.close()