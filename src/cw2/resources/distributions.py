import numpy as np
import matplotlib.pyplot as plt

# Settings
num_samples = 10000
interval = 0.1

# Sampling
uniform = np.random.uniform(-interval, interval, num_samples)
# stddev is smaller to keep range tight
gaussian = np.random.normal(0, interval / 2, num_samples)
laplace = np.random.laplace(0, interval / np.sqrt(2), num_samples)
exponential = np.random.exponential(
    interval, num_samples) - interval  # Shift to center around 0
beta = (np.random.beta(2, 5, num_samples) - 0.5) * \
    2 * interval  # Beta in [-0.1, 0.1]

# Plot
fig, axs = plt.subplots(2, 3, figsize=(8, 4), constrained_layout=True)
bins = 35

axs[0, 0].hist(uniform, bins=bins, color='skyblue', edgecolor='black')
axs[0, 0].set_title("Uniform $\\mathcal{U}[-0.1, 0.1)$")

axs[0, 1].hist(gaussian, bins=bins, color='lightcoral', edgecolor='black')
axs[0, 1].set_title("Gaussian $\\mathcal{N}(0, \\sigma^2)$")

axs[0, 2].hist(laplace, bins=bins, color='palegreen', edgecolor='black')
axs[0, 2].set_title("Laplace $\\text{Laplace}(0, b)$")

axs[1, 0].hist(exponential, bins=bins, color='plum', edgecolor='black')
axs[1, 0].set_title("Shifted Exponential")

axs[1, 1].hist(beta, bins=bins, color='gold', edgecolor='black')
axs[1, 1].set_title("Scaled Beta $(\\alpha{=}2, \\beta{=}5)$")

for ax in axs.flat:
    ax.set_xlim(-0.2, 0.2)
    ax.grid(True)
    ax.set_yticks([])

# Turn off the last (empty) subplot
axs[1, 2].axis("off")


fig.suptitle(
    "Sampling Distributions for $\\delta_i$", fontsize=14)
plt.savefig("sampling-distr.png", dpi=300)
