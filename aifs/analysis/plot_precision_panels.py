# DKE of the Gaussian noise sweep, bf16-mixed vs fp32, one panel per precision.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Root for the npy files and the DKE cache.
aifs_scratch = os.environ["aifs_scratch"]

stds = ["1.0", "0.1", "0.01", "0.001", "0.0001"]
precisions = ["bf16", "fp32"]
cache = f"{aifs_scratch}/prec_test/dke_curves.npy"

# Latitude weights
lats_deg = 90.0 - 0.25 * np.arange(721)
cos_lats = np.cos(np.deg2rad(lats_deg))
lat_weights = cos_lats / cos_lats.mean()

# Reading all ten 10.8 GB arrays takes ~15 min, so the 13-point curves are cached.
if os.path.exists(cache):
    curves = np.load(cache)
else:
    curves = np.empty((len(precisions), len(stds), 13))
    for p in range(len(precisions)):
        for s in range(len(stds)):
            path = f"{aifs_scratch}/inf_pert_ens_50_{precisions[p]}_{stds[s]}.npy"
            members = np.load(path, mmap_mode="r")
            n = members.shape[0]
            s1 = np.zeros(members.shape[1:], np.float64)
            s2 = np.zeros(members.shape[1:], np.float64)
            for i in range(n):
                x = np.asarray(members[i])
                s1 = s1 + x
                s2 = s2 + x * x
            var = (s2 - s1 * s1 / n) / (n - 1) # variance trick!
            dke = var[0] + var[1]
            curves[p, s] = np.sum(lat_weights * dke.sum(axis=2), axis=1) / (721 * 1440)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, curves)

# Plotting: one panel per precision, all noise scales
xx = np.arange(0, 73, 6)
titles = ["bf16-mixed", "fp32"]
colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#7b3294"]

fig, axes = plt.subplots(1, len(precisions), figsize=(10, 4.4), sharey=True)

for p in range(len(precisions)):
    ax = axes[p]
    for s in range(len(stds)):
        ax.plot(xx, curves[p, s], color=colors[s], label=r"$\varepsilon = " + stds[s] + "$", alpha=0.6)
    ax.set_xlabel("Time [Hours]")
    ax.set_yscale("log")
    ax.set_ylim(1e-9, 1e+3)
    ax.set_xticks(xx)
    ax.set_xlim(-3, 75)
    ax.set_title(titles[p])
    ax.grid()

axes[0].set_ylabel("DKE")
title = fig.suptitle("DKE of AIFS-CRPS | bf16-mixed vs fp32")

axes[0].legend(loc="lower right")
fig.tight_layout()
# centre the title over the two panels, not the whole figure
title.set_x((axes[0].get_position().x0 + axes[1].get_position().x1) / 2)
title.set_y(0.95)

out_dir = "figures/precision"
os.makedirs(out_dir, exist_ok=True)
out_path = f"{out_dir}/dke_bf16_vs_fp32_panels.png"
fig.savefig(out_path, bbox_inches="tight")
plt.close(fig)
print(f"saved {out_path}")
