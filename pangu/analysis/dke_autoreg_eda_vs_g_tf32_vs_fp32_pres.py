# TF32 vs FP32 DKE at 100% and 0.1% perturbation.
import os
import numpy as np
from pangu_paths import load_diagnostics
import matplotlib.pyplot as plt
from plot_config_pangu import params
plt.rcParams.update(params)

plt.rcParams.update({
    "axes.titlesize": 23,
    "axes.labelsize": 21,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 20,
    "lines.linewidth": 2.8,
    "savefig.dpi": 300,
})

total_hours = 72
step = 1
ensembles = 5

sources = {
    r"EDA, 100\%":    ("E", "1"),
    r"EDA, 0.1\%":    ("E", "1e-3"),
    "Gaussian, 1":    ("G", "1"),
    "Gaussian, 1e-3": ("G", "0.001"),
}
legs = {"TF32 (reduced precision)": "tf32", "FP32 (single precision)": "fp32"}

# Latitude weights
lats_deg = 90.0 - 0.25 * np.arange(721)
cos_lats = np.cos(np.deg2rad(lats_deg))
lat_weights = cos_lats / cos_lats.mean()

def compute_dke(pred_u, pred_v):
    # one hour at a time, so the member variance never holds all hours in memory
    dkes = np.empty(pred_u.shape[1])
    for j in range(pred_u.shape[1]):
        dke = pred_u[:, j].var(axis=0, ddof=1) + pred_v[:, j].var(axis=0, ddof=1)
        dkes[j] = np.sum(lat_weights * dke.sum(axis=1)) / (721 * 1440)
    return dkes

dkes_by_leg = {}
for leg, precision in legs.items():
    dkes_by_leg[leg] = {}
    for label, (noise, sc) in sources.items():
        arrays, _ = load_diagnostics(precision, "autoreg", "oper", noise, sc, step=step)
        dkes_by_leg[leg][label] = compute_dke(arrays["u300"][:ensembles],
                                              arrays["v300"][:ensembles])

# Plotting
xx = np.arange(0, total_hours + 1, step)
fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

colors = {r"EDA, 100\%": "blue", r"EDA, 0.1\%": "blue",
          "Gaussian, 1": "red", "Gaussian, 1e-3": "red"}
# scale=1 solid, 1e-3 dashed, so the pair colour reads as the perturbation type
styles = {r"EDA, 100\%": "-", r"EDA, 0.1\%": "--",
          "Gaussian, 1": "-", "Gaussian, 1e-3": "--"}

for ax, (leg, dkes_by_source) in zip(axes, dkes_by_leg.items()):
    for label, dkes in dkes_by_source.items():
        ax.plot(xx, dkes, color=colors[label], linestyle=styles[label], label=label)
    ax.set_xlabel("Time [Hours]")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1e+3)
    ax.set_xticks(np.arange(0, total_hours + 1, 12))
    ax.set_xlim(-3, 75)
    ax.set_title(leg)
    ax.legend(loc="lower right")
    ax.grid()

axes[0].set_ylabel("DKE")
plt.tight_layout(pad=1.6)

out_dir = "figures/precision"
os.makedirs(out_dir, exist_ok=True)
out_path = f"{out_dir}/eda_vs_g_tf32_vs_fp32_pres_{step}h_{ensembles}.png"
plt.savefig(out_path)
plt.close(fig)
print(f"saved {out_path}")
