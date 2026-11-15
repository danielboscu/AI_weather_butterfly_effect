# DKE across the four autoregressive model steps, TF32 against FP32.
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
    "legend.fontsize": 17,
    "lines.linewidth": 2.8,
    "savefig.dpi": 300,
})

total_hours = 72
ensembles = 5
steps = (1, 3, 6, 24)

noises = {"E": ("1e-3", "blue", r"EDA 0.1\%"),
          "G": ("0.001", "red", "Gaussian 1e-3")}
styles = {1: "-", 3: "--", 6: ":", 24: "-."}
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
    for noise, (sc, _, _) in noises.items():
        for step in steps:
            arrays, _ = load_diagnostics(precision, "autoreg", "oper", noise, sc, step=step)
            d = compute_dke(arrays["u300"][:ensembles], arrays["v300"][:ensembles])
            dkes_by_leg[leg][(noise, step)] = d

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), sharey=True)

for ax, (leg, dkes_by_key) in zip(axes, dkes_by_leg.items()):
    for noise, (_, color, stem) in noises.items():
        for step in steps:
            xx = np.arange(0, total_hours + 1, step)
            ax.plot(xx, dkes_by_key[(noise, step)], color=color,
                    linestyle=styles[step], label=f"{stem}, {step}h")
    ax.set_xlabel("Time [Hours]")
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 1e+3)
    ax.set_xticks(np.arange(0, total_hours + 1, 12))
    ax.set_xlim(-3, 75)
    ax.set_title(leg)
    ax.grid()

axes[0].set_ylabel("DKE")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4,
           frameon=True, handlelength=3.0, columnspacing=1.8)

fig.tight_layout(rect=[0, 0.17, 1, 1])

out_dir = "figures/model_step"
os.makedirs(out_dir, exist_ok=True)
out_path = f"{out_dir}/autoreg_by_step_eda_vs_g_tf32_vs_fp32_pres_{ensembles}.png"
plt.savefig(out_path)
plt.close(fig)
print(f"saved {out_path}")
