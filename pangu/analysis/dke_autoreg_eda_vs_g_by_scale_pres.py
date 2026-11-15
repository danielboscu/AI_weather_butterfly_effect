# EDA vs Gaussian DKE by perturbation scale, one model step per run.
import argparse
import os
import numpy as np
from pangu_paths import load_diagnostics
import matplotlib.pyplot as plt
from plot_config_pangu import params
plt.rcParams.update(params)

plt.rcParams.update({
    "axes.titlesize": 20,
    "axes.labelsize": 21,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 22,
    "lines.linewidth": 2.6,
    "savefig.dpi": 300,
})

parser = argparse.ArgumentParser()
parser.add_argument('--step', type=int, required=True, choices=(1, 3, 6, 24),
                    help='autoreg model step in hours')
args = parser.parse_args()

total_hours = 72
step = args.step
ensembles = 5

scales = {
    r"EDA 100\% / Gaussian 1":     ("1", "1"),
    r"EDA 10\% / Gaussian 1e-1":   ("1e-1", "0.1"),
    r"EDA 1\% / Gaussian 1e-2":    ("1e-2", "0.01"),
    r"EDA 0.1\% / Gaussian 1e-3":  ("1e-3", "0.001"),
    r"EDA 0.01\% / Gaussian 1e-4": ("1e-4", "0.0001"),
}

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

dkes_by_scale = {}
for title, (eda_sc, g_sc) in scales.items():
    dkes_by_scale[title] = {}
    for noise, sc in (("E", eda_sc), ("G", g_sc)):
        arrays, _ = load_diagnostics("fp32", "autoreg", "oper", noise, sc, step=step)
        dkes_by_scale[title][noise] = compute_dke(arrays["u300"][:ensembles],
                                                  arrays["v300"][:ensembles])

# Plotting
xx = np.arange(0, total_hours + 1, step)
fig = plt.figure(figsize=(13, 7.4))
gs = fig.add_gridspec(2, 6)
slots = [gs[0, 0:2], gs[0, 2:4], gs[0, 4:6], gs[1, 0:2], gs[1, 2:4]]
axes = []
for slot in slots:
    axes.append(fig.add_subplot(slot))

colors = {"E": "blue", "G": "red"}
labels = {"E": "EDA", "G": "Gaussian"}
marker = None
if step == 24:
    marker = "o"

for ax, (title, dkes_by_noise) in zip(axes, dkes_by_scale.items()):
    for noise, dkes in dkes_by_noise.items():
        ax.plot(xx, dkes, color=colors[noise], label=labels[noise], marker=marker)
    ax.set_yscale("log")
    ax.set_ylim(1e-9, 1e+3)
    ax.set_xticks(np.arange(0, total_hours + 1, 12))
    ax.set_xlim(-3, 75)
    ax.set_title(title, pad=6)
    ax.grid()

for ax in (axes[1], axes[2], axes[4]):
    ax.tick_params(labelleft=False)

axes[0].set_ylabel("DKE")
axes[3].set_ylabel("DKE")

leg_ax = fig.add_subplot(gs[1, 4:6])
leg_ax.axis("off")
handles, leg_labels = axes[0].get_legend_handles_labels()
leg_ax.legend(handles, leg_labels, loc="center", frameon=True,
              handlelength=3.0, borderpad=1.0, labelspacing=1.2)

# One shared x-label instead of five copies of the same string.
fig.supxlabel("Time [Hours]", y=0.04)
fig.tight_layout()

out_dir = "figures/perturbation_scale"
os.makedirs(out_dir, exist_ok=True)
out_path = f"{out_dir}/eda_vs_g_by_scale_pres_{step}h_{ensembles}.png"
plt.savefig(out_path)
plt.close(fig)
print(f"saved {out_path}")
