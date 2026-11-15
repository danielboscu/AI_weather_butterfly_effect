# DKE of greedy runs: ERA5 vs OPER initial conditions, Gaussian vs EDA, at scale 1e-3.
import os
import numpy as np
from pangu_paths import load_diagnostics
import matplotlib.pyplot as plt
from plot_config_pangu import params
plt.rcParams.update(params)

total_hours = 72
scale = 1e-3

sources = {
    "oper_g (Gaussian, 1e-3)":     (("oper", "G", str(scale)), 50),
    "oper_eda (scaled EDA, 1e-3)": (("oper", "E", "1e-3"),     50),
    "era5_g (Gaussian, 1e-3)":     (("era5", "G", str(scale)),  9),
    "era5_eda (scaled EDA, 1e-3)": (("era5", "E", "1e-3"),      9),
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

dkes_by_source = {}
for label, ((src, noise, sc), n_members) in sources.items():
    arrays, _ = load_diagnostics("fp32", "greedy", src, noise, sc)
    dkes_by_source[label] = compute_dke(arrays["u300"][:n_members], arrays["v300"][:n_members])

# Plotting
xx = np.arange(0, total_hours + 1, 1)
fig, ax = plt.subplots(figsize=(7, 6))

colors = {"oper_g (Gaussian, 1e-3)": "tab:blue", "oper_eda (scaled EDA, 1e-3)": "tab:red",
          "era5_g (Gaussian, 1e-3)": "#78C6E0", "era5_eda (scaled EDA, 1e-3)": "#E46BA8"}
for label, dkes in dkes_by_source.items():
    ax.plot(xx, dkes, color=colors[label], label=label, marker='o', markersize=4)

ax.set_xlabel("Time [Hours]")
ax.set_ylabel("DKE")
ax.set_yscale("log")
ax.set_ylim(1e-12, 1e+3)
ax.set_xticks(np.arange(0, total_hours + 1, 6))
ax.set_xlim(-3, 75)
ax.set_title("DKE of PanguWeather | ERA5 vs OPER, Gaussian vs EDA | scale=1e-3")
ax.legend()
ax.grid()

out_dir = "figures/dataset"
os.makedirs(out_dir, exist_ok=True)
out_path = f"{out_dir}/compare_4way_{scale}.png"
plt.savefig(out_path)
plt.close(fig)
print(f"saved {out_path}")
