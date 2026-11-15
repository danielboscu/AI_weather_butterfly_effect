import os
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import numpy as np
from pangu_paths import load_diagnostics
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from plot_config_pangu import params
plt.rcParams.update(params)

step = 1
ensembles = 5
t = 12

# Latitudes
lats_deg = 90.0 - 0.25 * np.arange(721)
lon = np.arange(0, 360, 0.25)

def add_cyclic(field_2d):
    # Close the 359.75 -> 0 seam.
    return add_cyclic_point(field_2d, coord=lon)

def z500_spread(precision, noise, scale):
    # Ensemble standard deviation of z500 in geopotential metres at hour t.
    arrays, hours = load_diagnostics(precision, "autoreg", "oper", noise, scale, step=step)
    ti = int(np.flatnonzero(hours == t)[0])
    z = np.asarray(arrays["z500"][:ensembles, ti]) / 9.80665   # m^2/s^2 -> gpm
    return z.std(axis=0, ddof=1)

cmap = LinearSegmentedColormap.from_list(
    "inferno_trim", plt.get_cmap("inferno")(np.linspace(0.10, 0.92, 256)))

def make_figure(fields, titles, fname):
    # 2x2 of absolute z500 spread on one shared log scale.
    pooled = np.concatenate([fields[0].ravel(), fields[1].ravel(), fields[2].ravel(), fields[3].ravel()])
    pooled = pooled[pooled > 0]
    vmax = 10.0 ** np.ceil(np.log10(np.percentile(pooled, 99.99)))
    vmin = max(10.0 ** np.floor(np.log10(np.percentile(pooled, 1.0))), vmax / 1e5)
    if pooled.min() < vmin and pooled.max() > vmax:
        extend = "both"
    elif pooled.min() < vmin:
        extend = "min"
    elif pooled.max() > vmax:
        extend = "max"
    else:
        extend = "neither"
    norm = LogNorm(vmin=vmin, vmax=vmax)

    proj = ccrs.Robinson(central_longitude=0)

    left, right, wspace = 0.02, 0.98, 0.02
    top, bottom, hspace = 0.91, 0.14, 0.16   # hspace must clear the row-2 titles
    aspect = ((proj.x_limits[1] - proj.x_limits[0]) /
              (proj.y_limits[1] - proj.y_limits[0]))
    fig_w = 13.0
    ax_w = fig_w * (right - left) / (2 + wspace)
    fig_h = (2 + hspace) * (ax_w / aspect) / (top - bottom)

    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h), subplot_kw={"projection": proj})

    ax = axes[0, 0]
    cf = ax.pcolormesh(lonc, lats_deg, fields[0], norm=norm, cmap=cmap,
                       shading="nearest", transform=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5, color="k", path_effects=[pe.withStroke(linewidth=1.3, foreground="white")])
    ax.set_global()
    ax.set_title(titles[0])

    ax = axes[0, 1]
    ax.pcolormesh(lonc, lats_deg, fields[1], norm=norm, cmap=cmap,
                  shading="nearest", transform=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5, color="k", path_effects=[pe.withStroke(linewidth=1.3, foreground="white")])
    ax.set_global()
    ax.set_title(titles[1])

    ax = axes[1, 0]
    ax.pcolormesh(lonc, lats_deg, fields[2], norm=norm, cmap=cmap,
                  shading="nearest", transform=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5, color="k", path_effects=[pe.withStroke(linewidth=1.3, foreground="white")])
    ax.set_global()
    ax.set_title(titles[2])

    ax = axes[1, 1]
    ax.pcolormesh(lonc, lats_deg, fields[3], norm=norm, cmap=cmap,
                  shading="nearest", transform=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5, color="k", path_effects=[pe.withStroke(linewidth=1.3, foreground="white")])
    ax.set_global()
    ax.set_title(titles[3])

    cb_h, cb_gap = 0.024, 0.018
    cax = fig.add_axes([0.22, bottom - cb_gap - cb_h, 0.56, cb_h])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2], extend=extend)
    cb.set_label(r"$z$500 ensemble spread (gpm)")

    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom,
                        wspace=wspace, hspace=hspace)

    out_dir = "figures/precision"
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/{fname}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir}/{fname}")

# EDA
eda_tf32_100 = z500_spread("tf32", "E", "1")
eda_tf32_01 = z500_spread("tf32", "E", "1e-3")
eda_fp32_100 = z500_spread("fp32", "E", "1")
eda_fp32_01 = z500_spread("fp32", "E", "1e-3")

eda_tf32_100, lonc = add_cyclic(eda_tf32_100)
eda_tf32_01, lonc = add_cyclic(eda_tf32_01)
eda_fp32_100, lonc = add_cyclic(eda_fp32_100)
eda_fp32_01, lonc = add_cyclic(eda_fp32_01)

make_figure([eda_tf32_100, eda_tf32_01, eda_fp32_100, eda_fp32_01],
            [r"TF32, EDA $100\%$", r"TF32, EDA $0.1\%$",
             r"FP32, EDA $100\%$", r"FP32, EDA $0.1\%$"],
            f"z500_spread_tf32_vs_fp32_eda_t{t}.png")
