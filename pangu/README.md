# Perturbation growth in Pangu-Weather

These experiments perturb Pangu-Weather's initial conditions, run ensembles out to
72 h, and measure how the spread grows. Spread is measured as **difference kinetic
energy (DKE) at 300 hPa**, the latitude-weighted ensemble variance of u and v
(ddof = 1). The comparisons are:

- **Perturbation type:** ECMWF EDA perturbations (flow-dependent, scaled as a
  fraction of the EDA spread) vs white Gaussian noise (in m/s and K).
- **Initial-condition source:** ECMWF operational analysis vs ERA5.
- **Forecast scheme:** *greedy*, Pangu's hierarchical combination of its
  24/6/3/1 h models, vs *autoregressive*, iterating a single 1, 3, 6 or 24 h model.
- **GPU precision:** TF32 vs true FP32 matrix multiplies.

The initial conditions are 2021-06-26 00 UTC.

## Main findings

- **Precision matters for small perturbations.** ONNX Runtime turns on TF32 by
  default on A100s, which keeps only 10 mantissa bits. That adds about 1.8e-2 m/s
  of round-off error per model call at 300 hPa, 16× larger than a 0.1 % EDA
  perturbation. Under TF32, small-perturbation experiments measure round-off,
  not error growth. All inference here sets `use_tf32=0` and
  `NVIDIA_TF32_OVERRIDE=0`.
- **The model step decides whether errors grow.** In FP32 a 1 h autoregressive
  rollout grows the perturbation by six orders of magnitude over 72 h, and
  saturates regardless of the initial size. A 24 h rollout barely grows it (×1.3).
  The 6 h rollout grows linearly and independently of scale (~×2.7), which is
  not chaotic behaviour.

Figures are grouped by question in [`figures/`](figures/), whose README explains each one.

| Figure | Script |
|---|---|
| `model_step/autoreg_by_step_eda_vs_g_tf32_vs_fp32_pres_5.png`: growth by model step, EDA vs Gaussian, TF32 vs FP32 | `analysis/dke_autoreg_by_step_tf32_vs_fp32_pres.py` |
| `perturbation_scale/eda_vs_g_by_scale_pres_{1,3,6,24}h_5.png`: EDA vs Gaussian across perturbation sizes, one figure per model step | `analysis/dke_autoreg_eda_vs_g_by_scale_pres.py --step N` |
| `precision/eda_vs_g_tf32_vs_fp32_pres_1h_5.png`: TF32 vs FP32 at 100 % and 0.1 % perturbation | `analysis/dke_autoreg_eda_vs_g_tf32_vs_fp32_pres.py` |
| `precision/z500_spread_tf32_vs_fp32_eda_t12.png`: map of 500 hPa geopotential spread, TF32 vs FP32 | `analysis/cartopy_z500_tf32_vs_fp32.py` |
| `dataset/compare_4way_0.001.png`: greedy {operational, ERA5} × {Gaussian, EDA} | `analysis/greedy_dke_compare_oper_g_oper_eda_0.001.py` |

## Layout

```
data/        download initial conditions (operational: MARS; ERA5: CDS)
inference/
  autoreg/   eda_oper/, g_oper/: iterate one model step
  greedy/    oper_eda/, oper_g/, era5_eda/, era5_g/: Pangu's hierarchical scheme
analysis/    distill_run.py   reduce raw output to diag/{u300,v300,u850,z500}.npy
             *_pres.py, cartopy_*, *compare*   figures
             pangu_paths.py   output-directory layout (shared)
jobs/        PBS job scripts (NCAR Derecho), one per step
figures/     results, grouped into model_step/, perturbation_scale/, precision/, dataset/
```

## Setup

1. Install the Python dependencies with `pip install -r requirements.txt`.
   The plots also need a LaTeX install (`module load texlive` on Derecho).
2. Download the Pangu-Weather ONNX models (`pangu_weather_{1,3,6,24}.onnx`) from the
   [official repo](https://github.com/198808xc/Pangu-Weather)
   into `pangu/models/`. They are under the authors' non-commercial licence.
3. Set up your API credentials: `~/.ecmwfapirc` for MARS (operational/EDA) and
   `~/.cdsapirc` for ERA5.
4. Set `pangu_scratch` to a large scratch directory (the full campaign is several TB).
   The job scripts set it to `/glade/derecho/scratch/$USER/sensitivity_analysis/pangu`.

Everything runs from `pangu/`. The job scripts `cd` to `/glade/u/home/$USER/AI_weather_butterfly_effect/pangu`, so clone the repo there or edit that line.

## Pipeline

```bash
# 1. initial conditions -> $pangu_scratch/orig_data/
python3 data/obtain_data_oper.py
python3 data/obtain_data_era5.py

cd jobs
# 2. inference -> $pangu_scratch/data/{fp32,tf32}/{greedy,autoreg}/{oper,era5}/{E,G}/<scale>[/<step>h]
qsub -v script=oper_eda/oper_eda_greedy.py,members=50,scale=1e-3 greedy_fp32.sh
qsub -v members=5,scale=1e-3 autoreg_eda_oper_fp32.sh
qsub -v members=5,scale=0.001,models=1 autoreg_g_oper_fp32.sh   # or models=3+6+24
# (the *_tf32.sh jobs run the TF32 versions)

# 3. reduce each run to the diagnostics the figures read
qsub -v precision=fp32,source=oper,noise=E,scale=1e-3,members=5,step=1 distill_autoreg.sh
qsub -v precision=fp32,source=oper,noise=E,scale=1e-3,members=50 distill_greedy.sh

# 4. figures -> pangu/figures/
qsub -v step=1 dke_autoreg_eda_vs_g_by_scale_pres.sh
qsub dke_autoreg_by_step_tf32_vs_fp32_pres.sh
# ...one job script per figure script
```

## Notes on the method

- **EDA member indexing:** index 0 of the EDA array is the control. Members use
  `[:, m]` for m = 1..N, so the control is never treated as a perturbation.
  Operational EDA gives up to 50 members and ERA5 EDA up to 9.
- **Gaussian noise is seeded on member only** (`default_rng([20210626, m])`). Every
  scale therefore perturbs the same field, so a scale sweep changes only the amplitude.
- **"1e-3" means different things for the two perturbation types.** For EDA it is
  dimensionless (0.1 % of the EDA spread). For Gaussian it is 1e-3 m/s in u and v
  and 1e-3 K in T, applied uniformly on all 13 levels.

Pangu-Weather is by Bi et al. (2023). See the [top-level README](../README.md) for
the citation.
