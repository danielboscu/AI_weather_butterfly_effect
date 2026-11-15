# Perturbation growth in PanguWeather

In these experiments, we perturb the initial conditions of PanguWeather, run ensembles
out to 72 hours, and measure how the spread between members grows. The spread is
measured with **Difference Kinetic Energy (DKE) at 300 hPa**, which is the
latitude-weighted ensemble variance of u and v (ddof = 1). We compare the following:

- **Perturbation type:** EDA perturbations from ECMWF, which are physical and scaled as
  a fraction of the EDA spread, against Gaussian white noise (in m/s and K).
- **Initial-condition source:** ECMWF operational analysis (OPER) against ERA5.
- **Forecast algorithm:** the *greedy* algorithm, which always uses the largest of the
  24/6/3/1h models, against *autoregressive* rollouts that step a single 1h, 3h, 6h or
  24h model forward.
- **GPU precision:** TF32 against true FP32 matrix multiplications.

All initial conditions are from 2021-06-26 00 UTC.

## Main findings

- **Numerical precision is significant for small perturbations.** ONNX Runtime turns
  TF32 on by default on A100s, and TF32 only keeps 10 mantissa bits. This adds about
  1.8e-2 m/s of round-off error per model call at 300 hPa, which is 16× larger than a
  0.1% EDA perturbation. As a result, small-perturbation experiments run in TF32 measure
  round-off error rather than error growth. All inference here therefore sets
  `use_tf32=0` and `NVIDIA_TF32_OVERRIDE=0`.
- **The model step decides whether errors grow.** In FP32, the 1h autoregressive rollout
  grows the perturbation by six orders of magnitude over 72 hours and saturates
  regardless of its initial size. The 24h rollout, however, barely grows it (×1.3). The
  6h rollout grows linearly by the same factor (~×2.7) at every scale, which is not
  chaotic behaviour.

The figures are grouped by question in [`figures/`](figures/), whose README explains
each one.

| Figure | Script |
|---|---|
| `model_step/autoreg_by_step_eda_vs_g_tf32_vs_fp32_pres_5.png`: growth by model step, EDA vs Gaussian, TF32 vs FP32 | `analysis/dke_autoreg_by_step_tf32_vs_fp32_pres.py` |
| `perturbation_scale/eda_vs_g_by_scale_pres_{1,3,6,24}h_5.png`: EDA vs Gaussian across perturbation sizes, one figure per model step | `analysis/dke_autoreg_eda_vs_g_by_scale_pres.py --step N` |
| `precision/eda_vs_g_tf32_vs_fp32_pres_1h_5.png`: TF32 vs FP32 at 100% and 0.1% perturbation | `analysis/dke_autoreg_eda_vs_g_tf32_vs_fp32_pres.py` |
| `precision/z500_spread_tf32_vs_fp32_eda_t12.png`: map of 500 hPa geopotential spread, TF32 vs FP32 | `analysis/cartopy_z500_tf32_vs_fp32.py` |
| `dataset/compare_4way_0.001.png`: greedy {operational, ERA5} × {Gaussian, EDA} | `analysis/greedy_dke_compare_oper_g_oper_eda_0.001.py` |

## Layout

```
data/        download initial conditions (operational: MARS; ERA5: CDS)
inference/
  autoreg/   eda_oper/, g_oper/: step one model forward autoregressively
  greedy/    oper_eda/, oper_g/, era5_eda/, era5_g/: PanguWeather's greedy algorithm
analysis/    distill_run.py   reduce raw output to diag/{u300,v300,u850,z500}.npy
             *_pres.py, cartopy_*, *compare*   figures
             pangu_paths.py   output-directory layout (shared)
             plot_config_pangu.py  matplotlib style (shared)
jobs/        PBS job scripts (NCAR Derecho), one per step
figures/     results, grouped into model_step/, perturbation_scale/, precision/, dataset/
```

## Setup

1. Install the Python dependencies with `pip install -r requirements.txt`. The plots
   also need a LaTeX install (`module load texlive` on Derecho).
2. Download the PanguWeather ONNX models (`pangu_weather_{1,3,6,24}.onnx`) from the
   [official repository](https://github.com/198808xc/Pangu-Weather) into
   `pangu/models/`. These are under the authors' non-commercial licence.
3. Set up your API credentials: `~/.ecmwfapirc` for MARS (operational and EDA) and
   `~/.cdsapirc` for ERA5.
4. Set `pangu_scratch` to a large scratch directory, since the full campaign is several
   TB. The job scripts set it to `/glade/derecho/scratch/$USER/sensitivity_analysis/pangu`.

Everything runs from `pangu/`. The job scripts `cd` to
`/glade/u/home/$USER/AI_weather_butterfly_effect/pangu`, so either clone the repository
there or edit that line.

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
  `[:, m]` for m = 1..N, so the control is never treated as a perturbation. The
  operational EDA gives up to 50 members and the ERA5 EDA up to 9.
- **The Gaussian noise is seeded on the member only** (`default_rng([42, m])`), in both
  the greedy and the autoregressive runs. Every scale therefore perturbs the same field,
  so a scale sweep only changes the amplitude. The published figures were made with seed
  20210626 for the autoregressive runs and unseeded greedy runs, so a rerun gives
  slightly different curves.
- **"1e-3" means different things for the two perturbation types.** For EDA it is
  dimensionless (0.1% of the EDA spread). For Gaussian it is 1e-3 m/s in u and v and
  1e-3 K in T, applied uniformly on all 13 levels.

PanguWeather is by Bi et al. (2023). See the [top-level README](../README.md) for the
citation.
