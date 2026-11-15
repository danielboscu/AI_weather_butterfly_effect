# Perturbation growth in AIFS

In these experiments, we perturb the initial conditions of ECMWF's AIFS-ENS 1.0, run
50-member ensembles out to 72 hours, and measure how the spread between members grows.
As for PanguWeather, the spread is measured with **Difference Kinetic Energy (DKE) at
300 hPa**, which is the latitude-weighted ensemble variance of u and v (ddof = 1). We
compare the following:

- **Perturbation scale:** Gaussian white noise with standard deviation ε = 1, 0.1, 0.01,
  0.001 and 0.0001 (in m/s for u and v, and K for t), added on all levels.
- **GPU precision:** bf16-mixed, the precision AIFS-ENS was trained in, against full fp32.

The model seed is fixed at 42 for every member, so members differ only by their initial
conditions. All initial conditions are from 2021-06-26 00 UTC.

## Main findings

- **Numerical precision is significant for small perturbations.** bf16 accounts for only
  about 2.4 decimal digits. At ε = 1 the two precisions end only 1.05× apart, but at
  ε = 0.001 bf16 ends about **1200× higher** than fp32, because it is measuring its own
  round-off error rather than error growth.
- **In bf16 the scales collapse onto each other.** Within the first 6 hours, every scale
  from ε = 0.01 down jumps to the same DKE, and at 72 hours they end within 7% of each
  other. This looks like the Butterfly Effect, but it disappears in fp32, where each scale
  stays roughly an order of magnitude apart.
- **fp32 is also far more reproducible.** Two identical bf16 runs differ by 5.0e-3
  (median relative RMS), whereas two fp32 runs differ by only 5.6e-6, which is about 900×
  smaller.

The figure is explained in [`figures/`](figures/README.md).

| Figure | Script |
|---|---|
| `precision/dke_bf16_vs_fp32_panels.png`: DKE for all five scales, bf16-mixed (left) vs fp32 (right) | `analysis/plot_precision_panels.py` |

## Layout

```
data/        perturb_ic.py: add Gaussian noise to the initial conditions, one file per member
inference/   run_member.py: run one member with anemoi-inference
             fp32_attention.py: exact fp32 attention, since flash-attention rejects fp32
             member_bf16.yaml, member_fp32.yaml: the two precision configs
analysis/    grib_to_npy_members.py: u and v at 300 hPa, regridded to 0.25 deg
             plot_precision_panels.py: DKE and the figure
jobs/        SLURM job scripts (TACC Stampede3)
figures/     results
```

## Setup

1. Create two Python environments in `aifs/` and install `requirements.txt` in both:
   `.venv` for the CPU jobs, and `.venv-blackwell` with `torch==2.9.1+cu128` for the
   RTX PRO 6000 Blackwell (`rtx-small`) GPU nodes.
2. Download the AIFS-ENS 1.0 checkpoint from
   [HuggingFace](https://huggingface.co/ecmwf/aifs-ens-1.0) into
   `aifs/models/aifs-ens-1.0.ckpt`. The weights are under CC BY 4.0.
3. Place the initial conditions in `$SCRATCH/aifs/ic_2021062600.grib`. These are pulled
   from MARS (`class=od, stream=oper, type=an`) for both 2021-06-25 18 UTC and
   2021-06-26 00 UTC, since the model needs two input times (184 fields on N320).

Everything runs from `aifs/`. The job scripts `cd` to
`$STOCKYARD/AI_weather_butterfly_effect/aifs` and set `aifs_scratch=$SCRATCH/aifs`, so
either clone the repository there or edit those lines.

## Pipeline

```bash
cd jobs
# 1. perturbed initial conditions -> $aifs_scratch/ics_pert/
for s in 1.0 0.1 0.01 0.001 0.0001; do sbatch --export=ALL,noise_std=$s perturb_ic.sh; done

# 2. inference and npy conversion, from a login node (TACC blocks sbatch on compute nodes)
#    -> $aifs_scratch/members_pert/member_<precision>_<std>_<member>.grib
#    -> $aifs_scratch/inf_pert_ens_50_<precision>_<std>.npy
nohup ./sweep.sh bf16 > sweep_bf16.log 2>&1 &
nohup ./sweep.sh fp32 > sweep_fp32.log 2>&1 &

# 3. figure -> aifs/figures/precision/
sbatch plot_precision_panels.sh
```

## The fp32 attention change

- **Setting `precision: "32"` is not enough on its own.** AIFS-ENS uses flash-attention,
  which only supports fp16 and bf16, so an fp32 run crashes on the first step.
- Full attention compares all **40,320 queries** with all **40,320 keys** of the hidden
  mesh. In fp32 this needs a 48.4 GiB score matrix, which runs out of memory.
- The attention, however, is **windowed**: each query only attends to keys within 1120
  of it. `fp32_attention.py` therefore processes the queries in tiles, and each tile of
  **4,096 queries** only needs **6,336 keys**. Its score matrix is only about 0.8 GiB.
  The result is exact, not an approximation.
- **Why the block size is 4096:** a tile of 1024 gave the same peak memory for the whole
  run (34.4 GiB), since the model weights and graph mappers set that peak rather than the
  attention. We therefore kept 4096, which needs only 10 passes instead of 40.
- The kernel change is 8× smaller than the precision effect, so it does not affect the
  fp32 results.

## Notes on the method

- **The noise is added on the native N320 grid and then regridded to 0.25°.** The
  regridding smooths the white noise, so the DKE at t = 0 for ε = 1 is about 1.0 rather
  than 2.0. As a result, ε is not directly comparable to the PanguWeather scales.
- **Member i uses noise seed 42 + i.** The published figure was made with seed 1000 + i,
  so a rerun gives slightly different curves.
- **The forecasts run to 72 hours**, since only 0–72 h is analysed. The published runs
  went to 240 hours, which does not change the first 72 hours but takes about 3× longer.
- **The rtx-small QOS allows only 2 jobs per account at a time**, so `sweep.sh` submits one
  array task at a time and waits for a free slot.
- **`plot_precision_panels.py` caches the DKE curves** in
  `$aifs_scratch/prec_test/dke_curves.npy`, since reading all ten npy files (10.8 GB each)
  takes about 15 minutes.

AIFS-ENS is by Lang et al. (2024). See the [top-level README](../README.md) for the
citation.
