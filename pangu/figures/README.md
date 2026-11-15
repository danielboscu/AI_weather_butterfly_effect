# Figures

All figures show **Difference Kinetic Energy (DKE)** at 300 hPa, which is used to
measure error growth between ensemble members:

$$DKE = \mathrm{variance}(u) + \mathrm{variance}(v)$$

Here u and v are the zonal and meridional components of the wind. Perturbations are
applied in two ways: **Gaussian (G)** is purely white noise, whereas **EDA (E)** comes
from ECMWF's Ensemble Data Assimilation and is more physical and realistic.

## `model_step/`: does the prediction time step matter?

- `autoreg_by_step_eda_vs_g_tf32_vs_fp32_pres_5.png` displays DKE for the 1h, 3h, 6h
  and 24h autoregressive rollouts, for EDA (blue) and Gaussian (red), under TF32 (left)
  and FP32 (right).
- In FP32, **the model step decides whether error grows at all.** The 1h rollout grows
  by about six orders of magnitude over 72 hours, whereas the 24h rollout grows by only
  31%.
- In TF32 every step diverges, and the larger steps sit 2–3 orders of magnitude above
  their FP32 counterparts. If we were to use TF32, we would be measuring round-off
  error rather than the atmosphere.

## `perturbation_scale/`: does the size of the perturbation matter?

- `eda_vs_g_by_scale_pres_{1h,3h,6h,24h}_5.png` display one figure per model step, with
  one panel per perturbation scale (100% down to 0.01%). FP32, operational EDA, 5
  members.
- **1h:** we observe an increase in DKE growth as the perturbation scale decreases, and
  **the spread between ensemble members collapses.** The approximately 8 orders of
  magnitude at the start collapse to about 1 at 72 hours. This saturation is consistent
  with the Butterfly Effect.
- **3h:** the saturation stops, and each scale stays roughly an order of magnitude
  apart. This is also where EDA and Gaussian separate the most: at 0.1%, Gaussian ends
  37× above EDA.
- **6h:** the growth is **linear**. Every scale grows by the same ~2.7× regardless of
  its starting size, and EDA and Gaussian overlap, which a chaotic system would not do.
- **24h:** there is essentially no error growth. The Gaussian perturbation is even
  *smaller* at 72 hours than at the start.

## `precision/`: does the GPU's numerical precision matter?

TF32 accounts for only about 3.3 decimal digits, whereas FP32 accounts for about 7.2.
TF32 is the default on the most recent NVIDIA GPU architectures (Ampere onward: A100,
H100, H200).

- `eda_vs_g_tf32_vs_fp32_pres_1h_5.png` displays DKE under TF32 (left) and FP32 (right)
  at 100% (solid) and 0.1% (dashed) perturbation, for the 1h autoregressive rollout.
  - At 100%, the two precisions end only 1.1× apart.
  - At 0.1%, however, the EDA perturbation sits at the round-off threshold, and TF32
    ends up to 8.7× higher. In FP32 the two scales separate cleanly. **If we were to
    mistakenly choose TF32, this separation would not have emerged.**
- `z500_spread_tf32_vs_fp32_eda_t12.png` displays the ensemble spread of 500 hPa
  geopotential height at 12h lead, TF32 (top) against FP32 (bottom), at 100% (left) and
  0.1% (right) EDA.
  - At 100% there is no visual difference between the two maps.
  - At 0.1%, however, there is a clear distinction: the FP32 map shows a significantly
    smaller spread than its TF32 counterpart.

## `dataset/`: does the source of the initial conditions matter?

- `compare_4way_0.001.png` uses the greedy algorithm at perturbation scale 1e-3, with
  operational (OPER) against ERA5 initial conditions and Gaussian against EDA
  perturbations.
- **The difference between ERA5 and OPER was insignificant.** The OPER and ERA5 curves
  sit on top of each other for both noise types.
- Since the greedy algorithm always uses the highest step model available, DKE barely
  grows over 72 hours. The drops at 24h and 48h are where the 24h model takes over.
