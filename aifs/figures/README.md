# Figures

The figure shows **Difference Kinetic Energy (DKE)** at 300 hPa, which is used to measure
error growth between ensemble members:

$$DKE = \mathrm{variance}(u) + \mathrm{variance}(v)$$

Here u and v are the zonal and meridional components of the wind. The perturbations are
Gaussian white noise of standard deviation ε, added to u, v and t on all levels.

## `precision/`: does the GPU's numerical precision matter?

bf16 accounts for only about 2.4 decimal digits, whereas fp32 accounts for about 7.2.
AIFS-ENS was trained in bf16-mixed, so this is the precision it runs in by default.

- `dke_bf16_vs_fp32_panels.png` displays DKE for all five perturbation scales, from
  ε = 1 down to ε = 0.0001, under bf16-mixed (left) and fp32 (right). 50 members.
- In bf16, we observe that **every scale from ε = 0.01 down jumps to the same DKE within
  the first 6 hours**, and the curves then collapse onto each other, ending within 7% at
  72 hours.
- In fp32, however, there is a clear distinction: the scales stay ordered, roughly an
  order of magnitude apart, and the smallest perturbation (ε = 0.0001) grows by about
  1400× over 72 hours.
- At ε = 1 there is no visual difference between the two precisions (1.05× apart at
  72 hours). At ε = 0.001, however, bf16 ends about 1200× higher than fp32. **If we were to
  run AIFS in its default bf16, the collapse would look like the Butterfly Effect, when it
  is round-off error.**
