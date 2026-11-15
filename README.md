# AI weather models and the butterfly effect

Weather is sensitive to small perturbations in its initial conditions, which results in
the Butterfly Effect. AI weather models, however, have struggled to reproduce this
error growth. In this repository, we test whether AI weather models amplify tiny
initial-condition errors the way the real atmosphere does, and under which conditions.

| Folder | Model | Status |
|---|---|---|
| [`pangu/`](pangu/) | [PanguWeather](https://github.com/198808xc/Pangu-Weather) | done |
| `aifs/` | [ECMWF AIFS](https://github.com/ecmwf/anemoi-inference) | coming |

**This repository contains our experiment and analysis code only.** The models were
developed by their original authors. We run them as released, and their weights are not
redistributed here.

## Credit

PanguWeather:

> Bi, K., Xie, L., Zhang, H., Chen, X., Gu, X., & Tian, Q. (2023).
> Accurate medium-range global weather forecasting with 3D neural networks.
> *Nature*, 619, 533–538. https://doi.org/10.1038/s41586-023-06185-3

The initial conditions are obtained from ECMWF operational analyses (MARS) and ERA5
(Copernicus Climate Data Store).
