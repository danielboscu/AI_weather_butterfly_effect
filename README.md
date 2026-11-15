# AI weather models and the butterfly effect

Do machine-learning weather models amplify tiny initial-condition errors the way
the real atmosphere does? This repository holds experiments that measure how small
perturbations grow (or don't) in AI forecast models.

| Folder | Model | Status |
|---|---|---|
| [`pangu/`](pangu/) | [Pangu-Weather](https://github.com/198808xc/Pangu-Weather) | done |
| `aifs/` | [ECMWF AIFS](https://github.com/ecmwf/anemoi-inference) | coming |

**This repo contains analysis code only.** I did not develop the models. They are
run as released by their authors, and their weights are not redistributed here.

## Credit

Pangu-Weather:

> Bi, K., Xie, L., Zhang, H., Chen, X., Gu, X., & Tian, Q. (2023).
> Accurate medium-range global weather forecasting with 3D neural networks.
> *Nature*, 619, 533–538. https://doi.org/10.1038/s41586-023-06185-3

Initial conditions come from ECMWF operational analyses (MARS) and ERA5
(Copernicus Climate Data Store).
