# AI weather models and the butterfly effect

Weather is sensitive to small perturbations in its initial conditions, which results in
the Butterfly Effect. AI weather models, however, have struggled to reproduce this
error growth. In this repository, we test whether AI weather models amplify tiny
initial-condition errors the way the real atmosphere does, and under which conditions.

| Folder | Model | Status |
|---|---|---|
| [`pangu/`](pangu/) | [PanguWeather](https://github.com/198808xc/Pangu-Weather) | done |
| [`aifs/`](aifs/) | [ECMWF AIFS-ENS 1.0](https://huggingface.co/ecmwf/aifs-ens-1.0) | done |

**This repository contains our experiment and analysis code only.** The models were
developed by their original authors. We run them as released, and their weights are not
redistributed here.

## Credit

PanguWeather:

> Bi, K., Xie, L., Zhang, H., Chen, X., Gu, X., & Tian, Q. (2023).
> Accurate medium-range global weather forecasting with 3D neural networks.
> *Nature*, 619, 533–538. https://doi.org/10.1038/s41586-023-06185-3

AIFS-ENS 1.0, run with [anemoi-inference](https://github.com/ecmwf/anemoi-inference)
(Apache-2.0). The model weights are published by ECMWF under CC BY 4.0:

> Lang, S., Alexe, M., et al. (2024). AIFS-CRPS: Ensemble forecasting using a model
> trained with a loss function based on the Continuous Ranked Probability Score.
> arXiv:2412.15832. https://doi.org/10.57967/hf/5902

The initial conditions are obtained from ECMWF operational analyses (MARS) and ERA5
(Copernicus Climate Data Store).
