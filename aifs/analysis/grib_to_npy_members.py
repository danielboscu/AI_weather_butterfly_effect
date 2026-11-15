# Collect u and v at 300 hPa, 0-72 h, from every member and regrid N320 -> 0.25 deg.
import argparse
import os
import earthkit.data as ekd
import earthkit.regrid as ekr
import numpy as np

# Root for the member GRIB files and the npy output.
aifs_scratch = os.environ["aifs_scratch"]

parser = argparse.ArgumentParser()
parser.add_argument("--precision", type=str, required=True, choices=("bf16", "fp32"))
parser.add_argument("--noise-std", type=str, required=True)
parser.add_argument("--members", type=int, default=50)
args = parser.parse_args()

data = None
for i in range(args.members):
    path = f"{aifs_scratch}/members_pert/member_{args.precision}_{args.noise_std}_{i}.grib"
    source = ekd.from_source("file", path).sel(shortName=["u", "v"], step=list(range(0, 73, 6)), level=[300])
    fields = ekr.interpolate(source, out_grid={"grid": [0.25, 0.25]}).to_xarray().to_array().values
    if data is None:
        data = np.empty((args.members,) + fields.shape, dtype=fields.dtype)
    data[i] = fields

out_path = f"{aifs_scratch}/inf_pert_ens_50_{args.precision}_{args.noise_std}.npy"
np.save(out_path, data)
print(f"saved {out_path}")
