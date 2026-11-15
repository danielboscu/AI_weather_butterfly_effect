# Add Gaussian noise to u, v and t of the initial conditions, one file per member.
import argparse
import os
import earthkit.data as ekd
import numpy as np

# Root for the initial conditions (ic_2021062600.grib, ics_pert/) and run output.
aifs_scratch = os.environ["aifs_scratch"]

parser = argparse.ArgumentParser()
parser.add_argument("--noise-std", type=float, default=1.0, help="std of the noise in m/s (u, v) and K (t)")
parser.add_argument("--members", type=int, default=50)
args = parser.parse_args()

perturb_vars = ("u", "v", "t")
ic_path = f"{aifs_scratch}/ic_2021062600.grib"
out_dir = f"{aifs_scratch}/ics_pert"
os.makedirs(out_dir, exist_ok=True)

ic = ekd.from_source("file", ic_path)

for i in range(args.members):
    # member i uses seed 42 + i
    rng = np.random.default_rng(42 + i)
    fields = []
    for f in ic:
        if f.metadata("shortName") in perturb_vars:
            f = f.copy(values=f.values + rng.standard_normal(f.values.size) * args.noise_std)
        fields.append(f)
    out = f"{out_dir}/ic_2021062600_pert_{args.noise_std}_{i}.grib"
    ekd.FieldList.from_fields(fields).to_target("file", out)

print(f"saved {args.members} members to {out_dir}")
