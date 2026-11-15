# Distill one run to the four planes the analysis scripts actually read.
import argparse
import glob
import os
import numpy as np
from pangu_paths import run_dir

total_hours = 72
# output_upper is (variable, level, lat, lon); variables are z,q,t,u,v
idx_z, idx_u, idx_v = 0, 3, 4
idx_300, idx_500, idx_850 = 7, 5, 2

parser = argparse.ArgumentParser()
parser.add_argument('--precision', type=str, default='fp32', choices=('fp32', 'tf32'))
parser.add_argument('--mode', type=str, required=True, choices=('greedy', 'autoreg'))
parser.add_argument('--source', type=str, required=True, choices=('era5', 'oper'))
parser.add_argument('--noise', type=str, required=True, choices=('E', 'G'))
parser.add_argument('--scale', type=str, required=True, help='used verbatim as the dir name')
parser.add_argument('--step', type=int, default=None, help='autoreg model step in hours')
parser.add_argument('--members', type=int, required=True)
args = parser.parse_args()

def member_files(path, m):
    files = []
    if args.mode == 'greedy':
        files.append(f"{path}/{m}/output_upper_greedy_0.npy")
        for i in range(1, total_hours + 1):
            files.append(glob.glob(f"{path}/{m}/{i}_*h_upper.npy")[0])
    else:
        for j in range(total_hours // args.step + 1):
            files.append(f"{path}/{m}/output_upper_{args.step}_{j}.npy")
    return files

path = run_dir(args.precision, args.mode, args.source, args.noise, args.scale, step=args.step)
if args.mode == 'greedy':
    hours = np.arange(total_hours + 1)
else:
    hours = np.arange(0, total_hours + 1, args.step)
out = os.path.join(path, 'diag')
os.makedirs(out, exist_ok=True)

# memmaps write straight to disk, not RAM
shape = (args.members, len(hours), 721, 1440)
u300 = np.lib.format.open_memmap(f"{out}/u300.npy", mode='w+', dtype=np.float32, shape=shape)
v300 = np.lib.format.open_memmap(f"{out}/v300.npy", mode='w+', dtype=np.float32, shape=shape)
u850 = np.lib.format.open_memmap(f"{out}/u850.npy", mode='w+', dtype=np.float32, shape=shape)
z500 = np.lib.format.open_memmap(f"{out}/z500.npy", mode='w+', dtype=np.float32, shape=shape)

for m in range(1, args.members + 1):
    files = member_files(path, m)
    for j in range(len(files)):
        d = np.load(files[j], mmap_mode='r')
        u300[m - 1, j] = d[idx_u, idx_300]
        v300[m - 1, j] = d[idx_v, idx_300]
        u850[m - 1, j] = d[idx_u, idx_850]
        z500[m - 1, j] = d[idx_z, idx_500]

u300.flush()
v300.flush()
u850.flush()
z500.flush()
np.save(f"{out}/hours.npy", hours)

open(f"{out}/.distilled_ok", 'w').close()
print(f"wrote {out}")
