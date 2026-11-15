# Run one AIFS-ENS member: python3 run_member.py <yaml> input.grib=<ic> output.grib=<out>
import sys
import torch
from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.runners import create_runner
import fp32_attention

# Fixed model seed, so members differ only by their initial conditions.
torch.manual_seed(42)

config = RunConfiguration.load(sys.argv[1], sys.argv[2:])
runner = create_runner(config)

# flash-attention rejects fp32 inputs, so fp32 runs swap in the exact fp32 attention.
if runner.autocast == torch.float32:
    fp32_attention.patch(runner.model)

runner.execute()
