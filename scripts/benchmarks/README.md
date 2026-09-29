# mjlab Nightly Benchmarks

This directory contains scripts for automated nightly benchmarking of mjlab.

## Overview

The nightly benchmark system:
1. Trains a tracking policy on the latest commit
2. Evaluates the policy across 1024 trials
3. Measures simulation throughput
4. Generates an HTML report with historical trends
5. Publishes results to GitHub Pages

## Usage

### Run the full nightly benchmark

```bash
./scripts/benchmarks/nightly_train.sh
```

### Skip training (regenerate report only)

```bash
SKIP_TRAINING=1 ./scripts/benchmarks/nightly_train.sh
```

### Skip training and throughput

```bash
SKIP_TRAINING=1 SKIP_THROUGHPUT=1 ./scripts/benchmarks/nightly_train.sh
```

### Regenerate report directly (no git operations)

```bash
uv run python scripts/benchmarks/generate_report.py \
  --entity gcbc_researchers \
  --tag nightly \
  --output-dir benchmark_results
```

### Measure throughput only

```bash
uv run python scripts/benchmarks/measure_throughput.py \
  --num-envs 4096 \
  --output-dir benchmark_results
```

### Collect PyTorch profiles

```bash
uv run scripts/benchmarks/measure_throughput.py \
  --tasks "['Mjlab-Velocity-Flat-Unitree-Go1']" \
  --num-envs 4096 \
  --output-dir benchmark_results \
  --profile True \
  --profile-steps 20 \
  --profile-warmup-steps 5
```

Throughput is measured first with profiling disabled. Two additional passes then
record raw physics stepping and full environment stepping. Each profiler step
represents one environment step; the physics pass runs `decimation` simulation
steps per profiler step. Initialization, environment resets before each pass, and
profiler warmup are excluded from the traces. Automatic resets during environment
stepping are included.

Each task writes `physics.trace.json`, `env.trace.json`, `physics.txt`, and `env.txt`
under `benchmark_results/profiles/<task>/<timestamp>/`. The text files contain
operator timing summaries. Open the JSON traces in [Perfetto](https://ui.perfetto.dev/)
or Chrome's tracing viewer. The trace directory is also recorded in the throughput
JSON results. Use `--profile-dir PATH` to choose a different trace root. Without
`--output-dir`, traces still default to `benchmark_results/profiles/`.

Optional `--profile-record-shapes True`, `--profile-memory True`, and
`--profile-with-stack True` flags enable additional PyTorch data collection.
These can increase profiling overhead and trace size. Memory tracking covers
PyTorch allocations but does not account for all MuJoCo/Warp allocations.
CPU activity is always recorded; CUDA runs also request CUDA activity. CUDA graph
replays remain enabled, so Warp physics appears under simulation ranges and graph
launches; detailed GPU kernel visibility depends on PyTorch/CUPTI support. The
benchmark also accepts `--device cpu` for CPU traces.

The `mjlab/` ranges identify environment, simulation, scene, entity, manager, term,
actuator, and sensor work. To include these ranges in a custom profiling loop:

```python
import torch
from mjlab.utils.profiling import profiling

with torch.profiler.profile() as profiler, profiling():
  env.step(action)
profiler.export_chrome_trace("env.trace.json")
```

## Configuration

Environment variables for `nightly_train.sh`:

- `CUDA_DEVICE` - GPU device to use (default: 0)
- `WANDB_TAGS` - Comma-separated tags for the run (default: nightly)
- `SKIP_TRAINING` - Set to "1" to skip training
- `SKIP_THROUGHPUT` - Set to "1" to skip throughput benchmarking

## Automated Setup

See [systemd/README.md](systemd/README.md) for instructions on setting up automated nightly runs using systemd timers.

## Report Options

The `generate_report.py` script supports:

- `--eval-limit N` - Maximum number of NEW runs to evaluate per invocation (default: 10)
  - Set to 0 for no limit
  - Historical cached results are always preserved
- `--tag TAG` - Filter runs by WandB tag (default: "nightly")
- `--num-envs N` - Number of parallel environments for evaluation (default: 1024)

## Viewing Results

Reports are published to: https://mujocolab.github.io/mjlab/nightly/
