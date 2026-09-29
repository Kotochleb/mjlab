"""Measure environment throughput for regression tracking.

This script measures physics and environment step throughput across canonical tasks
to catch performance regressions in the manager-based API.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch
import tyro
import wandb

import mjlab
import mjlab.tasks  # noqa: F401 - registers tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg
from mjlab.utils.profiling import profiling


@dataclass
class BenchmarkResult:
  """Results from a single benchmark run."""

  task: str
  num_envs: int
  num_steps: int
  decimation: int
  physics_sps: float
  env_sps: float
  overhead_pct: float
  profile_dir: str | None = None

  def __str__(self) -> str:
    return (
      f"{self.task} (dec={self.decimation}):\n"
      f"  Physics SPS: {self.physics_sps:,.0f}\n"
      f"  Env SPS:     {self.env_sps:,.0f}\n"
      f"  Overhead:    {self.overhead_pct:.1f}%"
    )

  def to_dict(self) -> dict:
    return asdict(self)


@dataclass
class ThroughputConfig:
  """Configuration for throughput benchmarking."""

  num_envs: int = 4096
  """Number of parallel environments."""

  num_steps: int = 200
  """Number of steps to measure (after warmup)."""

  warmup_steps: int = 50
  """Number of warmup steps before measuring."""

  device: str = "cuda:0"
  """Device to run on."""

  tasks: list[str] = field(
    default_factory=lambda: [
      "Mjlab-Velocity-Flat-Unitree-Go1",
      "Mjlab-Tracking-Flat-Unitree-G1",
      "Mjlab-Lift-Cube-Yam",
    ]
  )
  """Tasks to benchmark."""

  tracking_motion: str = "rll_humanoid/wandb-registry-Motions/lafan_cartwheel:latest"
  """W&B artifact path for tracking task motion (entity/project/name:alias)."""

  output_dir: Path | None = None
  """Output directory for JSON results. If None, results are only printed."""

  profile: bool = False
  """Collect PyTorch traces in separate passes after measuring throughput."""

  profile_dir: Path | None = None
  """Trace root. Defaults to <output_dir or benchmark_results>/profiles."""

  profile_steps: int = 20
  """Environment steps to record per trace (physics steps include decimation)."""

  profile_warmup_steps: int = 5
  """Profiler warmup steps before each capture."""

  profile_record_shapes: bool = False
  """Record tensor shapes in profiling data."""

  profile_memory: bool = False
  """Record PyTorch tensor memory allocations and deallocations."""

  profile_with_stack: bool = False
  """Record source locations for PyTorch operations."""

  def __post_init__(self) -> None:
    if self.num_envs <= 0 or self.num_steps <= 0:
      raise ValueError("num_envs and num_steps must be positive")
    if self.warmup_steps < 0:
      raise ValueError("warmup_steps must be nonnegative")
    if self.profile_steps <= 0 or self.profile_warmup_steps < 0:
      raise ValueError("profile_steps must be positive and profile warmup nonnegative")


def synchronize(device: str) -> None:
  """Wait for work on the benchmark's device, including non-default CUDA devices."""
  if torch.device(device).type == "cuda":
    torch.cuda.synchronize(device)


def measure_physics_sps(env: ManagerBasedRlEnv, num_steps: int) -> float:
  """Measure raw physics stepping in env steps per second.

  Runs num_steps worth of physics (i.e., num_steps * decimation sim.step calls)
  and reports throughput in env steps/sec for direct comparison with env.step().
  """
  decimation = env.cfg.decimation
  total_physics_steps = num_steps * decimation

  synchronize(env.device)
  start = time.perf_counter()

  for _ in range(total_physics_steps):
    env.sim.step()

  synchronize(env.device)
  elapsed = time.perf_counter() - start

  # Report in env steps/sec (not physics steps/sec) for fair comparison.
  return (num_steps * env.num_envs) / elapsed


def measure_env_sps(env: ManagerBasedRlEnv, num_steps: int) -> float:
  """Measure full environment step throughput in env steps per second."""
  action_dim = sum(env.action_manager.action_term_dim)
  action = torch.zeros(env.num_envs, action_dim, device=env.device)

  synchronize(env.device)
  start = time.perf_counter()

  for _ in range(num_steps):
    env.step(action)

  synchronize(env.device)
  elapsed = time.perf_counter() - start

  return (num_steps * env.num_envs) / elapsed


def collect_profiles(env: ManagerBasedRlEnv, task: str, cfg: ThroughputConfig) -> Path:
  """Export physics and environment traces without changing throughput timings."""
  root = cfg.profile_dir or (cfg.output_dir or Path("benchmark_results")) / "profiles"
  timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
  output_dir = root / task / timestamp
  output_dir.mkdir(parents=True, exist_ok=False)

  activities = [torch.profiler.ProfilerActivity.CPU]
  if torch.device(env.device).type == "cuda":
    activities.append(torch.profiler.ProfilerActivity.CUDA)
  action_dim = sum(env.action_manager.action_term_dim)
  action = torch.zeros(env.num_envs, action_dim, device=env.device)

  for phase in ("physics", "env"):
    env.reset()
    synchronize(env.device)
    with (
      torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
          wait=0,
          warmup=cfg.profile_warmup_steps,
          active=cfg.profile_steps,
          repeat=1,
        ),
        record_shapes=cfg.profile_record_shapes,
        profile_memory=cfg.profile_memory,
        with_stack=cfg.profile_with_stack,
      ) as profiler,
      profiling(),
    ):
      profiler.add_metadata("task", task)
      profiler.add_metadata("phase", phase)
      profiler.add_metadata_json(
        "benchmark",
        json.dumps(
          {
            "num_envs": env.num_envs,
            "decimation": env.cfg.decimation,
            "profile_steps": cfg.profile_steps,
            "device": env.device,
          }
        ),
      )
      for _ in range(cfg.profile_warmup_steps + cfg.profile_steps):
        if phase == "physics":
          with torch.profiler.record_function("benchmark/physics_step"):
            for _ in range(env.cfg.decimation):
              env.sim.step()
        else:
          with torch.profiler.record_function("benchmark/env_step"):
            env.step(action)
        profiler.step()
      synchronize(env.device)

    profiler.export_chrome_trace(str(output_dir / f"{phase}.trace.json"))
    averages = profiler.key_averages(group_by_input_shape=cfg.profile_record_shapes)
    sort_by = (
      "self_cuda_time_total"
      if torch.device(env.device).type == "cuda"
      else "self_cpu_time_total"
    )
    (output_dir / f"{phase}.txt").write_text(
      averages.table(sort_by=sort_by, row_limit=-1)
    )

  print(f"PyTorch profiles saved to {output_dir}")
  return output_dir


def benchmark_task(task: str, cfg: ThroughputConfig) -> BenchmarkResult:
  """Benchmark a single task."""
  print(f"\nBenchmarking {task}...")

  env_cfg = load_env_cfg(task)
  env_cfg.scene.num_envs = cfg.num_envs

  # Handle tracking task motion file.
  if len(env_cfg.commands) > 0:
    motion_cmd = env_cfg.commands.get("motion")
    if isinstance(motion_cmd, MotionCommandCfg):
      api = wandb.Api()
      artifact = api.artifact(cfg.tracking_motion)
      motion_dir = artifact.download()
      motion_cmd.motion_file = str(Path(motion_dir) / "motion.npz")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  try:
    env.reset()

    # Warmup.
    action_dim = sum(env.action_manager.action_term_dim)
    action = torch.zeros(env.num_envs, action_dim, device=env.device)
    for _ in range(cfg.warmup_steps):
      env.step(action)
    synchronize(env.device)

    physics_sps = measure_physics_sps(env, cfg.num_steps)

    env.reset()
    synchronize(env.device)
    env_sps = measure_env_sps(env, cfg.num_steps)

    profile_dir = collect_profiles(env, task, cfg) if cfg.profile else None
    return BenchmarkResult(
      task=task,
      num_envs=cfg.num_envs,
      num_steps=cfg.num_steps,
      decimation=env.cfg.decimation,
      physics_sps=physics_sps,
      env_sps=env_sps,
      overhead_pct=100 * (1 - env_sps / physics_sps),
      profile_dir=str(profile_dir) if profile_dir is not None else None,
    )
  finally:
    env.close()


def get_git_commit() -> str:
  """Get current git commit SHA."""
  try:
    result = subprocess.run(
      ["git", "rev-parse", "HEAD"],
      capture_output=True,
      text=True,
      check=True,
    )
    return result.stdout.strip()[:7]
  except subprocess.CalledProcessError:
    return "unknown"


def save_results(results: list[BenchmarkResult], output_dir: Path) -> None:
  """Save benchmark results to JSON, appending to existing data."""
  output_dir.mkdir(parents=True, exist_ok=True)
  data_file = output_dir / "throughput_data.json"

  # Load existing data.
  existing: list[dict] = []
  if data_file.exists():
    with open(data_file) as f:
      existing = json.load(f)

  # Create new run entry.
  run_entry = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "commit": get_git_commit(),
    "results": [r.to_dict() for r in results],
  }

  existing.append(run_entry)

  with open(data_file, "w") as f:
    json.dump(existing, f, indent=2)

  print(f"\nResults saved to {data_file}")


def main(cfg: ThroughputConfig) -> list[BenchmarkResult]:
  """Run throughput benchmarks on all configured tasks."""
  print("Throughput Benchmark")
  print(f"  Envs: {cfg.num_envs}")
  print(f"  Steps: {cfg.num_steps} (+ {cfg.warmup_steps} warmup)")
  print(f"  Device: {cfg.device}")

  results = []
  for task in cfg.tasks:
    result = benchmark_task(task, cfg)
    results.append(result)
    print(result)

  print("\n" + "=" * 74)
  print("Summary (all values in env steps per second):")
  print("  Physics SPS: sim.step() only (×decimation per env step)")
  print("  Env SPS: full env.step() including managers")
  print("  Overhead: time spent on non-physics work (observations, rewards, etc.)")
  print("=" * 74)
  print(f"{'Task':<35} {'Dec':>4} {'Physics SPS':>12} {'Env SPS':>12} {'Overhead':>8}")
  print("-" * 74)
  for r in results:
    task_short = r.task.replace("Mjlab-", "").replace("-Unitree-", "-")
    print(
      f"{task_short:<35} {r.decimation:>4} {r.physics_sps:>12,.0f} {r.env_sps:>12,.0f} {r.overhead_pct:>7.1f}%"
    )

  if cfg.output_dir:
    save_results(results, cfg.output_dir)

  return results


if __name__ == "__main__":
  cfg = tyro.cli(ThroughputConfig, config=mjlab.TYRO_FLAGS)
  main(cfg)
