"""Exercise benchmark profiling with a real CPU environment and exported traces."""

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.cartpole.cartpole_env_cfg import cartpole_balance_env_cfg


@pytest.fixture(scope="module")
def benchmark():
  path = Path(__file__).parents[1] / "scripts" / "benchmarks" / "measure_throughput.py"
  spec = importlib.util.spec_from_file_location("measure_throughput", path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  yield module
  del sys.modules[spec.name]


@pytest.fixture
def env():
  cfg = cartpole_balance_env_cfg()
  cfg.scene.num_envs = 2
  cfg.decimation = 2
  # Exercise automatic resets within every recorded environment step.
  cfg.episode_length_s = cfg.sim.mujoco.timestep * cfg.decimation
  instance = ManagerBasedRlEnv(cfg, device="cpu")
  yield instance
  instance.close()


def test_profile_cli(benchmark, tmp_path):
  cfg = tyro.cli(
    benchmark.ThroughputConfig,
    config=mjlab.TYRO_FLAGS,
    args=[
      "--tasks",
      "['Mjlab-Cartpole-Balance']",
      "--profile",
      "True",
      "--profile-dir",
      str(tmp_path),
      "--profile-steps",
      "2",
      "--profile-memory",
      "True",
    ],
  )
  assert cfg.profile and cfg.profile_memory
  assert cfg.profile_steps == 2
  assert cfg.profile_dir == tmp_path
  assert cfg.tasks == ["Mjlab-Cartpole-Balance"]


def test_profile_exports_warmed_up_steps_and_nested_ranges(benchmark, env, tmp_path):
  cfg = benchmark.ThroughputConfig(
    num_envs=2,
    num_steps=2,
    warmup_steps=1,
    device="cpu",
    profile=True,
    profile_dir=tmp_path,
    profile_steps=2,
    profile_warmup_steps=1,
    profile_record_shapes=True,
    profile_memory=True,
    profile_with_stack=True,
  )
  output = benchmark.collect_profiles(env, "cartpole", cfg)
  for phase in ("physics", "env"):
    trace = json.loads((output / f"{phase}.trace.json").read_text())
    counts = Counter(event.get("name") for event in trace["traceEvents"])
    assert trace["task"] == "cartpole"
    assert trace["phase"] == phase
    assert trace["benchmark"]["decimation"] == env.cfg.decimation
    assert counts[f"benchmark/{phase}_step"] == cfg.profile_steps
    assert counts["mjlab/Simulation.step"] == cfg.profile_steps * env.cfg.decimation
    # Explicit resets happen before profiling, but automatic resets are captured.
    assert counts["mjlab/ManagerBasedRlEnv.reset"] == 0
    assert (output / f"{phase}.txt").stat().st_size > 0
    if phase == "env":
      for name in (
        "ManagerBasedRlEnv.step",
        "ManagerBasedRlEnv._reset_idx",
        "ActionManager.process_action",
        "Scene.write_data_to_sim",
        "Entity.write_data_to_sim",
        "Simulation.forward",
        "RewardManager.compute/smooth_reward",
        "ObservationManager.term/actor/cart_pos",
        "EventManager.term/reset/reset_slider",
      ):
        assert counts[f"mjlab/{name}"] > 0, name
    else:
      assert counts["mjlab/ManagerBasedRlEnv.step"] == 0


@pytest.mark.parametrize("profile", [False, True])
def test_benchmark_profiles_after_timing(benchmark, monkeypatch, tmp_path, profile):
  env = Mock(device="cpu", num_envs=2)
  env.cfg.decimation = 2
  env.action_manager.action_term_dim = [1]
  env_cfg = Mock(commands={})
  monkeypatch.setattr(benchmark, "load_env_cfg", lambda task: env_cfg)
  monkeypatch.setattr(benchmark, "ManagerBasedRlEnv", lambda **kwargs: env)
  calls = []

  def measure_physics(*args):
    calls.append("physics")
    return 100.0

  def measure_env(*args):
    calls.append("env")
    return 80.0

  def collect(*args):
    calls.append("profile")
    return tmp_path

  monkeypatch.setattr(benchmark, "measure_physics_sps", measure_physics)
  monkeypatch.setattr(benchmark, "measure_env_sps", measure_env)
  monkeypatch.setattr(benchmark, "collect_profiles", collect)
  cfg = benchmark.ThroughputConfig(warmup_steps=0, profile=profile)
  result = benchmark.benchmark_task("cartpole", cfg)
  assert calls == ["physics", "env"] + (["profile"] if profile else [])
  assert result.overhead_pct == pytest.approx(20.0)
  assert result.profile_dir == (str(tmp_path) if profile else None)
  env.close.assert_called_once()


def test_benchmark_closes_environment_on_profile_failure(benchmark, monkeypatch):
  env = Mock(device="cpu", num_envs=2)
  env.action_manager.action_term_dim = [1]
  monkeypatch.setattr(benchmark, "load_env_cfg", lambda task: Mock(commands={}))
  monkeypatch.setattr(benchmark, "ManagerBasedRlEnv", lambda **kwargs: env)
  monkeypatch.setattr(benchmark, "measure_physics_sps", lambda *args: 100.0)
  monkeypatch.setattr(benchmark, "measure_env_sps", lambda *args: 80.0)
  monkeypatch.setattr(
    benchmark, "collect_profiles", Mock(side_effect=RuntimeError("export failed"))
  )
  cfg = benchmark.ThroughputConfig(warmup_steps=0, profile=True)
  with pytest.raises(RuntimeError, match="export failed"):
    benchmark.benchmark_task("cartpole", cfg)
  env.close.assert_called_once()


def test_synchronize_uses_requested_device(benchmark, monkeypatch):
  sync = Mock()
  monkeypatch.setattr(torch.cuda, "synchronize", sync)
  benchmark.synchronize("cpu")
  sync.assert_not_called()
  benchmark.synchronize("cuda:1")
  sync.assert_called_once_with("cuda:1")
