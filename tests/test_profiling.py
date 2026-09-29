"""Opt-in ranges must preserve function behavior and profiler context state."""

import pytest
import torch

from mjlab.utils.profiling import profile_scope, profiled, profiling


@profiled
def _add(value: torch.Tensor, *, amount: float) -> torch.Tensor:
  return value + amount


def test_disabled_ranges_do_not_call_record_function(monkeypatch):
  def unexpected_range(*args, **kwargs):
    pytest.fail("Profiling must be opt-in")

  monkeypatch.setattr(torch.profiler, "record_function", unexpected_range)
  with profile_scope("disabled"):
    assert _add(torch.ones(1), amount=2).item() == 3


def test_nested_profiling_restores_state_after_exception():
  value = torch.ones(1)
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
    with profiling():
      with pytest.raises(ValueError), profiling(), profile_scope("inner"):
        raise ValueError("test")
      with profile_scope("outer"):
        assert _add(value, amount=2).item() == 3
    with profile_scope("disabled"):
      _add(value, amount=1)

  events = p.events()
  assert events is not None
  names = [event.name for event in events]
  assert names.count("mjlab/_add") == 1
  assert "mjlab/inner" in names
  assert "mjlab/outer" in names
  assert "mjlab/disabled" not in names
  add_event = next(event for event in events if event.name == "mjlab/_add")
  assert add_event.cpu_parent is not None
  assert add_event.cpu_parent.name == "mjlab/outer"
