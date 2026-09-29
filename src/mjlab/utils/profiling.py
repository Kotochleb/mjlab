"""Opt-in PyTorch ranges for simulation and environment profiling.

Use ``with torch.profiler.profile(...), profiling():`` to collect mjlab ranges.
Outside ``profiling()``, decorated functions bypass ``record_function`` entirely.
"""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
from typing import ParamSpec, TypeVar

import torch

_P = ParamSpec("_P")
_R = TypeVar("_R")
_enabled = ContextVar("mjlab_profiling", default=False)
_null_scope = nullcontext()


@contextmanager
def profiling() -> Iterator[None]:
  """Enable mjlab profiler ranges in this context, restoring state on exit."""
  token = _enabled.set(True)
  try:
    yield
  finally:
    _enabled.reset(token)


def profile_scope(*names: str) -> AbstractContextManager:
  """Record a named range while profiling; otherwise reuse a no-op context."""
  if _enabled.get():
    return torch.profiler.record_function("mjlab/" + "/".join(names))
  return _null_scope


def profiled(func: Callable[_P, _R]) -> Callable[_P, _R]:
  """Annotate a method with its qualified name only when profiling is enabled."""
  name = "mjlab/" + getattr(func, "__qualname__", type(func).__name__)

  @wraps(func)
  def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
    if not _enabled.get():
      return func(*args, **kwargs)
    with torch.profiler.record_function(name):
      return func(*args, **kwargs)

  return wrapped
