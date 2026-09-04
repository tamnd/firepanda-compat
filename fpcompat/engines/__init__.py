"""The two engines.

The oracle is pandas, imported in the worker's process, pinned in `pixi.toml`, and
its version recorded in every result file. Nothing about it is clever.

The subject is firepanda, and it has two forms because the project is not finished:
importable after M3, and a Mojo driver over Arrow IPC files before it. Both live
behind the same three methods, so the runner does not know which one it has.
"""

from __future__ import annotations

from typing import Any, Protocol


class Engine(Protocol):
    """What the runner needs from an engine."""

    name: str

    def module(self) -> Any:
        """The module a case expression receives as its first argument."""

    def frame(self, name: str) -> Any:
        """Loads one corpus frame in this engine's frame type.

        Args:
            name: The corpus frame name.

        Returns:
            The frame.
        """

    def versions(self) -> dict[str, str]:
        """What goes in the result file, because a claim without a version is not one."""


def load(name: str) -> Engine:
    """Builds an engine by name.

    Args:
        name: `pandas` or `firepanda`.

    Returns:
        The engine.

    Raises:
        ValueError: When there is no such engine.
    """
    if name == "pandas":
        from fpcompat.engines.pandas_engine import PandasEngine

        return PandasEngine()
    if name == "firepanda":
        from fpcompat.engines.firepanda_engine import FirepandaEngine

        return FirepandaEngine()
    raise ValueError(f"no engine called {name}, there is pandas and there is firepanda")
