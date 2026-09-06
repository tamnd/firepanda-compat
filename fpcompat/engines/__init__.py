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

    def shape_of(self, answer: Any) -> str | None:
        """What kind of thing one of this engine's own answers is.

        `frame`, `series`, `index` or None for anything the engine does not recognise
        as one of its own types, which includes every scalar and every pandas object.

        The comparison layer reads an unfamiliar answer through the Arrow C data
        interface, and that interface can say everything about it except one thing: a
        series and an index both cross as a single array, and which one it is, is a
        fact about the engine's type system rather than about its data. So the engine
        says. It is the only place in this package allowed to import its subject, and
        this is the one question that needs the import.

        Args:
            answer: What a case expression returned.

        Returns:
            The shape, or None.
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
