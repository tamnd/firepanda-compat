"""firepanda, the subject.

Two forms, because the project is not finished.

After M3 there is an importable `firepanda` and this binds it as the module the case
expression receives, which makes every case a direct comparison in one interpreter.
That is three lines and it is the form the whole design is aimed at.

Before M3 there is no Python module, so `drivers/firepanda/main.mojo` is a program
that takes a case id and a corpus directory, runs the firepanda spelling of that case
and writes the answer as an Arrow IPC file. That driver does not exist yet.

Until one of the two arrives, this engine reports itself as unavailable and the
runner marks every case unimplemented. That is the honest outcome and it is a real
one: unimplemented counts against the score exactly as hard as a failure does, per
document 01. What it must not do is crash the run or produce an empty result file,
because a suite that cannot run at all and a suite that scores zero look identical in
a log and are very different things.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fpcompat import corpus

DRIVER = corpus.ROOT / "drivers" / "firepanda" / "firepanda-compat-driver"


class EngineUnavailable(RuntimeError):
    """Neither form of the subject is present."""


class FirepandaEngine:
    """The library under test, in whichever form is available."""

    name = "firepanda"

    def __init__(self) -> None:
        self._module: Any = None
        self._driver: Path | None = None
        try:
            # Imported here because the import is the availability check. A machine
            # with no firepanda on it still has to be able to run the pandas oracle.
            import firepanda
        except ImportError:
            firepanda = None
        if firepanda is not None:
            self._module = firepanda
        elif DRIVER.exists():
            self._driver = DRIVER

    @property
    def available(self) -> bool:
        """Whether anything can actually be run."""
        return self._module is not None or self._driver is not None

    @property
    def form(self) -> str:
        """Which of the two forms is in use, for the result file."""
        if self._module is not None:
            return "module"
        if self._driver is not None:
            return "driver"
        return "absent"

    def module(self) -> Any:
        """The firepanda module.

        Returns:
            The module.

        Raises:
            EngineUnavailable: When there is no importable firepanda, which the runner
                turns into `unimplemented` for every case rather than into a crash.
        """
        if self._module is None:
            raise EngineUnavailable(
                "no importable firepanda and no built driver at "
                f"{DRIVER.relative_to(corpus.ROOT)}. Build the driver against a "
                "firepanda checkout, or wait for M3 and the Python module"
            )
        return self._module

    def frame(self, name: str) -> Any:
        """Loads one corpus frame as a firepanda frame.

        Args:
            name: The corpus frame name.

        Returns:
            The frame.

        Raises:
            EngineUnavailable: When there is no firepanda.
        """
        return self.module().read_arrow(str(corpus.CORPUS / f"{name}.arrow"))

    def versions(self) -> dict[str, str]:
        """What goes in the result file.

        Returns:
            The firepanda version and form, or the absence of both, which is itself
            the thing a reader of an all unimplemented result file needs to know.
        """
        version = getattr(self._module, "__version__", None) if self._module else None
        return {
            "firepanda": version or "absent",
            "form": self.form,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        }
