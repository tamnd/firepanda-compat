"""firepanda, the subject.

Two forms, because the project is not finished.

After M3 there is an importable `firepanda` and this binds it as the module the case
expression receives, which makes every case a direct comparison in one interpreter.
That is three lines and it is the form the whole design is aimed at.

Before M3 there is no Python module, so `drivers/firepanda/main.mojo` is a program
that takes a case id and a corpus directory, runs the firepanda spelling of that case
and writes the answer as an Arrow IPC file. `fpcompat.driver` is this side of that,
and it hands back the same `Answer` the pandas side builds, so nothing downstream of
`compare` can tell the two forms apart.

The driver form runs a case in a process, which the module form will not, and that is
worth roughly a millisecond per case in exchange for being able to measure a library
that has no Python bindings at all. It is a scaffold and it is meant to be deleted the
day the module arrives, which is why `module()` is still the path a case expression
takes and the driver is a separate method rather than a fake module.

Until one of the two arrives, this engine reports itself as unavailable and the
runner marks every case unimplemented. That is the honest outcome and it is a real
one: unimplemented counts against the score exactly as hard as a failure does, per
document 01. What it must not do is crash the run or produce an empty result file,
because a suite that cannot run at all and a suite that scores zero look identical in
a log and are very different things.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from fpcompat import corpus
from fpcompat.cases import Case
from fpcompat.compare import Answer
from fpcompat.driver import Driver

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
        self._runner = None if self._driver is None else Driver(self._driver, corpus.CORPUS)

    @property
    def available(self) -> bool:
        """Whether anything can actually be run."""
        return self._module is not None or self._driver is not None

    @property
    def out_of_process(self) -> bool:
        """Whether the runner should call `run` instead of the case expression.

        The runner reads this rather than asking whether a `run` method exists, because
        this engine has one either way and in the module form it is the wrong path.
        """
        return self._module is None and self._runner is not None

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

    def run(self, case: Case, frame_name: str) -> Answer:
        """Runs one case out of process, when that is the form available.

        The runner calls this instead of evaluating the case expression, because there
        is no module for the expression to be evaluated against. The case expression is
        still the definition of the case, and the driver's job is to hold the firepanda
        spelling of the same thing, which is a duplication and a real cost: a case whose
        expression changes and whose driver entry does not is a case measuring the old
        question. It is the price of measuring a library with no bindings and it goes
        away with the module.

        Args:
            case: The case.
            frame_name: The corpus frame.

        Returns:
            The normalized answer.

        Raises:
            EngineUnavailable: When there is no driver, which cannot happen through the
                runner, since it only calls this when `form` is `driver`.
        """
        if self._runner is None:
            raise EngineUnavailable("there is no built driver to run a case against")
        return self._runner.run(case.id, frame_name)

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

        A conformance number that does not say which firepanda produced it is not a
        number anybody can act on, so this works harder than it looks like it should
        have to. The module form has `__version__`. The driver form has no version to
        ask for, because firepanda has no version constant in Mojo, so the build script
        writes `stamp.json` next to the binary with the version, the commit and whether
        the checkout was dirty, and that is read back here.

        Returns:
            The firepanda version and form, or the absence of both, which is itself
            the thing a reader of an all unimplemented result file needs to know.
        """
        versions = {
            "firepanda": "absent",
            "form": self.form,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        }
        if self._module is not None:
            versions["firepanda"] = getattr(self._module, "__version__", None) or "unknown"
            return versions
        if self._driver is not None:
            versions["firepanda"] = "unstamped"
            stamp = self._driver.parent / "stamp.json"
            if stamp.exists():
                try:
                    recorded = json.loads(stamp.read_text())
                except json.JSONDecodeError:
                    return versions
                versions["firepanda"] = str(recorded.get("firepanda", "unstamped"))
                versions["commit"] = str(recorded.get("commit", "unknown"))
                versions["mojo"] = str(recorded.get("mojo", "unknown"))
                versions["built"] = str(recorded.get("built", "unknown"))
        return versions
