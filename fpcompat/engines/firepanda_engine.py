"""firepanda, the subject.

Two forms, and they are two instruments rather than a scaffold and its replacement.

`drivers/firepanda/main.mojo` is a program that takes a case id and a corpus
directory, runs the firepanda spelling of that case and writes the answer as an Arrow
IPC file. `fpcompat.driver` is this side of that, and it hands back the same `Answer`
the pandas side builds, so nothing downstream of `compare` can tell the two forms
apart. It costs a process per case, roughly a millisecond, and it buys the ability to
measure a library that has no Python bindings at all.

The other form is an importable `firepanda`, bound as the module the case expression
receives, which makes a case a direct comparison in one interpreter.

This file used to say the driver was a scaffold to be deleted the day the module
arrived, and that was wrong in a way worth recording, because it was costing the
board 2452 runs. The driver can only answer a question somebody hand wrote an entry
for. 2452 of the 4081 runs on the board are generated, one per public pandas name,
and they ask whether the name resolves and whether its signature matches. Nobody is
ever going to hand write 2452 driver entries, and a driver entry could not answer
those questions if they did, because "does `DataFrame.pivot` exist" is a question
about a module and the driver is handed a case id. Only reflection answers a
reflection question.

So the engine holds both at once and routes per case, and the routing rule is the
level rather than a list of sections. L0 asks whether a name resolves and L1 whether
its signature matches, and both are questions about the API, answerable by looking at
the module and by nothing else. L2 and L3 ask whether an answer is right, which is a
question about data, and the driver is what runs those today. That is a rule with a
reason in it rather than a lookup table, and it stays correct when a section is added.

L4 joins the reflection levels, for the same kind of reason and not the same reason.
An L4 case asks which exception class comes out, and `fpcompat.driver` says at length
why the driver cannot answer that: Mojo has one `Error` carrying a message, so every
raise arrives here as a `SubjectRaised` with no class on it, and guessing a class from
the message would be inventing a result that nothing downstream could tell from a real
one. The driver is not slightly worse at L4, it is structurally unable, and it shows:
there is not one `errors/` entry in `drivers/firepanda/main.mojo` and there was never
going to be. All 46 L4 runs were scored unimplemented on that account, which was
accurate and was measuring the harness rather than the library.

The module has the classes. `firepanda.errors` maps a tagged message back to a real
Python class at the boundary, so `except KeyError` fires on a missing column and an
`IntCastingNaNError` arrives as one. Routing L4 to the module asks firepanda the
question the level exists to ask, and a wrong answer is now a failure with a reason in
it rather than a gap with the harness's name on it.

An engine with only one form uses it for everything it can and reports the rest as
unimplemented, which is the truthful reading: a firepanda with no extension built has
not answered the reflection questions, and a firepanda with no driver has not answered
the data ones.

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

# Where `drivers/firepanda/build.sh` stages the Python extension, laid out the way a
# wheel has it. Looked for beside the driver rather than on the ambient path so that
# the module and the driver describe the same firepanda: a run that measured a branch
# with the driver and whatever happened to be pip installed with the module would be
# reporting one version and comparing two.
STAGED = corpus.ROOT / "drivers" / "firepanda" / "python"

# The levels the module answers and the driver cannot. See the module docstring. L0
# and L1 are questions about the API rather than about an answer, and they are two
# thirds of the board. L4 is a question about which exception class comes out, and one
# Error type carrying one message has no class in it to report.
IN_PROCESS = ("L0", "L1", "L4")


class EngineUnavailable(RuntimeError):
    """The form of the subject a case needs is not present."""


def _import_staged() -> Any:
    """Imports the staged firepanda, or an ambient one, or nothing.

    The staged copy wins, because it is the one the driver beside it was built from
    and a run has to measure one firepanda rather than two. An ambient `firepanda` is
    accepted when nothing is staged, which is how a machine with a real installation
    and no driver still gets the reflection cases scored.

    An import failure is not an error here. A firepanda too old to have an extension,
    an extension built for another interpreter and a machine with no firepanda at all
    all arrive here the same way, and all three mean the same thing to the board,
    which is that no reflection question has been answered.

    Returns:
        The module, or None.
    """
    if STAGED.is_dir() and str(STAGED) not in sys.path:
        # Appended rather than inserted, so a staged copy can never shadow something
        # the caller deliberately put in front of it.
        sys.path.append(str(STAGED))
    try:
        import firepanda
    except Exception:  # noqa: BLE001  an extension that will not load is an absence
        return None
    return firepanda


class FirepandaEngine:
    """The library under test, in whichever form is available."""

    name = "firepanda"

    def __init__(self) -> None:
        self._module: Any = _import_staged()
        self._driver: Path | None = DRIVER if DRIVER.exists() else None
        self._runner = None if self._driver is None else Driver(self._driver, corpus.CORPUS)

    @property
    def available(self) -> bool:
        """Whether anything can actually be run."""
        return self._module is not None or self._driver is not None

    def out_of_process_for(self, case: Case) -> bool:
        """Whether the runner should call `run` instead of the case expression.

        A reflection case goes to the module, because the driver cannot answer it: the
        driver is handed a case id and asked to produce an answer, and "does this name
        resolve" is not a question with an answer to produce. An L4 case goes to the
        module for the same kind of reason, which is that the class an exception has is
        not something a message can carry. Everything else goes to the driver when
        there is one, because the driver is where the firepanda spelling of each hand
        written case lives.

        The runner asks this per case rather than reading a property once per run,
        which is the change that let the two forms coexist. While it was a property the
        two were mutually exclusive across the whole board, so an importable firepanda
        would have taken every hand written case away from the driver and given it to a
        binding that answers far fewer of them. That was a trap: the first person to
        put firepanda on the path would have watched the board fall off a cliff and
        would have had no reason to suspect the harness.

        Args:
            case: The case about to be run.

        Returns:
            True when the driver should run it.
        """
        if case.level in IN_PROCESS:
            return False
        return self._runner is not None

    @property
    def form(self) -> str:
        """Which forms are in use, for the result file."""
        if self._module is not None and self._driver is not None:
            return "module+driver"
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
                "no importable firepanda, so no reflection case can be answered. "
                f"`pixi run driver <checkout>` stages one in {STAGED.relative_to(corpus.ROOT)} "
                "when the checkout has tools/build_extension.sh and it links"
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

        This goes through the same `corpus.load` the pandas side uses and then hands
        the table across by the Arrow C data interface, rather than asking firepanda to
        open the file. Two reasons. firepanda has no reader for the Arrow IPC file
        format and is not going to grow one just so this suite can load a corpus, and
        more importantly a second reader would mean the two sides of every comparison
        were reading the bytes by two different paths, so a difference in the answer
        could be a difference in the loading. One reader and one handover removes that.

        Args:
            name: The corpus frame name.

        Returns:
            The frame.

        Raises:
            EngineUnavailable: When there is no firepanda.
        """
        return self.module().from_arrow(corpus.load(name))

    def shape_of(self, answer: Any) -> str | None:
        """Which of firepanda's three types an answer is, if it is one of them.

        Asked by `isinstance` against the module's own classes and not by looking for
        an attribute, because attributes are what the comparison layer already uses to
        decide everything it can decide without an import, and this method exists for
        the one question that cannot be answered that way. A firepanda `Series` and a
        firepanda `Index` both cross the Arrow interface as a single array carrying a
        name, and nothing about the bytes says which is which.

        The names are looked up with `getattr` rather than imported at the top of a
        method, so a firepanda that predates one of the three types answers None for it
        instead of raising, and None sends the answer down the ordinary route. That is
        the same reasoning as everywhere else here: a missing name is a gap in the
        subject and has to read as one, not as a crash in the harness.

        Args:
            answer: What a case expression returned.

        Returns:
            `frame`, `series`, `index`, or None for anything else, including a scalar.
        """
        if self._module is None:
            return None
        for shape, attribute in (("frame", "DataFrame"), ("series", "Series"), ("index", "Index")):
            kind = getattr(self._module, attribute, None)
            if isinstance(kind, type) and isinstance(answer, kind):
                return shape
        return None

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
