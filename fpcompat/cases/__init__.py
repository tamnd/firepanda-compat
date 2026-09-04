"""The case registry.

Implements the case shape from `docs/specs/03-harness.md`.

A case is a declaration and not a function body, because everything except the
expression has to be machine readable for the scoreboard to say anything. The
expression takes the module and a frame and returns an answer, and the same lambda
runs on pandas and on firepanda because the module is a parameter. That is the whole
point of an API that is a copy of another API, and it means a case cannot
accidentally be written against one engine's spelling.

Three fields do the work that makes the published numbers mean something.

`id` is stable forever. A divergence registry entry, an expected failure, a
performance measurement and a bug report all refer to a case by id, so renaming one
is a breaking change to this repository.

`api` is checked against the committed pandas inventory at registration. A case that
names `Series.str.zfil` does not register, it raises, and the run stops. A registry
full of quietly misspelled names produces coverage numbers that are wrong in the
flattering direction.

`covers` is what makes L3 measurable, and it is checked against the parameter list
the surface tool read from `inspect.signature`. A case claiming to cover `min_period`
on a method whose parameter is `min_periods` would otherwise show up as coverage of a
parameter that does not exist while the real one stays untested.

Importing this module registers nothing. `registry()` imports the case modules and
returns them, so a tool that only wants the `Case` type does not pay for the corpus.
"""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import cache
from typing import Any

from fpcompat import corpus, surface
from fpcompat.compare import Rules, resolve_error

# One module per group of cases. The module name becomes the case's `section` field,
# which is one field nobody has to keep in sync with anything.
#
# This is the suite's own organisation and it is not the same list as the parity
# sections the scoreboard counts against, which are a partition of the pandas surface.
# These are files. The two lists mostly agree, and where they do not it is deliberate:
# `errors.py` holds an L4 case about `dt.tz_localize` and that case is evidence about
# the temporal part of pandas rather than about an errors part.
CASE_MODULES = (
    "basics",
    "indexing",
    "strings",
    "groupby",
    "reshape",
    "stats",
    "windows",
    "categorical",
    "temporal",
    "nested",
    "errors",
    "divergences",
    "resolution",
    "signature",
)

LEVELS = ("L0", "L1", "L2", "L3", "L4")

# Lower case, slash separated, and readable. A case id ends up in a divergence
# registry entry, in a bug report and in a performance table, so it has to survive
# being typed by a person into a filter.
ID_PATTERN = re.compile(r"^[a-z0-9]+(/[a-z0-9][a-z0-9._-]*)+$")


# The operator forms. The surface tool reads public names, and a name starting with an
# underscore is not one, so none of these appear in the inventory. They are still part
# of the pandas API and they are the part people write most, so a case has to be able
# to name one. The escape hatch is a closed list written out by hand rather than a rule
# that lets any underscore name through, and a case naming one of these cannot claim to
# cover a parameter, because there is no signature in the inventory to check it against.
OPERATORS = frozenset(
    {
        "DataFrame.__getitem__",
        "DataFrame.__setitem__",
        "DataFrame.__len__",
        "DataFrame.__iter__",
        "DataFrame.__contains__",
        "DataFrame.__invert__",
        "Series.__getitem__",
        "Series.__setitem__",
        "Series.__len__",
        "Series.__iter__",
        "Series.__contains__",
        "Series.__neg__",
        "Series.__pos__",
        "Series.__invert__",
        "Series.__and__",
        "Series.__or__",
        "Series.__xor__",
        "Index.__getitem__",
        "Index.__len__",
        "Index.__contains__",
    }
)


class CaseError(ValueError):
    """A case declaration is wrong.

    Always fatal and never an outcome. A broken case is a bug in this repository and
    counting it as a conformance failure would blame the library for a typo here.
    """


# The sentinel for a case that asserts nothing at all was warned, which is different
# from a case that does not care. `None` means do not check, because most cases do
# not, and checking everywhere would make every pandas deprecation a hundred failures.
NO_WARNING = ("", "")


@dataclass(frozen=True)
class Case:
    """One conformance case.

    Attributes:
        id: Stable forever. Renaming one breaks the divergence registry.
        api: The pandas name this case is evidence about, as it appears in the
            inventory, for example `Series.str.pad` or `DataFrame.groupby`.
        section: The parity section, taken from the module the case is declared in.
        milestone: Which milestone owns this name.
        level: What the case claims. L2 is default behaviour, L3 exercises named
            parameters, L4 is about the failure.
        covers: Parameter names this case exercises, checked against the inventory.
        frames: Corpus frame names. The case runs once per frame.
        expr: Takes the engine module and a frame, returns an answer.
        rules: What the comparison is allowed to relax, per document 05.
        raises: The exception type name and a message substring, for an L4 case.
        warns: The warning type name and a substring, `NO_WARNING` to assert that
            nothing was warned, or None to not look.
        note: Why this case exists, when that is not obvious from the id.
    """

    id: str
    api: str
    section: str
    milestone: str
    level: str
    covers: tuple[str, ...]
    frames: tuple[str, ...]
    expr: Callable[[Any, Any], Any]
    rules: Rules = field(default_factory=Rules)
    raises: tuple[str, str] | None = None
    warns: tuple[str, str] | None = None
    note: str = ""

    def describe(self) -> dict[str, Any]:
        """The machine readable part of the declaration, for a result file.

        Returns:
            Everything except the expression, which does not serialize and which the
            scoreboard does not need.
        """
        return {
            "id": self.id,
            "api": self.api,
            "section": self.section,
            "milestone": self.milestone,
            "level": self.level,
            "covers": list(self.covers),
            "relaxations": sorted(self.rules.relaxations),
            "tolerance": self.rules.tolerance.name,
            "reason": self.rules.reason,
            "raises": list(self.raises) if self.raises else None,
        }


_REGISTERED: dict[str, Case] = {}
_CURRENT_SECTION: list[str] = []


@cache
def _inventory() -> dict[str, Any]:
    """Reads the committed pandas inventory.

    Returns:
        The document written by the surface tool.

    Raises:
        CaseError: When there is no committed inventory, since without it the api and
            covers checks cannot run and the registry would validate nothing.
    """
    import pandas as pd

    path = surface.path_for(pd.__version__)
    if not path.exists():
        raise CaseError(
            f"no committed inventory at {path.name}. The registry checks every api "
            "name and every covered parameter against it, so run `pixi run surface` "
            "first rather than registering cases that nothing can check"
        )
    return json.loads(path.read_text())


@cache
def _frame_names() -> frozenset[str]:
    """The corpus frame names, read from the committed manifest.

    From the manifest rather than from `corpus.frames()`, because building all 56
    frames to check a spelling would cost a second on every import of the registry,
    and the manifest is the committed description of exactly those frames.

    Returns:
        The names.
    """
    if not corpus.MANIFEST.exists():
        raise CaseError(
            "no corpus/manifest.json, so a case naming a frame cannot be checked. "
            "Run `pixi run corpus` first"
        )
    return frozenset(json.loads(corpus.MANIFEST.read_text())["frames"])


def _members(api: str) -> tuple[str, str, dict[str, Any]]:
    """Resolves an api string against the inventory.

    The namespace is the second to last component, which handles `Series.str.pad` as
    `str.pad` and `pandas.concat` as `pandas.concat` without a table of special cases.

    Args:
        api: The pandas name.

    Returns:
        The namespace, the member and the member's inventory entry.

    Raises:
        CaseError: When the name is not in the inventory.
    """
    parts = api.split(".")
    if len(parts) < 2:
        raise CaseError(f"{api} is not a pandas name, it needs a namespace in front")
    member = parts[-1]
    spaces = _inventory()["namespaces"]
    # Inventory namespaces are one component, so `Series.str.pad` is `str.pad`, except
    # for `api.types` which is two. Try the long form first and fall back to the short.
    namespace = ".".join(parts[:-1])
    if namespace not in spaces:
        namespace = parts[-2]
    if namespace not in spaces:
        raise CaseError(
            f"{api} names the namespace {namespace}, which the inventory does not "
            f"have. The 21 it has are {', '.join(sorted(spaces))}"
        )
    entry = spaces[namespace]["members"].get(member)
    if entry is None:
        raise CaseError(
            f"{api} does not exist in pandas {_inventory()['pandas']}. A case that "
            "names something misspelled would count as coverage of a name nobody has "
            "tested"
        )
    return namespace, member, entry


def _check_covers(api: str, covers: Iterable[str], entry: dict[str, Any]) -> None:
    """Checks that every covered parameter is a parameter of that pandas callable.

    Args:
        api: The pandas name, for the message.
        covers: The parameter names the case claims.
        entry: The inventory entry.

    Raises:
        CaseError: When a name is not a parameter of that callable.
    """
    signature = entry.get("signature") or []
    known = {param["name"] for param in signature}
    unknown = sorted(set(covers) - known)
    if unknown:
        raise CaseError(
            f"{api} has no parameter called {', '.join(unknown)}. It takes "
            f"{', '.join(sorted(known)) or 'nothing'}, and a case that covers a "
            "parameter that does not exist inflates the coverage number while the "
            "real parameter stays untested"
        )


def case(
    id: str,  # noqa: A002  the field is called id in document 03 and in every result file
    api: str,
    milestone: str = "M6",
    level: str = "L2",
    covers: Iterable[str] = (),
    frames: Iterable[str] = (),
    expr: Callable[[Any, Any], Any] | None = None,
    rules: Rules | None = None,
    raises: tuple[str, str] | None = None,
    warns: tuple[str, str] | None = None,
    note: str = "",
) -> Case:
    """Declares and registers one case.

    Args:
        id: Stable forever, lower case and slash separated.
        api: The pandas name, checked against the inventory.
        milestone: Which milestone owns the name.
        level: What the case claims.
        covers: Parameter names, checked against the inventory.
        frames: Corpus frame names. The case runs once per frame.
        expr: Takes the module and a frame, returns an answer.
        rules: What the comparison may relax.
        raises: For an L4 case, the exception type and a message substring.
        warns: The expected warning, `NO_WARNING`, or None to not look.
        note: Why the case exists.

    Returns:
        The registered case.

    Raises:
        CaseError: When anything about the declaration is wrong. Always fatal.
    """
    if not ID_PATTERN.match(id):
        raise CaseError(f"{id!r} is not a usable case id, it has to match {ID_PATTERN.pattern}")
    if id in _REGISTERED:
        raise CaseError(
            f"{id} is already registered. Ids are stable forever because a divergence "
            "entry and a bug report both point at one, so two cases cannot share one"
        )
    if level not in LEVELS:
        raise CaseError(f"{level} is not one of {', '.join(LEVELS)}")
    if expr is None:
        raise CaseError(f"{id} has no expression, so there is nothing to run")

    if api in OPERATORS:
        if covers:
            raise CaseError(
                f"{id} covers parameters of {api}, which is an operator form with no "
                "signature in the inventory, so there is nothing to check the claim "
                "against and the coverage number would be taking the case's word for it"
            )
    else:
        _, _, entry = _members(api)
        _check_covers(api, covers, entry)

    frames = tuple(frames)
    if not frames:
        raise CaseError(f"{id} names no frames, and a case that runs on nothing is not a case")
    unknown = sorted(set(frames) - _frame_names())
    if unknown:
        raise CaseError(
            f"{id} names the frames {', '.join(unknown)}, which the corpus does not "
            "have. `python -m fpcompat.corpus --list` prints the names"
        )
    if raises and level != "L4":
        raise CaseError(f"{id} expects an exception, which makes it an L4 case rather than {level}")
    if raises:
        # Checked here rather than at run time, because a case naming an exception type
        # that does not exist is the same class of mistake as one naming a method that
        # does not exist, and it belongs in the same place: fatal, at import, with the
        # run refusing to start rather than dying halfway through.
        try:
            resolve_error(raises[0])
        except LookupError as error:
            raise CaseError(f"{id} expects {raises[0]}, and {error}") from error

    declared = Case(
        id=id,
        api=api,
        section=_CURRENT_SECTION[-1] if _CURRENT_SECTION else "unknown",
        milestone=milestone,
        level=level,
        covers=tuple(covers),
        frames=frames,
        expr=expr,
        rules=rules or Rules(),
        raises=raises,
        warns=warns,
        note=note,
    )
    _REGISTERED[id] = declared
    return declared


def section(name: str) -> None:
    """Names the parity section every case declared after this call belongs to.

    Called once at the top of each module in this package.

    Args:
        name: The module name, which must be one of `CASE_MODULES`.

    Raises:
        CaseError: When the name is not a known section.
    """
    if name not in CASE_MODULES:
        raise CaseError(f"{name} is not a case module, the modules are {', '.join(CASE_MODULES)}")
    _CURRENT_SECTION.append(name)


def registry() -> dict[str, Case]:
    """Imports every case module and returns the whole registry.

    Returns:
        Case id to case, in declaration order, which is the order the runner uses so
        that a failure list reads in the same order as the source.
    """
    for name in CASE_MODULES:
        importlib.import_module(f"fpcompat.cases.{name}")
    return dict(_REGISTERED)


def select(pattern: str | None = None) -> list[Case]:
    """The cases matching a filter.

    Args:
        pattern: A substring of the id, of the api or of the section. None for all.

    Returns:
        The selected cases in declaration order.
    """
    cases = list(registry().values())
    if not pattern:
        return cases
    return [
        item
        for item in cases
        if pattern in item.id or pattern in item.api or pattern == item.section
    ]
