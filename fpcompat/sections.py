"""Which parity section each pandas name belongs to.

The scoreboard's first rule is that the denominator is the pandas surface and not our
case list, and the section table needs that surface cut into sections before it can
report one. This module is the cut. Every public pandas name in the committed
inventory lands in exactly one of the eleven parity sections, and a test asserts that
the partition is total and disjoint against the inventory, so a name that pandas adds
shows up as a failing test with the name in it rather than quietly falling out of the
denominator.

There is no rule that could compute this. `sort_index` is indexing and `sort_values`
is indexing and `rank` is statistics, and no amount of string matching gets that right.
So it is a table, written out by hand, argued once and then checked mechanically
forever. Whole namespaces are assigned in one line where the namespace is the section,
which is 657 of the names. The remaining 468 are `DataFrame`, `Series` and the top
level, and those are listed name by name.

Two notes on the shape of it.

The frame table is keyed by the member name rather than by the qualified name, because
`DataFrame.sort_values` and `Series.sort_values` are the same decision and writing it
twice is how the two copies come to disagree. 159 of the 207 frame members exist on
both.

A case is attributed to the section of the pandas name it is evidence about rather
than to the module it is declared in. Those two mostly agree, and where they do not it
is on purpose: `fpcompat/cases/errors.py` holds an L4 case about `dt.tz_localize` and
that case is evidence about the temporal section. The module a case lives in organises
the suite. The section a case counts toward is a fact about pandas.
"""

from __future__ import annotations

import json
from functools import cache
from typing import Any

from fpcompat import surface

# The eleven, in the order the report prints them. `divergences` is not here: it is a
# case module and not a parity section, because there is no part of the pandas surface
# that is the divergence part.
SECTIONS = (
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
)


class SectionError(LookupError):
    """A pandas name has no section.

    Fatal, and never reported as a conformance outcome. A name with no section is a
    hole in the denominator, which is the one number in this repository that is not
    allowed to be approximate.
    """


# Namespaces that are a section on their own. This is most of the surface and it is the
# uninteresting part, which is why it is four lines instead of six hundred.
NAMESPACE = {
    "str": "strings",
    "cat": "categorical",
    "list": "nested",
    "struct": "nested",
    "errors": "errors",
    "GroupBy": "groupby",
    "Resampler": "groupby",
    "Rolling": "windows",
    "Expanding": "windows",
    "ExponentialMovingWindow": "windows",
    "Index": "indexing",
    "MultiIndex": "indexing",
    "dt": "temporal",
    "DatetimeIndex": "temporal",
    "Timestamp": "temporal",
    "Timedelta": "temporal",
    "offsets": "temporal",
    # The predicates a third party library calls on our frame. They are not a section
    # of their own in the parity checklist and they belong with the other things a
    # user does before doing anything interesting.
    "api.types": "basics",
}

# ---------------------------------------------------------------------------
# DataFrame and Series
# ---------------------------------------------------------------------------

# Keyed by member name, shared between the two frame types. Anything not named here is
# basics, which is deliberate: basics is the residual and the residual has to have a
# home, otherwise a new pandas method silently disappears from the denominator instead
# of landing somewhere a person will argue with.

FRAME = {}


def _assign(section: str, names: str) -> None:
    """Puts a whitespace separated list of frame members into a section."""
    for name in names.split():
        if name in FRAME:
            raise SectionError(f"{name} is in two sections, {FRAME[name]} and {section}")
        FRAME[name] = section


# Label and position based selection, and everything that changes which rows or which
# labels you have without changing the values. Workstream A in document 08.
_assign(
    "indexing",
    """
    loc iloc at iat index axes xs take reindex reindex_like align filter get keys
    head tail sample truncate squeeze pop droplevel reorder_levels swaplevel
    set_axis set_index reset_index sort_index sort_values rename rename_axis
    add_prefix add_suffix first_valid_index last_valid_index searchsorted argsort
    repeat isetitem item
    """,
)

# The string namespace ships whole, so the accessor is the section.
_assign("strings", "str")
_assign("categorical", "cat")
_assign(
    "temporal",
    "dt asfreq asof at_time between_time to_period to_timestamp tz_convert tz_localize",
)
_assign("groupby", "groupby resample")
_assign("windows", "rolling expanding ewm")
_assign("nested", "explode list struct")

# Changing the shape rather than the labels. `join` and `merge` are here rather than in
# indexing because what a user is doing with them is combining two tables, even though
# what the implementation is doing is an index operation.
_assign("reshape", "T transpose stack unstack pivot pivot_table melt join merge to_frame")

# Workstream H, statistics and the rest of the frame. Everything that reduces, ranks,
# accumulates or fills between known values.
_assign(
    "stats",
    """
    all any count nunique value_counts sum prod product mean median min max mode
    std var sem skew kurt kurtosis quantile describe corr cov corrwith autocorr
    cummax cummin cumprod cumsum diff pct_change shift rank idxmax idxmin
    argmax argmin nlargest nsmallest interpolate replace between
    """,
)

# ---------------------------------------------------------------------------
# The top level
# ---------------------------------------------------------------------------

TOP_LEVEL = {}


def _top(section: str, names: str) -> None:
    """Puts a whitespace separated list of top level names into a section."""
    for name in names.split():
        if name in TOP_LEVEL:
            raise SectionError(f"{name} is in two sections, {TOP_LEVEL[name]} and {section}")
        TOP_LEVEL[name] = section


_top("indexing", "Index MultiIndex RangeIndex IndexSlice")
_top("categorical", "Categorical CategoricalDtype CategoricalIndex")
_top(
    "temporal",
    """
    Timestamp Timedelta TimedeltaIndex DatetimeIndex DatetimeTZDtype DateOffset
    Period PeriodDtype PeriodIndex NaT offsets tseries
    date_range bdate_range period_range timedelta_range interval_range infer_freq
    to_datetime to_timedelta
    """,
)
_top("groupby", "Grouper NamedAgg")
_top(
    "reshape",
    """
    concat merge merge_asof merge_ordered pivot pivot_table melt crosstab
    get_dummies from_dummies lreshape wide_to_long
    """,
)
_top("stats", "cut qcut")
_top("nested", "json_normalize")
_top("errors", "errors")

# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def section_of(api: str) -> str:
    """The parity section a pandas name belongs to.

    Args:
        api: A name as it appears in the inventory, for example `Series.str.pad`,
            `DataFrame.groupby` or `pandas.concat`.

    Returns:
        One of `SECTIONS`.

    Raises:
        SectionError: When the name has no namespace, which means it is not a pandas
            name this repository knows how to talk about.
    """
    parts = api.split(".")
    if len(parts) < 2:
        raise SectionError(f"{api} is not a pandas name, it needs a namespace in front")
    member = parts[-1]
    # Inventory namespaces are mostly one component, so `Series.str.pad` is `str.pad`,
    # and `api.types` is the one that is two. Try the long form first so that the one
    # two component namespace resolves, then fall back to the short one.
    namespace = ".".join(parts[:-1])
    if namespace not in NAMESPACE:
        namespace = parts[-2]

    if namespace in NAMESPACE:
        return NAMESPACE[namespace]
    if namespace in ("DataFrame", "Series"):
        return FRAME.get(member, "basics")
    if namespace == "pandas":
        return TOP_LEVEL.get(member, "basics")
    raise SectionError(
        f"{api} names the namespace {namespace}, which has no section. Add it to "
        "fpcompat/sections.py rather than letting the denominator quietly shrink"
    )


@cache
def inventory(version: str | None = None) -> dict[str, Any]:
    """The committed pandas inventory for one pandas version.

    The version is a parameter so that the report can render a result file produced
    on another machine without a pandas installed here. A result file records the
    pandas it ran against, and the inventory for that pandas is committed, so the two
    together are enough. Only when no version is given does this import pandas to ask
    which one is present.

    Args:
        version: The pandas version, or None for whichever one is installed.

    Returns:
        The document the surface tool wrote.
    """
    if version is None:
        import pandas as pd

        version = pd.__version__
    return json.loads(surface.path_for(version).read_text())


@cache
def denominator(version: str | None = None) -> dict[str, int]:
    """How many pandas callables each section has.

    This is the denominator of every percentage the report prints, and it comes from
    pandas rather than from the case list, because a suite that reports a pass rate
    over its own cases reports how good it is at writing cases it passes.

    Args:
        version: The pandas version, or None for the installed one.

    Returns:
        Section name to callable count, covering all eleven.
    """
    counts = dict.fromkeys(SECTIONS, 0)
    for space, entry in inventory(version)["namespaces"].items():
        for member, info in entry["members"].items():
            if info["kind"] == "callable":
                counts[section_of(f"{space}.{member}")] += 1
    return counts


@cache
def callables(version: str | None = None) -> dict[str, str]:
    """Every pandas callable and its section.

    Args:
        version: The pandas version, or None for the installed one.

    Returns:
        Qualified name to section name.
    """
    found = {}
    for space, entry in inventory(version)["namespaces"].items():
        for member, info in entry["members"].items():
            if info["kind"] == "callable":
                name = f"{space}.{member}"
                found[name] = section_of(name)
    return found


@cache
def parameters(version: str | None = None) -> dict[str, tuple[str, ...]]:
    """Every pandas callable and the parameters it takes.

    Args:
        version: The pandas version, or None for the installed one.

    Returns:
        Qualified name to parameter names, in declaration order.
    """
    found = {}
    for space, entry in inventory(version)["namespaces"].items():
        for member, info in entry["members"].items():
            if info["kind"] != "callable":
                continue
            signature = info.get("signature") or []
            found[f"{space}.{member}"] = tuple(param["name"] for param in signature)
    return found
