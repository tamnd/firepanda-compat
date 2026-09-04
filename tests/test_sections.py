"""Tests for the partition of the pandas surface into parity sections.

There is one property here that matters more than the rest. Every public pandas name
lands in exactly one section, and it lands there because somebody wrote it down rather
than because a rule guessed. If that stops being true the denominator is wrong, and a
wrong denominator makes every percentage in the report a number about nothing.

So the first two tests are total and disjoint against the committed inventory, and they
are written to fail with the offending name in the message, because the day this breaks
is the day pandas adds something and the useful output is which thing it added.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from fpcompat import sections, surface

INVENTORY = json.loads(surface.path_for(pd.__version__).read_text())
NAMES = [
    f"{space}.{member}"
    for space, entry in INVENTORY["namespaces"].items()
    for member in entry["members"]
]


# ---------------------------------------------------------------------------
# The partition
# ---------------------------------------------------------------------------


def test_every_pandas_name_has_a_section():
    """Total. A name with no section is a hole in the denominator."""
    homeless = []
    for name in NAMES:
        try:
            sections.section_of(name)
        except sections.SectionError:
            homeless.append(name)
    assert not homeless, f"no section for {homeless}"


def test_every_section_is_one_of_the_eleven():
    """Disjoint by construction, since a name resolves to one string, and this is the
    check that the string is one we print."""
    for name in NAMES:
        assert sections.section_of(name) in sections.SECTIONS


def test_the_two_hand_written_tables_do_not_overlap():
    """`_assign` and `_top` raise on a repeat, so importing the module is the test.
    This asserts the guard itself still bites, because a guard nobody has seen fire is
    a guard nobody knows is wired up."""
    with pytest.raises(sections.SectionError):
        sections._assign("stats", "loc")
    with pytest.raises(sections.SectionError):
        sections._top("stats", "concat")


def test_a_name_with_no_namespace_is_fatal():
    with pytest.raises(sections.SectionError):
        sections.section_of("concat")


def test_an_unknown_namespace_says_where_to_fix_it():
    with pytest.raises(sections.SectionError) as caught:
        sections.section_of("Sparkles.glitter")
    assert "fpcompat/sections.py" in str(caught.value)


# ---------------------------------------------------------------------------
# The decisions worth pinning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("api", "section"),
    [
        # The one namespace with a dot in it. Resolving this as `types` was a real bug.
        ("api.types.is_bool_dtype", "basics"),
        # Same member name on both frame types is one decision, written once.
        ("DataFrame.sort_values", "indexing"),
        ("Series.sort_values", "indexing"),
        # Sorting is indexing and ranking is statistics, which no rule gets right.
        ("DataFrame.sort_index", "indexing"),
        ("DataFrame.rank", "stats"),
        # Combining two tables, even though the implementation is an index operation.
        ("DataFrame.merge", "reshape"),
        ("pandas.merge_asof", "reshape"),
        # Whole namespaces.
        ("Series.str.pad", "strings"),
        ("Series.dt.tz_localize", "temporal"),
        ("Series.cat.codes", "categorical"),
        ("Rolling.mean", "windows"),
        ("GroupBy.agg", "groupby"),
        ("errors.MergeError", "errors"),
        # Basics is the residual, so anything nobody argued about lands there.
        ("DataFrame.astype", "basics"),
    ],
)
def test_named_decisions(api, section):
    assert sections.section_of(api) == section


# ---------------------------------------------------------------------------
# The denominator
# ---------------------------------------------------------------------------


def test_the_denominator_is_every_callable_pandas_has():
    assert sum(sections.denominator(pd.__version__).values()) == INVENTORY["totals"]["callables"]


def test_the_callable_map_agrees_with_the_denominator():
    counts = dict.fromkeys(sections.SECTIONS, 0)
    for section in sections.callables(pd.__version__).values():
        counts[section] += 1
    assert counts == sections.denominator(pd.__version__)


def test_the_parameter_map_covers_every_callable():
    parameters = sections.parameters(pd.__version__)
    assert set(parameters) == set(sections.callables(pd.__version__))
    total = sum(len(names) for names in parameters.values())
    assert total == INVENTORY["totals"]["parameters"]


def test_a_result_file_can_be_read_without_the_pandas_it_names():
    """The version is a parameter so the report can render somebody else's run. This
    only proves the plumbing takes one, which is the part that rots."""
    assert sections.denominator(pd.__version__) == sections.denominator()
