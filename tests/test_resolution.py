"""Tests for the generated L0 and L1 cases.

Two things are being checked here. That the generator produced a case for every name
the inventory has, which is what makes the L0 number a statement about pandas rather
than about our imagination, and that a missing name comes out as `unimplemented`
rather than as `fail`.

The second one is the one with teeth. The runner tells those two apart by traceback
depth, which is a subtle mechanism, and an expression written one frame deeper than
intended would silently reclassify every absent method in the library as a bug. That
would turn a schedule into a bug list overnight and nobody would notice, because both
numbers would still add up.
"""

from __future__ import annotations

import json
import types

import pandas as pd
import pytest

from fpcompat import runner, surface
from fpcompat.cases import registry
from fpcompat.cases.resolution import RESOLUTION_CASES
from fpcompat.cases.signature import SIGNATURE_CASES
from fpcompat.engines import load

ORACLE = load("pandas")
INVENTORY = json.loads(surface.path_for(pd.__version__).read_text())
CASES = registry()


def module_with(**members) -> types.ModuleType:
    """A stand in engine module holding only what a test gives it."""
    module = types.ModuleType("firepanda")
    for name, value in members.items():
        setattr(module, name, value)
    return module


class Subject:
    """An engine whose module is whatever the test built."""

    def __init__(self, module):
        self._module = module

    def module(self):
        return self._module

    def frame(self, name):
        return ORACLE.frame(name)

    def versions(self):
        return {"firepanda": "0.0.0"}


def run(case_id: str, module) -> dict:
    """Runs one generated case against a stand in module."""
    return runner.run_case(CASES[case_id], ORACLE, Subject(module), "empty")


# ---------------------------------------------------------------------------
# The generator against the inventory
# ---------------------------------------------------------------------------


def test_there_is_one_resolution_case_per_name():
    """The denominator is pandas, so the case count is pandas' name count."""
    assert INVENTORY["totals"]["names"] == RESOLUTION_CASES


def test_there_is_one_signature_case_per_readable_callable():
    readable = sum(
        1
        for entry in INVENTORY["namespaces"].values()
        for info in entry["members"].values()
        if info["kind"] == "callable" and info.get("signature") is not None
    )
    assert readable == SIGNATURE_CASES


def test_no_signature_case_for_a_callable_pandas_cannot_introspect():
    """Both engines failing to introspect is not evidence, so no case claims it is."""
    for space, entry in INVENTORY["namespaces"].items():
        for member, info in entry["members"].items():
            if info["kind"] == "callable" and info.get("signature") is None:
                assert f"signature/{space.lower()}.{member.lower()}" not in CASES


def test_every_generated_case_names_a_real_pandas_name():
    """Which the registry enforces at declaration, so this is the check on the check."""
    names = {
        f"{space}.{member}"
        for space, entry in INVENTORY["namespaces"].items()
        for member in entry["members"]
    }
    for case_id, item in CASES.items():
        if case_id.startswith(("resolution/", "signature/")):
            assert item.api in names


# ---------------------------------------------------------------------------
# What an absent name looks like
# ---------------------------------------------------------------------------


def test_a_missing_member_is_unimplemented():
    """A schedule, not a bug list."""
    record = run("resolution/dataframe.sort_values", module_with(DataFrame=type("Frame", (), {})))
    assert record["outcome"] == runner.UNIMPLEMENTED
    assert "DataFrame.sort_values does not exist" in record["detail"]


def test_a_missing_accessor_is_unimplemented_for_every_name_under_it():
    """A library with no `.str` at all has 57 names to write, not 57 bugs to fix."""
    record = run("resolution/str.lower", module_with(Series=type("Series", (), {})))
    assert record["outcome"] == runner.UNIMPLEMENTED
    assert "the str namespace could not be built" in record["detail"]


def test_a_missing_member_is_unimplemented_at_l1_too():
    record = run("signature/dataframe.sort_values", module_with(DataFrame=type("Frame", (), {})))
    assert record["outcome"] == runner.UNIMPLEMENTED


# ---------------------------------------------------------------------------
# What a wrong name looks like
# ---------------------------------------------------------------------------


def test_a_method_where_pandas_has_a_property_fails():
    """`frame.shape` and `frame.shape()` are different programs."""

    class Frame:
        def shape(self):
            return ()

    record = run("resolution/dataframe.shape", module_with(DataFrame=Frame))
    assert record["outcome"] == runner.FAIL
    assert "property" in record["detail"]


def test_a_signature_with_the_wrong_parameter_names_fails():
    class Frame:
        def sort_values(self, columns=None, ascending=True):
            return None

    record = run("signature/dataframe.sort_values", module_with(DataFrame=Frame))
    assert record["outcome"] == runner.FAIL
    assert "columns" in record["detail"] or "by" in record["detail"]


def test_a_signature_that_cannot_be_read_fails():
    """Pandas could read this one, so an engine that cannot has not matched it."""

    class Frame:
        sort_values = print  # a C level callable with no readable signature

    record = run("signature/dataframe.sort_values", module_with(DataFrame=Frame))
    assert record["outcome"] == runner.FAIL


def test_a_matching_signature_passes():
    """The oracle proves this for all 1034, and this is the readable version of it."""
    record = run("signature/dataframe.sort_values", pd)
    assert record["outcome"] == runner.PASS


# ---------------------------------------------------------------------------
# The one thing the ids have to guarantee
# ---------------------------------------------------------------------------


def test_generated_ids_do_not_collide():
    generated = [case_id for case_id in CASES if case_id.startswith(("resolution/", "signature/"))]
    assert len(generated) == len(set(generated))
    assert len(generated) == RESOLUTION_CASES + SIGNATURE_CASES


@pytest.mark.parametrize("case_id", ["resolution/api.types.is_bool_dtype", "resolution/str.pad"])
def test_the_two_awkward_namespaces_generated(case_id):
    """`api.types` is the only namespace with a dot in it and `str` is a descriptor."""
    assert case_id in CASES
