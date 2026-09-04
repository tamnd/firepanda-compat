"""Tests for the surface inventory.

These are tests of the instrument rather than of pandas or of firepanda. What they
protect is the denominator: every coverage and conformance percentage this
repository publishes is computed against the committed inventory, so an inventory
that is nondeterministic, incomplete or stale makes every number downstream of it
meaningless in a way nobody would notice.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fpcompat import surface

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def doc():
    return surface.inventory()


def test_every_namespace_is_available(doc):
    """A namespace that could not be built would silently shrink the denominator."""
    unavailable = {
        name: record["reason"]
        for name, record in doc["namespaces"].items()
        if not record["available"]
    }
    assert unavailable == {}


def test_the_inventory_is_deterministic():
    """Two runs against one pandas must produce the same bytes.

    pandas uses bare sentinel objects as defaults and their repr carries an
    address, which made the first version of this tool report a diff every time it
    ran and would have made `--check` useless.
    """
    assert surface.dumps(surface.inventory()) == surface.dumps(surface.inventory())


def test_no_default_carries_an_address(doc):
    for name, record in doc["namespaces"].items():
        for member, entry in record["members"].items():
            if entry.get("kind") != "callable" or not entry.get("signature"):
                continue
            for param in entry["signature"]:
                default = param["default"] or ""
                assert " at 0x" not in default, f"{name}.{member}({param['name']})"


def test_totals_are_the_sum_of_the_namespaces(doc):
    for key in ("names", "callables", "parameters", "properties"):
        assert doc["totals"][key] == sum(
            record["counts"][key] for record in doc["namespaces"].values()
        )


def test_self_is_never_counted_as_a_parameter(doc):
    """Counting `self` would inflate every method on a class by one."""
    for record in doc["namespaces"].values():
        for member, entry in record["members"].items():
            if entry.get("kind") == "callable" and entry.get("signature"):
                assert "self" not in {p["name"] for p in entry["signature"]}, member


def test_the_surface_is_the_size_we_think_it_is(doc):
    """A floor rather than an equality, so a pandas patch release does not fail this.

    The numbers in the specification are 1413 names and 1125 callables. If a pandas
    upgrade takes the surface below these, something has been removed upstream and
    that is a thing to find out from a failing test rather than from a coverage
    percentage that quietly improved.
    """
    assert doc["totals"]["names"] >= 1400
    assert doc["totals"]["callables"] >= 1100
    assert doc["totals"]["parameters"] >= 3200
    assert len(doc["namespaces"]) == 21


def test_the_committed_inventory_is_current(doc):
    """The same check CI runs, so it fails locally before it fails in review."""
    target = surface.path_for(doc["pandas"])
    if not target.exists():
        pytest.fail(
            f"no committed inventory for pandas {doc['pandas']}: "
            "run `pixi run surface` and commit it"
        )
    assert target.read_text() == surface.dumps(doc)


def test_the_committed_inventory_parses_and_carries_its_versions(doc):
    stored = json.loads(surface.path_for(doc["pandas"]).read_text())
    assert stored["pandas"] and stored["pyarrow"] and stored["python"]
    assert stored["namespaces"]["str"]["counts"]["callables"] >= 50


def test_gaps_reports_what_a_document_does_not_mention(doc, tmp_path):
    markdown = tmp_path / "partial.md"
    markdown.write_text("We support `sum` and `Series.mean` and nothing else.\n")
    missing = surface.gaps(doc, markdown)
    assert "sum" not in missing["DataFrame"]
    assert "mean" not in missing["DataFrame"]
    assert "pivot_table" in missing["DataFrame"]


def test_gaps_against_the_parity_checklist_if_it_is_checked_out(doc):
    """The measurement that produced document 02, run again.

    Skipped rather than failed when the library repository is not beside this one,
    because a developer with one checkout should still be able to run the tests.
    This is the one place a skip is allowed, and it is a skip about the environment
    rather than about a case.
    """
    parity = ROOT.parent / "firepanda" / "docs" / "specs" / "06-pandas-parity.md"
    if not parity.exists():
        pytest.skip("firepanda checkout not beside this one")
    missing = surface.gaps(doc, parity)
    assert "union" in missing["Index"]
    assert "case_when" in missing["Series"]
