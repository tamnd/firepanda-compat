"""Tests for the scoreboard.

The report is the part of this repository that other people quote, so the tests are
mostly about the five rules document 07 sets and the ways a scoreboard usually breaks
them. A denominator that quietly becomes the case list. A divergence that quietly
becomes a pass. An untouched parameter that quietly becomes covered. A losing run that
quietly does not get printed. Each of those has a test here that fails loudly.

The documents are synthetic and tiny. They name real pandas callables, because the
denominator comes from the committed inventory either way, but they carry a handful of
records rather than four thousand, so a test that fails says which rule broke instead of
which of four thousand numbers moved.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pandas as pd
import pytest

from fpcompat import report, sections

VERSION = pd.__version__


def document(records, declarations=None, **overrides):
    """A result file with the given records and defaults for everything else."""
    outcomes = dict.fromkeys(("pass", "fail", "divergent", "unimplemented"), 0)
    for record in records:
        outcomes[record["outcome"]] += 1
    base = {
        "when": "2026-09-05T00:00:00+00:00",
        "engine": "firepanda",
        "oracle": False,
        "cases": len(records),
        "runs": len(records),
        "seconds": 1.0,
        "versions": {"pandas": VERSION, "firepanda": "0.1.0", "python": "3.14.7"},
        "totals": outcomes,
        "records": records,
        "declarations": declarations or {},
    }
    base.update(overrides)
    return base


def record(case_id, api, level, outcome, divergence=""):
    return {
        "id": case_id,
        "api": api,
        "level": level,
        "outcome": outcome,
        "divergence": divergence,
        "frame": "empty",
        "detail": "",
        "seconds": 0.001,
        "section": sections.section_of(api),
    }


# ---------------------------------------------------------------------------
# The denominator is the pandas surface
# ---------------------------------------------------------------------------


def test_the_denominator_is_pandas_and_not_the_case_list():
    """One passing case does not make a suite that is one for one."""
    computed = report.score(document([record("a/one", "DataFrame.sort_values", "L3", "pass")]))
    assert computed["totals"]["callables"] == sections.inventory(VERSION)["totals"]["callables"]
    assert computed["totals"]["levels"]["L3"] == 1


def test_a_name_with_no_case_is_counted_as_untested():
    computed = report.score(document([]))
    assert computed["totals"]["untested"] == computed["totals"]["callables"]
    assert computed["totals"]["levels"]["L0"] == 0


def test_a_case_on_a_property_is_not_in_the_denominator():
    """`DataFrame.shape` is real and it is not a callable, so it is evidence in the
    detail listing rather than a row in the section table."""
    computed = report.score(document([record("basics/shape", "DataFrame.shape", "L2", "pass")]))
    assert computed["totals"]["levels"]["L2"] == 0
    assert "DataFrame.shape" not in computed["names"]


# ---------------------------------------------------------------------------
# The level ladder
# ---------------------------------------------------------------------------


def test_a_level_is_cumulative_downward():
    """A name with a passing L2 case and no L1 case counts at L1 too, because
    producing the right default answer required the call to be accepted."""
    computed = report.score(document([record("a/one", "DataFrame.sort_values", "L2", "pass")]))
    levels = computed["totals"]["levels"]
    assert levels["L0"] == levels["L1"] == levels["L2"] == 1
    assert levels["L3"] == 0


def test_a_failure_low_down_stops_the_climb():
    """An L3 pass on a name whose L1 case failed is not an L3 name. If it were, the
    fastest way to raise the score would be to write the hard cases and skip the easy
    ones."""
    computed = report.score(
        document(
            [
                record("a/one", "DataFrame.sort_values", "L1", "fail"),
                record("a/two", "DataFrame.sort_values", "L3", "pass"),
            ]
        )
    )
    levels = computed["totals"]["levels"]
    # Nothing at all, not even L0, because there is no passing L0 case here either.
    # The inference runs downward from a level a name reached and never upward from
    # one it did not, so a failing L1 does not get to argue that L0 must be fine.
    assert levels["L0"] == levels["L1"] == levels["L2"] == levels["L3"] == 0
    assert computed["totals"]["fail"] == 1


def test_the_climb_stops_where_the_failure_is_and_keeps_what_is_below_it():
    """The realistic shape of it, now that every callable has a generated L0 case. The
    name resolves, its signature is wrong, and whatever an L3 case says about it does
    not undo that."""
    computed = report.score(
        document(
            [
                record("a/zero", "DataFrame.sort_values", "L0", "pass"),
                record("a/one", "DataFrame.sort_values", "L1", "fail"),
                record("a/two", "DataFrame.sort_values", "L3", "pass"),
            ]
        )
    )
    levels = computed["totals"]["levels"]
    assert levels["L0"] == 1
    assert levels["L1"] == levels["L2"] == levels["L3"] == 0


def test_unimplemented_stops_the_climb_without_being_a_failure():
    computed = report.score(
        document([record("a/one", "DataFrame.sort_values", "L2", "unimplemented")])
    )
    assert computed["totals"]["unimplemented"] == 1
    assert computed["totals"]["fail"] == 0
    assert computed["totals"]["levels"]["L2"] == 0


# ---------------------------------------------------------------------------
# A divergence is displayed and not subtracted
# ---------------------------------------------------------------------------


def test_a_divergence_is_neither_a_pass_nor_a_failure():
    computed = report.score(
        document(
            [record("a/one", "DataFrame.sort_values", "L2", "divergent", divergence="sort-kind")]
        )
    )
    totals = computed["totals"]
    assert totals["divergent"] == 1
    assert totals["fail"] == 0
    assert totals["levels"]["L2"] == 0
    assert totals["untested"] == totals["callables"] - 1


# ---------------------------------------------------------------------------
# An untouched parameter is a hole and not a pass
# ---------------------------------------------------------------------------


def test_coverage_counts_the_parameters_a_case_declared():
    declared = {"a/one": {"covers": ["by", "ascending"]}}
    computed = report.score(
        document([record("a/one", "DataFrame.sort_values", "L3", "pass")], declared)
    )
    assert computed["totals"]["covered"] == 2
    assert computed["totals"]["parameters"] == sections.inventory(VERSION)["totals"]["parameters"]


def test_a_parameter_that_is_not_in_the_signature_is_not_covered():
    """A case can claim whatever it likes. What counts is the intersection with the
    parameter list pandas actually has."""
    declared = {"a/one": {"covers": ["by", "invented"]}}
    computed = report.score(
        document([record("a/one", "DataFrame.sort_values", "L3", "pass")], declared)
    )
    assert computed["totals"]["covered"] == 1


def test_a_result_file_with_no_declarations_reports_no_coverage():
    """Older result files predate the declarations block. They render, and they render
    as zero covered, which is the truth about what they can prove."""
    doc = document([record("a/one", "DataFrame.sort_values", "L3", "pass")])
    del doc["declarations"]
    computed = report.score(doc)
    assert computed["totals"]["covered"] == 0
    assert computed["totals"]["levels"]["L3"] == 1


def test_the_work_list_names_the_individual_parameters():
    declared = {"a/one": {"covers": ["by"]}}
    doc = document([record("a/one", "DataFrame.sort_values", "L3", "pass")], declared)
    lines = report.coverage_lines(doc, report.score(doc))
    mine = [line for line in lines if line.startswith("DataFrame.sort_values ")]
    assert len(mine) == 1
    assert "covered: by" in mine[0]
    assert "ascending" in mine[0].split("uncovered:")[1]


def test_a_fully_covered_callable_leaves_the_work_list():
    parameters = sections.parameters(VERSION)["DataFrame.sort_values"]
    declared = {"a/one": {"covers": list(parameters)}}
    doc = document([record("a/one", "DataFrame.sort_values", "L3", "pass")], declared)
    lines = report.coverage_lines(doc, report.score(doc))
    assert not [line for line in lines if line.startswith("DataFrame.sort_values ")]


def test_the_unmeasurable_list_is_the_callables_pandas_cannot_introspect():
    names = report.unmeasurable(document([]))
    inventory = sections.inventory(VERSION)
    expected = sum(
        1
        for entry in inventory["namespaces"].values()
        for info in entry["members"].values()
        if info["kind"] == "callable" and info.get("signature") is None
    )
    assert len(names) == expected
    assert names == sorted(names)


# ---------------------------------------------------------------------------
# The oracle stands in front of the score
# ---------------------------------------------------------------------------


def test_an_imperfect_oracle_run_puts_a_warning_where_the_number_goes():
    doc = document(
        [record("a/one", "DataFrame.sort_values", "L2", "fail")], oracle=True, engine="pandas"
    )
    warning = report.oracle_warning(doc)
    assert "THE ORACLE IS NOT PERFECT" in warning
    assert warning in report.front_page(doc, report.score(doc))


def test_a_clean_oracle_run_says_nothing():
    doc = document(
        [record("a/one", "DataFrame.sort_values", "L2", "pass")], oracle=True, engine="pandas"
    )
    assert report.oracle_warning(doc) == ""


def test_an_imperfect_oracle_run_is_still_printed(tmp_path, capsys):
    """The run that lost is published. There is no threshold below which the report
    declines to print, and the exit code carries the bad news instead."""
    doc = document(
        [record("a/one", "DataFrame.sort_values", "L2", "fail")], oracle=True, engine="pandas"
    )
    path = tmp_path / "run.json"
    path.write_text(json.dumps(doc))
    assert report.main(["--results", str(path)]) == 1
    printed = capsys.readouterr().out
    assert "THE ORACLE IS NOT PERFECT" in printed
    assert "| basics |" in printed


def test_a_subject_run_that_lost_badly_exits_zero(tmp_path, capsys):
    """A low score is a fact and not an error. Only the ratchet and a broken oracle
    fail the process."""
    path = tmp_path / "run.json"
    path.write_text(json.dumps(document([record("a/one", "DataFrame.sort_values", "L2", "fail")])))
    assert report.main(["--results", str(path)]) == 0
    assert "L3 0/" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The ratchet
# ---------------------------------------------------------------------------


def test_a_section_going_backwards_is_a_regression():
    computed = report.score(document([]))
    recorded = {"l3": {"basics": 40}, "fail": 0}
    problems = report.check_ratchet(computed, recorded)
    assert len(problems) == 1
    assert "basics went from L3 40 to 0" in problems[0]
    assert "ratchet.json" in problems[0]


def test_going_forwards_is_not_a_regression():
    computed = report.score(document([record("a/one", "DataFrame.sort_values", "L3", "pass")]))
    assert (
        report.check_ratchet(computed, {"l3": dict.fromkeys(sections.SECTIONS, 0), "fail": 0}) == []
    )


def test_a_new_failure_is_a_regression():
    computed = report.score(document([record("a/one", "DataFrame.sort_values", "L2", "fail")]))
    problems = report.check_ratchet(computed, {"l3": {}, "fail": 0})
    assert len(problems) == 1
    assert "failures went from 0 to 1" in problems[0]


def test_more_unimplemented_names_is_not_a_regression():
    """Unimplemented is a schedule and a failure is a bug. Only the bug count is
    forbidden from going up."""
    computed = report.score(
        document([record("a/one", "DataFrame.sort_values", "L2", "unimplemented")])
    )
    assert report.check_ratchet(computed, {"l3": {}, "fail": 0}) == []


def test_the_floor_covers_every_section():
    computed = report.score(document([]))
    assert set(report.floors(computed)["l3"]) == set(sections.SECTIONS)


# ---------------------------------------------------------------------------
# Rendering somebody else's run
# ---------------------------------------------------------------------------


def test_the_report_renders_with_no_pandas_installed(tmp_path):
    """The point of the declarations block and of passing the version around.

    A result file names the pandas it ran against, the inventory for that pandas is
    committed, and the declarations travel in the file, so nothing about rendering a
    run needs the library or the case registry. This runs in a subprocess with pandas
    made unimportable, because the only way to test an absent import from inside a
    test session that has already imported pandas is not to be in it.
    """
    doc = document(
        [record("a/one", "DataFrame.sort_values", "L3", "pass")], {"a/one": {"covers": ["by"]}}
    )
    path = tmp_path / "run.json"
    path.write_text(json.dumps(doc))

    program = textwrap.dedent(
        """
        import sys

        class Refuse:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in ("pandas", "pyarrow"):
                    raise AssertionError(f"the report imported {name}")
                return None

        sys.meta_path.insert(0, Refuse())
        from fpcompat import report

        sys.exit(report.main(["--results", sys.argv[1], "--site", sys.argv[2]]))
        """
    )
    finished = subprocess.run(
        [sys.executable, "-c", program, str(path), str(tmp_path / "site")],
        capture_output=True,
        text=True,
        cwd=report.ROOT,
    )
    assert finished.returncode == 0, finished.stderr
    assert (tmp_path / "site" / "index.md").read_text().count("| basics |") == 1


# ---------------------------------------------------------------------------
# What the pages say
# ---------------------------------------------------------------------------


def test_the_summary_names_both_engines_and_both_numbers():
    doc = document([record("a/one", "DataFrame.sort_values", "L3", "pass")])
    line = report.summary(doc, report.score(doc))
    assert "firepanda 0.1.0 vs pandas" in line
    assert "untested" in line
    assert "parameters" in line


def test_the_oracle_summary_says_pandas_on_both_sides():
    doc = document([], oracle=True, engine="pandas")
    assert report.summary(doc, report.score(doc)).startswith(f"pandas {VERSION} vs pandas")


def test_the_section_table_has_a_row_per_section_and_a_total():
    doc = document([])
    table = report.section_table(report.score(doc))
    for section in sections.SECTIONS:
        assert f"| {section} |" in table
    assert "| **all** |" in table


def test_the_front_page_refuses_to_call_ninety_percent_done():
    doc = document([])
    page = report.front_page(doc, report.score(doc))
    assert "90 percent and not done" in page
    assert "neither number is quotable without the other" in page


def test_the_site_is_three_pages(tmp_path):
    doc = document([record("a/one", "DataFrame.sort_values", "L3", "pass")])
    written = report.write_site(doc, report.score(doc), tmp_path / "site")
    assert [path.name for path in written] == ["index.md", "coverage.md", "divergences.md"]
    assert all(path.read_text().strip() for path in written)


def test_percent_of_nothing_is_a_dash():
    assert report.percent(0, 0).strip() == "-"
    assert report.percent(1, 4) == "25.0%"


def test_a_missing_result_file_says_which_command_makes_one(tmp_path):
    with pytest.raises(FileNotFoundError) as caught:
        report.read(tmp_path / "nothing.json")
    assert "pixi run oracle" in str(caught.value)
