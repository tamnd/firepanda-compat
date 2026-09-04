"""Tests for the relaxation sweep.

The sweep is a check on the cases rather than on an engine, so almost everything
here is built out of cases written in this file. A test that asserts the real
registry comes back clean tells you the registry is clean today, which is what the
task itself prints. A test that hands the sweep a case with a relaxation nothing
depends on, and asserts it gets caught, tells you the sweep can tell the difference,
and that is the only version worth having.

One test does run against the real registry, and it is the one about the thing the
sweep exists to correct: that disabling a relaxation in a pandas against pandas run
changes nothing, so the check the specification originally described would report
every declaration in the repository as unnecessary.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from fpcompat import sweep
from fpcompat.cases import Case, select
from fpcompat.compare import Rules, compare
from fpcompat.engines import load

ORACLE = load("pandas")

ORDER = Rules(relaxations=frozenset({"row_order"}), reason="a test case, so no reason applies")
GROUPED = Rules(
    relaxations=frozenset({"grouped_order"}), reason="a test case, so no reason applies"
)


def build(**overrides) -> Case:
    """One case, built directly rather than registered, so nothing global moves."""
    fields = {
        "id": "basics/scratch",
        "api": "DataFrame.head",
        "section": "basics",
        "milestone": "M6",
        "level": "L2",
        "covers": (),
        "frames": ("keys_awkward",),
        "expr": lambda pd, df: df,
        "rules": ORDER,
    }
    fields.update(overrides)
    return Case(**fields)


# ---------------------------------------------------------------------------
# The permutation
# ---------------------------------------------------------------------------


def test_a_frame_with_a_default_index_keeps_one_after_being_permuted():
    # Without this the permuted side would carry 3, 2, 1, 0 and the comparison would
    # fail on the index rather than on the row order, which would make every
    # relaxation look load bearing for a reason that has nothing to do with order.
    frame = pd.DataFrame({"a": [1, 2, 3, 4]})
    permuted = sweep.permute(frame)
    assert list(permuted["a"]) == [4, 3, 2, 1]
    assert list(permuted.index) == [0, 1, 2, 3]


def test_a_real_index_travels_with_its_rows():
    # Which is what a genuinely reordered answer looks like: the same pairs of label
    # and value, in a different order.
    frame = pd.DataFrame({"a": [1, 2]}, index=pd.Index(["x", "y"], name="key"))
    permuted = sweep.permute(frame)
    assert list(permuted.index) == ["y", "x"]
    assert list(permuted["a"]) == [2, 1]


def test_a_series_permutes_the_same_way():
    series = pd.Series([1, 2, 3], index=["a", "b", "c"])
    assert list(sweep.permute(series).index) == ["c", "b", "a"]


def test_an_index_permutes():
    assert list(sweep.permute(pd.Index([1, 2, 3]))) == [3, 2, 1]


def test_one_row_cannot_be_permuted():
    assert sweep.permute(pd.DataFrame({"a": [1]})) is None
    assert sweep.permute(pd.Series([1])) is None
    assert sweep.permute(pd.Index([1])) is None


def test_a_scalar_has_no_order_to_permute():
    assert sweep.permute(7) is None
    assert sweep.permute("seven") is None


# ---------------------------------------------------------------------------
# The two questions
# ---------------------------------------------------------------------------


def test_a_relaxation_that_carries_the_case_is_load_bearing():
    frame = pd.DataFrame({"a": [1, 2, 3]})
    verdict = sweep.probe(build(), "row_order", frame, sweep.permute(frame))
    assert verdict["state"] == sweep.NEEDED


def test_a_relaxation_the_case_does_not_lean_on_is_reported():
    # Every row is the same row, so no permutation of it is distinguishable from it
    # and sorting before comparing achieves nothing.
    frame = pd.DataFrame({"a": [5, 5, 5]})
    verdict = sweep.probe(build(), "row_order", frame, frame.iloc[::-1].reset_index(drop=True))
    assert verdict["state"] == sweep.UNNECESSARY
    assert "nothing about this case depends on it" in verdict["detail"]


def test_a_relaxation_that_does_not_survive_a_permutation_is_a_bug_in_the_layer():
    # Not a fact about the case. If a declared relaxation cannot absorb a reordering
    # of the answer, the relaxation is not doing what its name says.
    frame = pd.DataFrame({"a": [1, 2, 3]})
    verdict = sweep.probe(build(rules=Rules()), "row_order", frame, sweep.permute(frame))
    assert verdict["state"] == sweep.BROKEN
    assert "not delivering the order insensitivity" in verdict["detail"]


# ---------------------------------------------------------------------------
# One case at a time
# ---------------------------------------------------------------------------


def test_a_case_whose_answer_is_a_scalar_declares_a_relaxation_that_is_dead():
    case = build(expr=lambda pd, df: len(df))
    verdicts = sweep.sweep_case(case, ORACLE)
    assert verdicts["row_order"]["state"] == sweep.NO_ORDER


def test_a_case_whose_answer_is_one_row_everywhere_declares_a_relaxation_that_is_dead():
    case = build(expr=lambda pd, df: df.head(1))
    verdicts = sweep.sweep_case(case, ORACLE)
    assert verdicts["row_order"]["state"] == sweep.NO_ORDER


def test_a_case_that_needs_its_relaxation_says_which_frame_proved_it():
    case = build(frames=("keys_awkward", "keys_two_column"))
    verdicts = sweep.sweep_case(case, ORACLE)
    assert verdicts["row_order"]["state"] == sweep.NEEDED
    assert verdicts["row_order"]["frame"] == "keys_awkward"


def test_one_frame_is_enough_to_prove_a_relaxation():
    # A declaration is load bearing if it is load bearing anywhere, so the sweep
    # stops at the first frame that proves it rather than running the rest.
    seen: list[str] = []

    def watched(pd_module, df):
        seen.append(len(df))
        return df

    case = build(frames=("keys_awkward", "keys_two_column", "keys_unique"), expr=watched)
    sweep.sweep_case(case, ORACLE)
    assert len(seen) == 1


def test_a_redundant_pair_of_relaxations_is_caught_as_a_pair():
    # `_apply_ordering` applies grouped_order or row_order and never both, so a case
    # declaring the pair has one line too many. One at a time reports both of them,
    # because removing either one on its own changes nothing, and that is the honest
    # answer: the sweep can say there is a redundancy and it cannot say which of the
    # two lines the author meant to keep. Nothing in the registry does this today and
    # the point is that it would be caught if it did.
    case = build(
        rules=Rules(
            relaxations=frozenset({"grouped_order", "row_order"}),
            reason="a test case, so no reason applies",
        )
    )
    verdicts = sweep.sweep_case(case, ORACLE)
    assert verdicts["grouped_order"]["state"] == sweep.UNNECESSARY
    assert verdicts["row_order"]["state"] == sweep.UNNECESSARY


def test_a_case_that_raises_is_reported_rather_than_swallowed():
    def broken(pd_module, df):
        raise ValueError("the case is wrong")

    verdicts = sweep.sweep_case(build(expr=broken), ORACLE)
    assert verdicts["row_order"]["state"] == sweep.BROKEN
    assert "no answer to permute" in verdicts["row_order"]["detail"]


def test_a_grouped_answer_needs_the_grouped_relaxation():
    case = build(
        api="DataFrame.groupby",
        frames=("keys_10",),
        expr=lambda pd, df: df.groupby("key", sort=False).sum(),
        rules=GROUPED,
    )
    assert sweep.sweep_case(case, ORACLE)["grouped_order"]["state"] == sweep.NEEDED


# ---------------------------------------------------------------------------
# The whole thing
# ---------------------------------------------------------------------------


def test_a_whole_section_of_the_real_registry_sweeps_clean():
    # A section rather than the registry, because the registry is what the task does
    # and it takes a couple of minutes, most of it on one ten million row self join.
    # This is here so that the wiring is covered by something that runs on every
    # commit, and `pixi run sweep` in CI is what actually gates the whole thing.
    document = sweep.sweep("groupby")
    assert document["clean"], document["by_state"]
    assert document["declarations"] == len(document["by_state"][sweep.NEEDED])


def test_every_case_that_declares_a_relaxation_is_swept(monkeypatch):
    # The walk, not the comparisons, so this can cover the whole registry without
    # paying for it. What it catches is a case that declares a relaxation and gets
    # dropped on the floor, which would look exactly like a clean sweep.
    monkeypatch.setattr(sweep, "probe", lambda *a: {"state": sweep.NEEDED, "detail": ""})
    declared = {case.id for case in select() if case.rules.relaxations}
    swept = {finding["case"] for finding in sweep.sweep()["findings"].values()}
    assert swept == declared


def test_the_check_the_specification_described_would_delete_every_relaxation():
    # The reason this module exists, asserted rather than argued. Running the oracle
    # a second time with a relaxation disabled is pandas against pandas with the same
    # expression on the same frame, so both sides come back in the same order and
    # nothing fails. Under the specified rule every declaration in the repository is
    # unnecessary, and every one of them is needed.
    survived = 0
    declarations = 0
    for case in select():
        if not case.rules.relaxations:
            continue
        answer, error, _ = sweep.run_expression(case, ORACLE, case.frames[0])
        if error is not None or sweep._rows(answer) > sweep.LARGE:
            continue
        again, _, _ = sweep.run_expression(case, ORACLE, case.frames[0])
        for name in sorted(case.rules.relaxations):
            declarations += 1
            survived += bool(compare(answer, again, case.rules.without(name)).equal)
    assert declarations > 0
    assert survived == declarations


def test_a_dirty_sweep_exits_non_zero(monkeypatch, capsys):
    dead = build(expr=lambda pd, df: len(df))
    monkeypatch.setattr(sweep, "select", lambda pattern=None: [dead])
    assert sweep.main([]) == 1
    assert "dead on every engine" in capsys.readouterr().out


def test_a_clean_sweep_exits_zero(monkeypatch, capsys):
    good = build()
    monkeypatch.setattr(sweep, "select", lambda pattern=None: [good])
    assert sweep.main([]) == 0
    assert "load bearing" in capsys.readouterr().out


def test_the_document_can_be_written_out(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sweep, "select", lambda pattern=None: [build()])
    out = tmp_path / "deep" / "sweep.json"
    sweep.main(["--json", str(out)])
    document = json.loads(out.read_text())
    assert document["check"] == "relaxation sweep"
    assert document["findings"]["basics/scratch row_order"]["state"] == sweep.NEEDED
    capsys.readouterr()


def test_a_filter_that_matches_nothing_says_so_rather_than_reporting_clean(monkeypatch, capsys):
    monkeypatch.setattr(sweep, "select", lambda pattern=None: [])
    sweep.main(["--filter", "nothing-is-called-this"])
    assert "fact about the filter" in capsys.readouterr().out


@pytest.mark.parametrize("state", [sweep.UNNECESSARY, sweep.NO_ORDER, sweep.BROKEN])
def test_every_state_that_is_not_needed_makes_the_sweep_dirty(state):
    document = {
        "check": "relaxation sweep",
        "cases": 1,
        "declarations": 1,
        "findings": {
            "x y": {
                "case": "x",
                "relaxation": "y",
                "state": state,
                "detail": "d",
                "frame": "",
                "reason": "r",
            }
        },
        "by_state": {
            sweep.NEEDED: [],
            sweep.UNNECESSARY: ["x y"] if state == sweep.UNNECESSARY else [],
            sweep.NO_ORDER: ["x y"] if state == sweep.NO_ORDER else [],
            sweep.BROKEN: ["x y"] if state == sweep.BROKEN else [],
        },
        "clean": False,
    }
    assert "exits non zero" in sweep.render(document)
