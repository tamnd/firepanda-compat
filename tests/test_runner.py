"""Tests for the runner.

Four outcomes and no fifth, so there are four tests that each produce exactly one of
them, built out of cases written here rather than taken from the registry. Building
them here is the point: a test that asserts a real case passes tells you the case
passes, while a test that asserts a deliberately wrong case fails tells you the
harness can tell the difference, and only the second one is worth having. If the
runner ever stops noticing a wrong answer, every number this repository publishes
becomes a hundred percent and none of it means anything.

The last two tests are about the crash recovery, which is the part that is hardest to
get right and easiest to get wrong quietly. A worker that dies has to be attributed to
the case that killed it and the run has to carry on, because a suite that loses six
hundred results to one segfault is a suite people stop running.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from fpcompat import runner
from fpcompat.cases import NO_WARNING, Case
from fpcompat.compare import Rules
from fpcompat.engines import load

ORACLE = load("pandas")


def build(**overrides) -> Case:
    """One case, built directly rather than registered, so nothing global moves."""
    fields = {
        "id": "basics/scratch",
        "api": "DataFrame.head",
        "section": "basics",
        "milestone": "M6",
        "level": "L2",
        "covers": (),
        "frames": ("two",),
        "expr": lambda pd, df: df.head(1),
        "rules": Rules(),
    }
    fields.update(overrides)
    return Case(**fields)


class Lying:
    """An engine that is pandas with one thing about it deliberately wrong.

    Wrapping the real engine rather than writing a fake one keeps everything except
    the single lie identical, so a test that fails is failing about the lie and not
    about some other difference between a fake and the real thing.
    """

    def __init__(self, module=None):
        self._module = module

    def module(self):
        return self._module if self._module is not None else pd

    def frame(self, name):
        return ORACLE.frame(name)

    def versions(self):
        return ORACLE.versions()


class Shadow:
    """pandas under a different identity.

    An expression that wants to answer differently for the subject has to be able to
    tell which engine it was handed, and against pandas as its own oracle the module
    is the same object both times. This is the same module wearing a different name,
    which is exactly what firepanda will be.
    """

    def __getattr__(self, name):
        return getattr(pd, name)


def bare():
    """A real empty module, which is what a name nobody has written looks like.

    A real module and not a class with a __getattr__ on it, because the depth of the
    traceback is the whole signal the runner reads and a Python level __getattr__ adds
    a frame to it. An empty module raises from C at the same depth firepanda will.
    """
    import types

    return types.ModuleType("firepanda")


def run_one(case: Case, subject=None, frame_name: str = "two") -> dict:
    """Runs one case in this process, against pandas as the oracle."""
    return runner.run_case(case, ORACLE, subject or ORACLE, frame_name)


# ---------------------------------------------------------------------------
# The four outcomes
# ---------------------------------------------------------------------------


def test_matching_answers_pass():
    record = run_one(build())
    assert record["outcome"] == runner.PASS
    assert record["detail"] == ""


def test_a_wrong_answer_fails():
    """The most important test here.

    The expression asks the subject for two rows and the oracle for one, so the two
    answers differ in exactly one way. A runner that calls this a pass is a runner
    that would call anything a pass.
    """
    seen = {"count": 0}

    def expr(module, df):
        seen["count"] += 1
        return df.head(2 if seen["count"] == 2 else 1)

    record = run_one(build(expr=expr))
    assert record["outcome"] == runner.FAIL
    assert "row" in record["detail"] or "shape" in record["detail"]


def test_an_off_by_one_float_fails():
    """A tolerance class is not a licence to be wrong by a whole unit."""

    def expr(module, df):
        answer = df["b"].sum()
        return answer + 1.0 if module is not pd else answer

    record = run_one(build(api="Series.sum", expr=expr), subject=Lying(module=Shadow()))
    assert record["outcome"] == runner.FAIL


def test_a_missing_method_is_unimplemented_and_not_a_failure():
    """The distinction the whole scoreboard rests on.

    A method firepanda has not written yet is a gap in the plan. A method that is
    there and wrong is a bug. Counting the first as the second makes an early
    milestone look broken, and counting the second as the first hides a real defect,
    so the runner separates them by looking at how deep the AttributeError came from.
    """

    def expr(module, df):
        return module.not_written_yet(df)

    record = run_one(build(expr=expr), subject=Lying(module=bare()))
    assert record["outcome"] == runner.UNIMPLEMENTED
    assert "not_written_yet" in record["detail"]


def test_a_deep_attribute_error_is_a_failure_and_not_unimplemented():
    """An AttributeError from inside a working method is a bug, not a gap.

    Three frames deep means something went wrong while computing, and calling that
    unimplemented would let a library hide real breakage behind the word not yet.
    """

    def expr(module, df):
        if module is pd:
            return df.head(1)

        def inner():
            def deeper():
                raise AttributeError("something inside broke")

            return deeper()

        return inner()

    record = run_one(build(expr=expr), subject=Lying(module=Shadow()))
    assert record["outcome"] == runner.FAIL


# The divergent outcome is produced by the registry in fpcompat/divergences.py and it
# is tested in tests/test_divergences.py, next to the rule that a registered divergence
# has to actually diverge. There is nothing left to assert about it from here.


def test_the_four_outcomes_are_the_only_ones():
    """There is no skip, and adding one would be a change to the specification."""
    assert runner.OUTCOMES == ("pass", "fail", "divergent", "unimplemented")


# ---------------------------------------------------------------------------
# Level four
# ---------------------------------------------------------------------------


def test_l4_matching_exception_passes():
    record = run_one(
        build(
            id="errors/scratch",
            api="DataFrame.__getitem__",
            level="L4",
            expr=lambda pd, df: df["not_a_column"],
            raises=("KeyError", "not_a_column"),
        )
    )
    assert record["outcome"] == runner.PASS


def test_l4_wrong_exception_type_fails():
    """A subclass is not what the caller wrote in their except clause."""
    record = run_one(
        build(
            id="errors/scratch",
            api="DataFrame.__getitem__",
            level="L4",
            expr=lambda pd, df: df["not_a_column"],
            raises=("ValueError", "not_a_column"),
        )
    )
    assert record["outcome"] == runner.FAIL
    assert "wrong about pandas" in record["detail"]


def test_l4_case_that_is_wrong_about_pandas_says_so():
    """A broken case has to blame this repository and not the library under test."""
    record = run_one(
        build(
            id="errors/scratch",
            api="DataFrame.head",
            level="L4",
            expr=lambda pd, df: df.head(1),
            raises=("KeyError", "nothing is raised here"),
        )
    )
    assert record["outcome"] == runner.FAIL
    assert "wrong about pandas" in record["detail"]


def test_an_unexpected_exception_from_pandas_fails():
    """Either the expression is wrong or the case should have been an L4 one."""
    record = run_one(build(expr=lambda pd, df: df["not_a_column"]))
    assert record["outcome"] == runner.FAIL
    assert "should be an L4 one" in record["detail"]


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


def test_a_case_that_asserts_silence_fails_when_something_warns():
    def expr(module, df):
        import warnings

        if module is not pd:
            warnings.warn("noisy", UserWarning, stacklevel=1)
        return df.head(1)

    record = run_one(build(warns=NO_WARNING, expr=expr), subject=Lying(module=Shadow()))
    assert record["outcome"] == runner.FAIL


def test_a_case_that_asserts_silence_passes_when_nothing_warns():
    record = run_one(build(warns=NO_WARNING))
    assert record["outcome"] == runner.PASS


# ---------------------------------------------------------------------------
# The runner surviving its own bugs
# ---------------------------------------------------------------------------


def test_a_comparison_that_raises_is_a_loud_failure_and_not_a_crash(monkeypatch):
    """The comparison blowing up must not take six hundred results with it."""

    def explode(expected, actual, rules):
        raise RuntimeError("the normalizer is broken")

    monkeypatch.setattr(runner, "compare", explode)
    record = run_one(build())
    assert record["outcome"] == runner.FAIL
    assert "bug in fpcompat.compare" in record["detail"]


def test_a_worker_that_dies_is_attributed_to_the_case_that_killed_it():
    """End to end, with a real subprocess that really exits in the middle.

    The case that kills the worker is the second of three, so the assertion is not
    only that the run finished but that the right one got the blame and the third one
    still ran. Any harness that gets this wrong loses results silently.
    """
    work = [("basics/shape", "two"), ("basics/kill", "two"), ("basics/shape", "single")]
    records = _drive_with_a_dying_worker(work)
    assert len(records) == len(work)
    assert [item["id"] for item in records] == [item[0] for item in work]
    assert records[0]["outcome"] == runner.PASS
    assert records[1]["outcome"] == runner.FAIL
    assert "the worker died" in records[1]["detail"]
    assert records[2]["outcome"] == runner.PASS


def _drive_with_a_dying_worker(work):
    """Runs `drive` against a worker that exits when it sees `basics/kill`."""
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import json, os, sys
        from fpcompat import runner
        from fpcompat.cases import registry
        from fpcompat.engines import load
        engine = load("pandas")
        cases = registry()
        for line in sys.stdin:
            key = line.strip()
            if not key:
                continue
            case_id, frame = key.split("\\t")
            print(json.dumps({"event": "start", "id": case_id, "frame": frame}), flush=True)
            if case_id == "basics/kill":
                os._exit(1)
            record = runner.run_case(cases[case_id], engine, engine, frame)
            print(json.dumps({"event": "result", "record": record}), flush=True)
        """
    )

    def spawn(engine_name, oracle):
        return subprocess.Popen(
            [sys.executable, "-c", script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            cwd=runner.ROOT,
        )

    original = runner._spawn
    runner._spawn = spawn
    try:
        return runner.drive(list(work), "pandas", True)
    finally:
        runner._spawn = original


def test_a_worker_that_never_works_gives_up():
    """Twenty deaths is a broken environment and not a conformance result."""
    import subprocess
    import sys

    def spawn(engine_name, oracle):
        return subprocess.Popen(
            [sys.executable, "-c", "import os; os._exit(1)"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            cwd=runner.ROOT,
        )

    original = runner._spawn
    runner._spawn = spawn
    try:
        with pytest.raises(RuntimeError, match="broken environment"):
            runner.drive([("basics/shape", "two")], "pandas", True)
    finally:
        runner._spawn = original


# ---------------------------------------------------------------------------
# The result document
# ---------------------------------------------------------------------------


def test_the_result_document_is_serializable_and_counted(tmp_path):
    document = runner.run("pandas", True, "basics/shape")
    assert document["totals"]["pass"] == len(document["records"])
    assert sum(document["totals"].values()) == len(document["records"])
    assert set(document["totals"]) == set(runner.OUTCOMES)
    path = tmp_path / "out.json"
    path.write_text(json.dumps(document, indent=2))
    assert json.loads(path.read_text())["engine"] == "pandas"


def test_a_filter_that_matches_nothing_is_not_a_silent_success():
    """Zero of zero is not a hundred percent, and the runner has to say so."""
    with pytest.raises(RuntimeError, match="matches no cases"):
        runner.run("pandas", True, "nothing/matches/this")
