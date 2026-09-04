"""The cost matrix.

What is worth testing here is not the timings, which are different every run. It is
everything around them: that the corpus is the same corpus at two sizes, that a row
which did not run stays in the table, that the ratios point the way the caption says
they do, and that the gate catches a regression instead of quietly passing.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from fpcompat import budget, corpus

# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------


def test_the_budget_corpus_has_the_frames_the_operations_ask_for():
    built = budget.frames(1_000)
    assert set(built) == set(budget.FRAMES)
    for item in budget.OPERATIONS:
        for name in item.needs:
            assert name in built, f"{item.id} needs a frame called {name}"


def test_every_frame_is_the_requested_size_except_the_lookup():
    built = budget.frames(2_048)
    for name, table in built.items():
        expected = budget.GROUPS if name == "lookup" else 2_048
        assert table.num_rows == expected, name


def test_the_same_size_generates_the_same_bytes_twice():
    # The whole point of a seeded corpus. If this ever fails, every number the matrix
    # has ever published was measured on inputs nobody can reproduce.
    first = budget.frames(1_000)
    second = budget.frames(1_000)
    for name in first:
        assert budget.sample_digest(first[name]) == budget.sample_digest(second[name])


def test_a_bigger_corpus_extends_the_same_stream():
    # A counter stream, so the first hundred rows of the ten thousand row corpus are
    # the first hundred rows of the thousand row one. That is what makes a result at
    # one size comparable with a result at another, and it is not automatic: it would
    # be false the moment somebody seeded a frame from its own row count.
    small = budget.frames(1_000)
    large = budget.frames(10_000)
    for name in ("numeric", "temporal", "strings"):
        assert small[name].slice(0, 100).to_pylist() == large[name].slice(0, 100).to_pylist()


def test_the_unique_key_is_the_one_column_that_scales_with_the_corpus():
    # id3 is the high cardinality key, so its bound is the row count by design and it
    # is the one column that does not line up between two sizes. Asserted rather than
    # left as a surprise for whoever writes the next cross size comparison.
    small = budget.frames(1_000)["keyed"]
    large = budget.frames(10_000)["keyed"]
    for column in ("id1", "id2", "v1", "v2"):
        assert small[column][:100].to_pylist() == large[column][:100].to_pylist()
    assert max(small["id3"].to_pylist()) < 1_000 < max(large["id3"].to_pylist())


def test_the_corpus_uses_the_correctness_seed():
    # Same generator and same seed as fpcompat/corpus.py, so a Mojo driver can produce
    # these columns without reading a file.
    assert budget.frames(64)["numeric"]["a"].to_pylist()[:4] == [
        int(value)
        for value in budget.signed(corpus.splitmix64(corpus.SEED ^ corpus.salt("budget-a"), 4))
    ]


def test_the_strings_have_a_length_distribution_and_a_space_in_them():
    values = budget.frames(5_000)["strings"]["s"].to_pylist()
    lengths = {len(value) for value in values}
    assert min(lengths) == 4 and max(lengths) == 27
    assert any(" " in value for value in values), "str.split needs something to split on"


def test_the_float_column_is_exactly_representable():
    # Fifty three bits and no more, so no engine gets to be faster by rounding
    # differently and no ratio is really a comparison of two different answers.
    values = budget.frames(1_000)["numeric"]["c"].to_pylist()
    assert all(0.0 <= value < 1.0 for value in values)
    assert all(value * (1 << 53) == int(value * (1 << 53)) for value in values)


def test_the_manifest_digest_covers_both_ends(monkeypatch):
    import pyarrow as pa

    monkeypatch.setattr(budget, "SAMPLE", 2)
    head = pa.table({"a": [9, 1, 2, 3, 4]})
    tail = pa.table({"a": [0, 1, 2, 3, 9]})
    middle = pa.table({"a": [0, 1, 9, 3, 4]})
    plain = pa.table({"a": [0, 1, 2, 3, 4]})
    assert budget.sample_digest(head) != budget.sample_digest(plain)
    assert budget.sample_digest(tail) != budget.sample_digest(plain)
    # A change in the middle is missed, which is the price of a sampled digest and is
    # why this asserts it rather than pretending otherwise.
    assert budget.sample_digest(middle) == budget.sample_digest(plain)


def test_the_digest_notices_a_dtype_change():
    import pyarrow as pa

    wide = pa.table({"a": pa.array([1, 2, 3], type=pa.int64())})
    narrow = pa.table({"a": pa.array([1, 2, 3], type=pa.int32())})
    assert budget.sample_digest(wide) != budget.sample_digest(narrow)


def test_the_frame_name_addresses_the_budget_corpus_through_the_normal_loader():
    assert budget.frame_name(1_000_000, "keyed") == "budget/1000000/keyed"
    assert budget.path_of(1_000_000, "keyed").name == "keyed.arrow"


# ---------------------------------------------------------------------------
# The operations
# ---------------------------------------------------------------------------


def test_no_two_operations_share_an_id():
    assert len(budget.registry()) == len(budget.OPERATIONS)


def test_there_are_at_least_twenty_chained_operations():
    # Twenty is what document 09 asks for. Chains are where a tenth of the memory is
    # available, because a single reduction cannot use much less memory than its input.
    chained = [item for item in budget.OPERATIONS if item.chained]
    assert len(chained) >= 20, "the chained rows are the ones the memory goal is about"
    assert len(budget.OPERATIONS) >= 40


def test_every_section_with_a_conformance_score_has_at_least_one_row():
    # Otherwise a section can be fully conformant and completely unmeasured, and the
    # first anybody hears about it is a user whose program is sixty percent that
    # section. errors and nested are out: raising is not an operation with a cost, and
    # pandas handles nested columns thinly enough that timing it compares two
    # different amounts of work.
    from fpcompat import sections

    measured = {item.section for item in budget.OPERATIONS}
    missing = set(sections.SECTIONS) - measured - {"errors", "nested"}
    assert not missing, f"no cost matrix row for {sorted(missing)}"


def test_every_operation_names_a_section_the_scoreboard_knows():
    from fpcompat import sections

    for item in budget.OPERATIONS:
        assert item.section in sections.SECTIONS, item.id


def test_every_operation_says_which_pandas_names_it_covers():
    for item in budget.OPERATIONS:
        assert item.covers, item.id


def test_a_chained_operation_covers_more_than_one_name():
    for item in budget.OPERATIONS:
        if item.chained:
            assert len(item.covers) > 1, f"{item.id} is marked chained and does one thing"


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------


class Lazy:
    """An answer that has not been computed yet."""

    def __init__(self):
        self.collected = False

    def collect(self):
        self.collected = True
        return [1, 2, 3]


def test_consume_forces_a_lazy_answer():
    # firepanda is lazy underneath after M4, so without this every row would read as
    # instant and the matrix would be a table of plan construction times.
    answer = Lazy()
    assert budget.consume(answer) == 3
    assert answer.collected


def test_consume_counts_rows_of_a_frame():
    import pandas as pd

    assert budget.consume(pd.DataFrame({"a": [1, 2, 3]})) == 3
    assert budget.consume(pd.Series([1, 2])) == 2
    assert budget.consume(7) == 1


def test_iqr_is_zero_below_four_samples():
    assert budget.iqr([1.0, 2.0, 3.0]) == 0.0
    assert budget.iqr([1.0, 2.0, 3.0, 10.0]) == pytest.approx(5.0)


def test_a_raising_operation_is_a_result_and_not_a_crash():
    def boom(module, frames):
        raise NotImplementedError("no groupby yet")

    item = budget.Operation("x", "basics", (), ("a",), boom)
    record = budget.measure(None, item, {}, 3)
    assert record["ok"] is False
    assert record["unimplemented"] is True
    assert "no groupby yet" in record["reason"]


def test_an_error_is_kept_apart_from_an_unimplemented():
    def boom(module, frames):
        raise ValueError("that is a bug")

    item = budget.Operation("x", "basics", (), ("a",), boom)
    record = budget.measure(None, item, {}, 1)
    assert record["ok"] is False and record["unimplemented"] is False


def test_a_measurement_reports_everything_the_specification_asks_for():
    item = budget.Operation("x", "basics", (), ("a",), lambda module, frames: [1, 2, 3])
    record = budget.measure(None, item, {}, 7)
    for field in (
        "median_s",
        "iqr_s",
        "peak_rss_bytes",
        "rss_delta_bytes",
        "cpu_user_s",
        "cpu_sys_s",
        "minor_faults",
        "major_faults",
        "threads_peak",
    ):
        assert field in record, field
    assert record["repeats"] == 7 and record["rows"] == 3


def test_a_dead_worker_becomes_a_row_rather_than_an_exception(monkeypatch):
    # An operation that segfaults an engine is exactly what this table exists to
    # publish, and losing the whole sweep to it means nobody ever sees the row.
    class Dead:
        returncode = -11
        stdout = ""
        stderr = "Segmentation fault"

    monkeypatch.setattr(budget.subprocess, "run", lambda *a, **k: Dead())
    record = budget.child("firepanda", "sum", 1_000, 3)
    assert record["ok"] is False
    assert "-11" in record["reason"] and "Segmentation fault" in record["reason"]


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------


def sweep(engine, records, rows=1_000_000, repeats=7):
    """A result document with whatever records the test needs."""
    return {
        "when": "2026-09-05T00:00:00+00:00",
        "engine": engine,
        "rows": rows,
        "repeats": repeats,
        "machine": {
            "host": "somewhere",
            "platform": "linux",
            "processor": "a cpu",
            "cores": 8,
            "python": "3.14.0",
        },
        "seconds": 1.0,
        "operations": {},
        "records": records,
    }


def ran(median_s, peak_mb, rows=1_000):
    """One measurement that worked."""
    return {
        "ok": True,
        "rows": rows,
        "repeats": 7,
        "median_s": median_s,
        "min_s": median_s,
        "max_s": median_s,
        "iqr_s": 0.0,
        "peak_rss_bytes": peak_mb * (1 << 20),
        "rss_delta_bytes": 0,
        "cpu_user_s": median_s,
        "cpu_sys_s": 0.0,
        "minor_faults": 0,
        "major_faults": 0,
        "threads_peak": 1,
    }


def test_the_ratio_is_above_one_when_the_subject_is_faster():
    name = budget.OPERATIONS[0].id
    text = budget.matrix(
        sweep("pandas", {name: ran(0.100, 800)}),
        sweep("firepanda", {name: ran(0.010, 200)}),
    )
    assert "10.00x" in text and "4.00x" in text


def test_the_ratio_is_below_one_when_the_subject_loses():
    # Publishing this row is the whole discipline. A table that hides it is an
    # advertisement, and the first reader who notices discounts every other number.
    name = budget.OPERATIONS[0].id
    text = budget.matrix(
        sweep("pandas", {name: ran(0.010, 200)}),
        sweep("firepanda", {name: ran(0.020, 400)}),
    )
    assert "0.50x" in text
    assert "| rows below pandas | | 1 | 1 |" in text


def test_a_row_only_one_engine_ran_is_left_out_of_the_ratios_and_named_below():
    first, second = budget.OPERATIONS[0].id, budget.OPERATIONS[1].id
    text = budget.matrix(
        sweep("pandas", {first: ran(0.100, 800), second: ran(0.100, 800)}),
        sweep(
            "firepanda",
            {
                first: ran(0.010, 200),
                second: {"ok": False, "reason": "no rolling yet", "unimplemented": True},
            },
        ),
    )
    assert "every row both engines ran | 1 " in text
    assert "1 operations firepanda did not run" in text
    assert "| unimplemented | no rolling yet |" in text


def test_the_chained_summary_is_taken_from_the_chained_rows_alone():
    single = next(item for item in budget.OPERATIONS if not item.chained)
    chained = next(item for item in budget.OPERATIONS if item.chained)
    text = budget.matrix(
        sweep("pandas", {single.id: ran(0.100, 100), chained.id: ran(0.100, 1000)}),
        sweep("firepanda", {single.id: ran(0.100, 100), chained.id: ran(0.010, 100)}),
    )
    # The single row ties on both and the chained row is ten times on both, so the
    # median over everything is the midpoint and the chained line is the real one.
    assert "| the chained rows | 1 | 10.00x | 10.00x |" in text


def test_the_median_is_used_and_not_the_mean():
    # A hundred times on one operation and a tie everywhere else is not a hundred
    # times engine, and an arithmetic mean says it is.
    names = [item.id for item in budget.OPERATIONS[:3]]
    text = budget.matrix(
        sweep("pandas", {name: ran(0.010, 100) for name in names}),
        sweep(
            "firepanda",
            {
                names[0]: ran(0.010, 100),
                names[1]: ran(0.010, 100),
                names[2]: ran(0.0001, 100),
            },
        ),
    )
    assert "| every row both engines ran | 3 | 1.00x | 1.00x |" in text


def test_the_baseline_alone_still_renders_a_table():
    # On purpose rather than as a fallback. A baseline measured after the subject is a
    # baseline measured to make the subject look a particular way.
    name = budget.OPERATIONS[0].id
    text = budget.matrix(sweep("pandas", {name: ran(0.100, 800)}), None)
    assert "One engine here, which is pandas" in text
    assert "| peak MB | delta MB |" in text
    assert "100.0" in text


def test_the_refusals_are_published_even_with_one_engine():
    text = budget.matrix(
        sweep("pandas", {"sum": {"ok": False, "reason": "it broke", "unimplemented": False}}),
        None,
    )
    assert "1 operations pandas did not run" in text and "| error | it broke |" in text


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_the_gate_passes_a_run_inside_the_slack():
    name = budget.OPERATIONS[0].id
    base = budget.baseline_of(sweep("firepanda", {name: ran(0.010, 100)}))
    assert budget.regressions(sweep("firepanda", {name: ran(0.0105, 100)}), base) == []


def test_the_gate_catches_a_row_that_got_slower():
    name = budget.OPERATIONS[0].id
    base = budget.baseline_of(sweep("firepanda", {name: ran(0.010, 100)}))
    found = budget.regressions(sweep("firepanda", {name: ran(0.020, 100)}), base)
    assert len(found) == 1 and "time went from" in found[0] and "2.00 times" in found[0]


def test_the_gate_catches_a_row_that_got_heavier():
    name = budget.OPERATIONS[0].id
    base = budget.baseline_of(sweep("firepanda", {name: ran(0.010, 100)}))
    found = budget.regressions(sweep("firepanda", {name: ran(0.010, 200)}), base)
    assert len(found) == 1 and "peak memory went from" in found[0]


def test_a_row_that_stopped_running_is_the_largest_regression_available():
    # A gate that only compares the rows both sides have scores this as a clean pass,
    # which is how an engine loses an operation and the table says nothing.
    name = budget.OPERATIONS[0].id
    base = budget.baseline_of(sweep("firepanda", {name: ran(0.010, 100)}))
    found = budget.regressions(
        sweep("firepanda", {name: {"ok": False, "reason": "it broke", "unimplemented": False}}),
        base,
    )
    assert len(found) == 1 and "did not run here" in found[0] and "it broke" in found[0]


def test_a_faster_run_is_not_a_regression():
    name = budget.OPERATIONS[0].id
    base = budget.baseline_of(sweep("firepanda", {name: ran(0.010, 100)}))
    assert budget.regressions(sweep("firepanda", {name: ran(0.001, 10)}), base) == []


def test_the_baseline_keeps_two_numbers_per_row_and_no_more():
    # A baseline carrying page fault counts is a baseline that changes when the
    # allocator changes, and then it gets regenerated until nobody reads the diff.
    name = budget.OPERATIONS[0].id
    base = budget.baseline_of(sweep("firepanda", {name: ran(0.010, 100)}))
    assert set(base["rows_measured"][name]) == {"median_s", "peak_rss_bytes"}


def test_the_baseline_leaves_out_a_row_that_did_not_run():
    # Otherwise a broken operation would be committed as a floor of zero and never
    # noticed again.
    base = budget.baseline_of(
        sweep("firepanda", {"sum": {"ok": False, "reason": "no", "unimplemented": True}})
    )
    assert base["rows_measured"] == {}


def test_the_machine_key_is_not_the_hostname():
    key = budget.machine_key()
    assert "core" in key and key == key.lower()


def test_the_gate_refuses_a_baseline_from_another_processor(tmp_path, monkeypatch):
    name = budget.OPERATIONS[0].id
    monkeypatch.setattr(budget, "RESULTS", tmp_path)
    monkeypatch.setattr(budget, "ROOT", tmp_path)
    here = sweep("firepanda", {name: ran(0.010, 100)})
    there = sweep("firepanda", {name: ran(0.010, 100)})
    there["machine"]["processor"] = "a different cpu"
    (tmp_path / "budget-firepanda-1000000.json").write_text(json.dumps(here))
    (tmp_path / "baselines").mkdir()
    path = tmp_path / "baselines" / f"{budget.machine_key()}-firepanda-1000000.json"
    path.write_text(json.dumps(budget.baseline_of(there)))
    assert budget.gate("firepanda", 1_000_000) == 1


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_a_small_sweep_runs_every_operation_against_pandas(tmp_path, monkeypatch):
    # The one test that actually calls pandas. It runs at a thousand rows, where the
    # timings mean nothing and the point is that all forty five operations are spelled
    # correctly, which no unit test can tell you.
    monkeypatch.setattr(budget, "BUDGET", tmp_path / "budget")
    budget.write(1_000)
    from fpcompat.engines import load

    engine = load("pandas")
    module = engine.module()
    import pyarrow as pa

    def frame(name):
        path = budget.path_of(1_000, name.rsplit("/", 1)[-1])
        with pa.memory_map(str(path), "rb") as source:
            return pa.ipc.open_file(source).read_all().to_pandas()

    monkeypatch.setattr(engine, "frame", frame)
    broken = []
    for item in budget.OPERATIONS:
        loaded = {name: frame(name) for name in item.needs}
        record = budget.measure(module, item, loaded, 1)
        if not record["ok"]:
            broken.append(f"{item.id}: {record['reason']}")
    assert not broken


def test_the_worker_prints_one_json_line_and_nothing_else():
    # The parent parses the last line of stdout. Anything the worker prints alongside
    # it, a warning or a progress line, turns a measured row into a dead one.
    subprocess.run(
        [sys.executable, "-m", "fpcompat.budget", "--corpus", "--rows", "10000"],
        capture_output=True,
        cwd=budget.ROOT,
        check=True,
    )
    finished = subprocess.run(
        [
            sys.executable,
            "-m",
            "fpcompat.budget",
            "--worker",
            "--engine",
            "pandas",
            "--operation",
            "sum",
            "--rows",
            "10000",
            "--repeats",
            "1",
        ],
        capture_output=True,
        text=True,
        cwd=budget.ROOT,
        check=True,
    )
    lines = finished.stdout.strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["ok"] and record["id"] == "sum"


def test_the_baseline_does_not_carry_the_hostname():
    # These files are public and a hostname is somebody's laptop. The processor and
    # the core count are what actually make two timings comparable.
    name = budget.OPERATIONS[0].id
    base = budget.baseline_of(sweep("firepanda", {name: ran(0.010, 100)}))
    assert "host" not in base["machine"]
    assert base["machine"]["processor"] == "a cpu"
