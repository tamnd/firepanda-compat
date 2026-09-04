"""Tests for the out of process form of the subject.

Almost all of these run against a fake driver written here rather than against the real
one, and that is deliberate twice over. The first reason is that the real driver is a
compiled Mojo binary which most machines running these tests will not have. The second
is more important: a test against the real driver tells you what firepanda does today,
which changes, while a test against a fake one tells you the protocol is read correctly,
which must not. A driver that reports `absent` has to become `unimplemented` and a
driver that reports `raised` has to become a failure, and if those two ever swap the
score moves for a reason nobody would find.

The fake driver is a Python script with a shebang, because the protocol is a program on
disk that prints a line and writes a file, and testing it through anything less than a
real process would not be testing the protocol.

The last few tests use the real binary and skip when it is absent. They are worth
having despite the skip, because everything above them proves this repository agrees
with itself and nothing above them proves it agrees with the driver.
"""

from __future__ import annotations

import json
import stat
import textwrap
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from fpcompat import corpus, runner
from fpcompat.cases import Case
from fpcompat.compare import Rules
from fpcompat.driver import Absent, Driver, DriverBroken, SubjectRaised
from fpcompat.engines.firepanda_engine import DRIVER, FirepandaEngine

BODY = """
    #!/usr/bin/env python3
    import json, sys
    import pyarrow as pa, pyarrow.ipc as ipc

    options = dict(a.removeprefix("--").split("=", 1) for a in sys.argv[1:])
    header = json.loads({script!r})
    table = {table!r}

    if table is not None:
        with ipc.new_file(options["out"], pa.schema(
            [pa.field(n, pa.type_for_alias(t)) for n, t in table["schema"]]
        )) as writer:
            writer.write_table(pa.table(
                {{n: c for (n, _), c in zip(table["schema"], table["columns"])}},
                schema=pa.schema(
                    [pa.field(n, pa.type_for_alias(t)) for n, t in table["schema"]]
                ),
            ))
    print(json.dumps(header))
    sys.exit({code})
"""


def fake(
    tmp_path: Path,
    header: dict,
    table: dict | None = None,
    code: int = 0,
    stdout: str = "",
) -> Driver:
    """Writes a driver that always answers the same thing.

    Args:
        tmp_path: Where to write it.
        header: The JSON line it prints.
        table: A schema and columns for the answer file, or None to write no file.
        code: The exit code.
        stdout: Anything to print before the protocol line.

    Returns:
        A Driver pointed at it.
    """
    path = tmp_path / "fake-driver"
    source = textwrap.dedent(BODY).strip().format(script=json.dumps(header), table=table, code=code)
    if stdout:
        line = "print(json.dumps(header))"
        source = source.replace(line, f"print({stdout!r})\n{line}")
    path.write_text(source + "\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return Driver(path, corpus.CORPUS)


def case(case_id: str = "basics/scratch") -> Case:
    """One case, for the runner tests, which only read its id."""
    return Case(
        id=case_id,
        api="DataFrame.head",
        section="basics",
        milestone="M6",
        level="L2",
        covers=(),
        frames=("two",),
        expr=lambda pd_, frame: frame.head(),
        rules=Rules(),
    )


# ---------------------------------------------------------------------------
# The four statuses
# ---------------------------------------------------------------------------


def test_absent_is_its_own_exception_and_not_a_failure(tmp_path):
    driver = fake(tmp_path, {"status": "absent"})
    with pytest.raises(Absent):
        driver.run("basics/nothing", "two")


def test_raised_carries_the_type_the_driver_reported(tmp_path):
    driver = fake(tmp_path, {"status": "raised", "type": "KeyError", "message": "no column q"})
    with pytest.raises(SubjectRaised) as caught:
        driver.run("basics/scratch", "two")
    assert caught.value.type_name == "KeyError"
    assert "no column q" in str(caught.value)


def test_broken_is_a_bug_here_rather_than_a_result(tmp_path):
    driver = fake(tmp_path, {"status": "broken", "message": "no corpus frame"}, code=1)
    with pytest.raises(DriverBroken, match="no corpus frame"):
        driver.run("basics/scratch", "two")


def test_a_status_nobody_has_defined_is_broken_rather_than_ignored(tmp_path):
    # The failure mode this is about is a driver newer than this file. Treating an
    # unknown status as a pass would turn a protocol mismatch into a perfect score.
    driver = fake(tmp_path, {"status": "fine"})
    with pytest.raises(DriverBroken, match="which has no meaning"):
        driver.run("basics/scratch", "two")


def test_a_driver_that_says_ok_and_writes_nothing_is_broken(tmp_path):
    driver = fake(tmp_path, {"status": "ok", "kind": "scalar"})
    with pytest.raises(DriverBroken, match="wrote no file"):
        driver.run("basics/scratch", "two")


def test_a_driver_that_prints_nothing_is_broken(tmp_path):
    path = tmp_path / "silent"
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    with pytest.raises(DriverBroken, match="printed nothing"):
        Driver(path, corpus.CORPUS).run("basics/scratch", "two")


def test_a_binary_that_is_not_there_is_broken_rather_than_a_crash(tmp_path):
    with pytest.raises(DriverBroken, match="could not run"):
        Driver(tmp_path / "missing", corpus.CORPUS).run("basics/scratch", "two")


def test_a_signal_is_a_result_about_the_subject(tmp_path):
    # A driver that segfaults on a case is a real finding, and it is the finding a
    # suite is most tempted to lose, because the process produced no output to read.
    path = tmp_path / "crasher"
    path.write_text("#!/bin/sh\nkill -SEGV $$\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    with pytest.raises(SubjectRaised, match="signal 11"):
        Driver(path, corpus.CORPUS).run("basics/scratch", "two")


def test_debug_output_before_the_protocol_line_does_not_break_the_protocol(tmp_path):
    # A `print` left in the driver during a debugging session should cost that person
    # an afternoon and not cost the project a run of three thousand cases.
    driver = fake(tmp_path, {"status": "absent"}, stdout="about to do the thing")
    with pytest.raises(Absent):
        driver.run("basics/scratch", "two")


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


def test_a_scalar_keeps_its_arrow_type(tmp_path):
    # The reason a scalar travels as a table. An int32 answer where pandas gives int64
    # is a conformance failure, and through JSON both of them are the characters `7`.
    driver = fake(
        tmp_path,
        {"status": "ok", "kind": "scalar"},
        {"schema": [["value", "int32"]], "columns": [[7]]},
    )
    answer = driver.run("basics/scratch", "two")
    assert answer.kind == "scalar"
    assert answer.value == 7
    assert answer.type_name == "int32"


def test_a_null_scalar_comes_back_as_none_and_not_as_a_nan(tmp_path):
    driver = fake(
        tmp_path,
        {"status": "ok", "kind": "scalar"},
        {"schema": [["value", "double"]], "columns": [[None]]},
    )
    answer = driver.run("basics/scratch", "two")
    assert answer.value is None
    assert answer.type_name == "double"


def test_a_frame_is_renamed_positionally_and_keeps_its_labels(tmp_path):
    driver = fake(
        tmp_path,
        {"status": "ok", "kind": "frame", "index": 0, "columns": ["a", "b"]},
        {"schema": [["a", "int64"], ["b", "double"]], "columns": [[1, 2], [1.5, 2.5]]},
    )
    answer = driver.run("basics/scratch", "two")
    assert answer.kind == "frame"
    assert answer.table.column_names == ["c0", "c1"]
    assert answer.columns == ("a", "b")
    assert answer.n_index == 0
    assert answer.default_index is True


def test_a_series_data_column_is_named_the_way_the_comparison_wants(tmp_path):
    driver = fake(
        tmp_path,
        {"status": "ok", "kind": "series", "index": 0, "name": "a"},
        {"schema": [["__value__", "int64"]], "columns": [[1, 2]]},
    )
    answer = driver.run("basics/scratch", "two")
    assert answer.kind == "series"
    assert answer.table.column_names == ["__value__"]
    assert answer.name == "a"


def test_an_index_is_carried_as_an_index_and_not_flattened_to_an_array(tmp_path):
    # `columns` and every index accessor give an Index in pandas, and a library that
    # hands back a Series of the same values has returned the wrong thing. The kind is
    # what carries that, so it has to survive the trip through the protocol.
    driver = fake(
        tmp_path,
        {"status": "ok", "kind": "index", "name": None},
        {"schema": [["__value__", "string"]], "columns": [["a", "b"]]},
    )
    answer = driver.run("basics/scratch", "two")
    assert answer.kind == "index"
    assert answer.table.column_names == ["__value__"]
    assert answer.table.column(0).to_pylist() == ["a", "b"]
    assert answer.name is None


def test_an_index_answer_of_more_than_one_column_is_broken(tmp_path):
    driver = fake(
        tmp_path,
        {"status": "ok", "kind": "index"},
        {"schema": [["a", "int64"], ["b", "int64"]], "columns": [[1], [2]]},
    )
    with pytest.raises(DriverBroken, match="an index answer has one column"):
        driver.run("basics/scratch", "two")


def test_a_tuple_keeps_a_part_per_column_with_its_own_type(tmp_path):
    # This is what `shape` comes back as. The widths are the point: a pair of ints
    # through JSON has no width at all, and an int32 where pandas gives int64 is a
    # result somebody should see rather than something the transport rounds off.
    driver = fake(
        tmp_path,
        {"status": "ok", "kind": "tuple"},
        {"schema": [["a", "int64"], ["b", "int32"]], "columns": [[10000], [3]]},
    )
    answer = driver.run("basics/scratch", "two")
    assert answer.kind == "tuple"
    assert len(answer.parts) == 2
    assert [p.value for p in answer.parts] == [10000, 3]
    assert [p.type_name for p in answer.parts] == ["int64", "int32"]


def test_a_tuple_answer_of_more_than_one_row_is_broken(tmp_path):
    driver = fake(
        tmp_path,
        {"status": "ok", "kind": "tuple"},
        {"schema": [["a", "int64"]], "columns": [[1, 2]]},
    )
    with pytest.raises(DriverBroken, match="a tuple answer is one row"):
        driver.run("basics/scratch", "two")


def test_a_frame_whose_header_and_file_disagree_is_broken(tmp_path):
    # Not a failure. A driver that says three columns and writes two has a bug in it,
    # and reporting that as firepanda getting the answer wrong sends somebody looking
    # in the wrong repository.
    driver = fake(
        tmp_path,
        {"status": "ok", "kind": "frame", "index": 0, "columns": ["a", "b", "c"]},
        {"schema": [["a", "int64"]], "columns": [[1]]},
    )
    with pytest.raises(DriverBroken, match="reported 3 column names"):
        driver.run("basics/scratch", "two")


def test_a_scalar_that_is_not_one_row_is_broken(tmp_path):
    driver = fake(
        tmp_path,
        {"status": "ok", "kind": "scalar"},
        {"schema": [["value", "int64"]], "columns": [[1, 2]]},
    )
    with pytest.raises(DriverBroken, match="one row and one column"):
        driver.run("basics/scratch", "two")


def test_a_kind_the_protocol_does_not_have_is_broken(tmp_path):
    driver = fake(
        tmp_path,
        {"status": "ok", "kind": "panel"},
        {"schema": [["value", "int64"]], "columns": [[1]]},
    )
    with pytest.raises(DriverBroken, match="not one of scalar, frame, series, index or tuple"):
        driver.run("basics/scratch", "two")


# ---------------------------------------------------------------------------
# What the runner does with all that
# ---------------------------------------------------------------------------


class FakeEngine:
    """An engine in the driver form, for the runner tests."""

    name = "firepanda"
    out_of_process = True

    def __init__(self, driver: Driver) -> None:
        self._driver = driver

    def run(self, item: Case, frame_name: str):
        return self._driver.run(item.id, frame_name)


def test_absent_is_scored_unimplemented_by_type_and_not_by_depth(tmp_path):
    # `_unimplemented` reads traceback depth for the in process form, and the depth of
    # an Absent is a fact about how many functions fpcompat.driver has rather than
    # about firepanda. This is the test that says so.
    record = runner.run_case(
        case(), runner.load("pandas"), FakeEngine(fake(tmp_path, {"status": "absent"})), "two"
    )
    assert record["outcome"] == runner.UNIMPLEMENTED


def test_a_driver_raise_is_scored_a_failure(tmp_path):
    driver = fake(tmp_path, {"status": "raised", "type": "Error", "message": "no chunks"})
    record = runner.run_case(case(), runner.load("pandas"), FakeEngine(driver), "two")
    assert record["outcome"] == runner.FAIL
    assert "no chunks" in record["detail"]


def test_a_right_answer_from_a_driver_passes(tmp_path):
    # `two` is one row of each of three columns twice over, and `head` returns all of
    # it. The point is that the whole path works, from a real process through the file
    # to a pass, without any of the comparison knowing a driver was involved.
    frame = ipc.open_file(corpus.CORPUS / "two.arrow").read_all()
    driver = fake(
        tmp_path,
        {"status": "ok", "kind": "frame", "index": 0, "columns": list(frame.column_names)},
        {
            "schema": [
                [n, str(t)] for n, t in zip(frame.column_names, frame.schema.types, strict=True)
            ],
            "columns": [c.to_pylist() for c in frame.columns],
        },
    )
    record = runner.run_case(case(), runner.load("pandas"), FakeEngine(driver), "two")
    assert record["outcome"] == runner.PASS, record["detail"]


def test_a_wrong_answer_from_a_driver_fails(tmp_path):
    frame = ipc.open_file(corpus.CORPUS / "two.arrow").read_all()
    columns = [c.to_pylist() for c in frame.columns]
    columns[0] = [value + 1 for value in columns[0]]
    driver = fake(
        tmp_path,
        {"status": "ok", "kind": "frame", "index": 0, "columns": list(frame.column_names)},
        {
            "schema": [
                [n, str(t)] for n, t in zip(frame.column_names, frame.schema.types, strict=True)
            ],
            "columns": columns,
        },
    )
    record = runner.run_case(case(), runner.load("pandas"), FakeEngine(driver), "two")
    assert record["outcome"] == runner.FAIL


# ---------------------------------------------------------------------------
# The real binary, when there is one
# ---------------------------------------------------------------------------

built = pytest.mark.skipif(
    not DRIVER.exists(), reason="no built driver, run drivers/firepanda/build.sh"
)


@built
def test_the_engine_picks_the_driver_up_and_says_so():
    engine = FirepandaEngine()
    assert engine.available
    assert engine.form == "driver"
    assert engine.out_of_process


@built
def test_the_result_file_can_say_which_firepanda_produced_it():
    # A conformance number with no version attached is not a number anybody can act
    # on, and the driver form has no `__version__` to ask for, so the build stamps it.
    versions = FirepandaEngine().versions()
    assert versions["form"] == "driver"
    assert versions["firepanda"] not in ("absent", "unstamped"), (
        "build the driver with drivers/firepanda/build.sh rather than by hand"
    )


@built
def test_the_real_driver_agrees_with_pandas_about_the_length_of_a_frame():
    answer = Driver(DRIVER, corpus.CORPUS).run("basics/len", "tall")
    assert answer.kind == "scalar"
    assert answer.value == 10000
    assert answer.type_name == "int64"


@built
def test_the_real_driver_reports_absent_for_a_case_it_has_no_entry_for():
    with pytest.raises(Absent):
        Driver(DRIVER, corpus.CORPUS).run("basics/no-such-case", "two")


@built
def test_a_corpus_frame_firepanda_cannot_read_is_a_result_and_not_a_broken_harness():
    # firepanda has no dictionary encoded column to read a categorical into. That is a
    # gap in firepanda and it has to be scored as one. Reporting it as a broken harness
    # would hide a real limitation behind a message about the driver, and this test is
    # here because the driver did exactly that until it was pointed at the file first.
    with pytest.raises(SubjectRaised, match="dictionary"):
        Driver(DRIVER, corpus.CORPUS).run("basics/len", "categorical_ordered")


@built
def test_the_real_driver_writes_a_file_pyarrow_can_read():
    answer = Driver(DRIVER, corpus.CORPUS).run("basics/head", "tall")
    assert isinstance(answer.table, pa.Table)
    assert answer.table.num_rows == 5
