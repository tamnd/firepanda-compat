"""Talks to the firepanda driver, the out of process form of the subject.

firepanda is a Mojo library with no Python module yet, so a case cannot be handed the
library the way a case is handed pandas. `drivers/firepanda/main.mojo` stands in for
that: it takes a case id and a corpus frame, runs the firepanda spelling of that case,
prints one line of JSON saying what shape came back, and writes the answer itself as an
Arrow IPC file. This module is the other end of that conversation. It runs the binary,
reads the line, reads the file, and builds the same `Answer` the pandas side builds, so
that everything downstream of `compare` cannot tell which form of the subject produced
a result. See `docs/specs/03-harness.md`.

Three things this deliberately does not do, all of them for the same reason, which is
that a harness which papers over the subject's limits is measuring the harness.

There is no index. firepanda has none, the driver reports `index: 0` on every answer,
and this module writes that down rather than manufacturing a range. A pandas answer
whose index is a plain 0 to n-1 range compares equal to one with no index at all, per
the global rule in document 05, and a pandas answer whose index is anything else fails.
`DataFrame.tail` is the first case that fails this way and it is supposed to.

There is no exception type. Mojo has one `Error` carrying a message, so a raise comes
back as `SubjectRaised` and every L4 case that names a pandas exception type fails
against the driver. Guessing a type name out of the message text would be inventing a
result, and it would be an invisible invention, since nothing downstream could tell a
guessed type from a real one.

There are no warnings. A separate process has no way to hand a `WarningMessage` back
and this module does not pretend otherwise, so a case that declares a warning fails on
the subject side until firepanda has a warning channel to report through.

The exception classes are how the runner tells the four cases apart. `Absent` is the
driver having no entry for a case, which is `unimplemented`. `SubjectRaised` is
firepanda raising, which is a result about firepanda. `DriverBroken` is this repository
or the build being wrong, which is a bug here rather than a fact about anything.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.ipc as ipc

from fpcompat.compare import INDEX_PREFIX, UNNAMED_LEVEL, VALUE, Answer, canonical_type

# Five minutes per case, which is two orders of magnitude above the slowest thing in
# the corpus and is here for a driver that loops rather than for one that is slow. A
# run of three thousand cases against a hung driver is otherwise a run that never ends
# and produces no file at all, which is the one outcome worse than a bad score.
TIMEOUT = 300.0


class Absent(NotImplementedError):
    """The driver has no entry for this case.

    A `NotImplementedError` subclass because that is what the runner already means by
    the name not existing, and it is recognised by type rather than by traceback depth,
    which is how the in process form has to be recognised.
    """


class SubjectRaised(Exception):
    """firepanda raised.

    Attributes:
        type_name: What the driver called it, which is `Error` for everything today
            because Mojo has one exception type.
    """

    def __init__(self, type_name: str, message: str) -> None:
        super().__init__(f"{type_name}: {message}")
        self.type_name = type_name


class DriverBroken(RuntimeError):
    """The driver failed at its own job.

    Not a conformance result. A missing corpus file, a truncated line, an answer file
    that is not Arrow. The runner records it as a failure because it has nowhere else
    to put it, and the message says whose bug it is so that a person reading a result
    file is not told firepanda cannot add up when the real problem is a stale build.
    """


def _table(path: Path) -> pa.Table:
    """Reads the answer file the driver wrote.

    Args:
        path: The file.

    Returns:
        The table, fully in memory, because the file is deleted immediately after.

    Raises:
        DriverBroken: When the file is not readable as Arrow IPC.
    """
    try:
        with ipc.open_file(path) as reader:
            return reader.read_all()
    except pa.ArrowInvalid as error:
        raise DriverBroken(
            f"the driver said it wrote an answer and {path.name} is not an Arrow IPC "
            f"file: {error}. That is a bug in the driver rather than a result"
        ) from error


def _index_names(header: dict[str, Any], n_index: int) -> tuple[str, ...]:
    """Reads the index level names a driver reported, in the spelling `_label` uses.

    A driver sends `null` for an unnamed level, because that is what the level is
    called in every library that has one, and the comparison holds level names in the
    rendering `_label` produces so that the string `"None"` and no name at all stay
    apart. So the translation happens here rather than in the driver, which should not
    have to know how this harness spells a label.

    A driver that sends no `index_names` at all and reports index levels anyway is
    taken to mean they are unnamed, which is the common case and keeps a driver that
    only ever produces one unnamed level from having to send the field.

    Args:
        header: The parsed JSON line.
        n_index: How many index levels the driver said it wrote.

    Returns:
        The level names, one per level.

    Raises:
        DriverBroken: When the driver sent a different number of names than levels.
    """
    if "index_names" not in header:
        return (UNNAMED_LEVEL,) * n_index
    names = tuple(header["index_names"])
    if len(names) != n_index:
        raise DriverBroken(
            f"the driver reported {len(names)} index level names and {n_index} index levels"
        )
    return tuple(UNNAMED_LEVEL if name is None else str(name) for name in names)


def _answer(header: dict[str, Any], table: pa.Table) -> Answer:
    """Builds the normalized answer from the header line and the answer file.

    The driver names the columns whatever firepanda named them, because a person
    opening the file with pyarrow should see a readable table. The positional names
    the comparison wants are applied here, and the labels the driver reported go into
    `columns`, where they are compared against the pandas labels as labels.

    `default_index` is true and `n_index` is zero, which together say that this answer
    carries no index. That is the same thing `normalize` records for a bare Arrow
    table, and it is what makes a firepanda frame comparable to a pandas frame whose
    index is a plain range and not comparable to one whose index is data.

    Args:
        header: The parsed JSON line.
        table: What the driver wrote.

    Returns:
        The answer.

    Raises:
        DriverBroken: When the header and the file disagree, or the kind is not one
            the protocol has.
    """
    kind = header.get("kind")
    n_index = int(header.get("index", 0))

    if kind == "scalar":
        if table.num_columns != 1 or table.num_rows != 1:
            raise DriverBroken(
                f"a scalar answer has to be one row and one column and the driver wrote "
                f"{table.num_rows} by {table.num_columns}"
            )
        column = table.column(0)
        return Answer(
            kind="scalar",
            value=column[0].as_py(),
            type_name=canonical_type(column.type),
        )

    if kind == "frame":
        columns = tuple(header.get("columns", ()))
        if len(columns) != table.num_columns - n_index:
            raise DriverBroken(
                f"the driver reported {len(columns)} column names and wrote "
                f"{table.num_columns - n_index} data columns"
            )
        names = [f"{INDEX_PREFIX}{i}" for i in range(n_index)]
        names += [f"c{i}" for i in range(table.num_columns - n_index)]
        return Answer(
            kind="frame",
            table=table.rename_columns(names),
            n_index=n_index,
            columns=columns,
            index_names=_index_names(header, n_index),
            default_index=bool(header.get("default_index", True)),
        )

    if kind == "series":
        if table.num_columns != n_index + 1:
            raise DriverBroken(
                f"a series answer has one data column and the driver wrote "
                f"{table.num_columns - n_index}"
            )
        names = [f"{INDEX_PREFIX}{i}" for i in range(n_index)] + [VALUE]
        name = header.get("name")
        return Answer(
            kind="series",
            table=table.rename_columns(names),
            n_index=n_index,
            columns=(VALUE,),
            index_names=_index_names(header, n_index),
            default_index=bool(header.get("default_index", True)),
            name=None if name is None else str(name),
        )

    if kind == "index":
        # An Index is not a Series and not an array. pandas returns one from
        # `columns` and from every index accessor, and a library that hands back a
        # Series where pandas hands back an Index has returned the wrong thing even
        # when every value matches, so the kind is carried rather than folded in.
        if table.num_columns != 1:
            raise DriverBroken(
                f"an index answer has one column and the driver wrote {table.num_columns}"
            )
        name = header.get("name")
        return Answer(
            kind="index",
            table=table.rename_columns([VALUE]),
            columns=(VALUE,),
            name=None if name is None else str(name),
        )

    if kind == "tuple":
        # One row of as many columns as the tuple has parts. Each part comes back as
        # a scalar with its Arrow type, for the reason a scalar does not travel
        # through JSON: `shape` is a pair of integers and an integer through JSON has
        # no width, while an int32 where pandas gives int64 is a conformance result.
        if table.num_rows != 1:
            raise DriverBroken(
                f"a tuple answer is one row of one column per part and the driver "
                f"wrote {table.num_rows} rows"
            )
        parts = []
        for i in range(table.num_columns):
            column = table.column(i)
            parts.append(
                Answer(
                    kind="scalar",
                    value=column[0].as_py(),
                    type_name=canonical_type(column.type),
                )
            )
        return Answer(kind="tuple", parts=tuple(parts))

    raise DriverBroken(
        f"the driver reported kind {kind!r}, which is not one of scalar, frame, "
        "series, index or tuple. Either the driver is newer than this file or the "
        "line is corrupt"
    )


def _line(output: str) -> dict[str, Any]:
    """Parses the driver's last line of stdout.

    The last line and not the whole of stdout, because a debug print left in the driver
    should not turn every case into a protocol error. The last line is the protocol and
    anything before it is somebody's `print`.

    Args:
        output: What the driver printed.

    Returns:
        The parsed object.

    Raises:
        DriverBroken: When there is no line or it is not JSON.
    """
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        raise DriverBroken("the driver printed nothing, so there is no result to read")
    try:
        header = json.loads(lines[-1])
    except json.JSONDecodeError as error:
        raise DriverBroken(f"the driver's last line is not JSON: {lines[-1]!r}") from error
    if not isinstance(header, dict) or "status" not in header:
        raise DriverBroken(f"the driver's last line has no status: {lines[-1]!r}")
    return header


class Driver:
    """One built driver binary and the corpus it reads.

    Attributes:
        path: The binary.
        corpus: The directory of corpus frames it is pointed at.
    """

    def __init__(self, path: Path, corpus: Path) -> None:
        self.path = path
        self.corpus = corpus

    def run(self, case_id: str, frame_name: str) -> Answer:
        """Runs one case on one frame and returns the answer.

        Args:
            case_id: The case.
            frame_name: The corpus frame.

        Returns:
            The normalized answer.

        Raises:
            Absent: When the driver has no entry for the case.
            SubjectRaised: When firepanda raised.
            DriverBroken: When the driver itself failed.
        """
        with tempfile.TemporaryDirectory(prefix="fpcompat-") as directory:
            out = Path(directory) / "answer.arrow"
            command = [
                str(self.path),
                f"--case={case_id}",
                f"--frame={frame_name}",
                f"--corpus={self.corpus}",
                f"--out={out}",
            ]
            try:
                finished = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=TIMEOUT,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                raise DriverBroken(
                    f"the driver did not finish {case_id} on {frame_name} in "
                    f"{TIMEOUT} seconds and was killed"
                ) from error
            except OSError as error:
                raise DriverBroken(f"could not run {self.path}: {error}") from error

            if finished.returncode < 0:
                # A signal. Loud, because a driver that segfaults on a case is a real
                # finding about firepanda, and it is the one finding a suite is most
                # tempted to lose track of.
                raise SubjectRaised(
                    "Signal",
                    f"the driver died with signal {-finished.returncode} on {case_id}",
                )

            header = _line(finished.stdout)
            status = header["status"]

            if status == "absent":
                raise Absent(f"the firepanda driver has no entry for {case_id}")
            if status == "raised":
                raise SubjectRaised(header.get("type", "Error"), header.get("message", ""))
            if status == "broken":
                raise DriverBroken(header.get("message", "the driver reported no message"))
            if status != "ok":
                raise DriverBroken(f"the driver reported status {status!r}, which has no meaning")

            if not out.exists():
                raise DriverBroken(f"the driver reported an answer for {case_id} and wrote no file")
            return _answer(header, _table(out))
