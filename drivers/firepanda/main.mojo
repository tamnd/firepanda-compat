"""The firepanda side of the conformance suite, run as a separate process.

firepanda is a Mojo library with no Python module yet, so a case cannot be handed
the library the way a case is handed pandas. This program stands in for that. It
takes a case id and a corpus frame, runs the firepanda spelling of that case, and
writes the answer as an Arrow IPC file that the harness reads back with pyarrow and
compares against the pandas answer. See `docs/specs/03-harness.md`.

There is one rule here and everything else follows from it. This program writes down
what firepanda does and never what pandas does. It would be easy to make the numbers
look better: `DataFrame.tail` in pandas keeps the original row labels, firepanda has
no index at all, and five lines here could manufacture an index column and turn a
failure into a pass. That would be a lie, and worse than a lie it would be an
invisible one, since nothing downstream can tell a real pass from a manufactured one.
So the driver runs the firepanda spelling, reports the shape firepanda actually
produced, and the cases that need an index fail until firepanda has an index. That
failure is the M6 A1 workstream, and it is supposed to be visible.

A case id this program has no entry for is reported as absent, which the harness
scores as unimplemented, which counts against the score exactly as hard as a failure
does. That is deliberate too. The alternative, a driver that quietly skips what it
does not know, makes a suite whose score goes up when you delete cases.

The protocol is one line of JSON on stdout and nothing else. stderr is free.

    {"status": "ok", "kind": "frame", "index": 0, "columns": ["a", "b"]}
    {"status": "ok", "kind": "series", "index": 0, "name": "a"}
    {"status": "ok", "kind": "scalar"}
    {"status": "absent"}
    {"status": "raised", "type": "KeyError", "message": "no column named q"}

`kind` is the vocabulary of `fpcompat.compare.Answer`, because the answer's shape is
part of the answer: a Series and a one column DataFrame are different results and a
suite that folded them together would pass a library that returned the wrong one.
`index` is how many leading columns of the written table came from an index, which is
always zero today and is in the protocol because it will not always be. A scalar is
written as a table of one row and one column, so that its type travels as an Arrow
type rather than through JSON, where a float would lose its last bits and an integer
would lose its width.

Usage:
    firepanda-compat-driver --case=basics/head --frame=two --corpus=corpus \\
        --out=/tmp/answer.arrow
"""

from std.os.path import exists
from std.sys import argv, exit

from firepanda.array.array import Array
from firepanda.dtype import Field, LogicalType, Schema
from firepanda.frame.frame import DataFrame
from firepanda.frame.series import Series
from firepanda.frame.groupby import AggSpec
from firepanda.io import read_arrow, write_arrow
from firepanda.kernel import AggKind

# The exit status is not the protocol, the JSON line is, and this is only here so
# that a harness reading a truncated line has something to say about it. Zero means a
# line was printed, whatever that line said, including `raised` and `absent`. Anything
# else means this program failed to do its own job, which is a bug here rather than a
# result about firepanda.
comptime OK = 0
comptime BROKEN = 1


def quote(text: String) -> String:
    """Escapes a string for JSON.

    Small on purpose. The only strings that go through here are column names from the
    corpus and error messages from firepanda, and a dependency on a JSON library for
    that would be a dependency the harness has to build.

    Args:
        text: The string.

    Returns:
        The string with quotes, backslashes and control characters escaped.
    """
    var out = String('"')
    for char in text.codepoints():
        var code = Int(char)
        if code == 34:
            out += '\\"'
        elif code == 92:
            out += "\\\\"
        elif code == 10:
            out += "\\n"
        elif code == 13:
            out += "\\r"
        elif code == 9:
            out += "\\t"
        elif code < 32:
            comptime DIGITS = "0123456789abcdef"
            out += "\\u00"
            out += DIGITS[byte = code // 16 : code // 16 + 1]
            out += DIGITS[byte = code % 16 : code % 16 + 1]
        else:
            out += String(char)
    return out + '"'


def names_json(frame: DataFrame) -> String:
    """Renders a frame's column names as a JSON array.

    Args:
        frame: The frame.

    Returns:
        The array.
    """
    var out = String("[")
    var names = frame.names()
    for i in range(len(names)):
        if i > 0:
            out += ","
        out += quote(names[i])
    return out + "]"


def option(args: List[String], name: String) -> String:
    """Reads one `--name=value` argument.

    Args:
        args: The command line.
        name: The option name, without the dashes.

    Returns:
        The value, or the empty string when the option was not given.
    """
    var prefix = String("--") + name + "="
    for i in range(len(args)):
        var arg = args[i]
        if arg.startswith(prefix):
            return String(arg[byte = prefix.byte_length() :])
    return String("")


def scalar_frame[dt: DType](name: String, value: Scalar[dt]) raises -> DataFrame:
    """Wraps one value as a frame of one row and one column.

    A scalar answer travels as a table rather than as a number in the JSON line, so
    that its type is an Arrow type. A float in JSON is a decimal string and the last
    bits do not survive the trip, and an integer in JSON has no width at all, while
    the difference between an int32 and an int64 answer is a conformance result.

    Args:
        name: The column name, which the harness ignores and a person reading the
            file with pyarrow does not.
        value: The value.

    Parameters:
        dt: The value's dtype.

    Returns:
        The frame.

    Raises:
        Error: If the one column frame cannot be built, which would be a bug here.
    """
    var column = Array[dt](1)
    column[0] = value
    var series = List[Series]()
    series.append(Series(name, column^))
    return DataFrame.from_series(series^)


def reduce(frame: DataFrame, column: String, kind: AggKind) raises -> DataFrame:
    """Runs one reduction over one column, the firepanda spelling.

    `DataFrame.agg` rather than a hand written loop, because that is the method a
    firepanda user would call and the suite is measuring firepanda's methods rather
    than this file's arithmetic.

    Args:
        frame: The frame.
        column: The column to reduce.
        kind: Which reduction.

    Returns:
        A frame of one row and one column.

    Raises:
        Error: Whatever `agg` raises, which the caller turns into a `raised` line.
    """
    var specs = List[AggSpec]()
    specs.append(AggSpec(column, kind, "value"))
    return frame.agg(specs)


def main() raises:
    var args = argv()
    var arguments = List[String]()
    for i in range(len(args)):
        arguments.append(String(args[i]))

    var case_id = option(arguments, "case")
    var frame_name = option(arguments, "frame")
    var corpus = option(arguments, "corpus")
    var out = option(arguments, "out")
    if case_id.byte_length() == 0 or frame_name.byte_length() == 0 or corpus.byte_length() == 0 or out.byte_length() == 0:
        print(
            '{"status":"broken","message":"needs --case, --frame, --corpus and --out"}'
        )
        exit(BROKEN)
        return

    var source = corpus + "/" + frame_name + ".arrow"
    if not exists(source):
        # A missing corpus file is the harness's own job and not a result about
        # firepanda, so this is broken rather than raised. A harness that scored it as
        # a firepanda failure would report a missing file as a conformance regression.
        print(
            '{"status":"broken","message":'
            + quote("no corpus frame at " + source)
            + "}"
        )
        exit(BROKEN)
        return

    var frame: DataFrame
    try:
        frame = read_arrow(source)
    except error:
        # The file is there and firepanda would not read it, which is the opposite
        # case and is a result about firepanda. It is what happens today for a
        # dictionary encoded column and for a temporal type firepanda has no reader
        # for, and calling that a broken harness would hide a real gap behind a
        # message about this program. The case fails, and it should.
        print(
            '{"status":"raised","type":"Error","message":'
            + quote("cannot read " + source + ": " + String(error))
            + "}"
        )
        exit(OK)
        return

    # From here on, anything firepanda raises is a result about firepanda and is
    # reported as one. The harness decides what it means; this program only reports.
    try:
        if case_id == "basics/len":
            write_arrow(scalar_frame[DType.int64]("value", Int64(len(frame))), out)
            print('{"status":"ok","kind":"scalar"}')
        elif case_id == "basics/size":
            write_arrow(
                scalar_frame[DType.int64]("value", Int64(len(frame) * frame.width())),
                out,
            )
            print('{"status":"ok","kind":"scalar"}')
        elif case_id == "basics/ndim":
            write_arrow(scalar_frame[DType.int64]("value", Int64(2)), out)
            print('{"status":"ok","kind":"scalar"}')
        elif case_id == "basics/empty":
            write_arrow(
                scalar_frame[DType.bool](
                    "value", Scalar[DType.bool](len(frame) == 0 or frame.width() == 0)
                ),
                out,
            )
            print('{"status":"ok","kind":"scalar"}')
        elif case_id == "basics/head":
            var answer = frame.head()
            write_arrow(answer, out)
            print(
                '{"status":"ok","kind":"frame","index":0,"columns":'
                + names_json(answer)
                + "}"
            )
        elif case_id == "basics/head-n":
            var answer = frame.head(3)
            write_arrow(answer, out)
            print(
                '{"status":"ok","kind":"frame","index":0,"columns":'
                + names_json(answer)
                + "}"
            )
        elif case_id == "basics/tail":
            var answer = frame.tail()
            write_arrow(answer, out)
            print(
                '{"status":"ok","kind":"frame","index":0,"columns":'
                + names_json(answer)
                + "}"
            )
        elif case_id == "basics/tail-n":
            var answer = frame.tail(3)
            write_arrow(answer, out)
            print(
                '{"status":"ok","kind":"frame","index":0,"columns":'
                + names_json(answer)
                + "}"
            )
        elif case_id == "basics/copy":
            var answer = DataFrame(copy=frame)
            write_arrow(answer, out)
            print(
                '{"status":"ok","kind":"frame","index":0,"columns":'
                + names_json(answer)
                + "}"
            )
        elif case_id == "basics/column-select":
            var first = frame.names()[0]
            var series = List[Series]()
            series.append(frame.column(first).rename("__value__"))
            write_arrow(DataFrame.from_series(series^), out)
            print(
                '{"status":"ok","kind":"series","index":0,"name":'
                + quote(first)
                + "}"
            )
        elif case_id == "basics/sum":
            write_arrow(reduce(frame, "value", AggKind.SUM), out)
            print('{"status":"ok","kind":"scalar"}')
        elif case_id == "basics/mean":
            write_arrow(reduce(frame, "value", AggKind.MEAN), out)
            print('{"status":"ok","kind":"scalar"}')
        elif case_id == "basics/min":
            write_arrow(reduce(frame, "value", AggKind.MIN), out)
            print('{"status":"ok","kind":"scalar"}')
        elif case_id == "basics/max":
            write_arrow(reduce(frame, "value", AggKind.MAX), out)
            print('{"status":"ok","kind":"scalar"}')
        elif case_id == "basics/count":
            write_arrow(reduce(frame, "value", AggKind.COUNT), out)
            print('{"status":"ok","kind":"scalar"}')
        else:
            print('{"status":"absent"}')
    except error:
        # There is no exception type to report. Mojo's `Error` is one type carrying a
        # message, so the harness gets `Error` and the message, and every L4 case that
        # names a pandas exception type fails against this driver. That is the honest
        # reading: firepanda does not have those types yet, workstream J is where it
        # gets them, and a driver that guessed at a type name from the message text
        # would be inventing a result.
        print(
            '{"status":"raised","type":"Error","message":' + quote(String(error)) + "}"
        )
    exit(OK)
