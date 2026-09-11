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
    {"status": "ok", "kind": "index", "name": null}
    {"status": "ok", "kind": "tuple"}
    {"status": "absent"}
    {"status": "raised", "type": "KeyError", "message": "no column named q"}

`kind` is the vocabulary of `fpcompat.compare.Answer`, because the answer's shape is
part of the answer: a Series and a one column DataFrame are different results and a
suite that folded them together would pass a library that returned the wrong one.
`index` is the kind `DataFrame.columns` returns, an ordered set of labels that is not
a Series. `tuple` is what `shape` returns, and it travels as one row of as many
columns as the tuple has parts, each part read back as a scalar with its Arrow type,
for the same reason a scalar does not travel through JSON.
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
from firepanda.array.strings import StringArray, StringBuilder
from firepanda.array.value import Value
from firepanda.dtype import Field, LogicalType, Schema, TimeUnit, TimeZone
from firepanda.frame.frame import DataFrame, dt_isocalendar
from firepanda.frame.index import Index
from firepanda.frame.series import Series
from firepanda.frame.groupby import AggSpec
from firepanda.io import read_arrow, write_arrow
from firepanda.kernel import AggKind, BinaryOp
from firepanda.kernel.window import WindowEdge, WindowOp

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


def scalar_frame[
    dt: DType
](name: String, value: Scalar[dt]) raises -> DataFrame:
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


def reduce_pair(
    frame: DataFrame, left: String, right: String, kind: AggKind
) raises -> DataFrame:
    """Runs one two column reduction, the firepanda spelling.

    `Series.corr` and `Series.cov` in pandas are a method on one series taking
    another, and in firepanda they are an `AggSpec` naming two columns of the same
    frame handed to `agg`. Both cases in the suite take their two columns out of one
    frame, so nothing is lost in the translation.

    Args:
        frame: The frame.
        left: The first column.
        right: The second column.
        kind: Which pair reduction.

    Returns:
        A frame of one row and one column.

    Raises:
        Error: Whatever `agg` raises.
    """
    var specs = List[AggSpec]()
    specs.append(AggSpec(left, right, kind, "value"))
    return frame.agg(specs^)


def index_header(index: Index) -> String:
    """Renders the three header fields that describe a frame's row labels.

    A default index is reported as no index at all, which is what this driver has
    always sent and is why every case that passes today keeps passing. That is not a
    dodge: a pandas index that is a plain zero to n minus one range carries no
    information, the harness compares such a frame to one with no index, and sending
    the labels would be sending a column of nothing. Anything else is reported as one
    level with its labels written in front of the data.

    An unnamed level goes over as `null` rather than as any particular string, because
    the name of a level that has no name is not the driver's business to spell.

    Args:
        index: The row labels.

    Returns:
        The fields, without the surrounding braces.

    """
    if index.is_default():
        return '"index":0,"index_names":[],"default_index":true'
    var name = String("null")
    if index.name:
        name = quote(index.name.value())
    return '"index":1,"index_names":[' + name + '],"default_index":false'


def with_index(frame: DataFrame) raises -> DataFrame:
    """Puts a frame's row labels in front of its data columns.

    The harness reads the leading columns positionally and renames them, so the label
    this gives the column is never compared and only has to not collide with a real
    one. It is the name the harness uses for the same thing, which makes a file
    written by this driver readable by hand.

    Args:
        frame: The answer.

    Returns:
        The frame with one more column at the front, or the frame unchanged when its
        index is the default and there is nothing to say.

    Raises:
        Error: If the labels cannot be built or the columns cannot be copied.
    """
    if frame.index.is_default():
        return DataFrame(copy=frame)
    var series = List[Series]()
    series.append(Series("__index__0", frame.index.materialize()))
    var names = frame.names()
    for i in range(len(names)):
        series.append(frame.column(names[i]))
    return DataFrame.from_series(series^)


def emit_scalar(frame: DataFrame, path: String) raises:
    """Writes a one row one column frame as a scalar answer and prints its line.

    Args:
        frame: The one row frame carrying the value.
        path: Where to write it.

    Raises:
        Error: If it cannot be written.
    """
    write_arrow(frame, path)
    print('{"status":"ok","kind":"scalar"}')


def string_scalar_frame(name: String, value: String) raises -> DataFrame:
    """Wraps one string as a frame of one row and one column.

    The typed `scalar_frame` cannot do this, because text has no dtype in firepanda
    and a `StringArray` is not an `Array[dt]`. Two answers need it, the spelling of a
    dtype and the name of a column, and both of those are strings in pandas.

    Args:
        name: The column name.
        value: The value.

    Returns:
        The frame.

    Raises:
        Error: If the one column frame cannot be built, which would be a bug here.
    """
    var builder = StringBuilder(capacity=1)
    builder.append(value.as_bytes())
    var series = List[Series]()
    series.append(Series(name, builder^.finish()))
    return DataFrame.from_series(series^)


def tuple_frame(values: List[Int64]) raises -> DataFrame:
    """Wraps a tuple of integers as one row of as many columns as it has parts.

    `shape` is a tuple in pandas and a tuple is not a one row frame, so this is a
    transport shape rather than an answer shape. The harness turns each column back
    into a scalar part. Every part firepanda has to send today is an integer, which
    is why this takes a list rather than something more general: a tuple of mixed
    types would need a column per type and there is nothing to put in one.

    Args:
        values: The parts, in order.

    Returns:
        A frame of one row and `len(values)` columns.

    Raises:
        Error: If the frame cannot be built, which would be a bug here.
    """
    var series = List[Series]()
    for i in range(len(values)):
        var column = Array[DType.int64](1)
        column[0] = values[i]
        series.append(Series("part" + String(i), column^))
    return DataFrame.from_series(series^)


def emit_frame(frame: DataFrame, path: String) raises:
    """Writes a frame answer and prints its line.

    Args:
        frame: The answer.
        path: Where to write it.

    Raises:
        Error: If the frame cannot be written, which for an empty frame used to be
            most of them.
    """
    var header = index_header(frame.index)
    write_arrow(with_index(frame), path)
    print(
        '{"status":"ok","kind":"frame",'
        + header
        + ',"columns":'
        + names_json(frame)
        + "}"
    )


def scalar_case(case_id: String, name: String) -> Bool:
    """Says whether a case id is one of an operation's two scalar cases.

    Each of the seven operations has two of them, `basics/add-scalar` on the ten
    half null width frames and `basics/add-scalar-dense` on the ten dense ones.
    The driver runs identical code for both, so matching them together here is
    what keeps the chain seven branches long instead of fourteen.

    They are two cases in the corpus rather than one because pandas answers them
    differently, and for a reason that has nothing to do with the operation. A
    narrow numpy backed column with a null in it is a float64 before any
    arithmetic starts, so the half null family measures the read path and every
    one of its answers is a float64. The dense family is the only place the
    width rule itself is visible.

    Args:
        case_id: The case the runner asked for.
        name: The pandas method name, such as `add`.

    Returns:
        True if the id is either of that operation's two scalar cases.
    """
    var base = String("basics/", name, "-scalar")
    return case_id == base or case_id == base + "-dense"


def scalar_op(frame: DataFrame, op: BinaryOp, value: Int64) raises -> Series:
    """Applies one operation to the `value` column and one Python integer.

    The `weakened` is the whole point of the helper existing rather than the
    call being written out seven times. A Python `1` has no width and takes the
    column's, so `s + 1` on an int8 column is an int8 in pandas 3, and a
    constant that arrives at the kernel without that flag brings int64 with it
    and widens everything it touches. The Python binding weakens what it reads;
    a driver that did not would be measuring a call no user can make.

    Args:
        frame: The corpus frame, which has a `value` column in every width.
        op: The operation.
        value: The constant, always a small integer, because these cases are
            about what the width does and not about the number.

    Returns:
        The answer.

    Raises:
        Error: Whatever firepanda raises, which is a result about firepanda.
    """
    return frame.column("value").binary(Value(value).weakened(), op)


def dt_column(frame: DataFrame) raises -> String:
    """Which column of a temporal corpus frame the `dt` cases read.

    The cases are written as `df["us" if "us" in df else "second"]`, because the
    resolutions frame carries the same instants four times under the names of the
    four units and the range frame carries them once under `second`. This is that
    expression, and it is a function rather than a line repeated at every entry so
    that the two files cannot drift apart on which column was measured.

    Args:
        frame: The corpus frame.

    Returns:
        The column name.

    Raises:
        Error: Never, and it is declared because `names` is.
    """
    for name in frame.names():
        if name == "us":
            return "us"
    return "second"


def dt_column_name(frame: DataFrame) raises -> String:
    """Picks the datetime column the case means, the way the case picks it.

    The two cases that use this run over three differently shaped frames and
    name their column with a conditional on the pandas side rather than with a
    parameter, so this is that same conditional and not a lookup by type. Doing
    it by type would find the right column today and a different one the moment
    a frame gains a second datetime column, and then the two engines would be
    reducing different data and agreeing about it.

    Args:
        frame: The frame the case was handed.

    Returns:
        The name of the column to operate on.

    Raises:
        If the frame is none of the three the cases run over.
    """
    if frame.has("second"):
        return String("second")
    if frame.has("us"):
        return String("us")
    if frame.has("zoned"):
        return String("zoned")
    raise Error("no datetime column in this frame")


def first_instant(column: Series) raises -> Value:
    """Reads row zero of a timestamp series back out as a constant.

    This is `df["second"].iloc[0]`, which pandas hands back as a `Timestamp`. There
    is no `iloc` in firepanda yet, so the value is read off the column and rebuilt as
    a typed constant, and the resolution is read off the column with it. That last
    part is the whole reason this is a function: a constant built at the wrong
    resolution would still subtract, and the answer would come back in the wrong unit
    rather than come back wrong, which is the harder failure to notice.

    Args:
        column: The timestamp series.

    Returns:
        The constant standing for its first row.

    Raises:
        Error: If the column is empty or is not stored as int64, which for a
            timestamp column would be a bug in firepanda rather than here.
    """
    var type = column.logical()
    return Value.timestamp(
        column.values.as_typed_view[DType.int64]()[0],
        type.unit,
        TimeZone(copy=type.zone),
    )


def dt_field_name(case_id: String) -> String:
    """Turns a `temporal/` case id into the pandas `dt` name it asks for.

    The ids spell a field with hyphens and pandas spells it with underscores, so
    `temporal/days-in-month` is `days_in_month`. Anything under `temporal/` that is
    not one of the nineteen fields comes back empty, so that `temporal/strftime`
    and the rest are reported absent rather than reported as raising. Those two
    are scored the same and they read differently, and a case firepanda has no
    implementation for is an absence.

    Args:
        case_id: The case the runner asked for.

    Returns:
        The pandas field name, or an empty string.
    """
    if not case_id.startswith("temporal/"):
        return ""
    var tail = String(case_id[byte="temporal/".byte_length() :]).replace("-", "_")
    if (
        tail == "year"
        or tail == "month"
        or tail == "day"
        or tail == "hour"
        or tail == "minute"
        or tail == "second"
        or tail == "microsecond"
        or tail == "nanosecond"
        or tail == "dayofweek"
        or tail == "dayofyear"
        or tail == "quarter"
        or tail == "days_in_month"
        or tail == "is_leap_year"
        or tail == "is_month_start"
        or tail == "is_month_end"
        or tail == "is_quarter_start"
        or tail == "is_quarter_end"
        or tail == "is_year_start"
        or tail == "is_year_end"
    ):
        return tail
    return ""


def emit_series(name: String, var column: Series, path: String) raises:
    """Writes a series answer under its pandas name and prints its line.

    The written column is renamed, because the harness looks for one data column and
    not for a particular label, and the label travels in the line where it is compared
    as a label. A series whose name is right and whose values are wrong and one whose
    values are right and whose name is wrong are both failures and they are different
    failures.

    Args:
        name: The pandas name of the series, which is what gets compared.
        column: The values. Consumed.
        path: Where to write them.

    Raises:
        Error: If the series cannot be written.
    """
    var labels = Index(copy=column.index)
    var series = List[Series]()
    series.append(column^.rename("__value__"))
    var out = DataFrame.from_series(series^)
    # `from_series` gives the frame a fresh default index, since a list of columns
    # says nothing about what its rows are called. The labels being sent are the
    # series' own, so they are taken before the column is consumed and put back here.
    out.index = labels^
    var header = index_header(out.index)
    write_arrow(with_index(out), path)
    print(
        '{"status":"ok","kind":"series",'
        + header
        + ',"name":'
        + quote(name)
        + "}"
    )


def emit_series_unnamed(var column: Series, path: String) raises:
    """Writes a series answer whose pandas name is None.

    `groupby(...).size()` is the one that needs this. It returns a Series with no
    name, and a driver that sent the firepanda column label instead would fail the
    case for the wrong reason and then pass it for the wrong reason once somebody
    renamed the column.

    Args:
        column: The values. Consumed.
        path: Where to write them.

    Raises:
        Error: If the series cannot be written.
    """
    var labels = Index(copy=column.index)
    var series = List[Series]()
    series.append(column^.rename("__value__"))
    var out = DataFrame.from_series(series^)
    out.index = labels^
    var header = index_header(out.index)
    write_arrow(with_index(out), path)
    print('{"status":"ok","kind":"series",' + header + ',"name":null}')


def emit_index(var column: Series, path: String) raises:
    """Writes an index answer, which is what `DataFrame.columns` is.

    An index is not a series and not an array. pandas returns one from `columns`, from
    `unique` on some dtypes and from every index accessor, and a library that returned
    a Series where pandas returns an Index has returned the wrong thing even when the
    values match. The name is null on every index firepanda can produce today, since
    the only one is a column label list, which pandas leaves unnamed.

    Args:
        column: The labels. Consumed.
        path: Where to write them.

    Raises:
        Error: If it cannot be written.
    """
    var series = List[Series]()
    series.append(column^.rename("__value__"))
    write_arrow(DataFrame.from_series(series^), path)
    print('{"status":"ok","kind":"index","name":null}')


def labels_of(frame: DataFrame) raises -> Series:
    """Builds the column labels of a frame as a column of text.

    Args:
        frame: The frame.

    Returns:
        A one column series holding the names, in order.

    Raises:
        Error: If the string column cannot be built.
    """
    var names = frame.names()
    var builder = StringBuilder(capacity=len(names))
    for i in range(len(names)):
        builder.append(names[i].as_bytes())
    return Series("__value__", builder^.finish())


def one_key(name: String) -> List[String]:
    """Wraps a single key column name as the list `group_agg` takes.

    Args:
        name: The key column.

    Returns:
        A list of one name.
    """
    var keys = List[String]()
    keys.append(name)
    return keys^


def grouped(
    frame: DataFrame,
    keys: List[String],
    kind: AggKind,
    dropna: Bool = True,
    sort: Bool = True,
    as_index: Bool = False,
) raises -> DataFrame:
    """Runs one reduction per group over every column that is not a key.

    This is `df.groupby(keys).sum()` and its family, in the firepanda spelling.
    `as_index` asks for the pandas shape, where the key is the row labels rather
    than a leading column, and firepanda has that now.

    It is still a parameter rather than the only behaviour, for two reasons. The
    `flat-` cases beside every one of these ask pandas for `as_index=False` and
    exist to compare the arithmetic without the shape in the way, so they want the
    other form. And a two key grouping is a MultiIndex in pandas, which firepanda
    does not have, so asking for it there raises rather than handing back one of
    the two levels. Those cases stay flat and keep failing on the level count,
    which is the honest report: the answer really does have the wrong shape.

    Args:
        frame: The frame.
        keys: The key columns.
        kind: The reduction to apply to everything else.
        dropna: Whether a group whose key is null is dropped.
        sort: Whether the result comes back sorted by key.
        as_index: Whether the key comes back as the row labels.

    Returns:
        The aggregated frame.

    Raises:
        Error: Whatever `group_agg` raises.
    """
    return frame.group_agg(keys, kind, dropna, sort, as_index)


def rolled(frame: DataFrame, op: WindowOp) raises -> Series:
    """Runs one reduction over every five row window of the `value` column.

    This is `df["value"].rolling(5).sum()` and the four cases beside it, which
    ask for the plain form of each reduction with nothing but the width set. The
    width is five in all five of them, so it is written here once rather than at
    each arm.

    Args:
        frame: The corpus frame.
        op: The reduction.

    Returns:
        The windowed column.

    Raises:
        Error: Whatever `rolling` raises.
    """
    return frame.column("value").rolling(
        op, 5, None, False, WindowEdge.RIGHT, None
    )


def measured(frame: DataFrame, op: WindowOp) raises -> Series:
    """Runs one spread reduction over every eight row window of `value`.

    The three spread cases ask for a width of eight where the five plain ones ask
    for five, because a variance needs more than a handful of rows before the
    answer says anything, and eight is what the case list settled on. The degrees
    of freedom is one, which is what pandas defaults to and what the cases spell.

    Args:
        frame: The corpus frame.
        op: The reduction.

    Returns:
        The windowed column.

    Raises:
        Error: Whatever `rolling` raises.
    """
    return frame.column("value").rolling(
        op, 8, None, False, WindowEdge.RIGHT, None, 1
    )


def spread(frame: DataFrame, op: WindowOp) raises -> Series:
    """Runs one reduction over every window that starts at the first row.

    The `min_periods` is one rather than absent because the Mojo entry point
    takes a number here where the Python one takes an absence. That is not the
    driver choosing a default: one is what an expanding window defaults to in
    pandas, and it is the Python layer above this that turns the absence into it.

    Args:
        frame: The corpus frame.
        op: The reduction.

    Returns:
        The windowed column.

    Raises:
        Error: Whatever `expanding` raises.
    """
    return frame.column("value").expanding(op, 1)


def category_label(column: Series, at: Int) raises -> String:
    """Reads one of a category column's labels out, counting from either end.

    Four cases need a label of the column they are about to operate on, because
    the pandas spelling of them reads `cat.categories[0]` or `[-1]` rather than
    naming a string. Doing the same here keeps the two sides asking the same
    question of the same data instead of agreeing by hand written constant.

    Args:
        column: The category column.
        at: The position, and negative counts back from the end.

    Returns:
        The label.

    Raises:
        Error: If the column is not a category column, or the position is not
            in it.
    """
    var names = column.cat_categories()
    var where = at + len(names) if at < 0 else at
    return names.text(where)


def category_list(var labels: List[String]) raises -> StringArray:
    """Turns a list of words into the text column the category doors take.

    Args:
        labels: The labels, in the order to hold them. Consumed.

    Returns:
        The column.

    Raises:
        Error: If the column cannot be built.
    """
    var built = StringBuilder(capacity=len(labels))
    for label in labels:
        built.append(label.as_bytes())
    return built^.finish()


def emit_bool(value: Bool, path: String) raises:
    """Writes a scalar bool answer and prints its line.

    Args:
        value: The answer.
        path: Where to write it.

    Raises:
        Error: If it cannot be written.
    """
    write_arrow(
        scalar_frame[DType.bool]("value", Scalar[DType.bool](value)), path
    )
    print('{"status":"ok","kind":"scalar"}')


def main() raises:
    var args = argv()
    var arguments = List[String]()
    for i in range(len(args)):
        arguments.append(String(args[i]))

    var case_id = option(arguments, "case")
    var frame_name = option(arguments, "frame")
    var corpus = option(arguments, "corpus")
    var out = option(arguments, "out")
    if (
        case_id.byte_length() == 0
        or frame_name.byte_length() == 0
        or corpus.byte_length() == 0
        or out.byte_length() == 0
    ):
        print(
            '{"status":"broken","message":"needs --case, --frame, --corpus and'
            ' --out"}'
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
        # `widen_for_missing` is what turns the Arrow file into the frame a
        # pandas user would have been handed, which is the frame this suite is
        # about. pandas on the numpy backend has one missing value for a number
        # and it is NaN, so it widens an integer column with a missing row to
        # float64 at read time, and the oracle beside this program has already
        # had that done to it by `to_pandas`. Reading the file without it would
        # mean the two engines started from different data and every difference
        # after that would be attributed to the operation under test. See
        # firepanda #171 and document 20.
        frame = read_arrow(source).widen_for_missing()
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
            write_arrow(
                scalar_frame[DType.int64]("value", Int64(len(frame))), out
            )
            print('{"status":"ok","kind":"scalar"}')
        elif case_id == "basics/size":
            write_arrow(
                scalar_frame[DType.int64](
                    "value", Int64(len(frame) * frame.width())
                ),
                out,
            )
            print('{"status":"ok","kind":"scalar"}')
        elif case_id == "basics/ndim":
            write_arrow(scalar_frame[DType.int64]("value", Int64(2)), out)
            print('{"status":"ok","kind":"scalar"}')
        elif case_id == "basics/empty":
            write_arrow(
                scalar_frame[DType.bool](
                    "value",
                    Scalar[DType.bool](len(frame) == 0 or frame.width() == 0),
                ),
                out,
            )
            print('{"status":"ok","kind":"scalar"}')
        elif case_id == "basics/head":
            emit_frame(frame.head(), out)
        elif case_id == "basics/head-n":
            emit_frame(frame.head(3), out)
        elif case_id == "basics/tail":
            emit_frame(frame.tail(), out)
        elif case_id == "basics/tail-n":
            emit_frame(frame.tail(3), out)
        elif case_id == "basics/copy":
            emit_frame(DataFrame(copy=frame), out)
        elif case_id == "basics/column-select":
            var first = frame.names()[0]
            emit_series(first, frame.column(first), out)
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
        elif case_id == "basics/median":
            write_arrow(reduce(frame, "value", AggKind.MEDIAN), out)
            print('{"status":"ok","kind":"scalar"}')
        elif case_id == "basics/std":
            write_arrow(reduce(frame, "value", AggKind.STD), out)
            print('{"status":"ok","kind":"scalar"}')
        elif case_id == "basics/var":
            write_arrow(reduce(frame, "value", AggKind.VAR), out)
            print('{"status":"ok","kind":"scalar"}')
        elif case_id == "basics/nunique":
            # The first column and not "value", because the case reduces
            # `df.iloc[:, 0]` and the frames it runs on are the key frames.
            write_arrow(reduce(frame, frame.names()[0], AggKind.NUNIQUE), out)
            print('{"status":"ok","kind":"scalar"}')

        # Shape and labels. `shape` is a tuple in pandas rather than a frame, and
        # `columns` is an Index rather than a Series, and both of those distinctions
        # are answers rather than packaging.
        elif case_id == "basics/shape":
            write_arrow(
                tuple_frame([Int64(len(frame)), Int64(frame.width())]), out
            )
            print('{"status":"ok","kind":"tuple"}')
        elif case_id == "basics/series-shape":
            write_arrow(tuple_frame([Int64(len(frame))]), out)
            print('{"status":"ok","kind":"tuple"}')
        elif case_id == "basics/columns":
            emit_index(labels_of(frame), out)
        elif case_id == "basics/series-name":
            write_arrow(string_scalar_frame("value", frame.names()[0]), out)
            print('{"status":"ok","kind":"scalar"}')
        elif case_id == "basics/series-dtype":
            # firepanda's spelling of its own dtype, not a translation into
            # pandas'. Where the two differ that is the answer and not a bug here.
            write_arrow(
                string_scalar_frame(
                    "value", String(frame.column("value").dtype())
                ),
                out,
            )
            print('{"status":"ok","kind":"scalar"}')

        # Selecting and reshaping columns.
        elif case_id == "basics/column-list":
            var names = frame.names()
            emit_frame(frame.select([names[1], names[0]]), out)
        elif case_id == "basics/drop-column":
            emit_frame(frame.drop([frame.names()[0]]), out)
        elif case_id == "basics/rename":
            emit_frame(frame.rename(frame.names()[0], "renamed"), out)
        elif case_id == "basics/boolean-mask":
            emit_frame(
                frame.filter(frame.column("flag").as_typed[DType.bool]()),
                out,
            )
        elif case_id == "basics/head-negative":
            emit_frame(frame.head(-2), out)

        # Nulls.
        elif case_id == "basics/isna":
            emit_series(
                "value", Series("value", frame.column("value").is_null()), out
            )
        elif case_id == "basics/notna":
            emit_series(
                "value",
                Series("value", frame.column("value").is_not_null()),
                out,
            )
        elif case_id == "basics/dropna":
            emit_series("value", frame.column("value").drop_nulls(), out)
        elif case_id == "basics/frame-dropna":
            emit_frame(frame.drop_nulls(), out)
        elif case_id == "basics/ffill":
            emit_series("value", frame.column("value").fill_forward(), out)
        elif case_id == "basics/bfill":
            emit_series("value", frame.column("value").fill_backward(), out)

        # Moving a column along its own rows. The type is as much of the answer
        # as the values here: a shift that opens a gap in an integer column
        # answers float64 because pandas has no integer that means absent, and a
        # shift of nothing or a shift with a fill value opens no gap and stays
        # integer. All three go through the same call, so the three entries below
        # are one rule measured from three sides.
        elif case_id == "basics/shift":
            emit_series("value", frame.column("value").shift(), out)
        elif case_id == "basics/shift-negative":
            emit_series("value", frame.column("value").shift(-2), out)
        elif case_id == "basics/shift-fill":
            emit_series(
                "value",
                frame.column("value").shift(1, Value(Int64(0)).weakened()),
                out,
            )
        elif case_id == "basics/diff":
            emit_series("value", frame.column("value").diff(), out)
        elif case_id == "basics/pct-change":
            emit_series("value", frame.column("value").pct_change(), out)
        elif case_id == "basics/alignment-subtract-shifted":
            # Written out the long way rather than called as `diff`, because the
            # case is about the subtraction finding the same labels on both sides
            # and not about the difference it arrives at.
            var lagged = frame.column("value").shift(1)
            emit_series("value", frame.column("value") - lagged, out)

        # Running folds. These are the shift's opposite number on the missing
        # question: a shift makes a gap where there was none and has to widen for
        # it, while a scan is missing in exactly the rows its input was and has
        # nothing to widen. The four are here separately because the type rule is
        # not one rule: a running total widens a narrow integer column and a
        # running extreme does not.
        elif case_id == "basics/cumsum":
            emit_series("value", frame.column("value").cumsum(), out)
        elif case_id == "basics/cumprod":
            emit_series("value", frame.column("value").cumprod(), out)
        elif case_id == "basics/cummax":
            emit_series("value", frame.column("value").cummax(), out)
        elif case_id == "basics/cummin":
            emit_series("value", frame.column("value").cummin(), out)

        # Ordering. firepanda has no default for the null position and pandas'
        # default is last, so that default is written out here rather than left to
        # be guessed. Where firepanda puts a null when it is asked to put it last is
        # the thing being measured.
        #
        # The first two name a second key, and that is not firepanda needing the
        # help. pandas defaults to an unstable kind, so a sort on a column with
        # ties returns a permutation pandas does not promise and does not
        # reproduce, and a case comparing one of those is measuring numpy's
        # introsort rather than either library. Document 23.
        elif case_id == "basics/sort-values":
            emit_frame(
                frame.sort_values(
                    ["key", "value"], [False, False], [False, False]
                ),
                out,
            )
        elif case_id == "basics/sort-values-descending":
            emit_frame(
                frame.sort_values(
                    ["key", "value"], [True, True], [False, False]
                ),
                out,
            )
        # The one case where pandas does promise the tie order, because it was
        # asked to. firepanda has no kind argument and never needed one, since
        # its sort is a stable merge sort and stable is the only answer it can
        # give, so this case passes by construction and is here to say so.
        elif case_id == "basics/sort-values-stable":
            emit_frame(frame.sort_values(["key"], [False], [False]), out)
        # The first two columns, taken from the frame rather than named, because
        # the null bearing column is the key in one of these frames and the
        # value in the other, which is what the case is about.
        elif case_id == "basics/sort-values-na-first":
            var na_first_keys = frame.names()
            emit_frame(
                frame.sort_values(
                    [na_first_keys[0], na_first_keys[1]],
                    [False, False],
                    [True, True],
                ),
                out,
            )
        elif case_id == "basics/sort-two-columns":
            emit_frame(
                frame.sort_values(
                    ["left", "right"], [False, True], [False, False]
                ),
                out,
            )

        # Casts.
        elif case_id == "basics/astype-float":
            emit_series("value", frame.column("value").cast(DType.float64), out)
        elif case_id == "basics/astype-narrow":
            emit_series("value", frame.column("value").cast(DType.int8), out)
        elif case_id == "basics/astype-string":
            emit_series(
                "value", frame.column("value").cast(LogicalType.STRING), out
            )
        # The target is read off the column rather than named, because the case
        # runs on an integer frame and a float one and the whole question is
        # whether the column that comes back is the column that went out.
        elif case_id == "basics/astype-round-trip":
            var round_trip = frame.column("value")
            var as_text = round_trip.cast(LogicalType.STRING)
            if round_trip.values.type.is_float():
                emit_series("value", as_text.cast(DType.float64), out)
            else:
                emit_series("value", as_text.cast(DType.int64), out)

        # Arithmetic against a constant, on all ten widths, twice. The width is
        # the point of these rather than the arithmetic: `s + 1` answers int8 on
        # an int8 column in pandas 3, because a Python integer has no width of
        # its own and takes the column's, and a library that widens to int64
        # here doubles a frame on the first expression anybody writes and keeps
        # it doubled. Every constant is weakened, because that is what the
        # Python binding does to a Python scalar and a driver that skipped it
        # would be measuring a call no user can make.
        #
        # The dense half and the half null half of each pair run the same code,
        # which is why `scalar_case` takes both ids. They are separate cases in
        # the corpus rather than separate work here, because pandas answers them
        # differently for a reason that is nothing to do with the operation.
        elif scalar_case(case_id, "add"):
            emit_series("value", scalar_op(frame, BinaryOp.ADD, Int64(1)), out)
        elif scalar_case(case_id, "sub"):
            emit_series("value", scalar_op(frame, BinaryOp.SUB, Int64(1)), out)
        elif scalar_case(case_id, "mul"):
            emit_series("value", scalar_op(frame, BinaryOp.MUL, Int64(2)), out)
        elif scalar_case(case_id, "truediv"):
            emit_series("value", scalar_op(frame, BinaryOp.DIV, Int64(2)), out)
        elif scalar_case(case_id, "floordiv"):
            emit_series(
                "value", scalar_op(frame, BinaryOp.FLOORDIV, Int64(2)), out
            )
        elif scalar_case(case_id, "mod"):
            emit_series("value", scalar_op(frame, BinaryOp.MOD, Int64(3)), out)
        elif scalar_case(case_id, "pow"):
            emit_series("value", scalar_op(frame, BinaryOp.POW, Int64(2)), out)
        # The three bool answers. The other four raise, in both engines, and a
        # case that raises is an L4 case and belongs in the errors section.
        elif case_id == "basics/bool-add-column":
            emit_series(
                "flag",
                frame.column("flag").binary(frame.column("flag"), BinaryOp.ADD),
                out,
            )
        elif case_id == "basics/bool-mul-column":
            emit_series(
                "flag",
                frame.column("flag").binary(frame.column("flag"), BinaryOp.MUL),
                out,
            )
        elif case_id == "basics/bool-mod-column":
            emit_series(
                "flag",
                frame.column("flag").binary(frame.column("flag"), BinaryOp.MOD),
                out,
            )
        elif case_id == "basics/bool-add-scalar":
            emit_series(
                "flag",
                frame.column("flag").binary(
                    Value(True).weakened(), BinaryOp.ADD
                ),
                out,
            )
        elif case_id == "basics/add-edges":
            emit_series(
                "int8",
                frame.column("int8").binary(
                    Value(Int64(1)).weakened(), BinaryOp.ADD
                ),
                out,
            )
        elif case_id == "basics/div-by-zero":
            emit_series(
                "value",
                frame.column("value").binary(
                    Value(Int64(0)).weakened(), BinaryOp.DIV
                ),
                out,
            )
        elif case_id == "basics/column-arithmetic":
            # Unnamed, because pandas gives the product of two differently named
            # columns a name of None. firepanda gives it the empty string, which
            # is the registered name divergence and is a separate thing from the
            # values being right, so sending the label would fail this case for
            # a reason it is not about.
            var names = frame.names()
            emit_series_unnamed(
                frame.column(names[0]).binary(
                    frame.column(names[1]), BinaryOp.MUL
                ),
                out,
            )

        # Grouped aggregations. These used to produce a frame whose keys were
        # columns where the pandas answer has them in the index, so every one of them
        # failed on the shape and would have failed even if every number in it were
        # right. firepanda has the indexed shape now, so they ask for it and are
        # compared on their values like anything else. The two key cases are the
        # exception and stay flat, because two keys are a MultiIndex in pandas and
        # firepanda has none, so those still fail on the level count, which is the
        # truthful report rather than a gap being papered over.
        elif case_id == "groupby/sum":
            emit_frame(
                grouped(frame, one_key("key"), AggKind.SUM, True, True, True),
                out,
            )
        elif case_id == "groupby/mean":
            emit_frame(
                grouped(frame, one_key("key"), AggKind.MEAN, True, True, True),
                out,
            )
        elif case_id == "groupby/min":
            emit_frame(
                grouped(frame, one_key("key"), AggKind.MIN, True, True, True),
                out,
            )
        elif case_id == "groupby/max":
            emit_frame(
                grouped(frame, one_key("key"), AggKind.MAX, True, True, True),
                out,
            )
        elif case_id == "groupby/count":
            emit_frame(
                grouped(frame, one_key("key"), AggKind.COUNT, True, True, True),
                out,
            )
        elif case_id == "groupby/first":
            emit_frame(
                grouped(frame, one_key("key"), AggKind.FIRST, True, True, True),
                out,
            )
        elif case_id == "groupby/last":
            emit_frame(
                grouped(frame, one_key("key"), AggKind.LAST, True, True, True),
                out,
            )
        elif case_id == "groupby/median":
            emit_frame(
                grouped(
                    frame, one_key("key"), AggKind.MEDIAN, True, True, True
                ),
                out,
            )
        elif case_id == "groupby/nunique":
            emit_frame(
                grouped(
                    frame, one_key("key"), AggKind.NUNIQUE, True, True, True
                ),
                out,
            )
        elif case_id == "groupby/std":
            emit_frame(
                grouped(frame, one_key("key"), AggKind.STD, True, True, True),
                out,
            )
        elif case_id == "groupby/var":
            emit_frame(
                grouped(frame, one_key("key"), AggKind.VAR, True, True, True),
                out,
            )
        elif case_id == "groupby/sem":
            emit_frame(
                grouped(frame, one_key("key"), AggKind.SEM, True, True, True),
                out,
            )
        elif case_id == "groupby/skew":
            emit_frame(
                grouped(frame, one_key("key"), AggKind.SKEW, True, True, True),
                out,
            )
        elif case_id == "groupby/size":
            # pandas returns an unnamed Series here, one row per group, indexed by
            # the key, so the size column travels alone and carries the labels.
            emit_series_unnamed(
                frame.group_count(one_key("key"), True, True, True).column(
                    "size"
                ),
                out,
            )
        # The same reductions asked for with the keys left as columns, which is the
        # shape firepanda already produces, so these are the ones that say whether
        # the numbers are right. An indexed answer fails on the shape and the
        # comparison stops before it reads a single value, so without these the
        # section could not tell a correct grouped sum from a wrong one.
        elif case_id == "groupby/flat-sum":
            emit_frame(grouped(frame, one_key("key"), AggKind.SUM), out)
        elif case_id == "groupby/flat-mean":
            emit_frame(grouped(frame, one_key("key"), AggKind.MEAN), out)
        elif case_id == "groupby/flat-min":
            emit_frame(grouped(frame, one_key("key"), AggKind.MIN), out)
        elif case_id == "groupby/flat-max":
            emit_frame(grouped(frame, one_key("key"), AggKind.MAX), out)
        elif case_id == "groupby/flat-count":
            emit_frame(grouped(frame, one_key("key"), AggKind.COUNT), out)
        elif case_id == "groupby/flat-first":
            emit_frame(grouped(frame, one_key("key"), AggKind.FIRST), out)
        elif case_id == "groupby/flat-last":
            emit_frame(grouped(frame, one_key("key"), AggKind.LAST), out)
        elif case_id == "groupby/flat-median":
            emit_frame(grouped(frame, one_key("key"), AggKind.MEDIAN), out)
        elif case_id == "groupby/flat-nunique":
            emit_frame(grouped(frame, one_key("key"), AggKind.NUNIQUE), out)
        elif case_id == "groupby/flat-std":
            emit_frame(grouped(frame, one_key("key"), AggKind.STD), out)
        elif case_id == "groupby/flat-var":
            emit_frame(grouped(frame, one_key("key"), AggKind.VAR), out)
        elif case_id == "groupby/flat-sem":
            emit_frame(grouped(frame, one_key("key"), AggKind.SEM), out)
        elif case_id == "groupby/flat-skew":
            emit_frame(grouped(frame, one_key("key"), AggKind.SKEW), out)
        elif case_id == "groupby/flat-dropna-false":
            emit_frame(
                grouped(frame, one_key("key"), AggKind.SUM, False, True), out
            )
        elif case_id == "groupby/flat-sort-false":
            emit_frame(
                grouped(frame, one_key("key"), AggKind.SUM, True, False), out
            )
        elif case_id == "groupby/flat-two-keys":
            var flat_pair = List[String]()
            flat_pair.append("left")
            flat_pair.append("right")
            emit_frame(grouped(frame, flat_pair, AggKind.SUM), out)
        elif case_id == "groupby/two-keys":
            var pair = List[String]()
            pair.append("left")
            pair.append("right")
            emit_frame(grouped(frame, pair, AggKind.SUM), out)
        elif case_id == "groupby/two-keys-size":
            var pair_size = List[String]()
            pair_size.append("left")
            pair_size.append("right")
            emit_series_unnamed(
                frame.group_count(pair_size).column("size"), out
            )
        elif case_id == "groupby/series":
            emit_series(
                "value",
                grouped(
                    frame, one_key("key"), AggKind.SUM, True, True, True
                ).column("value"),
                out,
            )
        elif case_id == "groupby/sort-false":
            emit_frame(
                grouped(frame, one_key("key"), AggKind.SUM, True, False, True),
                out,
            )
        elif case_id == "groupby/dropna-false":
            emit_frame(
                grouped(frame, one_key("key"), AggKind.SUM, False, True, True),
                out,
            )
        elif case_id == "groupby/on-tall":
            emit_series(
                "value",
                grouped(
                    frame, one_key("key"), AggKind.MEAN, True, True, True
                ).column("value"),
                out,
            )
        elif case_id == "groupby/bool-key":
            emit_series(
                "value",
                grouped(
                    frame, one_key("flag"), AggKind.SUM, True, True, True
                ).column("value"),
                out,
            )
        elif case_id == "groupby/as-index-false":
            # The one aggregation in the section whose pandas answer has a default
            # index, because `as_index=False` asks for the keys as columns. It is
            # therefore the only one that can pass, and what it passes on is the
            # arithmetic rather than the shape.
            emit_frame(grouped(frame, one_key("key"), AggKind.SUM), out)
        elif case_id == "groupby/ngroups":
            write_arrow(
                scalar_frame[DType.int64](
                    "value",
                    Int64(len(frame.group_by(one_key("key"), List[AggSpec]()))),
                ),
                out,
            )
            print('{"status":"ok","kind":"scalar"}')
        # The dt accessor. Every one of these reads a column of instants and a
        # count of them per second and nothing else, so the nineteen fields are one
        # entry rather than nineteen. The name the answer travels under is the
        # column's own, because `df["us"].dt.year` is a Series called `us` in
        # pandas and a series whose values are right and whose name is wrong is a
        # failure the suite is supposed to see.
        elif dt_field_name(case_id) != "":
            var stamps = dt_column(frame)
            emit_series(
                stamps, frame.column(stamps).dt(dt_field_name(case_id)), out
            )
        elif case_id == "temporal/date":
            var stamps = dt_column(frame)
            emit_series(stamps, frame.column(stamps).dt_date(), out)
        elif case_id == "temporal/normalize":
            var stamps = dt_column(frame)
            emit_series(stamps, frame.column(stamps).dt_normalize(), out)
        elif case_id == "temporal/day-name":
            var stamps = dt_column(frame)
            emit_series(stamps, frame.column(stamps).dt_day_name(), out)
        elif case_id == "temporal/month-name":
            var stamps = dt_column(frame)
            emit_series(stamps, frame.column(stamps).dt_month_name(), out)
        elif case_id == "temporal/strftime":
            var stamps = dt_column(frame)
            emit_series(
                stamps,
                frame.column(stamps).dt_strftime("%Y-%m-%dT%H:%M:%S"),
                out,
            )
        elif case_id == "temporal/to-datetime-strings":
            # A round trip. The column is rendered to text and read back, and
            # the answer has to be the column it started as. This is the only
            # case on the board that exercises the reading half of the calendar
            # against a whole frame of instants rather than a handful of
            # literals, and the two halves share their format machinery, so a
            # directive that one of them reads differently from how the other
            # writes it shows up here as a column of wrong answers.
            var stamps = dt_column(frame)
            emit_series(
                stamps,
                frame.column(stamps)
                .dt_strftime("%Y-%m-%d %H:%M:%S")
                .to_datetime(),
                out,
            )
        elif case_id == "temporal/to-datetime-format":
            # The same round trip with the format said out loud, in an order
            # firepanda's guesser will not touch. Day first is exactly the shape
            # that cannot be guessed safely, since `01/02/2026` is a real date
            # under both readings and nothing later catches the wrong one.
            var stamps = dt_column(frame)
            emit_series(
                stamps,
                frame.column(stamps)
                .dt_strftime("%d/%m/%Y")
                .to_datetime("%d/%m/%Y"),
                out,
            )
        elif case_id == "temporal/isocalendar":
            # The one member of the accessor that answers a frame, which is why
            # it is a free function in firepanda and an `emit_frame` here. The
            # three column names travel with it and pandas' are `year`, `week`
            # and `day`.
            var stamps = dt_column(frame)
            emit_frame(dt_isocalendar(frame.column(stamps)), out)
        elif case_id == "temporal/floor":
            var stamps = dt_column(frame)
            emit_series(stamps, frame.column(stamps).dt_floor("h"), out)
        elif case_id == "temporal/ceil":
            var stamps = dt_column(frame)
            emit_series(stamps, frame.column(stamps).dt_ceil("h"), out)
        elif case_id == "temporal/round":
            var stamps = dt_column(frame)
            emit_series(stamps, frame.column(stamps).dt_round("h"), out)
        elif case_id == "temporal/round-minute":
            emit_series("second", frame.column("second").dt_round("min"), out)
        elif case_id == "temporal/as-unit":
            emit_series("ns", frame.column("ns").dt_as_unit("s"), out)
        elif case_id == "temporal/as-unit-up":
            emit_series("s", frame.column("s").dt_as_unit("ns"), out)
        elif case_id.startswith("temporal/nanosecond-"):
            # Unlike the nineteen above, these name their column in the id, because
            # the whole point of the family is that the same instant answers zero at
            # three resolutions and a real number at the fourth.
            var stamps = String(case_id[byte="temporal/nanosecond-".byte_length() :])
            emit_series(stamps, frame.column(stamps).dt("nanosecond"), out)
        elif case_id.startswith("temporal/unit-"):
            var stamps = String(case_id[byte="temporal/unit-".byte_length() :])
            write_arrow(
                string_scalar_frame(
                    "value", String(frame.column(stamps).logical().unit)
                ),
                out,
            )
            print('{"status":"ok","kind":"scalar"}')
        elif case_id.startswith("temporal/dtype-"):
            # The logical type and not the physical one. A timestamp column is
            # int64 underneath and answering `int64` here would be a true statement
            # about the buffer and a wrong answer to the question.
            var stamps = String(case_id[byte="temporal/dtype-".byte_length() :])
            write_arrow(
                string_scalar_frame(
                    "value", String(frame.column(stamps).logical())
                ),
                out,
            )
            print('{"status":"ok","kind":"scalar"}')
        # Elapsed times. A duration column is the answer to `s - s`, so almost
        # nothing here is reachable until subtracting two instants is, which is why
        # the ten arrived together rather than one at a time.
        elif case_id == "temporal/duration-dtype":
            emit_scalar(
                string_scalar_frame(
                    "value", String(frame.column("value").logical())
                ),
                out,
            )
        elif case_id == "temporal/total-seconds":
            emit_series("value", frame.column("value").dt_total_seconds(), out)
        elif case_id == "temporal/duration-days":
            emit_series("value", frame.column("value").dt_days(), out)
        elif case_id == "temporal/duration-sum":
            emit_scalar(reduce(frame, "value", AggKind.SUM), out)
        elif case_id == "temporal/duration-mean":
            emit_scalar(reduce(frame, "value", AggKind.MEAN), out)
        elif case_id == "temporal/duration-abs":
            emit_series("value", frame.column("value").abs(), out)
        elif case_id == "temporal/dst-difference":
            # The zoned column counts instants and not wall clocks, so the gap
            # across a transition comes out as the hour that was really there
            # rather than the hour the clock face suggests. Nothing in the call
            # says so; it falls out of the column keeping its zone through the
            # shift and the subtraction.
            emit_series("zoned", frame.column("zoned").diff(), out)
        elif case_id == "temporal/timestamp-minus-timestamp":
            var stamps = frame.column("second")
            emit_series(
                "second",
                stamps.binary(first_instant(stamps), BinaryOp.SUB),
                out,
            )
        elif case_id == "temporal/timestamp-plus-duration":
            # `pd.Timedelta(hours=1)` is a microsecond constant, so this answers a
            # microsecond column even though the column it was added to counts
            # seconds. That is pandas' rule and the next case is the same amount of
            # time spelled the other way, where it is not.
            emit_series(
                "second",
                frame.column("second").binary(
                    Value.duration(3_600_000_000, TimeUnit.MICRO), BinaryOp.ADD
                ),
                out,
            )
        elif case_id == "temporal/timedelta-construct":
            # `pd.Timedelta(90, unit='s')` is a second constant, so the answer stays
            # a second column. Same accessor as the case above, same kind of amount,
            # different answer type, and the difference is the spelling.
            emit_series(
                "second",
                frame.column("second").binary(
                    Value.duration(90, TimeUnit.SECOND), BinaryOp.ADD
                ),
                out,
            )
        elif case_id == "temporal/to-timedelta":
            emit_series("value", frame.column("value").to_timedelta("s"), out)
        # Two cases that needed no new operation and no new case, only for the
        # sort and the reduction to stop handing back the integer count they had
        # been reducing. Both had been reported absent since the temporal frames
        # first became readable, which made them look like missing features
        # rather than missing labels.
        elif case_id == "temporal/sort-timestamps":
            var instants = dt_column_name(frame)
            emit_series(
                instants, frame.column(instants).sort_values(), out
            )
        elif case_id == "temporal/max":
            emit_scalar(reduce(frame, dt_column_name(frame), AggKind.MAX), out)
        # The zone entries that need no zone database, which is more of them than
        # it sounds. Converting keeps the instant and changes the name it is read
        # against, so it never asks what the offset is and works for every zone
        # there is. Localising to UTC needs the offset and UTC's is zero. What is
        # missing is the rest of `tz_localize`, where the offset is a rule.
        elif case_id == "temporal/tz":
            emit_scalar(
                string_scalar_frame("value", frame.column("zoned").dt_tz()),
                out,
            )
        elif case_id == "temporal/tz-convert-utc":
            emit_series(
                "zoned", frame.column("zoned").dt_tz_convert("UTC"), out
            )
        elif case_id == "temporal/tz-convert-half-hour":
            emit_series(
                "zoned",
                frame.column("zoned").dt_tz_convert("Asia/Kolkata"),
                out,
            )
        elif case_id == "temporal/tz-convert-hour":
            emit_series(
                "zoned",
                frame.column("zoned").dt_tz_convert("UTC").dt("hour"),
                out,
            )
        elif case_id == "temporal/tz-localize":
            emit_series(
                "second", frame.column("second").dt_tz_localize("UTC"), out
            )
        # The stats section. Almost every case here answers with a scalar, which is
        # the one answer shape that does not go through an index, so this is the
        # section where firepanda's arithmetic can be compared to pandas without
        # #154 standing in front of every result. `kurt`, `mode`, `prod`, `rank`,
        # `nlargest`, `interpolate`, `searchsorted`, `cumsum` and `describe` have no
        # entry because firepanda has no such operation, and the four non linear
        # `interpolation` settings on `quantile` have none because firepanda's
        # quantile has no such parameter. Those are absences and the suite should
        # say so rather than have this file guess an answer.
        # The two divergence cases that assert the corrected moments. Both sides
        # ignore the frame they were handed and build the same five values, because
        # the divergence needs a column whose mean cannot be represented exactly and
        # no corpus frame is shaped that way. Two to the fifty second is where the
        # gap between neighbouring float64 values is exactly one, so every input is
        # exact and the rounded centre is the only thing either engine can get wrong.
        elif case_id.startswith("divergences/moment-precision/"):
            var base = 4503599627370496.0
            var shifted = Array[DType.float64](5)
            shifted[0] = base + 1.0
            shifted[1] = base + 2.0
            shifted[2] = base + 4.0
            shifted[3] = base + 8.0
            shifted[4] = base + 16.0
            var wide = List[Series]()
            wide.append(Series("value", shifted^))
            var built = DataFrame.from_series(wide^)
            var moment = AggKind.SKEW if case_id.endswith(
                "/skew"
            ) else AggKind.VAR
            emit_scalar(reduce(built, "value", moment), out)

        # The integer zero divisor. These build their own two columns for the
        # reason the moment cases do, which is that no corpus frame guarantees a
        # zero in the divisor, and the divergence is about nothing else. The
        # numerator is the same three rows every time so that the five answers can
        # be read against each other, and the divisor has two rows that divide
        # perfectly well in it, which is what shows that pandas charges the whole
        # column for one bad row.
        elif case_id.startswith("divergences/zero-divisor/"):
            var top = Array[DType.int64](3)
            top[0] = 7
            top[1] = -7
            top[2] = 0
            var numerator = Series("value", top^)
            var op = BinaryOp.DIV
            if case_id.startswith("divergences/zero-divisor/floordiv"):
                op = BinaryOp.FLOORDIV
            elif case_id.startswith("divergences/zero-divisor/mod"):
                op = BinaryOp.MOD
            if case_id.endswith("-scalar"):
                # Weakened, because the pandas side divides by a Python `0`, which
                # has no width of its own and takes the column's.
                emit_series(
                    "value",
                    numerator.binary(Value(Int64(0)).weakened(), op),
                    out,
                )
            else:
                var bottom = Array[DType.int64](3)
                bottom[0] = 3
                bottom[1] = 0
                bottom[2] = 2
                var divisor = Series("value", bottom^)
                emit_series("value", numerator.binary(divisor, op), out)
        elif case_id == "stats/std":
            emit_scalar(reduce(frame, "value", AggKind.STD), out)
        elif case_id == "stats/var":
            emit_scalar(reduce(frame, "value", AggKind.VAR), out)
        elif case_id == "stats/sem":
            emit_scalar(reduce(frame, "value", AggKind.SEM), out)
        elif case_id == "stats/skew":
            emit_scalar(reduce(frame, "value", AggKind.SKEW), out)
        elif case_id == "stats/std-ddof-zero":
            emit_scalar(reduce(frame, "value", AggKind.std_with(0)), out)
        elif case_id == "stats/var-ddof-zero":
            emit_scalar(reduce(frame, "value", AggKind.var_with(0)), out)
        elif case_id == "stats/std-single-row":
            # The `single` frame has no column called `value`, and the case asks for
            # `b`, which is its float one.
            emit_scalar(reduce(frame, "b", AggKind.STD), out)
        elif case_id == "stats/median":
            emit_scalar(reduce(frame, "value", AggKind.MEDIAN), out)
        elif case_id == "stats/quantile-linear":
            emit_scalar(reduce(frame, "value", AggKind.quantile_at(0.25)), out)
        elif case_id == "stats/quantile-nulls":
            emit_scalar(reduce(frame, "value", AggKind.quantile_at(0.5)), out)
        elif case_id == "stats/corr-pearson" or case_id == "stats/cov":
            # The case takes the last two columns of the frame by position, and
            # names them in that order: the last one is the series the method is
            # called on and the one before it is the argument.
            var labels = frame.names()
            var kind = (
                AggKind.CORR if case_id == "stats/corr-pearson" else AggKind.COV
            )
            emit_scalar(
                reduce_pair(
                    frame,
                    labels[len(labels) - 1],
                    labels[len(labels) - 2],
                    kind,
                ),
                out,
            )
        elif case_id == "stats/corr-with-nulls":
            emit_scalar(reduce_pair(frame, "value", "row", AggKind.CORR), out)
        elif case_id == "stats/monotonic":
            # The case reads the first column by position rather than by name,
            # because the three frames it runs on do not agree on one.
            emit_scalar(
                scalar_frame[DType.bool](
                    "value",
                    Scalar[DType.bool](
                        frame.column(frame.names()[0]).is_monotonic_increasing()
                    ),
                ),
                out,
            )
        elif case_id == "stats/hasnans":
            emit_scalar(
                scalar_frame[DType.bool](
                    "value",
                    Scalar[DType.bool](frame.column("value").null_count() > 0),
                ),
                out,
            )
        elif case_id == "stats/argsort":
            emit_series(
                "value",
                Series("value", frame.column("value").argsort()),
                out,
            )
        elif case_id == "stats/argsort-ties":
            # The key column rather than the value one, because ten distinct keys
            # over ten thousand rows means nearly every position in the answer is
            # the tie break and not a comparison.
            emit_series(
                "key",
                Series("key", frame.column("key").argsort()),
                out,
            )
        elif case_id == "stats/cumsum-tall":
            # Ten thousand rows, which is where a block scan and a row at a time
            # loop stop being able to hide a disagreement about the carry between
            # blocks. The short frames above would pass with the carry deleted.
            emit_series("value", frame.column("value").cumsum(), out)
        elif case_id == "stats/cumsum-float-edges":
            # The frame with the infinities and the signed zeros in it, compared
            # exactly rather than within a tolerance. A running total that adds
            # infinity to negative infinity produces a NaN, and that NaN is a
            # value rather than a missing row, so it is carried to the end of the
            # column. Reading it as missing would answer the rest of the column
            # instead, which is why this is exact.
            emit_series("value", frame.column("value").cumsum(), out)
        elif case_id == "categorical/categories":
            emit_index(frame.column("value").cat_categories(), out)
        elif case_id == "categorical/codes":
            # Unnamed, because pandas hands back a series with `None` there. The
            # codes are not the column, so they do not carry its name, and a
            # driver that sent the firepanda label would fail this for the wrong
            # reason and then pass it for the wrong reason.
            emit_series_unnamed(frame.column("value").cat_codes(), out)
        elif case_id == "categorical/ordered":
            emit_bool(frame.column("value").cat_ordered(), out)
        elif case_id == "categorical/dtype":
            write_arrow(
                string_scalar_frame(
                    "value", String(frame.column("value").logical())
                ),
                out,
            )
            print('{"status":"ok","kind":"scalar"}')
        elif case_id == "categorical/isna":
            emit_series(
                "value",
                Series("value", frame.column("value").is_null()),
                out,
            )
        elif case_id == "categorical/dropna":
            emit_series("value", frame.column("value").drop_nulls(), out)
        elif case_id == "categorical/remove-unused":
            emit_index(
                frame.column("value")
                .cat_drop_unused_categories()
                .cat_categories(),
                out,
            )
        elif case_id == "categorical/as-ordered":
            emit_bool(
                frame.column("value").cat_set_ordered(True).cat_ordered(), out
            )
        elif case_id == "categorical/as-unordered":
            emit_bool(
                frame.column("value").cat_set_ordered(False).cat_ordered(), out
            )
        elif case_id == "categorical/rename-categories":
            # The pandas side builds a mapping from the categories it already
            # has, so this builds the same list from the same place. A rename is
            # decided by position and no code moves, which is why the answer is
            # the whole column rather than the category list.
            var renaming = frame.column("value")
            var old = renaming.cat_categories()
            var upper = List[String]()
            for i in range(len(old)):
                upper.append(old.text(i).upper())
            emit_series(
                "value",
                renaming.cat_rename_categories(
                    category_list(upper^), renaming.cat_ordered()
                ),
                out,
            )
        elif case_id == "categorical/reorder-categories":
            # Reordering is `set_categories` with the same set in a different
            # order, which is the Mojo spelling document 27 leaves a caller. The
            # list arithmetic is on the pandas side of firepanda and this driver
            # runs the Mojo side, so the list is built here the way a Mojo caller
            # would build it.
            var reordering = frame.column("value")
            var held = reordering.cat_categories()
            var backwards = List[String]()
            for i in range(len(held)):
                backwards.append(held.text(len(held) - 1 - i))
            emit_series(
                "value",
                reordering.cat_set_categories(
                    category_list(backwards^), reordering.cat_ordered()
                ),
                out,
            )
        elif case_id == "categorical/set-categories":
            var setting = frame.column("value")
            var wanted: List[String] = ["a", "b", "c"]
            emit_series(
                "value",
                setting.cat_set_categories(
                    category_list(wanted^), setting.cat_ordered()
                ),
                out,
            )
        elif case_id == "categorical/compare-eq":
            var equated = frame.column("value")
            emit_series(
                "value",
                equated.binary(
                    Value(category_label(equated, 0)), BinaryOp.EQ
                ),
                out,
            )
        elif case_id == "categorical/compare-lt":
            var ordered = frame.column("value")
            emit_series(
                "value",
                ordered.binary(
                    Value(category_label(ordered, -1)), BinaryOp.LT
                ),
                out,
            )
        elif case_id == "categorical/astype-string":
            emit_series(
                "value", frame.column("value").cast(LogicalType.STRING), out
            )
        elif case_id == "categorical/astype-category":
            # The last column of a string frame rather than one called `value`,
            # because the frames this case names are the ordinary text ones and
            # the pandas spelling is `df.iloc[:, -1]`.
            var columns = frame.names()
            emit_index(
                frame.column(columns[len(columns) - 1])
                .cast(LogicalType.dictionary(DType.int32, False))
                .cat_categories(),
                out,
            )
        elif case_id == "strings/len":
            # The case the unicode frame exists for. A byte count would pass this
            # on the ascii frame and fail on the other three, which is the whole
            # reason the corpus has four text frames rather than one.
            emit_series("value", frame.column("value").chars_length(), out)
        elif case_id == "strings/slice":
            emit_series(
                "value",
                frame.column("value").chars_slice(Optional(1), Optional(4), 1),
                out,
            )
        elif case_id == "strings/slice-step":
            # Both bounds absent and a step of two, which is the shape where the
            # direction of a missing bound is decided. A resolver that reads a
            # missing stop as the low end rather than the far end answers the
            # empty string on every row here and gets everything else right.
            emit_series(
                "value",
                frame.column("value").chars_slice(None, None, 2),
                out,
            )
        elif case_id == "strings/slice-negative":
            emit_series(
                "value",
                frame.column("value").chars_slice(Optional(-3), None, 1),
                out,
            )
        elif case_id == "strings/slice-replace":
            emit_series(
                "value",
                frame.column("value").chars_slice_replace(
                    Optional(1), Optional(3), "XX"
                ),
                out,
            )
        elif case_id == "strings/get":
            # Past the end is a null here rather than the empty string a slice of
            # one character would give, which is the difference that makes this a
            # kernel of its own rather than a call into the slice above.
            emit_series("value", frame.column("value").chars_get(1), out)
        elif case_id == "strings/find":
            emit_series(
                "value",
                frame.column("value").chars_find("a", None, None, False),
                out,
            )
        elif case_id == "strings/rfind":
            emit_series(
                "value",
                frame.column("value").chars_find("a", None, None, True),
                out,
            )
        elif case_id == "strings/startswith":
            emit_series(
                "value", frame.column("value").chars_starts_with("a"), out
            )
        elif case_id == "strings/endswith":
            emit_series(
                "value", frame.column("value").chars_ends_with("z"), out
            )
        elif case_id == "strings/removeprefix":
            emit_series(
                "value", frame.column("value").chars_remove_prefix("a"), out
            )
        elif case_id == "strings/removesuffix":
            emit_series(
                "value", frame.column("value").chars_remove_suffix("z"), out
            )
        elif case_id == "windows/rolling-sum":
            emit_series("value", rolled(frame, WindowOp.SUM), out)
        elif case_id == "windows/rolling-mean":
            emit_series("value", rolled(frame, WindowOp.MEAN), out)
        elif case_id == "windows/rolling-min":
            emit_series("value", rolled(frame, WindowOp.MIN), out)
        elif case_id == "windows/rolling-max":
            emit_series("value", rolled(frame, WindowOp.MAX), out)
        elif case_id == "windows/rolling-count":
            # The one of the five with no hole at the top of a full column, and
            # the one that tests `min_periods` against how many rows the window
            # covers rather than how many of them hold a value.
            emit_series("value", rolled(frame, WindowOp.COUNT), out)
        elif case_id == "windows/rolling-var":
            emit_series("value", measured(frame, WindowOp.VAR), out)
        elif case_id == "windows/rolling-std":
            emit_series("value", measured(frame, WindowOp.STD), out)
        elif case_id == "windows/rolling-sem":
            emit_series("value", measured(frame, WindowOp.SEM), out)
        elif case_id == "windows/rolling-min-periods":
            emit_series(
                "value",
                frame.column("value").rolling(
                    WindowOp.SUM, 5, Optional(1), False, WindowEdge.RIGHT, None
                ),
                out,
            )
        elif case_id == "windows/rolling-min-periods-nulls":
            emit_series(
                "value",
                frame.column("value").rolling(
                    WindowOp.MEAN, 4, Optional(3), False, WindowEdge.RIGHT, None
                ),
                out,
            )
        elif case_id == "windows/rolling-center":
            emit_series(
                "value",
                frame.column("value").rolling(
                    WindowOp.SUM, 5, None, True, WindowEdge.RIGHT, None
                ),
                out,
            )
        elif case_id == "windows/rolling-center-even":
            # The case where which side gets the extra row is decided. A window
            # that leaned the other way would pass the odd width case above and
            # fail this one, which is why the suite has both.
            emit_series(
                "value",
                frame.column("value").rolling(
                    WindowOp.SUM, 4, None, True, WindowEdge.RIGHT, None
                ),
                out,
            )
        elif case_id == "windows/rolling-closed":
            emit_series(
                "value",
                frame.column("value").rolling(
                    WindowOp.SUM, 5, None, False, WindowEdge.LEFT, None
                ),
                out,
            )
        elif case_id == "windows/rolling-step":
            # The only window case whose answer is shorter than the frame, and
            # the labels it keeps are the labels of the rows it sampled. This
            # driver emits the index along with the values, so a stepped window
            # that relabelled its answer from zero would fail here rather than
            # looking right.
            emit_series(
                "value",
                frame.column("value").rolling(
                    WindowOp.SUM, 10, None, False, WindowEdge.RIGHT, Optional(3)
                ),
                out,
            )
        elif case_id == "windows/rolling-window-one":
            emit_series(
                "value",
                frame.column("value").rolling(
                    WindowOp.SUM, 1, None, False, WindowEdge.RIGHT, None
                ),
                out,
            )
        elif case_id == "windows/rolling-window-longer-than-frame":
            # A column called `b` rather than `value`, since the frames this
            # case names are the one row and two row ones.
            emit_series(
                "b",
                frame.column("b").rolling(
                    WindowOp.SUM, 100, None, False, WindowEdge.RIGHT, None
                ),
                out,
            )
        elif case_id == "windows/expanding-sum":
            emit_series("value", spread(frame, WindowOp.SUM), out)
        elif case_id == "windows/expanding-mean":
            emit_series("value", spread(frame, WindowOp.MEAN), out)
        elif case_id == "windows/expanding-min":
            emit_series("value", spread(frame, WindowOp.MIN), out)
        elif case_id == "windows/expanding-max":
            emit_series("value", spread(frame, WindowOp.MAX), out)
        elif case_id == "windows/expanding-count":
            emit_series("value", spread(frame, WindowOp.COUNT), out)
        elif case_id == "windows/expanding-var":
            emit_series("value", spread(frame, WindowOp.VAR), out)
        elif case_id == "windows/expanding-std":
            emit_series("value", spread(frame, WindowOp.STD), out)
        elif case_id == "windows/expanding-sem":
            emit_series("value", spread(frame, WindowOp.SEM), out)
        elif case_id == "windows/expanding-min-periods":
            emit_series(
                "value", frame.column("value").expanding(WindowOp.SUM, 5), out
            )
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
            '{"status":"raised","type":"Error","message":'
            + quote(String(error))
            + "}"
        )
    exit(OK)
