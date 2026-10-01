"""Construction, attributes, selection, arithmetic and the reductions.

The section everything else stands on. If `sum` over a column with nulls in it is
wrong then every groupby case and every window case is wrong too, and it is worth
knowing that from four cases here rather than from forty somewhere else.

The frames are chosen for the property being tested rather than for variety. A null
handling case runs on the half null and all null frames because those are where null
handling is visible, and an all null column is the case that separates a sum that
returns zero from one that returns null, which is a real difference that pandas has
an opinion about.
"""

from __future__ import annotations

import io

from fpcompat.cases import case, section
from fpcompat.compare import Rules, Tolerance

section("basics")

SHAPES = ("empty", "single", "two", "tall")
NUMERIC = ("int64_no_nulls", "int64_half_null", "int64_all_null")
FLOATS = ("float64_no_nulls", "float64_half_null", "float64_all_null")
WIDTHS = (
    "int8_half_null",
    "int16_half_null",
    "int32_half_null",
    "int64_half_null",
    "uint8_half_null",
    "uint16_half_null",
    "uint32_half_null",
    "uint64_half_null",
    "float32_half_null",
    "float64_half_null",
)
DENSE_WIDTHS = (
    "int8_no_nulls",
    "int16_no_nulls",
    "int32_no_nulls",
    "int64_no_nulls",
    "uint8_no_nulls",
    "uint16_no_nulls",
    "uint32_no_nulls",
    "uint64_no_nulls",
    "float32_no_nulls",
    "float64_no_nulls",
)

ACCUMULATED = Rules(
    tolerance=Tolerance.ACCUMULATION,
    reason="a sum over ten thousand doubles depends on the order they were added in",
)

# ---------------------------------------------------------------------------
# Shape and attributes
# ---------------------------------------------------------------------------

MEASURING = (
    "in process because there is no kernel to reach. A driver arm for one of these "
    "can only restate the case in Mojo, and one that did meant the board called "
    "`df.size` a pass through the whole period when the class had no such member. "
    "See spec 57 section 1"
)

VC_NOTE = (
    "In process because firepanda counts in its Python layer, with a group by and a "
    "stable sort, which the driver cannot reach"
)


case(
    "basics/shape",
    "DataFrame.shape",
    frames=(*SHAPES, "wide"),
    expr=lambda pd, df: df.shape,
    in_process=True,
    note="the empty frame is here because a zero row shape is where an off by one shows. "
    + MEASURING,
)
case("basics/len", "DataFrame.__len__", frames=SHAPES, expr=lambda pd, df: len(df))
case(
    "basics/size",
    "DataFrame.size",
    frames=(*SHAPES, "wide"),
    expr=lambda pd, df: df.size,
    in_process=True,
    note="cells rather than rows, so the wide frame is the one that tells the two apart. "
    + MEASURING,
)
case(
    "basics/ndim",
    "DataFrame.ndim",
    frames=SHAPES,
    expr=lambda pd, df: df.ndim,
    in_process=True,
    note="a constant, and the only thing worth measuring about it is that it answers. " + MEASURING,
)
case(
    "basics/columns",
    "DataFrame.columns",
    frames=(*SHAPES, "wide"),
    expr=lambda pd, df: df.columns,
)
case(
    "basics/dtypes",
    "DataFrame.dtypes",
    frames=(*SHAPES, "wide", "temporal_range", "categorical_ordered"),
    expr=lambda pd, df: df.dtypes.astype(str),
    in_process=True,
    note="compared as strings because a dtype object is not a value the comparison can "
    "hold. Text is spelt `str` on both sides now, so the frames with text in them agree "
    "exactly, and so does the temporal frame. " + MEASURING,
)
case(
    "basics/dtypes-labels",
    "DataFrame.dtypes",
    frames=("single", "wide"),
    expr=lambda pd, df: list(df.dtypes.index),
    in_process=True,
    note="the half that makes this a column rather than a list, which is the column "
    "names sitting beside the types. The values are checked next door and this is the "
    "labels on their own, because a right list of types under wrong labels reads as a "
    "pass. " + MEASURING,
)
case(
    "basics/index",
    "DataFrame.index",
    frames=SHAPES,
    expr=lambda pd, df: df.index,
    note="in process because the driver has no entry for this call, and the module answers it the "
    "way pandas does on every frame",
    in_process=True,
)
case(
    "basics/empty",
    "DataFrame.empty",
    frames=SHAPES,
    expr=lambda pd, df: df.empty,
    in_process=True,
    note="the empty frame is the whole case, because `empty` is a question about either "
    "axis rather than about rows and a frame of columns with no rows in them is one. " + MEASURING,
)
case(
    "basics/axes",
    "DataFrame.axes",
    frames=("single", "two"),
    expr=lambda pd, df: [len(axis) for axis in df.axes],
    in_process=True,
    note="the lengths rather than the axes, because the second entry is a list here and "
    "an Index in pandas, which is the divergence `columns` carries and is scored where "
    "`columns` is. What is scored here is that there are two of them and that the rows "
    "come first. " + MEASURING,
)
case(
    "basics/axes-labels",
    "DataFrame.axes",
    frames=("single", "two"),
    expr=lambda pd, df: list(df.axes[0]),
    in_process=True,
    note="the first axis on its own, which is the row labels and is an Index on both "
    "sides, so it can be compared as values. " + MEASURING,
)
case(
    "basics/series-dtype",
    "Series.dtype",
    frames=WIDTHS,
    expr=lambda pd, df: str(df["value"].dtype),
    note="every integer width, because pandas widens a nullable integer to float64 here "
    "and that is the single most surprising thing in the whole type system",
)
case(
    "basics/series-name",
    "Series.name",
    frames=SHAPES,
    expr=lambda pd, df: df[df.columns[0]].name,
)
case(
    "basics/series-shape",
    "Series.shape",
    frames=SHAPES,
    expr=lambda pd, df: df[df.columns[0]].shape,
    in_process=True,
    note="a tuple of one, which is the shape a caller writing code for both classes "
    "unpacks. " + MEASURING,
)
case(
    "basics/series-ndim",
    "Series.ndim",
    frames=SHAPES,
    expr=lambda pd, df: df[df.columns[0]].ndim,
    in_process=True,
    note="one where the frame says two, which is the pair that makes either constant "
    "worth having. " + MEASURING,
)
case(
    "basics/series-size",
    "Series.size",
    frames=SHAPES,
    expr=lambda pd, df: df[df.columns[0]].size,
    in_process=True,
    note="rows, where the frame's is cells, and the two names being the same is the "
    "thing to get wrong. " + MEASURING,
)
case(
    "basics/series-empty",
    "Series.empty",
    frames=SHAPES,
    expr=lambda pd, df: df[df.columns[0]].empty,
    in_process=True,
    note="one axis, so there is one way to be empty, which is the simple half of the "
    "rule the frame has the hard half of. " + MEASURING,
)
case(
    "basics/series-dtypes",
    "Series.dtypes",
    frames=DENSE_WIDTHS + WIDTHS,
    expr=lambda pd, df: str(df["value"].dtypes),
    in_process=True,
    note="the plural name on a thing with one type, which is what `dtype` answers and "
    "is run over every width for the same reason `dtype` is. The widths with gaps in "
    "them read as float64 on both sides, since `DataFrame.from_arrow` widens a gap the "
    "way pandas does. " + MEASURING,
)
case(
    "basics/series-axes",
    "Series.axes",
    frames=SHAPES,
    expr=lambda pd, df: len(df[df.columns[0]].axes),
    in_process=True,
    note="a list of one on a class with one axis, which reads like a mistake and is the "
    "shape that lets one loop walk either class. " + MEASURING,
)

INVARIANT = (
    "in process, and written as a question the two libraries answer the same way rather "
    "than as the number itself, because the number is scored under `divergences/nbytes`. "
    "These rules are still worth holding on their own, since every one of them can break "
    "without the byte count changing at all. " + MEASURING
)

case(
    "basics/series-nbytes",
    "Series.nbytes",
    frames=SHAPES,
    expr=lambda pd, df: df[df.columns[0]].nbytes > 0,
    in_process=True,
    note="whether the column weighs anything. The empty frame is the half that makes this a "
    "question rather than a constant. " + INVARIANT,
)
case(
    "basics/memory-columns",
    "DataFrame.memory_usage",
    level="L3",
    covers=("index",),
    frames=("single", "two", "tall"),
    expr=lambda pd, df: (
        df.memory_usage(index=False).tolist() == [df[name].nbytes for name in df.columns]
    ),
    in_process=True,
    note="the plural is the singular repeated, which holds in both libraries and is the "
    "rule an implementation that recomputed the columns under the flag would be free to "
    "break. " + INVARIANT,
)
case(
    "basics/memory-series",
    "Series.memory_usage",
    level="L3",
    covers=("index",),
    frames=("single", "two", "tall"),
    expr=lambda pd, df: df[df.columns[0]].memory_usage(index=False) == df[df.columns[0]].nbytes,
    in_process=True,
    note="the column's own weight with the labels left out, which is what `nbytes` "
    "answers on both sides, so the two members have to agree with each other. " + INVARIANT,
)
case(
    "basics/memory-series-index",
    "Series.memory_usage",
    level="L3",
    covers=("index",),
    frames=("single", "two", "tall"),
    expr=lambda pd, df: (
        df[df.columns[0]].memory_usage() - df[df.columns[0]].memory_usage(index=False)
        == df.index.nbytes
    ),
    in_process=True,
    note="what the flag is worth, which is exactly the index and nothing else. The two "
    "libraries put very different numbers in that gap, 132 there for labels nobody "
    "declared and nothing here, and the gap is the index either way. " + INVARIANT,
)
case(
    "basics/memory-deep",
    "DataFrame.memory_usage",
    level="L3",
    covers=("deep",),
    frames=("tall", "int64_no_nulls"),
    expr=lambda pd, df: df.memory_usage(deep=True).tolist() == df.memory_usage().tolist(),
    in_process=True,
    note="the flag changes nothing on a frame with no text in it, which is true in both "
    "libraries and is as far as the agreement goes. A frame with text in it was measured "
    "rather than assumed: pandas 3 charges the characters only under `deep`, so a two row "
    "text column goes from 16 to 84 there, while here the characters are always counted "
    "and there is no shallow answer to give. That difference belongs to the byte counts, "
    "which `engine/nbytes` already covers, and these two frames are what is left once it "
    "is taken out. " + INVARIANT,
)

REPORTED = (
    "in process, and read back out of a buffer, because `info` prints and answers nothing. "
    "Three of its lines are allowed to differ, the class, the spelling of the types and the "
    "memory, and all three are registered, so every case here reads a part of the report "
    "that is not one of those and asserts it exactly. " + MEASURING
)


def info_lines(frame: object, **kwargs: object) -> list[str]:
    """The report as lines, which is the only way to score a member that prints.

    Args:
        frame: The frame or the column being asked.
        kwargs: Passed straight through to `info`.

    Returns:
        The lines, with no newlines on them.
    """
    buf = io.StringIO()
    frame.info(buf=buf, **kwargs)  # type: ignore[attr-defined]
    return buf.getvalue().splitlines()


def info_rows(frame: object) -> list[str]:
    """The body of the column table, without its header or the two lines under it.

    The table starts three lines in, at the header, and the rule is the line after that, so
    the body is everything from the fifth line to the two lines that always end the report.

    Args:
        frame: The frame being asked.

    Returns:
        One string per column.
    """
    return info_lines(frame)[5:-2]


case(
    "basics/info-labels",
    "DataFrame.info",
    frames=SHAPES,
    expr=lambda pd, df: info_lines(df)[1],
    in_process=True,
    note="the line about the labels, which both libraries write the same way for a frame "
    "whose index was never declared, down to `RangeIndex` and the two ends. The empty "
    "frame is the one that stops after the count, because there is no first label. " + REPORTED,
)
case(
    "basics/info-shape",
    "DataFrame.info",
    frames=(*SHAPES, "wide"),
    expr=lambda pd, df: len(info_lines(df)),
    in_process=True,
    note="how many lines the report is, which is the cheapest thing that catches a table "
    "with a row too many or a heading that went missing. The wide frame is the one that "
    "takes the summary form instead, so the count there is five whatever the width. " + REPORTED,
)
case(
    "basics/info-columns",
    "DataFrame.info",
    frames=("single", "two", "tall"),
    expr=lambda pd, df: [line.split()[0:2] for line in info_rows(df)],
    in_process=True,
    note="the position and the name off each row of the table, which is the half of the "
    "table that has nothing to do with types. " + REPORTED,
)
case(
    "basics/info-counts",
    "DataFrame.info",
    frames=("single", "two", "tall"),
    expr=lambda pd, df: [line.split()[2] for line in info_rows(df)],
    in_process=True,
    note="how many rows each column says are not missing, which is what `count` says and "
    "is the one number in the report that is neither a shape nor a byte count. " + REPORTED,
)
case(
    "basics/info-summary",
    "DataFrame.info",
    level="L3",
    covers=("verbose",),
    frames=("single", "two", "wide"),
    expr=lambda pd, df: info_lines(df, verbose=False)[2],
    in_process=True,
    note="the one line form, which names the first column and the last one. The wide frame "
    "takes this form anyway and the two narrow ones only take it when asked, so the flag is "
    "doing the work on two of the three. " + REPORTED,
)
case(
    "basics/info-max-cols",
    "DataFrame.info",
    level="L3",
    covers=("max_cols",),
    frames=("two", "tall"),
    expr=lambda pd, df: info_lines(df, max_cols=1)[2].startswith("Columns:"),
    in_process=True,
    note="the width at which the table stops being worth printing, moved down to one so a "
    "frame of two or three columns is over it. pandas reads the default off an option and "
    "this library has it as a constant, which is why the case passes a number rather than "
    "leaning on the default. " + REPORTED,
)
case(
    "basics/info-show-counts",
    "DataFrame.info",
    level="L3",
    covers=("show_counts",),
    frames=("single", "two"),
    expr=lambda pd, df: (
        "Non-Null Count" not in "\n".join(info_lines(df, show_counts=False)),
        len(info_lines(df, show_counts=False)) == len(info_lines(df)),
    ),
    in_process=True,
    note="one column of the table goes and the report stays the same height, which is the "
    "pair that tells a flag that narrows the table from one that drops a row. " + REPORTED,
)
case(
    "basics/info-memory-off",
    "DataFrame.info",
    level="L3",
    covers=("memory_usage",),
    frames=("single", "two", "tall"),
    expr=lambda pd, df: info_lines(df, memory_usage=False) == info_lines(df)[:-1],
    in_process=True,
    note="the last line goes and nothing else moves, which can be asserted exactly even "
    "though the line itself holds a number the two libraries disagree about. " + REPORTED,
)
case(
    "basics/info-buf",
    "DataFrame.info",
    level="L3",
    covers=("buf",),
    frames=("single", "tall"),
    expr=lambda pd, df: (df.info(buf=io.StringIO()) is None, info_lines(df)[-1].endswith("s")),
    in_process=True,
    note="the call answers nothing and writes instead, which is the whole reason every "
    "other case here reads a buffer. The second half is the memory line ending in its "
    "unit, which is as much of that line as can be compared. " + REPORTED,
)
case(
    "basics/info-series",
    "Series.info",
    frames=("single", "two", "tall"),
    expr=lambda pd, df: info_lines(df[df.columns[0]])[2:4],
    in_process=True,
    note="the column's name and the heading under it, which is where the column's report "
    "parts company with the frame's: there is no position and no name in the table, so the "
    "name goes on a line of its own above it. " + REPORTED,
)

PRINTED = (
    "in process, and compared as the whole rendering rather than as the labels read out of "
    "it, because the spacing is as much of the contract as the labels are. These cases did "
    "read one field out of each line while the padding was a place out, which is what issue "
    "730 was, and a case that loose would have gone on passing after the padding broke "
    "again. " + MEASURING
)


def listing(column: object) -> list[str]:
    """The rows of a column's rendering, without the name above it or the footer below it.

    Worth having for the text columns only, where the two libraries still spell the type
    differently in the footer and comparing the whole rendering would be comparing that
    rather than the layout.

    A column with nothing in it renders as one line with no listing at all, and that is an
    empty list here rather than a line, because there are no labels in it to read.

    Args:
        column: The column being printed.

    Returns:
        One line per printed row.
    """
    lines = repr(column).splitlines()
    if lines[0].startswith("Series(["):
        return []
    head = 1 if column.index.name is not None else 0  # type: ignore[attr-defined]
    return lines[head:-1]


def relabelled(frame: object) -> object:
    """The second column of a frame, under labels taken from the first.

    Args:
        frame: The frame being read.

    Returns:
        A column whose labels are not a range.
    """
    names = frame.columns  # type: ignore[attr-defined]
    return frame.set_index(names[0])[names[1]]  # type: ignore[attr-defined]


case(
    "basics/repr-labels",
    "Series.__repr__",
    frames=("empty", "single", "two"),
    expr=lambda pd, df: repr(relabelled(df)),
    in_process=True,
    note="the labels a column prints down its left hand side, which used to be the row "
    "positions here whatever the labels were. Everything else about the column answered "
    "correctly, so nothing but a case that reads the rendering catches it. The float column "
    "is the one read, and on the two row frame it holds a negative, so this is also where "
    "the place kept in front of a value is compared. " + PRINTED,
)
case(
    "basics/repr-range",
    "Series.__repr__",
    frames=("empty", "single", "two"),
    expr=lambda pd, df: repr(df[df.columns[0]]),
    in_process=True,
    note="the same labels on a column that never left the default range, which is the case "
    "that was already right and had to stay right. " + PRINTED,
)
case(
    "basics/repr-level-name",
    "Series.__repr__",
    frames=("single", "two"),
    expr=lambda pd, df: repr(relabelled(df)).splitlines()[0],
    in_process=True,
    note="the name of the level, which pandas prints on a line of its own above the listing "
    "and which was not printed here at all. The whole line is compared because there is "
    "nothing else on it to pad. " + PRINTED,
)
case(
    "basics/repr-unnamed",
    "Series.__repr__",
    frames=("single", "two"),
    expr=lambda pd, df: repr(relabelled(df).rename_axis(None)),
    in_process=True,
    note="taking the name off puts the first label on the first line, so the rendering is "
    "one line shorter and starts with a label rather than a name. A column that printed the "
    "name line unconditionally would fail here and pass the case above. " + PRINTED,
)
case(
    "basics/repr-footer",
    "Series.__repr__",
    frames=("single", "two"),
    expr=lambda pd, df: repr(relabelled(df)).splitlines()[-1],
    in_process=True,
    note="the footer under the listing, which names the column and its type. The float "
    "column is the one read, because a text column used to spell its type differently "
    "here, which was `engine/dtype-spelling` rather than anything this member decides. " + PRINTED,
)
case(
    "basics/repr-elided",
    "Series.__repr__",
    frames=("int64_no_nulls", "keys_10"),
    expr=lambda pd, df: repr(relabelled(df)),
    in_process=True,
    note="a column with more rows in it than pandas will print, where the middle is left out "
    "and a footer says how long the whole thing was. Three things only this case reaches: "
    "the row of dots is two dots in a column too narrow for three, the label beside it is "
    "left blank, and the footer names the column before it gives the length. " + PRINTED,
)
case(
    "basics/repr-boolean",
    "Series.__repr__",
    frames=("tall",),
    expr=lambda pd, df: repr(df.set_index(df.columns[0])[df.columns[2]]),
    in_process=True,
    note="a boolean column, which pandas counts as numeric when it decides whether to hold "
    "the column's name in by a place, so the name sits one place further right than the same "
    "name over a text column. No boolean ever prints a sign, which is what makes this the "
    "one case where the rule can be seen on its own. " + PRINTED,
)
case(
    "basics/repr-float",
    "Series.__repr__",
    frames=("float64_no_nulls", "tall"),
    expr=lambda pd, df: repr(df.set_index(df.columns[0])[df.columns[1]]),
    in_process=True,
    note="a float column, where what a value prints as is a fact about the column rather "
    "than about the value. Both of these hold something over a million and print every value "
    "in scientific notation because of it, including the small ones, which is the whole point "
    "of reading the rendering rather than a value out of it. The first frame also carries a "
    "NaN, both infinities, a negative zero and the smallest subnormal there is, none of which "
    "count as numbers when the column decides its format and all of which have to survive "
    "being printed anyway. " + PRINTED,
)
case(
    "basics/repr-float-fixed",
    "Series.__repr__",
    frames=("single", "two"),
    expr=lambda pd, df: repr(df[df.columns[1]] / 3),
    in_process=True,
    note="the other side of the decision, and the clearest demonstration there is that the "
    "format belongs to the column. Nothing here is over a million or under the last place "
    "printed, so it stays in fixed point, and the division is there to put a value with six "
    "places in it beside a value with one. The half is printed as `0.5` on the one row frame "
    "and as `0.500000` on the two row frame, because the trailing zeros only come off while "
    "every value in the column still ends in one. " + PRINTED,
)
case(
    "basics/repr-text",
    "Series.__repr__",
    frames=("single",),
    expr=lambda pd, df: listing(df.set_index(df.columns[0])[df.columns[2]]),
    in_process=True,
    note="a text column, whose name is not held in and whose values are not read as numbers, "
    "so a word beginning with a minus does not take the place a number's sign takes. Only "
    "the one row frame is read: the two row frame has a missing value in its text column, "
    "which prints as `<NA>` here and as `NaN` there, and that is four characters against "
    "three so it moves the whole column as well. " + PRINTED,
)

# ---------------------------------------------------------------------------
# Selection and the head of the frame
# ---------------------------------------------------------------------------

case("basics/head", "DataFrame.head", frames=SHAPES, expr=lambda pd, df: df.head())
case(
    "basics/head-n",
    "DataFrame.head",
    level="L3",
    covers=("n",),
    frames=SHAPES,
    expr=lambda pd, df: df.head(3),
)
case(
    "basics/head-negative",
    "DataFrame.head",
    level="L3",
    covers=("n",),
    frames=SHAPES,
    expr=lambda pd, df: df.head(-2),
    note="a negative n means all but the last two, which is not what anyone guesses",
)
case("basics/tail", "DataFrame.tail", frames=SHAPES, expr=lambda pd, df: df.tail())
case(
    "basics/tail-n",
    "DataFrame.tail",
    level="L3",
    covers=("n",),
    frames=SHAPES,
    expr=lambda pd, df: df.tail(3),
)
case(
    "basics/column-select",
    "DataFrame.__getitem__",
    frames=SHAPES,
    expr=lambda pd, df: df[df.columns[0]],
)
case(
    "basics/column-list",
    "DataFrame.__getitem__",
    frames=SHAPES,
    expr=lambda pd, df: df[[df.columns[1], df.columns[0]]],
    note="the order asked for, not the order stored",
)
case(
    "basics/boolean-mask",
    "DataFrame.__getitem__",
    frames=("tall",),
    expr=lambda pd, df: df[df["flag"]],
)
case(
    "basics/row-slice-reversed",
    "DataFrame.__getitem__",
    frames=SHAPES,
    expr=lambda pd, df: df[::-1],
    in_process=True,
    note="a slice in the brackets picks rows, by position for whole numbers",
)
case(
    "basics/row-slice-stepped",
    "DataFrame.__getitem__",
    frames=SHAPES,
    expr=lambda pd, df: df[1:5:2],
    in_process=True,
    note="a stepped slice of whole numbers picks rows by position",
)
case(
    "basics/row-slice-backward-label",
    "DataFrame.loc",
    frames=SHAPES,
    expr=lambda pd, df: df.loc[3:0:-1],
    in_process=True,
    note="a negative step walks the labels back from the first bound",
)
case(
    "basics/copy",
    "DataFrame.copy",
    frames=SHAPES,
    expr=lambda pd, df: df.copy(),
)
case(
    "basics/copy-shallow",
    "DataFrame.copy",
    level="L3",
    covers=("deep",),
    frames=("two", "tall"),
    expr=lambda pd, df: df.copy(deep=False),
    note="the only parameter the frame's copy has, and the one answer both values of "
    "it get here. pandas shares the data when it is false and duplicates it when it "
    "is true, and the difference is only visible to a later write into one of the two "
    "objects. firepanda has no write into a frame at all, so the two are the same "
    "frame by every expression that can be written about them",
)
case(
    "basics/series-copy",
    "Series.copy",
    frames=("two", "tall"),
    expr=lambda pd, df: df.iloc[:, 1].copy(),
)
case(
    "basics/series-copy-shallow",
    "Series.copy",
    level="L3",
    covers=("deep",),
    frames=("two", "tall"),
    expr=lambda pd, df: df.iloc[:, 1].copy(deep=False),
)

# ---------------------------------------------------------------------------
# The reductions, which is where null handling becomes visible
# ---------------------------------------------------------------------------

for name in ("sum", "mean", "min", "max", "count", "median", "std", "var", "prod"):
    case(
        f"basics/{name}",
        f"Series.{name}",
        frames=NUMERIC + FLOATS,
        expr=(lambda method: lambda pd, df: getattr(df["value"], method)())(name),
        rules=Rules(
            tolerance=Tolerance.STATISTICAL,
            reason="a variance is computed in a different order by every implementation",
        )
        if name in ("std", "var")
        else Rules(),
        note="the all null frame is the one that matters: sum returns zero and mean "
        "returns nan, and no amount of reasoning tells you that in advance",
    )

FLAG_IN_PYTHON = (
    "the flag is a rule the Python layer applies to the answer the core gives, so the "
    "driver has no core call to make and the case runs in process, as document 36 allows"
)
"""Why the flag cases on a reduction run in process rather than through the driver."""

case(
    "basics/sum-skipna-false",
    "Series.sum",
    level="L3",
    covers=("skipna",),
    frames=NUMERIC + FLOATS,
    expr=lambda pd, df: df["value"].sum(skipna=False),
    in_process=True,
    note=FLAG_IN_PYTHON,
)
case(
    "basics/sum-min-count",
    "Series.sum",
    level="L3",
    covers=("min_count",),
    frames=NUMERIC + FLOATS,
    expr=lambda pd, df: df["value"].sum(min_count=1),
    in_process=True,
    note="min_count is the parameter that makes an all null sum return null instead of zero. "
    + FLAG_IN_PYTHON,
)
case(
    "basics/mean-skipna-false",
    "Series.mean",
    level="L3",
    covers=("skipna",),
    frames=FLOATS,
    expr=lambda pd, df: df["value"].mean(skipna=False),
    in_process=True,
    note=FLAG_IN_PYTHON,
)
case(
    "basics/sum-tall",
    "Series.sum",
    frames=("tall",),
    expr=lambda pd, df: df["value"].sum(),
    rules=ACCUMULATED,
    note="in process because the driver has no entry for this call, and the module answers it the "
    "way pandas does on every frame",
    in_process=True,
)
case(
    "basics/mean-tall",
    "Series.mean",
    frames=("tall",),
    expr=lambda pd, df: df["value"].mean(),
    rules=ACCUMULATED,
    note="in process because the driver has no entry for this call, and the module answers it the "
    "way pandas does on every frame",
    in_process=True,
)
case(
    "basics/frame-sum",
    "DataFrame.sum",
    frames=("single", "two", "wide"),
    expr=lambda pd, df: df.sum(numeric_only=True),
    level="L3",
    covers=("numeric_only",),
    note="unimplemented, and for a reason that has changed. It used to be that the "
    "answer is a Series with no name and an unnamed Series reported its name as an "
    "empty string here and as None in pandas, so the comparison failed on the name "
    "whatever the numbers did. The name is right now and what is left is the "
    "parameter: numeric_only refuses, because dropping the columns a reduction cannot "
    "read is not written yet. The cases below are the same reductions without it. In process "
    "because the driver has no entry for this call",
    in_process=True,
)
case(
    "basics/frame-sum-skipna-false",
    "DataFrame.sum",
    level="L3",
    covers=("skipna", "numeric_only"),
    frames=NUMERIC + FLOATS,
    expr=lambda pd, df: df.sum(skipna=False, numeric_only=True),
    in_process=True,
    note="a column with a gap is NaN and the answer widens to float64 only when one is. "
    + FLAG_IN_PYTHON,
)
case(
    "basics/frame-sum-min-count",
    "DataFrame.sum",
    level="L3",
    covers=("min_count", "numeric_only"),
    frames=NUMERIC + FLOATS,
    expr=lambda pd, df: df.sum(min_count=2, numeric_only=True),
    in_process=True,
    note="a column with fewer than two values is NaN rather than a total. " + FLAG_IN_PYTHON,
)
case(
    "basics/frame-max-skipna-false",
    "DataFrame.max",
    level="L3",
    covers=("skipna", "numeric_only"),
    frames=NUMERIC + FLOATS,
    expr=lambda pd, df: df.max(skipna=False, numeric_only=True),
    in_process=True,
    note=FLAG_IN_PYTHON,
)
case(
    "basics/frame-sum-whole-min-count",
    "DataFrame.sum",
    level="L3",
    covers=("axis", "min_count", "numeric_only"),
    frames=NUMERIC + FLOATS,
    expr=lambda pd, df: df.sum(axis=None, min_count=3, numeric_only=True),
    in_process=True,
    note="with axis=None the floor counts every cell in the frame rather than each "
    "column. " + FLAG_IN_PYTHON,
)
case(
    "basics/any",
    "Series.any",
    frames=("tall",),
    expr=lambda pd, df: df["flag"].any(),
)
case(
    "basics/all",
    "Series.all",
    frames=("tall",),
    expr=lambda pd, df: df["flag"].all(),
)
case(
    "basics/any-numbers",
    "Series.any",
    frames=NUMERIC + FLOATS,
    expr=lambda pd, df: df["value"].any(),
    note="a number is true when it is not zero, and the all null frame is the one "
    "worth reading because it answers False rather than a null",
)
case(
    "basics/all-numbers",
    "Series.all",
    frames=NUMERIC + FLOATS,
    expr=lambda pd, df: df["value"].all(),
    note="an all null column answers True, which is the identity of the operator and "
    "is the opposite of what the any case above gives on the same frame",
)
case(
    "basics/any-text",
    "Series.any",
    frames=("strings_ascii", "strings_null_heavy"),
    expr=lambda pd, df: df["value"].any(),
    note="the null heavy frame has empty strings sitting next to nulls, and an empty "
    "string is false while a null is neither",
)
case(
    "basics/all-text",
    "Series.all",
    frames=("strings_ascii", "strings_null_heavy"),
    expr=lambda pd, df: df["value"].all(),
)
case(
    "basics/any-bool-only",
    "Series.any",
    level="L3",
    covers=("bool_only",),
    frames=("tall",),
    expr=lambda pd, df: df["value"].any(bool_only=True),
    in_process=True,
    note="pandas takes this argument on a column, does nothing with it and answers, "
    "which is measurable and is not what the name suggests",
)
case(
    "basics/product",
    "Series.product",
    frames=NUMERIC + FLOATS,
    expr=lambda pd, df: df["value"].product(),
    note="pandas' second spelling of prod, which is the same method and has to be "
    "present under both names rather than aliased at the call site",
)
case(
    "basics/prod-nulls",
    "Series.prod",
    frames=("int64_half_null", "int64_all_null"),
    expr=lambda pd, df: df["value"].prod(),
    note="the case a product written like a sum fails, because a null holds a zero in "
    "an Arrow buffer and zero is the identity for the wrong operator",
)
# The per column forms on a frame. These answer a Series with no name, which is what
# kept them off the board until a series name became an `Optional[String]` rather than
# a `String` whose empty value was doing two jobs. The frames are the ones every column
# of which the reduction can read, because numeric_only is the parameter that drops the
# rest and it is not written yet.
for name in ("sum", "prod"):
    case(
        f"basics/frame-{name}-all-columns",
        f"DataFrame.{name}",
        frames=("int64_no_nulls", "float64_half_null", "wide"),
        expr=(lambda method: lambda pd, df: getattr(df, method)())(name),
        in_process=True,
        note="the answer is one value per column and is therefore about none of "
        "them, so it carries no name at all rather than one that is empty. The float "
        "frame carries the nulls because the integer one cannot: a whole frame answer "
        "is a typed column rather than a scalar, so it shows the widening the str.len "
        "entry in the registry already describes, where pandas turns an integer column "
        "with a missing row in it into a float64 and firepanda keeps the width. That is "
        "the same difference in a new place and it wants its own entry, which the "
        "registry says is a pull request of its own",
    )
for name in ("any", "all"):
    case(
        f"basics/frame-{name}",
        f"DataFrame.{name}",
        frames=("two", "wide"),
        expr=(lambda method: lambda pd, df: getattr(df, method)())(name),
        in_process=True,
        note="these two read a text column as well as a number, so the two frame is "
        "here where the arithmetic reductions above cannot have it",
    )
case(
    "basics/frame-name-of-a-reduction",
    "DataFrame.sum",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: df.sum().name,
    in_process=True,
    note="the attribute on its own, so that a regression in how the absence of a name "
    "is carried is reported as being about the name rather than about the reduction",
)
case(
    "basics/series-name-cleared",
    "Series.rename",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: df["value"].rename(None).name,
    in_process=True,
    note="rename(None) takes the name off and rename('') sets one that is empty, and "
    "they are two different calls because pandas tells the two states apart",
)
case(
    "basics/series-name-empty",
    "Series.rename",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: df["value"].rename("").name,
    in_process=True,
    note="the other half of the pair above, and the one that would come back wrong if "
    "the absence were put back by asking whether the name had come out empty",
)
for fold in ("sum", "prod", "min", "max", "any", "all"):
    case(
        f"basics/frame-fold-{fold}",
        f"DataFrame.{fold}",
        level="L3",
        covers=("axis",),
        frames=("wide", "int64_no_nulls"),
        expr=(lambda method: lambda pd, df: getattr(df, method)(axis=None))(fold),
        in_process=True,
        note="axis=None folds the whole frame to one value rather than meaning the "
        "default axis, and these six are the ones that can answer it by being asked "
        "a second time of their own per column answers",
    )

POSITIONS_IN_PROCESS = (
    "In process because firepanda finds the row in its Python layer, from the "
    "reduction and a filter, which the driver cannot reach"
)
case(
    "basics/idxmax",
    "Series.idxmax",
    frames=("float64_no_nulls", "int64_no_nulls", "tall"),
    expr=lambda pd, df: df["value"].idxmax(),
    in_process=True,
    note="the first maximum, and which one is first is the whole content of the case. "
    + POSITIONS_IN_PROCESS,
)
case(
    "basics/idxmin",
    "Series.idxmin",
    frames=("float64_no_nulls", "int64_no_nulls", "tall"),
    expr=lambda pd, df: df["value"].idxmin(),
    in_process=True,
    note=POSITIONS_IN_PROCESS,
)
case(
    "basics/idxmax-half-null",
    "Series.idxmax",
    frames=("float64_half_null",),
    expr=lambda pd, df: df["value"].idxmax(),
    in_process=True,
    note="a gap is passed over by default. " + POSITIONS_IN_PROCESS,
)
case(
    "basics/idxmax-skipna-false",
    "Series.idxmax",
    level="L4",
    covers=("skipna",),
    frames=("float64_half_null",),
    expr=lambda pd, df: df["value"].idxmax(skipna=False),
    raises=("ValueError", "Encountered an NA value with skipna=False"),
    in_process=True,
    note="pandas 3 raises where pandas 2 answered NaN. " + POSITIONS_IN_PROCESS,
)
FRAME_POSITIONS_IN_PROCESS = (
    "In process because firepanda reads each column's extreme, or each row's, in its "
    "Python layer, which the driver cannot reach"
)
case(
    "basics/frame-idxmax",
    "DataFrame.idxmax",
    frames=("keys_10", "float64_half_null"),
    expr=lambda pd, df: df.idxmax(),
    in_process=True,
    note="each column's first maximum, a gap passed over. " + FRAME_POSITIONS_IN_PROCESS,
)
case(
    "basics/frame-idxmin-across",
    "DataFrame.idxmin",
    covers=("axis",),
    frames=("keys_10",),
    expr=lambda pd, df: df.idxmin(axis=1),
    in_process=True,
    note="the column holding each row's minimum, the first on a tie. " + FRAME_POSITIONS_IN_PROCESS,
)
REPEAT_IN_PROCESS = (
    "In process because firepanda turns the counts into positions in its Python layer "
    "and takes them, which the driver cannot reach"
)
case(
    "basics/repeat",
    "Series.repeat",
    frames=("int64_no_nulls", "keys_awkward"),
    expr=lambda pd, df: df["value"].head(5).repeat(2),
    in_process=True,
    note="each value and its label twice, in order. " + REPEAT_IN_PROCESS,
)
case(
    "basics/repeat-each",
    "Series.repeat",
    covers=("repeats",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: df["value"].head(3).repeat([0, 1, 2]),
    in_process=True,
    note="one count per value, and a count of nought drops the value. " + REPEAT_IN_PROCESS,
)
SET_AXIS_IN_PROCESS = (
    "In process because firepanda puts the labels on in its Python layer, which the "
    "driver cannot reach"
)
case(
    "basics/set-axis-rows",
    "Series.set_axis",
    covers=("labels",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: df["value"].set_axis(list(range(len(df)))[::-1]),
    in_process=True,
    note="new row labels, one per row, counting down. " + SET_AXIS_IN_PROCESS,
)
case(
    "basics/set-axis-columns",
    "DataFrame.set_axis",
    covers=("labels", "axis"),
    frames=("keys_10",),
    expr=lambda pd, df: df.set_axis(["value", "key"], axis=1),
    in_process=True,
    note="the two column names swapped, so each label has to land on the right column. "
    + SET_AXIS_IN_PROCESS,
)
case(
    "basics/nunique",
    "Series.nunique",
    frames=("keys_10", "keys_1000", "keys_unique", "strings_null_heavy"),
    expr=lambda pd, df: df.iloc[:, 0].nunique(),
)
case(
    "basics/nunique-dropna-false",
    "Series.nunique",
    level="L3",
    covers=("dropna",),
    frames=("strings_null_heavy", "keys_awkward"),
    expr=lambda pd, df: df.iloc[:, 0].nunique(dropna=False),
    in_process=True,
    note=FLAG_IN_PYTHON,
)

# ---------------------------------------------------------------------------
# Nulls
# ---------------------------------------------------------------------------

case("basics/isna", "Series.isna", frames=NUMERIC + FLOATS, expr=lambda pd, df: df["value"].isna())
case(
    "basics/notna", "Series.notna", frames=NUMERIC + FLOATS, expr=lambda pd, df: df["value"].notna()
)
case(
    "basics/isna-float-edges",
    "Series.isna",
    frames=("float64_no_nulls", "float32_no_nulls"),
    expr=lambda pd, df: df["value"].isna(),
    note="the float frames carry a nan at offset zero and no null anywhere, so this is "
    "the case that says whether a nan counts as missing, which it does. In process because the "
    "driver has no entry for this call",
    in_process=True,
)
case(
    "basics/dropna",
    "Series.dropna",
    frames=NUMERIC + FLOATS + ("strings_null_heavy",),
    expr=lambda pd, df: df["value"].dropna(),
)
case(
    "basics/frame-dropna",
    "DataFrame.dropna",
    frames=("two", "strings_null_heavy"),
    expr=lambda pd, df: df.dropna(),
)
case(
    "basics/frame-dropna-how-all",
    "DataFrame.dropna",
    level="L3",
    covers=("how",),
    frames=("two", "int64_all_null"),
    expr=lambda pd, df: df.dropna(how="all"),
    in_process=True,
    note=FLAG_IN_PYTHON,
)
case(
    "basics/frame-dropna-thresh",
    "DataFrame.dropna",
    level="L3",
    covers=("thresh", "subset"),
    frames=("two", "strings_null_heavy"),
    expr=lambda pd, df: df.dropna(thresh=1, subset=list(df.columns[:2])),
    in_process=True,
    note="a row is kept with at least `thresh` values among the columns looked at. "
    + FLAG_IN_PYTHON,
)
case(
    "basics/frame-dropna-columns",
    "DataFrame.dropna",
    level="L3",
    covers=("axis", "how"),
    frames=("two", "strings_null_heavy"),
    expr=lambda pd, df: df.dropna(axis=1, how="all"),
    in_process=True,
    note="with axis=1 a column is dropped by the values in it, as a row is. " + FLAG_IN_PYTHON,
)
case(
    "basics/frame-dropna-ignore-index",
    "DataFrame.dropna",
    level="L3",
    covers=("ignore_index",),
    frames=("two", "strings_null_heavy"),
    expr=lambda pd, df: df.dropna(ignore_index=True),
    in_process=True,
    note="what is left is numbered again from zero. " + FLAG_IN_PYTHON,
)
case(
    "basics/fillna",
    "Series.fillna",
    level="L3",
    covers=("value",),
    frames=FLOATS,
    expr=lambda pd, df: df["value"].fillna(0.0),
)
case(
    "basics/fillna-signed-zero",
    "Series.fillna",
    level="L3",
    covers=("value",),
    frames=("float64_half_null",),
    expr=lambda pd, df: df["value"].fillna(-0.0),
    rules=Rules(
        signed_zero=True,
        reason="filling with a negative zero and getting a positive one back is a real "
        "difference that IEEE equality hides",
    ),
)
case(
    "basics/fillna-text",
    "Series.fillna",
    frames=("strings_null_heavy",),
    expr=lambda pd, df: df["value"].fillna("missing"),
    note="the float cases fill a number into a number and this one fills a word "
    "into a word, which is the other half of the rule that a column here is "
    "typed and stays typed",
)
case(
    "basics/frame-fillna",
    "DataFrame.fillna",
    frames=FLOATS,
    expr=lambda pd, df: df.fillna(0.0),
    note="the row column has no gaps and the value column does, so this is also "
    "the case that says a column with nothing missing is left alone whatever "
    "the value was, which is the rule pandas has and the reason filling a whole "
    "frame with one number is not a type error every time",
)
case(
    "basics/frame-fillna-dict",
    "DataFrame.fillna",
    level="L3",
    covers=("value",),
    frames=FLOATS,
    expr=lambda pd, df: df.fillna({"value": 0.0}),
)
case(
    "basics/fillna-map",
    "Series.fillna",
    level="L3",
    covers=("value",),
    frames=FLOATS,
    expr=lambda pd, df: df["value"].fillna({label: float(label) for label in range(df.shape[0])}),
    in_process=True,
    note="a dict on a column means row labels and not column names, which is the one "
    "shape that reads differently on a column than on a frame, and every row gets a "
    "different value so an implementation that lined the mapping up by position rather "
    "than by label would answer the same length and the wrong rows. The mapping covers "
    "every label because a row left missing on the firepanda side is a null and the "
    "same row on the pandas side is a NaN, which the harness is right to call a "
    "difference and which is about loading rather than about this method. The alignment "
    "lives in the Python layer rather than in the core, so a driver entry would have to "
    "write it. See spec 47",
)
case(
    "basics/fillna-column",
    "Series.fillna",
    level="L3",
    covers=("value",),
    frames=FLOATS,
    expr=lambda pd, df: df["value"].fillna(df["row"]),
    in_process=True,
    note="the row column is the position written down, so this fills each missing row "
    "with its own label and is the same question the mapping case asks with the other "
    "shape. It is also a fallback of a different type than the column it fills, which "
    "is allowed here because every whole number in it survives the trip to a float and "
    "back. Python layer, see spec 47",
)
case(
    "basics/frame-fillna-column",
    "DataFrame.fillna",
    level="L3",
    covers=("value",),
    frames=FLOATS,
    expr=lambda pd, df: df.fillna(
        type(df)({"name": ["value"], "fill": [0.0]}).set_index("name")["fill"]
    ),
    in_process=True,
    note="a column handed to a frame names columns rather than rows, so this is the "
    "dict case written the other way and it is worth a case of its own because the "
    "shape that lines up against the rows on a column lines up against the column names "
    "here. Python layer, see spec 47",
)
case(
    "basics/frame-fillna-frame",
    "DataFrame.fillna",
    level="L3",
    covers=("value",),
    frames=FLOATS,
    expr=lambda pd, df: df.fillna(type(df)({"value": [0.0] * df.shape[0]})),
    in_process=True,
    note="a frame is the one value that lines up on both axes, and the fallback here "
    "carries one of the two columns so the other one is left alone, which is the half "
    "of the rule that a shape lining up on one axis only cannot state. Python layer, "
    "see spec 47",
)
case(
    "basics/fillna-axis",
    "Series.fillna",
    level="L3",
    covers=("axis",),
    frames=("float64_half_null",),
    expr=lambda pd, df: df["value"].fillna(0.0, axis=0),
    note="one value per column means the two axes name the same answer, so the "
    "case is here to say the parameter is accepted and read rather than to "
    "measure a difference between two answers",
)
case(
    "basics/ffill",
    "Series.ffill",
    frames=(*FLOATS, "strings_null_heavy"),
    expr=lambda pd, df: df["value"].ffill(),
)
case(
    "basics/bfill",
    "Series.bfill",
    frames=(*FLOATS, "strings_null_heavy"),
    expr=lambda pd, df: df["value"].bfill(),
)

# ---------------------------------------------------------------------------
# Arithmetic, over every width, because overflow is width dependent
# ---------------------------------------------------------------------------

SCALAR_ARITHMETIC = (
    ("add", lambda s: s + 1),
    ("sub", lambda s: s - 1),
    ("mul", lambda s: s * 2),
    ("truediv", lambda s: s / 2),
    ("floordiv", lambda s: s // 2),
    ("mod", lambda s: s % 3),
    ("pow", lambda s: s**2),
)

for name, symbol in SCALAR_ARITHMETIC:
    case(
        f"basics/{name}-scalar",
        f"Series.{name}",
        frames=WIDTHS,
        expr=(lambda op: lambda pd, df: op(df["value"]))(symbol),
        note="every width, because what an int8 does at 127 is not what an int64 does",
    )

for name, symbol in SCALAR_ARITHMETIC:
    case(
        f"basics/{name}-scalar-dense",
        f"Series.{name}",
        frames=DENSE_WIDTHS,
        expr=(lambda op: lambda pd, df: op(df["value"]))(symbol),
        note="the same seven operations on the same ten widths with no nulls in them, which "
        "is the only place the width rule is visible at all. A narrow column with a null in "
        "it is already a float64 by the time pandas gets to the arithmetic, so the half null "
        "family above measures the read path rather than the arithmetic, and every one of its "
        "answers is float64 whatever the operation was",
    )

case(
    "basics/bool-add-column",
    "Series.add",
    frames=("tall",),
    expr=lambda pd, df: df["flag"] + df["flag"],
    note="the answer is the logical or and the dtype stays bool, which is numpy showing "
    "through pandas and is not what the symbol looks like",
)
case(
    "basics/bool-mul-column",
    "Series.mul",
    frames=("tall",),
    expr=lambda pd, df: df["flag"] * df["flag"],
    note="the logical and, for the same reason add is the or",
)
case(
    "basics/bool-mod-column",
    "Series.mod",
    frames=("tall",),
    expr=lambda pd, df: df["flag"] % df["flag"],
    note="the only one of the seven whose answer on two bools is a number. The other "
    "four raise, and they are in the errors section rather than here",
)
case(
    "basics/bool-add-scalar",
    "Series.add",
    frames=("tall",),
    expr=lambda pd, df: df["flag"] + True,
    note="a Python bool against a bool column is still the or, so what decides these is "
    "the dtype and not the shape of the other operand",
)
case(
    "basics/add-edges",
    "Series.add",
    frames=("integer_edges",),
    expr=lambda pd, df: df["int8"] + 1,
    note="the edges frame holds each width's maximum, so this is the overflow case",
)
case(
    "basics/div-by-zero",
    "Series.truediv",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: df["value"] / 0,
    note="pandas gives an infinity rather than raising, which is a decision and not an "
    "accident, and it has to be copied",
)
case(
    "basics/column-arithmetic",
    "Series.mul",
    frames=("two", "tall"),
    expr=lambda pd, df: df.iloc[:, 0] * df.iloc[:, 1],
)
case(
    "basics/abs",
    "Series.abs",
    frames=FLOATS + NUMERIC,
    expr=lambda pd, df: df["value"].abs(),
    note="in process because the driver has no entry for this call, and the module answers it the "
    "way pandas does on every frame",
    in_process=True,
)
case(
    "basics/round",
    "Series.round",
    level="L3",
    covers=("decimals",),
    frames=FLOATS,
    expr=lambda pd, df: df["value"].round(2),
    note="banker's rounding, and the tolerance is exact on purpose because a rounding "
    "case that tolerates a difference is testing nothing. In process because the driver has "
    "no entry for this call",
    rules=Rules(tolerance=Tolerance.EXACT),
    in_process=True,
)
case(
    "basics/round-tens",
    "Series.round",
    level="L3",
    covers=("decimals",),
    frames=FLOATS + NUMERIC,
    expr=lambda pd, df: df["value"].round(-1),
    note="a negative decimals is the one place an integer column rounds, half to even among "
    "the tens. In process because the driver has no entry for this call",
    rules=Rules(tolerance=Tolerance.EXACT),
    in_process=True,
)
case(
    "basics/frame-round",
    "DataFrame.round",
    level="L3",
    covers=("decimals",),
    frames=FLOATS + NUMERIC,
    expr=lambda pd, df: df.round(1),
    note="every column at once, so the text and key columns are the ones that have to come "
    "back unchanged. In process because the driver has no entry for this call",
    rules=Rules(tolerance=Tolerance.EXACT),
    in_process=True,
)
case(
    "basics/clip",
    "Series.clip",
    level="L3",
    covers=("lower", "upper"),
    frames=("float64_no_nulls", "int64_no_nulls"),
    expr=lambda pd, df: df["value"].clip(lower=-1, upper=1),
    in_process=True,
    note="written before the method existed and narrowed to the frames with no gaps "
    "when it arrived, because a row that holds nothing is not clipped by either "
    "library and so the answer carries a null here and a nan in the oracle, which is a "
    "difference about loading rather than about clipping. The method is written in the "
    "Python layer, where the shapes of the bounds and the type rules are, so a driver "
    "entry would have to write them again. See spec 49",
)
case(
    "basics/neg",
    "Series.__neg__",
    frames=FLOATS + NUMERIC,
    expr=lambda pd, df: -df["value"],
    note="in process because the driver has no entry for this call, and the module answers it the "
    "way pandas does on every frame",
    in_process=True,
)

COMPARISON_IN_PYTHON = (
    "The kernel answers null on a missing row and the Python layer gives pandas' answer "
    "for it, so the rule the case asks about is one the driver cannot reach and the case "
    "runs in process, as document 36 allows"
)
"""Why the comparison cases on a gap run in process rather than through the driver."""

for name, symbol in (
    ("eq", lambda s: s == 3),
    ("ne", lambda s: s != 3),
    ("lt", lambda s: s < 3),
    ("le", lambda s: s <= 3),
    ("gt", lambda s: s > 3),
    ("ge", lambda s: s >= 3),
):
    case(
        f"basics/{name}",
        f"Series.{name}",
        frames=("int64_half_null", "float64_half_null"),
        expr=(lambda op: lambda pd, df: op(df["value"]))(symbol),
        in_process=True,
        note="a comparison against a null is false rather than null, which is the "
        "numpy answer and not the SQL one. " + COMPARISON_IN_PYTHON,
    )

# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

case(
    "basics/sort-values",
    "DataFrame.sort_values",
    level="L3",
    covers=("by",),
    frames=("keys_10", "keys_1000", "keys_awkward"),
    expr=lambda pd, df: df.sort_values(["key", "value"]),
    note="the awkward frame has nulls and an empty string in the key, so this is where "
    "the null position rule shows. the value column is named second because pandas "
    "defaults to an unstable kind and ten distinct keys over ten thousand rows leaves "
    "the key alone deciding almost nothing, so sorting on the key by itself would "
    "compare a permutation pandas does not promise and does not reproduce across numpy "
    "builds",
)
case(
    "basics/sort-values-default",
    "DataFrame.sort_values",
    frames=("keys_unique",),
    expr=lambda pd, df: df.sort_values("value"),
    note="the plain call, with by the only argument given because by is the one this "
    "method requires. On the unique frame rather than one of the others, because a "
    "single key over a column with ties in it returns a permutation pandas does not "
    "promise, which is the reason every other case in this group names a second key",
)
case(
    "basics/sort-values-ignore-index",
    "DataFrame.sort_values",
    level="L3",
    covers=("by", "ignore_index"),
    frames=("keys_unique",),
    expr=lambda pd, df: df.sort_values("value", ignore_index=True),
    in_process=True,
    note="numbering the rows again after the sort, which in firepanda is a reset_index "
    "on the answer and lives in the Python layer rather than in the core, so a driver "
    "entry would have to write it. See spec 43",
)
case(
    "basics/sort-values-descending",
    "DataFrame.sort_values",
    level="L3",
    covers=("by", "ascending"),
    frames=("keys_10", "keys_awkward"),
    expr=lambda pd, df: df.sort_values(["key", "value"], ascending=False),
    note="the tiebreaker is here for the same reason it is on the ascending case, and "
    "descending is the direction where an implementation is most tempted to reverse "
    "the array instead of reversing the comparison, which scrambles every group of "
    "equal keys and is invisible unless the order inside a group is pinned",
)
case(
    "basics/sort-values-na-first",
    "DataFrame.sort_values",
    level="L3",
    covers=("by", "na_position"),
    frames=("keys_awkward", "strings_null_heavy"),
    expr=lambda pd, df: df.sort_values(list(df.columns[:2]), na_position="first"),
    note="both frames are short enough that numpy's introsort falls back to insertion "
    "sort and answers stably by accident, so the second key is here to make the case "
    "say what it means rather than to fix a failure that is showing today",
)
case(
    "basics/sort-values-stable",
    "DataFrame.sort_values",
    level="L3",
    covers=("by", "kind"),
    frames=("keys_10", "keys_1000"),
    expr=lambda pd, df: df.sort_values("key", kind="stable"),
    note="ten distinct keys over ten thousand rows means every group has a thousand "
    "ties, and kind is the only argument pandas has that decides what happens inside "
    "one, so this is the case that pins the tie order rather than working around it",
)
case(
    "basics/sort-two-columns",
    "DataFrame.sort_values",
    level="L3",
    covers=("by", "ascending"),
    frames=("keys_two_column",),
    expr=lambda pd, df: df.sort_values(["left", "right"], ascending=[True, False]),
)
case(
    "basics/sort-index",
    "DataFrame.sort_index",
    frames=("keys_10", "two"),
    expr=lambda pd, df: df.sort_values("value" if "value" in df else "b").sort_index(),
    note="in process because the driver has no entry for this call, and the module answers it the "
    "way pandas does on every frame",
    in_process=True,
)
case(
    "basics/rank",
    "Series.rank",
    frames=("keys_10", "float64_half_null"),
    expr=lambda pd, df: df["value"].rank(),
    note="in process because the driver has no entry for this call, and the module ranks "
    "in one sort and a pass over the ties",
    in_process=True,
)
case(
    "basics/rank-method-min",
    "Series.rank",
    level="L3",
    covers=("method",),
    frames=("keys_10", "float64_half_null"),
    expr=lambda pd, df: df["value"].rank(method="min"),
    note="in process because the driver has no entry for this call, and the module ranks "
    "in one sort and a pass over the ties",
    in_process=True,
)
case(
    "basics/rank-method-dense",
    "Series.rank",
    level="L3",
    covers=("method",),
    frames=("keys_10",),
    expr=lambda pd, df: df["key"].rank(method="dense"),
    note="in process because the driver has no entry for this call, and the module ranks "
    "in one sort and a pass over the ties",
    in_process=True,
)
case(
    "basics/frame-rank",
    "DataFrame.rank",
    frames=("keys_10", "float64_half_null"),
    expr=lambda pd, df: df.rank(),
    note="in process because the driver has no entry for this call, and the module ranks "
    "in one sort and a pass over the ties",
    in_process=True,
)
case(
    "basics/frame-rank-numeric",
    "DataFrame.rank",
    level="L3",
    covers=("numeric_only", "method", "ascending"),
    frames=("keys_10",),
    expr=lambda pd, df: df.rank(numeric_only=True, method="max", ascending=False),
    note="in process because the driver has no entry for this call, and the module ranks "
    "in one sort and a pass over the ties",
    in_process=True,
)

# ---------------------------------------------------------------------------
# Shifting and differences
# ---------------------------------------------------------------------------

case(
    "basics/shift",
    "Series.shift",
    frames=NUMERIC + FLOATS,
    expr=lambda pd, df: df["value"].shift(),
)
case(
    "basics/shift-negative",
    "Series.shift",
    level="L3",
    covers=("periods",),
    frames=NUMERIC + FLOATS,
    expr=lambda pd, df: df["value"].shift(-2),
)
case(
    "basics/shift-fill",
    "Series.shift",
    level="L3",
    covers=("periods", "fill_value"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: df["value"].shift(1, fill_value=0),
    note="without a fill value an integer column becomes a float one, which is the "
    "kind of quiet widening that only a type check catches",
)
case(
    "basics/diff",
    "Series.diff",
    frames=NUMERIC + FLOATS,
    expr=lambda pd, df: df["value"].diff(),
)
for name in ("cumsum", "cumprod", "cummax", "cummin"):
    case(
        f"basics/{name}",
        f"Series.{name}",
        frames=("int64_half_null", "float64_half_null", "float64_no_nulls"),
        expr=(lambda method: lambda pd, df: getattr(df["value"], method)())(name),
        note="a running total skips nulls and keeps them in place, which is not what a "
        "loop would do",
    )
case(
    "basics/pct-change",
    "Series.pct_change",
    frames=("float64_no_nulls", "tall"),
    expr=lambda pd, df: df["value"].pct_change(),
)

# ---------------------------------------------------------------------------
# Reshaping the frame itself
# ---------------------------------------------------------------------------

case(
    "basics/rename",
    "DataFrame.rename",
    level="L3",
    covers=("columns",),
    frames=("two", "tall"),
    expr=lambda pd, df: df.rename(columns={df.columns[0]: "renamed"}),
)

A_SCHEMA = (
    "renaming a column edits one field of a schema and reads no values, which is the "
    "half of pandas' rename that is here. The other half maps every row label through "
    "a dictionary or a callable, which is a pass over the index rather than a change "
    "to a schema, and it refuses. See spec 45"
)

case(
    "basics/rename-callable",
    "DataFrame.rename",
    level="L3",
    covers=("columns",),
    frames=("two", "tall"),
    expr=lambda pd, df: df.rename(columns=str.upper),
    in_process=True,
    note="a callable rather than a mapping. firepanda calls it in the Python layer and "
    "hands the core the two lists of names it worked out, so a driver entry would have "
    "to apply str.upper itself. " + A_SCHEMA,
)
case(
    "basics/rename-axis-columns",
    "DataFrame.rename",
    level="L3",
    covers=("mapper", "axis"),
    frames=("two", "tall"),
    expr=lambda pd, df: df.rename({df.columns[0]: "renamed"}, axis=1),
    note="the same door as basics/rename with the mapping passed positionally and the "
    "axis named, which is the spelling pandas puts first in its own signature. " + A_SCHEMA,
)
case(
    "basics/rename-swap",
    "DataFrame.rename",
    level="L3",
    covers=("columns",),
    frames=("two", "tall"),
    expr=lambda pd, df: df.rename(
        columns={df.columns[0]: df.columns[1], df.columns[1]: df.columns[0]}
    ),
    note="two columns trading names, which is the case that needs the whole rename to "
    "happen in one pass, because each half of a swap collides with a name the frame "
    "still has. " + A_SCHEMA,
)
case(
    "basics/rename-missing-ignored",
    "DataFrame.rename",
    level="L3",
    covers=("errors",),
    frames=("two",),
    expr=lambda pd, df: df.rename(columns={"not_a_column": "renamed"}),
    in_process=True,
    note="the default, which is that a name in the mapping that is not a column is "
    "skipped in silence and the caller is told nothing. What is being scored is the "
    "filtering, and that happens in the Python layer before the core is reached, so a "
    "driver entry would answer the frame unchanged without exercising anything. " + A_SCHEMA,
)
case(
    "basics/rename-missing-raised",
    "DataFrame.rename",
    level="L4",
    covers=("errors",),
    frames=("two",),
    expr=lambda pd, df: df.rename(columns={"not_a_column": "renamed"}, errors="raise"),
    raises=("KeyError", "not_a_column"),
    note="errors='raise' is the only way a caller finds out, and both libraries name "
    "what was missing in the message, which is what makes it worth raising. " + A_SCHEMA,
)
case(
    "basics/series-rename",
    "Series.rename",
    frames=("two", "tall"),
    expr=lambda pd, df: df[df.columns[0]].rename("renamed"),
    in_process=True,
    note="a scalar names the column, which is metadata and is here. A mapping or a "
    "callable in the same parameter maps the row labels instead and refuses, so the "
    "two doors of one pandas method land on opposite sides of the line spec 45 draws. "
    "firepanda reaches this through the same core call that sets Series.name, and a "
    "driver entry would be asserting the emitter rather than the rename",
)
case(
    "basics/drop-column",
    "DataFrame.drop",
    level="L3",
    covers=("columns",),
    frames=("two", "tall", "wide"),
    expr=lambda pd, df: df.drop(columns=[df.columns[0]]),
)

A_POINTER = (
    "dropping a column takes a pointer out of a schema and reads no values, which is "
    "the half of pandas' drop that costs nothing. The other half takes row labels out "
    "of the index and lives with the indexing cases, since what it is really about is "
    "the index. See spec 46"
)

case(
    "basics/drop-column-axis",
    "DataFrame.drop",
    level="L3",
    covers=("labels", "axis"),
    frames=("two", "tall"),
    expr=lambda pd, df: df.drop(df.columns[0], axis=1),
    note="the same door as basics/drop-column with the name passed positionally and the "
    "axis named, which is the spelling pandas puts first in its own signature. One name "
    "rather than a list of them as well, which matters because a string is iterable in "
    "Python and has to be read as a label anyway. " + A_POINTER,
)
case(
    "basics/drop-column-missing-ignored",
    "DataFrame.drop",
    level="L3",
    covers=("errors",),
    frames=("two",),
    expr=lambda pd, df: df.drop(columns=["not_a_column"], errors="ignore"),
    in_process=True,
    note="errors='ignore' skips a name that is not a column and drops the rest. The "
    "core resolves names against the schema and raises on the first one it cannot find, "
    "with no word for skipping, so the filtering happens in the Python layer before the "
    "core is reached and a driver entry would answer the frame unchanged without "
    "exercising anything. " + A_POINTER,
)
case(
    "basics/drop-both-axes",
    "DataFrame.drop",
    level="L3",
    covers=("index", "columns"),
    frames=("two",),
    expr=lambda pd, df: df.drop(index=[0], columns=[df.columns[0]]),
    note="naming both axes in one call, which drop allows and rename does not. rename "
    "refuses the pair because one of its halves refuses on its own, and here neither "
    "does, so the pair is two independent steps with nothing to decide between them. "
    "The labels are read against the default index this frame came with. " + A_POINTER,
)
case(
    "basics/drop-repeated-labels",
    "DataFrame.drop",
    level="L3",
    covers=("index",),
    frames=("two",),
    expr=lambda pd, df: df.set_axis([i % 3 for i in range(len(df))]).drop(index=[1]),
    in_process=True,
    note="row labels that repeat, where dropping one label drops every row holding it. "
    "reindex refuses a repeated label, so these rows are dropped by position in the "
    "Python layer",
)
case(
    "basics/drop-missing-label",
    "DataFrame.drop",
    level="L4",
    covers=("index",),
    frames=("two",),
    expr=lambda pd, df: df.drop(index=["not_a_label"]),
    raises=("KeyError", "['not_a_label'] not found in axis"),
    in_process=True,
    note="the message lists the missing labels as they were given, quotes and all, which "
    "firepanda's Python layer writes because the core prints them as bare text",
)
case(
    "basics/assign-categorical",
    "DataFrame.assign",
    level="L2",
    frames=("two",),
    expr=lambda pd, df: df.assign(
        extra=pd.Categorical([["x", "y", "z"][i % 3] for i in range(len(df))])
    )["extra"],
    in_process=True,
    note="a Categorical put in whole keeps its categories rather than being read back as "
    "its values",
)
case(
    "basics/assign",
    "DataFrame.assign",
    frames=("two", "tall"),
    expr=lambda pd, df: df.assign(extra=df.iloc[:, 0]),
    note="in process because the driver has no entry for this call, and the module adds "
    "the column beside the others without copying them",
    in_process=True,
)
case(
    "basics/assign-chained",
    "DataFrame.assign",
    level="L3",
    covers=("kwargs",),
    frames=("keys_10", "tall"),
    expr=lambda pd, df: df.assign(
        double=lambda f: f.value * 2, more=lambda f: f.double + 1, value=0
    ),
    note="each keyword sees the frame the ones before it made, and a replaced column "
    "keeps its place while a new one goes on the end. In process because the driver has "
    "no entry for this call",
    in_process=True,
)
case(
    "basics/assign-aligned",
    "DataFrame.assign",
    level="L3",
    covers=("kwargs",),
    frames=("keys_10",),
    expr=lambda pd, df: df.assign(back=df["value"].iloc[::-1].head(len(df) // 2)),
    note="a series is lined up on the row labels, so a reversed half lands on the rows it "
    "came from and the rest are missing. In process because the driver has no entry for "
    "this call",
    in_process=True,
)
case(
    "basics/insert-by-assignment",
    "DataFrame.__setitem__",
    frames=("two", "tall"),
    expr=lambda pd, df: df.assign(**{"new": 1}),
    note="the assign spelling, because a case that mutates its frame would change the "
    "input of the next case if frames were ever cached, and in process because the "
    "driver has no entry for this call",
    in_process=True,
)
case(
    "basics/astype-float",
    "Series.astype",
    level="L3",
    covers=("dtype",),
    frames=("int64_no_nulls", "int8_no_nulls"),
    expr=lambda pd, df: df["value"].astype("float64"),
)
case(
    "basics/astype-string",
    "Series.astype",
    level="L3",
    covers=("dtype",),
    frames=("int64_no_nulls", "float64_no_nulls"),
    expr=lambda pd, df: df["value"].astype("str"),
    note="how a float is spelled as text is a decision with a hundred edge cases in it, "
    "and the float frames start with nan and both infinities",
)
case(
    "basics/astype-round-trip",
    "Series.astype",
    level="L3",
    covers=("dtype",),
    frames=("int64_no_nulls", "float64_no_nulls"),
    expr=lambda pd, df: df["value"].astype("str").astype(df["value"].dtype),
    note="out to text and back, which is the only way this suite can reach the text to "
    "number direction today, because the corpus has no string frame of numerals with a "
    "missing row in it. The float frame is the one that makes it a question rather than a "
    "formality, since its nan has to leave the values on the way out and come back into "
    "them on the way in, and a library that writes the word nan instead passes the first "
    "leg and then reads a word",
)
case(
    "basics/astype-narrow",
    "Series.astype",
    level="L3",
    covers=("dtype",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: df["value"].astype("int8"),
    note="a narrowing cast wraps rather than raising, which is the numpy rule",
)
case(
    "basics/reset-index",
    "DataFrame.reset_index",
    level="L3",
    covers=("drop",),
    frames=("two", "keys_10"),
    expr=lambda pd, df: df.sort_values(list(df.columns)).reset_index(drop=True),
    note="sorted on every column and not only the first, because pandas sorts with an "
    "unstable kind by default and keys_10 has a thousand rows on each key, so the first "
    "column alone would compare an order of ties pandas does not promise rather than "
    "the new index. In process because the driver has no entry for this call",
    in_process=True,
)
case(
    "basics/transpose",
    "DataFrame.transpose",
    frames=("single", "two"),
    expr=lambda pd, df: df[[df.columns[0], df.columns[1]]].transpose(),
    in_process=True,
    note="the columns of a transpose are the old row labels, which are numbers here, and "
    "only firepanda's Python layer names a column with a number, so the case runs in process",
)
case(
    "basics/where",
    "Series.where",
    level="L3",
    covers=("cond", "other"),
    frames=(*FLOATS, "int64_no_nulls"),
    expr=lambda pd, df: df["value"].where(df["value"] > 0, -1),
    in_process=True,
    note="a condition written as a comparison is how this method is actually called, "
    "and it is also the shape that says what a comparison against a missing value "
    "does, since a row the condition says nothing about is replaced rather than kept. "
    "Every missing row is therefore filled, which is what keeps the case about the "
    "method rather than about a null on one side being a nan on the other. The two "
    "integer frames with gaps in them are not here for the same reason: the oracle "
    "loads an Arrow integer column with nulls as float64, so the answer would differ "
    "by type before the method had done anything. The alignment, the order the nulls "
    "are read in and the type rules are all in the Python layer, so a driver entry "
    "would have to write them again. See spec 48",
)
case(
    "basics/mask",
    "Series.mask",
    level="L3",
    covers=("cond", "other"),
    frames=("float64_no_nulls", "int64_no_nulls"),
    expr=lambda pd, df: df["value"].mask(df["value"] > 0, -1),
    in_process=True,
    note="the same condition and the opposite half of the column, on the frames with "
    "no nulls so that both libraries read the condition the same way everywhere. A "
    "comparison against a null answers a null here and a false in the oracle, which "
    "this method turns over and which would therefore be a difference about loading "
    "rather than about masking. The frames with gaps are the next case. Python layer, "
    "see spec 48",
)
case(
    "basics/where-column",
    "Series.where",
    level="L3",
    covers=("cond", "other"),
    frames=FLOATS,
    expr=lambda pd, df: df["value"].where(df["value"] > 0, df["row"]),
    in_process=True,
    note="the other side as a column rather than a value, lined up by label, and of a "
    "different type than the column it is going into, which is allowed because every "
    "whole number in it survives the trip to a float. Python layer, see spec 48",
)
case(
    "basics/where-leaves-missing",
    "Series.where",
    frames=("float64_half_null", "int64_no_nulls"),
    expr=lambda pd, df: df["value"].where(df["value"] > 0).isna(),
    in_process=True,
    note="the method called with nothing but the condition, which is what makes this "
    "the L2 case of the four. With no other side named the rows that were not kept "
    "hold nothing, and this compares which rows they are. The values are compared by "
    "`basics/where-widens`, since firepanda #1352 widens a column of whole numbers to "
    "float64 and puts a NaN in it as pandas does. Python layer",
)
case(
    "basics/mask-leaves-missing",
    "Series.mask",
    frames=("float64_no_nulls", "int64_no_nulls"),
    expr=lambda pd, df: df["value"].mask(df["value"] > 0).isna(),
    in_process=True,
    note="the same call with the condition turned over, on the frames with no nulls "
    "for the reason the mask case above gives. Python layer, see spec 48",
)
case(
    "basics/where-widens",
    "Series.where",
    frames=("int64_no_nulls", "float64_half_null"),
    expr=lambda pd, df: df["value"].where(df["value"] % 2 == 0),
    in_process=True,
    note="the values themselves this time. A gap in a column of whole numbers moves it "
    "to float64 on both sides since firepanda #1352, which widens the way pandas does "
    "because a numpy column has nowhere to keep a gap. Python layer",
)
case(
    "basics/flags-with-gap",
    "pandas.Series",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series([*(df["value"].head(3) % 2 == 0).tolist(), None]),
    in_process=True,
    note="flags beside a gap are an object column, since numpy has no flag that can be "
    "missing. Held as objects since firepanda #1357. Python layer",
)
case(
    "basics/flags-with-gap-frame",
    "pandas.DataFrame",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame({"flag": [True, None, False], "n": [1, 2, 3]}).dtypes.astype(
        str
    ),
    in_process=True,
    note="the same rule for a column handed to a frame in a mapping. Python layer",
)
case(
    "basics/flags-with-gap-convert",
    "Series.convert_dtypes",
    covers=("convert_boolean",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.concat(
        [
            pd.Series([True, None, False]).convert_dtypes().astype(str),
            pd.Series([True, None, False]).convert_dtypes(convert_boolean=False).astype(str),
        ],
        axis=1,
    ),
    in_process=True,
    note="an object column of flags becomes boolean, and stays objects with convert_boolean "
    "off. Python layer",
)
case(
    "basics/flags-as-text",
    "Series.astype",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: (df["value"] % 2 == 0).astype(str),
    in_process=True,
    note="a flag written as text is True or False, as Python spells it, since firepanda "
    "#1358. Python layer",
)
case(
    "basics/mask-widens-frame",
    "DataFrame.mask",
    frames=("int64_no_nulls", "float64_half_null"),
    expr=lambda pd, df: df[["value"]].mask(df[["value"]] > 0),
    in_process=True,
    note="the frame's mask on the same columns, which widens column by column. Python layer",
)
case(
    "basics/shift-flags",
    "Series.shift",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: repr((df["value"] > 0).head(4).shift(1)),
    in_process=True,
    note="a column of flags shifted along, which pandas makes object so the opened row "
    "can hold NaN, compared as printed. Python layer",
)
case(
    "basics/dtypes-printed",
    "DataFrame.dtypes",
    frames=("int64_no_nulls", "float64_half_null"),
    expr=lambda pd, df: repr(df.dtypes),
    in_process=True,
    note="the types as printed, where pandas' answer is an object column and so ends in "
    "`dtype: object`. Python layer",
)
case(
    "basics/frame-where-leaves-missing",
    "DataFrame.where",
    frames=("float64_no_nulls", "float64_half_null"),
    expr=lambda pd, df: df.where(df["value"] > 0).isna(),
    in_process=True,
    note="both columns are chosen over by one condition and the positions column is "
    "the one pandas widens, since it holds whole numbers and is about to hold a nan. "
    "Python layer, see spec 48",
)
case(
    "basics/frame-mask-leaves-missing",
    "DataFrame.mask",
    frames=("float64_no_nulls",),
    expr=lambda pd, df: df.mask(df["value"] > 0).isna(),
    in_process=True,
    note="the frame shaped mask with nothing but a condition. Python layer, see spec 48",
)
case(
    "basics/mask-nulls",
    "Series.mask",
    level="L3",
    covers=("cond", "other"),
    frames=FLOATS,
    expr=lambda pd, df: df["value"].mask(df["value"].isna(), 0.0),
    in_process=True,
    note="fillna written the other way round, which is worth a case because it is the "
    "one condition that picks out exactly the rows that hold nothing and because it "
    "fills every one of them. Python layer, see spec 48",
)
case(
    "basics/frame-where",
    "DataFrame.where",
    level="L3",
    covers=("cond", "other"),
    frames=FLOATS,
    expr=lambda pd, df: df.where(df["value"] > 0, 0),
    in_process=True,
    note="one column of flags against a frame is read down the rows and is shared by "
    "every column, so the positions column is chosen over by a condition computed from "
    "the value column. The other side is a whole number rather than 0.0 because the "
    "two columns are of two types and a whole number goes into both of them without "
    "either library widening anything. Python layer, see spec 48",
)
case(
    "basics/frame-mask",
    "DataFrame.mask",
    level="L3",
    covers=("cond", "other"),
    frames=FLOATS,
    expr=lambda pd, df: df.mask(df["value"].isna(), 0),
    in_process=True,
    note="the rows where one column holds nothing, replaced in every column, which is "
    "the frame shaped version of the condition that picks out the gaps. Python layer, "
    "see spec 48",
)
case(
    "basics/clip-float-edges",
    "Series.clip",
    frames=("float64_no_nulls",),
    expr=lambda pd, df: df["value"].clip(1e15, 8e15),
    in_process=True,
    note="a floor and a ceiling on the frame that carries the float edges, so the two "
    "infinities are caught by the bounds and the nan is left alone by both libraries "
    "without anybody writing a rule for it, which is the half of this method worth "
    "measuring. The bounds are floats because a fractional bound on whole numbers is "
    "the widening divergence rather than the method. Python layer, see spec 49",
)
case(
    "basics/clip-whole-numbers",
    "Series.clip",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: df["value"].clip(-4_611_686_017_000_000_000),
    in_process=True,
    note="a floor written as a whole number against whole numbers, which is the shape "
    "that does not widen in either library and so is the one where the values can be "
    "compared rather than the positions. The bound is inside the range the frame holds, "
    "so some rows are lifted and some are not. Python layer, see spec 49",
)
case(
    "basics/clip-ceiling",
    "Series.clip",
    level="L3",
    covers=("upper",),
    frames=("float64_no_nulls",),
    expr=lambda pd, df: df["value"].clip(upper=4e15),
    in_process=True,
    note="the ceiling on its own, which is the other half of the signature and is also "
    "the call that says an absent floor is not a floor of zero. Python layer, see "
    "spec 49",
)
case(
    "basics/clip-column",
    "Series.clip",
    level="L3",
    covers=("lower", "upper"),
    frames=("float64_no_nulls",),
    expr=lambda pd, df: df["value"].clip(df["row"] * 1e14, df["row"] * 1e15),
    in_process=True,
    note="both bounds carrying rows of their own and lined up by label, which is a "
    "different pair of bounds for every row and is the shape spec 49 section 4 spends "
    "the most code on. The two are built from the positions so they are in order "
    "everywhere and neither of them holds a gap. Python layer, see spec 49",
)
case(
    "basics/frame-clip",
    "DataFrame.clip",
    frames=("float64_no_nulls",),
    expr=lambda pd, df: df.clip(0.0),
    in_process=True,
    note="one bound for every column of the frame, where the positions column is never "
    "reached because no position is below zero. Both libraries hand that column back "
    "with its type, which is what keeps a fractional bound from being an error on a "
    "column of whole numbers it does not touch. Python layer, see spec 49",
)
case(
    "basics/frame-clip-per-column",
    "DataFrame.clip",
    level="L3",
    covers=("lower", "upper"),
    frames=("float64_no_nulls",),
    expr=lambda pd, df: df.clip([0, 1e15], [40, 8e15]),
    in_process=True,
    note="a run of values against a frame, which is one bound per column rather than "
    "one per row, so the positions get whole numbers and the values get floats and "
    "neither column is offered a bound of the other's type. Python layer, see spec 49",
)
case(
    "basics/frame-clip-down-the-rows",
    "DataFrame.clip",
    level="L3",
    covers=("lower", "axis"),
    frames=("float64_no_nulls",),
    expr=lambda pd, df: df.clip(df["row"], axis=0),
    in_process=True,
    note="a column of bounds read down the rows and shared by every column, which is "
    "the reading that needs an axis because the other reading is a bound per column and "
    "pandas will not guess between them. The positions column is bounded by itself and "
    "so nothing in it moves. Python layer, see spec 49",
)
case(
    "basics/replace",
    "Series.replace",
    level="L3",
    covers=("to_replace", "value"),
    frames=("keys_10",),
    expr=lambda pd, df: df["key"].replace(0, 99),
    in_process=True,
    note="one value swapped for another, which is the whole method in its smallest "
    "shape, and the replacement is a whole number so nothing here asks either library "
    "to widen a column. This case was written before the method existed and its frames "
    "are the ones it was written with. Python layer, see spec 50",
)
case(
    "basics/replace-run",
    "Series.replace",
    level="L3",
    covers=("to_replace", "value"),
    frames=("keys_10",),
    expr=lambda pd, df: df["key"].replace([0, 1, 2], [7, 8, 9]),
    in_process=True,
    note="a run of values against a run of replacements, which is three comparisons and "
    "three picks and is the shape that shows the pairs are judged against the column as "
    "it arrived. Every value here is one the column already holds, so nothing is a no "
    "op. Python layer, see spec 50",
)
case(
    "basics/replace-swap",
    "Series.replace",
    frames=("keys_10",),
    expr=lambda pd, df: df["key"].replace([3, 4], [4, 3]),
    in_process=True,
    note="two values swapped for each other in one call, which is only the right answer "
    "because no pair can see what the pair before it did. A method that walked the pairs "
    "against the running answer would send every three and every four to three. Python "
    "layer, see spec 50",
)
case(
    "basics/replace-mapping",
    "Series.replace",
    frames=("keys_10",),
    expr=lambda pd, df: df["key"].replace({0: 90, 5: 95}),
    in_process=True,
    note="a mapping of pairs on a column, where the keys are the values being replaced "
    "rather than anything to do with labels, which is the reading a frame only takes "
    "when nothing arrives beside it. Python layer, see spec 50",
)
case(
    "basics/replace-missing",
    "Series.replace",
    frames=("float64_half_null",),
    expr=lambda pd, df: df["value"].replace(float("nan"), 0.0),
    in_process=True,
    note="a missing value named for replacement, which is isna rather than a comparison "
    "because nothing equals a missing value, and so is the same answer as fillna. The "
    "frame is the one carrying gaps, so the rows that move are the ones that hold "
    "nothing. Python layer, see spec 50",
)
case(
    "basics/replace-nothing-holds",
    "Series.replace",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: df["value"].replace(0, 99),
    in_process=True,
    note="a value of the right type that no row holds, since every value in this frame "
    "is negative, so the answer is the column and the pair is dropped before any pick is "
    "built. Python layer, see spec 50",
)
case(
    "basics/frame-replace",
    "DataFrame.replace",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: df.replace(0, 99),
    in_process=True,
    note="one pair against every column of the frame, which reaches the first position "
    "and nothing in the values, so one column moves and one is handed back with its "
    "type. Python layer, see spec 50",
)
case(
    "basics/frame-replace-per-column",
    "DataFrame.replace",
    level="L3",
    covers=("to_replace", "value"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: df.replace({"row": 0}, 99),
    in_process=True,
    note="a mapping with a value beside it, which is the reading where the keys are "
    "column names, so only the positions column is offered the pair. The same mapping "
    "without the value would be a mapping of values and would do nothing at all, which "
    "is spec 50 section 5. Python layer, see spec 50",
)
case(
    "basics/frame-replace-mapping-of-mappings",
    "DataFrame.replace",
    level="L3",
    covers=("to_replace",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: df.replace({"row": {0: 90, 1: 91}}),
    in_process=True,
    note="a mapping of column names to mappings of pairs, which is the one shape read by "
    "column name with nothing beside it, decided on the whole mapping rather than per "
    "entry. Python layer, see spec 50",
)
case(
    "basics/frame-replace-value-per-column",
    "DataFrame.replace",
    level="L3",
    covers=("to_replace", "value"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: df.replace(0, {"row": 99}),
    in_process=True,
    note="a mapping as the value, which is a different replacement per column for the "
    "same thing replaced, and a column the mapping does not name is left alone rather "
    "than refused. Python layer, see spec 50",
)
case(
    "basics/isin",
    "Series.isin",
    level="L3",
    covers=("values",),
    frames=("keys_10", "keys_1000"),
    expr=lambda pd, df: df["key"].isin([0, 1, 2]),
)
case(
    "basics/isin-mixed",
    "Series.isin",
    level="L3",
    covers=("values",),
    frames=("keys_10",),
    expr=lambda pd, df: df["key"].isin([1, "b", 9.5, None, True]),
    in_process=True,
    note="a set holding one value of the column's kind and four that are not, which is "
    "the shape a set written by hand actually has. pandas compares by value and never "
    "refuses, and firepanda's kernel compares one type against one type and always "
    "refuses, so the whole of this case is the layer between them. Python layer, see "
    "spec 55 sections 3 and 4",
)
case(
    "basics/isin-nulls",
    "Series.isin",
    level="L3",
    covers=("values",),
    frames=("strings_null_heavy",),
    expr=lambda pd, df: df["value"].isin(["v1", None]),
    in_process=True,
    note="a missing row is in the set when the set holds a missing value and false when "
    "it does not, and pandas decides which missing value counts from the column's dtype "
    "where firepanda has one null for every dtype and decides it from the set. Python "
    "layer, see spec 55 section 5",
)
case(
    "basics/isin-empty",
    "Series.isin",
    level="L3",
    covers=("values",),
    frames=("keys_10",),
    expr=lambda pd, df: df["key"].isin([]),
    in_process=True,
    note="a set with nothing in it is false everywhere rather than an error, and it is "
    "worth its own case because a list with nothing in it has no type to build a column "
    "from and the obvious implementation raises there. Python layer, see spec 55",
)
case(
    "basics/frame-isin",
    "DataFrame.isin",
    level="L3",
    covers=("values",),
    frames=("keys_10", "two"),
    expr=lambda pd, df: df.isin([0, 1, "one"]),
    in_process=True,
    note="one set asked of every column, where each column finds its own half of it and "
    "a column that can hold none of it answers false all the way down. The answer is put "
    "back together through the frame constructor, since there is no other way to make a "
    "frame here. Python layer, see spec 55 section 8",
)
case(
    "basics/frame-isin-mapping",
    "DataFrame.isin",
    level="L3",
    covers=("values",),
    frames=("two",),
    expr=lambda pd, df: df.isin({"a": [1], "c": ["one"]}),
    in_process=True,
    note="a set per column, where the column the mapping does not name answers false all "
    "the way down rather than being left out of the answer. Python layer, see spec 55 "
    "section 8",
)
case(
    "basics/between",
    "Series.between",
    level="L3",
    covers=("left", "right"),
    frames=("float64_no_nulls", "keys_1000"),
    expr=lambda pd, df: df.iloc[:, 1].between(0, 100),
    in_process=True,
    note="In process because firepanda writes it as two comparisons and an and in its "
    "Python layer, which the driver cannot reach",
)
case(
    "basics/between-inclusive",
    "Series.between",
    level="L3",
    covers=("left", "right", "inclusive"),
    frames=("float64_half_null", "keys_1000"),
    expr=lambda pd, df: df.iloc[:, 1].between(10, 500, inclusive="neither"),
    in_process=True,
    note="both ends open, and a missing value is False. In process because firepanda "
    "writes it as two comparisons and an and in its Python layer, which the driver "
    "cannot reach",
)
case(
    "basics/unique",
    "Series.unique",
    frames=("keys_10", "keys_awkward", "strings_null_heavy"),
    expr=lambda pd, df: df.iloc[:, 0].unique(),
    in_process=True,
    note="unique keeps first seen order, which is the part people get wrong. In process "
    "because firepanda #1101 writes it as a duplicated mask in its Python layer",
)
case(
    "basics/value-counts",
    "Series.value_counts",
    frames=("keys_10", "keys_awkward", "strings_null_heavy"),
    expr=lambda pd, df: df.iloc[:, 0].value_counts(),
    rules=Rules(
        relaxations=frozenset({"row_order"}),
        reason="the counts are sorted by count and ten keys over sixty four rows means "
        "ties, and pandas does not promise how it breaks them",
    ),
    note=VC_NOTE,
    in_process=True,
)
case(
    "basics/value-counts-dropna-false",
    "Series.value_counts",
    level="L3",
    covers=("dropna",),
    frames=("keys_awkward", "strings_null_heavy"),
    expr=lambda pd, df: df.iloc[:, 0].value_counts(dropna=False),
    rules=Rules(
        relaxations=frozenset({"row_order"}),
        reason="same tie breaking as the case above",
    ),
    note=VC_NOTE,
    in_process=True,
)
case(
    "basics/value-counts-normalize",
    "Series.value_counts",
    level="L3",
    covers=("normalize",),
    frames=("keys_10",),
    expr=lambda pd, df: df["key"].value_counts(normalize=True),
    rules=Rules(
        relaxations=frozenset({"row_order"}),
        tolerance=Tolerance.SINGLE,
        reason="same tie breaking as the case above",
    ),
    note=VC_NOTE,
    in_process=True,
)
case(
    "basics/value-counts-ascending",
    "Series.value_counts",
    level="L3",
    covers=("sort", "ascending"),
    frames=("keys_10", "strings_null_heavy"),
    expr=lambda pd, df: df.iloc[:, 0].value_counts(ascending=True),
    rules=Rules(
        relaxations=frozenset({"row_order"}),
        reason="same tie breaking as the case above",
    ),
    note="the least common first. " + VC_NOTE,
    in_process=True,
)
case(
    "basics/value-counts-unsorted",
    "Series.value_counts",
    level="L3",
    covers=("sort",),
    frames=("keys_10", "strings_null_heavy"),
    expr=lambda pd, df: df.iloc[:, 0].value_counts(sort=False),
    note="in the order each value first appears, which is the order of pandas' hash "
    "table and has no ties to break, so the rows are compared in order. " + VC_NOTE,
    in_process=True,
)


# ---------------------------------------------------------------------------
# Automatic index alignment
# ---------------------------------------------------------------------------

# These were `divergences/alignment/*` until firepanda started aligning, and they are
# here rather than deleted because the behaviour they describe is worth checking now
# that both engines are supposed to do it. Two operands that share no labels give the
# union of both, filled with nulls, and an answer longer than either input.
#
# The two that run still fail, but not on the index. One reports nulls where pandas has
# NaN and the other keeps an integer column where pandas widens to double, which are
# the two open dtype questions and not anything to do with alignment. The shape and the
# labels of the answer already match.
#
# `head` and `tail` do the splitting rather than `iloc`, which is what the old versions
# used. That is not a stylistic change: `iloc` does not exist in firepanda yet, so the
# expression raised `AttributeError` before it reached any arithmetic and the case
# could not have told alignment from absence.
#
# None of these expressions has a lambda inside it, which is also not style. The
# unimplemented rule counts traceback frames below the case expression, so an inner
# lambda puts an absent name one frame too deep and the case is scored as a bug rather
# than as a gap. `align` does not exist yet and was being reported as a failure until
# the inner lambda came out.


def _top(df):
    """The first half of a frame."""
    return df.head(len(df) // 2)


def _bottom(df):
    """The second half, which shares no index label with the first."""
    return df.tail(len(df) - len(df) // 2)


case(
    "basics/alignment-series-add",
    "Series.add",
    frames=("tall", "float64_no_nulls"),
    expr=lambda pd, df: _top(df)["value"] + _bottom(df)["value"],
    note="two halves of one column added together. Every value in the answer is NaN, "
    "because no label appears in both, and the answer is twice as long as either "
    "operand. A user who meant to add them elementwise gets no error at all. In process "
    "because firepanda turns the gaps into NaN in its Python layer, which the driver "
    "cannot reach",
    in_process=True,
)
case(
    "basics/alignment-frame-add",
    "DataFrame.add",
    frames=("float64_no_nulls",),
    expr=lambda pd, df: _top(df) + _bottom(df),
    in_process=True,
    note="two halves of one frame added together, so every row is a gap and the integer "
    "row column widens to float64 to hold a NaN in each. In process because firepanda "
    "turns the gaps into NaN in its Python layer, which the driver cannot reach",
)
case(
    "basics/alignment-subtract-shifted",
    "Series.sub",
    frames=("tall",),
    expr=lambda pd, df: df["value"] - df["value"].shift(1),
    note="the one alignment that people rely on and that reads correctly, which is why "
    "it was the expensive part of the decision to leave it out",
)
case(
    "basics/series-from-mapping",
    "pandas.Series",
    frames=("two",),
    expr=lambda pd, df: pd.Series({"b": 2.5, "a": 1.5, "c": 0.5}, name="v"),
    in_process=True,
    note="the keys are the labels in the mapping's order and not sorted. In process because "
    "the driver builds frames and has no entry for a series made from literals",
)
case(
    "basics/series-from-mapping-picked",
    "pandas.Series",
    frames=("two",),
    expr=lambda pd, df: pd.Series({"a": 1, "b": 2}, index=["b", "z"]),
    in_process=True,
    note="index= beside a mapping picks keys out in its own order, and the key the mapping "
    "does not have makes the integers float64. In process for the same reason",
)


# A frame or a series built from literals. Every one is in process because the driver
# builds its frames from the corpus and has no entry for one made from literals.
BUILT = "In process because the driver has no entry for a frame or series made from literals"


def _numpy() -> object:
    """numpy, imported when a case runs rather than when the corpus loads."""
    import numpy

    return numpy


case(
    "basics/frame-from-records",
    "pandas.DataFrame",
    frames=("two",),
    expr=lambda pd, df: pd.DataFrame([{"a": 1, "b": "x"}, {"b": "y", "c": 2.5}]),
    in_process=True,
    note="the keys of every record are the columns in the order they are first seen, and "
    "the integer column with a record missing it widens to float64. " + BUILT,
)
case(
    "basics/frame-from-rows",
    "pandas.DataFrame",
    level="L3",
    covers=("columns", "index"),
    frames=("two",),
    expr=lambda pd, df: pd.DataFrame([(1, "x"), (2, "y")], columns=["a", "b"], index=[5, 6]),
    in_process=True,
    note="rows name no columns, so columns= names them, and index= labels the rows. " + BUILT,
)
case(
    "basics/frame-from-array",
    "pandas.DataFrame",
    level="L3",
    covers=("columns", "index"),
    frames=("two",),
    expr=lambda pd, df: pd.DataFrame(
        _numpy().arange(6, dtype="int32").reshape(3, 2), columns=["a", "b"], index=["x", "y", "z"]
    ),
    in_process=True,
    note="a two dimensional array is read column by column and keeps its int32. " + BUILT,
)
case(
    "basics/frame-scalar-broadcast",
    "pandas.DataFrame",
    level="L3",
    covers=("index",),
    frames=("two",),
    expr=lambda pd, df: pd.DataFrame({"a": 1, "b": "s"}, index=[5, 6]),
    in_process=True,
    note="a single value in the mapping is repeated down the rows index= names. " + BUILT,
)
case(
    "basics/frame-aligned-series",
    "pandas.DataFrame",
    frames=("two",),
    expr=lambda pd, df: pd.DataFrame(
        {
            "a": pd.Series([1, 2], index=["x", "y"]),
            "b": pd.Series([3, 4], index=["z", "y"]),
        }
    ),
    in_process=True,
    note="series in a mapping are lined up on the sorted union of their labels, and a "
    "label only one of them has leaves a gap in the other. " + BUILT,
)
case(
    "basics/frame-picked-columns",
    "pandas.DataFrame",
    level="L3",
    covers=("columns",),
    frames=("two",),
    expr=lambda pd, df: pd.DataFrame({"a": [1, 2], "b": [3, 4]}, columns=["b", "a"]),
    in_process=True,
    note="columns= beside a mapping picks the keys out in its own order. " + BUILT,
)
case(
    "basics/series-scalar-index",
    "pandas.Series",
    level="L3",
    covers=("index",),
    frames=("two",),
    expr=lambda pd, df: pd.Series(5.5, index=["a", "b"], name="n"),
    in_process=True,
    note="one value is repeated along the index given. " + BUILT,
)
case(
    "basics/series-reindexed",
    "pandas.Series",
    level="L3",
    covers=("index",),
    frames=("two",),
    expr=lambda pd, df: pd.Series(pd.Series([1, 2], index=["a", "b"]), index=["b", "c"]),
    in_process=True,
    note="a series with index= is a reindex, so the label it does not have makes the "
    "integers float64. " + BUILT,
)
case(
    "basics/series-from-array",
    "pandas.Series",
    frames=("two",),
    expr=lambda pd, df: pd.Series(_numpy().array([1, 2, 250], dtype="uint8")),
    in_process=True,
    note="an array keeps its own type rather than the one the values would be read as. " + BUILT,
)
case(
    "basics/alignment-power-gap",
    "Series.pow",
    frames=("two",),
    expr=lambda pd, df: pd.Series([1, 2, 3]).iloc[:2] ** pd.Series([0, 1, 2]).iloc[1:],
    in_process=True,
    note="a power across a gap, where the first row is only on the left and is 1, so "
    "numpy's `1 ** nan` answers 1 rather than NaN. In process because firepanda takes "
    "the power again over NaN in its Python layer, which the driver cannot reach",
)
case(
    "basics/alignment-power-gap-frame",
    "DataFrame.pow",
    frames=("two",),
    expr=lambda pd, df: (
        pd.DataFrame({"x": [1, 2], "y": [1, 4]})
        ** pd.DataFrame({"y": [0, 2], "z": [0, 6]}).iloc[1:]
    ),
    in_process=True,
    note="the frame form, where a 1 on the left and a 0 on the right each answer 1 "
    "across a gap. In process because firepanda takes the power again over NaN in its "
    "Python layer, which the driver cannot reach",
)
LOGICAL_IN_PROCESS = (
    "In process because firepanda answers the three logical operators in its Python "
    "layer, which the driver cannot reach"
)
case(
    "basics/logical-and-mask",
    "Series.__and__",
    frames=("tall",),
    expr=lambda pd, df: (df["key"] > 50) & df["flag"],
    in_process=True,
    note="a mask built from a comparison and a boolean column, which is what `&` is "
    "mostly written for. " + LOGICAL_IN_PROCESS,
)
case(
    "basics/logical-or-mask",
    "Series.__or__",
    frames=("tall",),
    expr=lambda pd, df: (df["key"] < 10) | df["flag"],
    in_process=True,
    note="the same with `|`. " + LOGICAL_IN_PROCESS,
)
case(
    "basics/logical-xor-scalar",
    "Series.__xor__",
    frames=("tall",),
    expr=lambda pd, df: True ^ df["flag"],
    in_process=True,
    note="a constant on the left, which Python sends to the reflected form, and which "
    "flips every row. " + LOGICAL_IN_PROCESS,
)
case(
    "basics/logical-or-gap",
    "Series.__or__",
    frames=("two",),
    expr=lambda pd, df: pd.Series([True, False, True]) | pd.Series([True, False]).iloc[1:],
    in_process=True,
    note="a row only the left side has, where pandas reads the right side as False. "
    + LOGICAL_IN_PROCESS,
)
case(
    "basics/logical-or-gap-reflected",
    "Series.__or__",
    frames=("two",),
    expr=lambda pd, df: pd.Series([True, False]).iloc[1:] | pd.Series([True, False, True]),
    in_process=True,
    note="the same two operands the other way round, where the row only the right side "
    "has answers False even though it is True there, so `a | b` and `b | a` differ. "
    + LOGICAL_IN_PROCESS,
)
case(
    "basics/logical-and-frame-gap",
    "DataFrame.__and__",
    frames=("two",),
    expr=lambda pd, df: (
        pd.DataFrame({"a": [True, False, True], "b": [True, True, False]})
        & pd.DataFrame({"a": [True, True, True], "c": [False, True, True]}).iloc[1:]
    ),
    in_process=True,
    note="two frames with a row and a column each only one side has, where the column "
    "is all NaN and the row is False. " + LOGICAL_IN_PROCESS,
)
case(
    "basics/logical-xor-frame-series",
    "DataFrame.__xor__",
    frames=("two",),
    expr=lambda pd, df: (
        pd.DataFrame({"b": [True, False], "a": [False, False]})
        ^ pd.DataFrame({"k": ["a", "c"], "v": [True, True]}).set_index("k")["v"]
    ),
    in_process=True,
    note="a series lined up against the columns, where the union of the labels comes "
    "back sorted and a column the frame lacks is all False. " + LOGICAL_IN_PROCESS,
)
case(
    "basics/alignment-align",
    "DataFrame.align",
    level="L3",
    covers=("other", "join"),
    frames=("float64_no_nulls",),
    expr=lambda pd, df: _top(df).align(_bottom(df), join="outer")[0],
    in_process=True,
    note="align is a Python method on the frame, answered in process",
)


# ---------------------------------------------------------------------------
# The series members that read as a frame
# ---------------------------------------------------------------------------

AS_A_FRAME = (
    "the series members that read as a frame are firepanda's Python layer on top of "
    "one door, and the core has no call for any of them, so a driver entry would have "
    "to build the frame of one column and then write the method itself. See spec 36"
)

case(
    "basics/series-duplicated",
    "Series.duplicated",
    frames=("keys_10",),
    expr=lambda pd, df: df["key"].duplicated(),
    in_process=True,
    note=AS_A_FRAME,
)
case(
    "basics/series-duplicated-keep",
    "Series.duplicated",
    level="L3",
    covers=("keep",),
    frames=("keys_10",),
    expr=lambda pd, df: df["key"].duplicated(keep="last"),
    in_process=True,
    note="ten distinct keys over ten thousand rows, so which end the rule keeps decides "
    "almost every row of this answer. " + AS_A_FRAME,
)
case(
    "basics/series-drop-duplicates",
    "Series.drop_duplicates",
    frames=("keys_10",),
    expr=lambda pd, df: df["key"].drop_duplicates(),
    in_process=True,
    note=AS_A_FRAME,
)
case(
    "basics/series-drop-duplicates-keep",
    "Series.drop_duplicates",
    level="L3",
    covers=("keep", "ignore_index"),
    frames=("keys_10",),
    expr=lambda pd, df: df["key"].drop_duplicates(keep="last", ignore_index=True),
    in_process=True,
    note="the labels a drop leaves behind are the positions the rows held before it, so "
    "numbering them again is the parameter worth covering beside the rule. " + AS_A_FRAME,
)

# ---------------------------------------------------------------------------
# Walking a frame and a column
#
# A frame walks its column names and a column walks its values, which is the
# one place the two classes deliberately disagree, and `in` follows the same
# split so it never looks at a value on a column. All of this is Python
# protocol rather than kernel, so every case is in process: the driver's
# boundary carries a frame, a column or a scalar, and a list of names, a run of
# pairs or a run of tuples is none of those. See spec 56.
# ---------------------------------------------------------------------------

WALKING = "Python protocol rather than kernel, see spec 56"

case(
    "basics/iter",
    "DataFrame.__iter__",
    frames=("empty", "single", "wide"),
    expr=lambda pd, df: list(df),
    in_process=True,
    note="the column names and not the rows, which reads like a mistake until you write "
    "the loop it was chosen for, which is a name followed by the column it names. The "
    "wide frame is here because a name at a time answer is where an order bug shows. " + WALKING,
)
case(
    "basics/iter-values",
    "Series.__iter__",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: list(df["value"]),
    in_process=True,
    note="the values and not the labels, which is the other half of the split above. "
    "The frame is free of gaps on purpose, because a missing row comes out as None "
    "here and as a nan in pandas, which would measure how the two libraries spell "
    "nothing rather than how they walk a column. " + WALKING,
)
case(
    "basics/contains",
    "DataFrame.__contains__",
    frames=("single", "two"),
    expr=lambda pd, df: [name in df for name in ("a", "c", "zz", 1, None)],
    in_process=True,
    note="a name, another name, a name that is not there, and two keys of a kind no "
    "column name could be, all of which are answered rather than refused because `in` "
    "is a question. " + WALKING,
)
case(
    "basics/contains-label",
    "Series.__contains__",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: [key in df["value"] for key in (0, 63, 64, -1, "zz")],
    in_process=True,
    note="the labels and never the values, which is the rule that surprises everybody. "
    "The frame's labels are its positions, so the first two are in and the third is one "
    "past the end. " + WALKING,
)
case(
    "basics/keys",
    "DataFrame.keys",
    frames=("single", "wide"),
    expr=lambda pd, df: list(df.keys()),
    in_process=True,
    note="the same answer as columns through a second name, wrapped in list because "
    "pandas hands back an Index there and firepanda hands back a list, which is the "
    "divergence columns already carries. " + WALKING,
)
case(
    "basics/keys-labels",
    "Series.keys",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: list(df["value"].keys()),
    in_process=True,
    note="the same answer as index through a second name. " + WALKING,
)
case(
    "basics/items",
    "DataFrame.items",
    frames=("empty", "single"),
    expr=lambda pd, df: [(name, held.tolist()) for name, held in df.items()],
    in_process=True,
    note="a name and its column at a time, lazily, because a frame may be a thousand "
    "columns wide and a caller who breaks out of the loop should not have paid for the "
    "rest. " + WALKING,
)
case(
    "basics/items-pairs",
    "Series.items",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: list(df["value"].items()),
    in_process=True,
    note="a label and a value at a time, which is the loop neither iterating nor asking "
    "in gives on its own. " + WALKING,
)
case(
    "basics/itertuples",
    "DataFrame.itertuples",
    frames=("empty", "single"),
    expr=lambda pd, df: [tuple(row) for row in df.itertuples()],
    in_process=True,
    note="the fast way to walk rows in pandas, and the tuples are compared as tuples "
    "rather than by type, because the type is built per call in both libraries and two "
    "of them are never the same object. Neither this nor the plain form runs on a frame "
    "with a gap in it, because a missing cell comes out as None here and as a nan in "
    "pandas. " + WALKING,
)
case(
    "basics/itertuples-fields",
    "DataFrame.itertuples",
    frames=("single", "two"),
    expr=lambda pd, df: list(next(iter(df.itertuples()))._fields),
    in_process=True,
    note="the field names, which are the column names with the row label in front under "
    "the name Index, and neither library writes that rule down. " + WALKING,
)
case(
    "basics/itertuples-plain",
    "DataFrame.itertuples",
    level="L3",
    covers=("index", "name"),
    frames=("empty", "single"),
    expr=lambda pd, df: list(df.itertuples(index=False, name=None)),
    in_process=True,
    note="both parameters at once, which turns the answer into plain tuples of the "
    "values with no label in front. " + WALKING,
)
TO_NUMPY_IN_PROCESS = (
    "In process because firepanda builds the numpy array in its Python layer, which the "
    "driver cannot reach. The case answers the numpy type and the values as a list, so "
    "the type is checked as well as what is in the array"
)
case(
    "basics/to-numpy",
    "Series.to_numpy",
    frames=("int64_no_nulls", "float64_half_null"),
    expr=lambda pd, df: [str(df["value"].to_numpy().dtype), df["value"].to_numpy().tolist()],
    in_process=True,
    note="whole numbers stay whole and floats keep NaN in a gap. " + TO_NUMPY_IN_PROCESS,
)
case(
    "basics/to-numpy-fill",
    "Series.to_numpy",
    covers=("na_value",),
    frames=("float64_half_null",),
    expr=lambda pd, df: df["value"].to_numpy(na_value=0.0).tolist(),
    in_process=True,
    note="a gap answered as the value given. " + TO_NUMPY_IN_PROCESS,
)
case(
    "basics/frame-to-numpy",
    "DataFrame.to_numpy",
    frames=("keys_10",),
    expr=lambda pd, df: [str(df.to_numpy().dtype), df.to_numpy().tolist()],
    in_process=True,
    note="a row per row in the one type the columns share, objects when text is one of "
    "them. " + TO_NUMPY_IN_PROCESS,
)
MAP_IN_PROCESS = (
    "In process because the function is Python and firepanda calls it on each value in "
    "its Python layer, as pandas does, which the driver cannot reach"
)
case(
    "basics/map-function",
    "Series.map",
    frames=("int64_no_nulls", "float64_half_null"),
    expr=lambda pd, df: df["value"].head(6).map(lambda v: v * 2),
    in_process=True,
    note="a function on each value, a gap reaching it as NaN. " + MAP_IN_PROCESS,
)
case(
    "basics/map-dict",
    "Series.map",
    covers=("func",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: df["value"].head(6).map({0: "zero", 1: "one"}),
    in_process=True,
    note="a mapping, a key it lacks answered as missing. " + MAP_IN_PROCESS,
)
case(
    "basics/map-ignore",
    "Series.map",
    covers=("na_action",),
    frames=("float64_half_null",),
    expr=lambda pd, df: df["value"].head(6).map(lambda v: v + 1, na_action="ignore"),
    in_process=True,
    note="a gap left alone rather than handed to the function. " + MAP_IN_PROCESS,
)
case(
    "basics/apply-function",
    "Series.apply",
    covers=("args",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: df["value"].head(6).apply(lambda v, k: v - k, args=(1,)),
    in_process=True,
    note="a function on each value with an extra argument. " + MAP_IN_PROCESS,
)
case(
    "basics/combine",
    "Series.combine",
    covers=("fill_value",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: df["value"].head(4).combine(df["value"].tail(3), max, fill_value=0),
    in_process=True,
    note="a function on each pair over both columns' labels, a label one side lacks read "
    "as the fill. " + MAP_IN_PROCESS,
)
case(
    "basics/frame-combine",
    "DataFrame.combine",
    covers=("fill_value",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: df.head(4).combine(
        df.tail(3), lambda a, b: a.where(a > b, b), fill_value=0
    ),
    in_process=True,
    note="a function on each pair of columns over both frames' labels, a label one side "
    "lacks read as the fill, and floats that are all whole cast back. " + MAP_IN_PROCESS,
)
case(
    "basics/convert-dtypes",
    "DataFrame.convert_dtypes",
    frames=("float64_no_nulls", "keys_10"),
    expr=lambda pd, df: df.convert_dtypes(),
    in_process=True,
    note="In process because the driver has no entry for it. pandas moves each column "
    "to its nullable type and every column here already holds a gap, so the one change "
    "is a float column of whole numbers becoming int64",
)
case(
    "basics/frame-map",
    "DataFrame.map",
    frames=("keys_10",),
    expr=lambda pd, df: df.map(lambda v: str(v)[:1]),
    in_process=True,
    note="a function on each value of each column. " + MAP_IN_PROCESS,
)
case(
    "basics/frame-apply",
    "DataFrame.apply",
    frames=("int64_no_nulls", "float64_half_null"),
    expr=lambda pd, df: df.apply(lambda c: c.max() - c.min()),
    in_process=True,
    note="a function on each column answering one value, gathered into a column. " + MAP_IN_PROCESS,
)
case(
    "basics/frame-apply-rows",
    "DataFrame.apply",
    covers=("axis",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: df.head(6).apply(lambda r: r.sum(), axis=1),
    in_process=True,
    note="a function on each row, the rows handed over as columns. " + MAP_IN_PROCESS,
)
case(
    "basics/frame-agg-list",
    "DataFrame.agg",
    covers=("func",),
    frames=("int64_no_nulls", "float64_half_null"),
    expr=lambda pd, df: df.agg(["sum", "min"]),
    in_process=True,
    note="a row per function, labelled by name. " + MAP_IN_PROCESS,
)
case(
    "basics/series-agg-list",
    "Series.agg",
    covers=("func",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: df["value"].agg(["sum", "max"]),
    in_process=True,
    note="a column of the reductions, labelled by name. " + MAP_IN_PROCESS,
)
case(
    "basics/series-transform",
    "Series.transform",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: df["value"].transform(lambda c: c - c.min()),
    in_process=True,
    note="a function answering a column of the same labels. " + MAP_IN_PROCESS,
)
case(
    "basics/frame-mode",
    "DataFrame.mode",
    frames=("keys_10",),
    expr=lambda pd, df: df.mode(),
    in_process=True,
    note="the most common values of each column, NaN after a column's last. " + MAP_IN_PROCESS,
)
XML_IN_PROCESS = "firepanda #1237 writes XML in the Python layer"
UPDATE_IN_PROCESS = (
    "In process because firepanda lines the other side up and puts the values in from "
    "its Python layer, which the driver cannot reach"
)


def _updated(df, other, **kwargs):
    """The frame after `update`, which answers None and changes the frame."""
    target = df.copy()
    target.update(other, **kwargs)
    return target


case(
    "basics/frame-update",
    "DataFrame.update",
    frames=("float64_half_null",),
    expr=lambda pd, df: _updated(df, df.fillna(-1.0) * 2),
    in_process=True,
    note="every value that is not missing on the other side put in. " + UPDATE_IN_PROCESS,
)
case(
    "basics/frame-update-keep",
    "DataFrame.update",
    covers=("overwrite",),
    frames=("float64_half_null",),
    expr=lambda pd, df: _updated(df, df.fillna(-1.0) * 2, overwrite=False),
    in_process=True,
    note="only the gaps filled, what is already there kept. " + UPDATE_IN_PROCESS,
)
case(
    "basics/from-records",
    "DataFrame.from_records",
    covers=("columns", "index"),
    frames=("single",),
    expr=lambda pd, df: pd.DataFrame.from_records(
        [(1, "a"), (2, "b")], columns=["n", "s"], index="s"
    ),
    in_process=True,
    note="tuples read against the names given, one column made the labels. " + UPDATE_IN_PROCESS,
)
case(
    "basics/series-filter",
    "Series.filter",
    covers=("like",),
    frames=("keys_awkward",),
    expr=lambda pd, df: df.set_index("key")["value"].filter(like="a"),
    in_process=True,
    note="the values whose labels hold the text. " + UPDATE_IN_PROCESS,
)
case(
    "basics/sample-seed",
    "Series.sample",
    covers=("random_state",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: df["value"].sample(5, random_state=42),
    in_process=True,
    note="the same seed draws the same rows, because both ask numpy for the positions. "
    + UPDATE_IN_PROCESS,
)
case(
    "basics/frame-sample-frac",
    "DataFrame.sample",
    covers=("frac", "random_state"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: df.sample(frac=0.01, random_state=7),
    in_process=True,
    note="a share of the rows drawn with a seed. " + UPDATE_IN_PROCESS,
)
case(
    "basics/case-when",
    "Series.case_when",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: df["value"].case_when([(df["value"] < 100, 0), (df["value"] > 900, 1000)]),
    in_process=True,
    note="the first condition that holds puts its value in. " + UPDATE_IN_PROCESS,
)
case(
    "basics/series-dot",
    "Series.dot",
    frames=("float64_no_nulls",),
    expr=lambda pd, df: float(df["value"].dot(df["value"])),
    in_process=True,
    note="the sum of the products, lined up by label. " + UPDATE_IN_PROCESS,
)
case(
    "basics/series-compare",
    "Series.compare",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: df["value"].compare(df["value"].where(df["value"] % 7 != 0, -1)),
    in_process=True,
    note="the values that differ, side by side. " + UPDATE_IN_PROCESS,
)
case(
    "basics/pivot-text-columns",
    "DataFrame.pivot",
    covers=("index", "columns", "values"),
    frames=("keys_two_column",),
    expr=lambda pd, df: df.drop_duplicates(subset=["left", "right"]).pivot(
        index="right", columns="left", values="value"
    ),
    in_process=True,
    note="the text key across so each column is named by text, which firepanda needs. "
    + UPDATE_IN_PROCESS,
)
case(
    "basics/pivot-table-text-columns",
    "DataFrame.pivot_table",
    covers=("index", "columns", "values", "aggfunc"),
    frames=("keys_two_column",),
    expr=lambda pd, df: df.pivot_table(
        index="right", columns="left", values="value", aggfunc="sum"
    ),
    in_process=True,
    note="the repeated pairs summed, with the text key across. " + UPDATE_IN_PROCESS,
)
case(
    "basics/to-numeric-text",
    "pandas.to_numeric",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.to_numeric(df["value"].astype(str)),
    in_process=True,
    note="whole numbers written as text read back as int64. " + UPDATE_IN_PROCESS,
)
case(
    "basics/to-numeric-coerce",
    "pandas.to_numeric",
    covers=("errors",),
    frames=("strings_ascii",),
    expr=lambda pd, df: pd.to_numeric(df["value"], errors="coerce"),
    in_process=True,
    note="text that is not a number read as NaN. " + UPDATE_IN_PROCESS,
)
case(
    "basics/to-numeric-downcast",
    "pandas.to_numeric",
    covers=("downcast",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.to_numeric(df["value"] % 100, downcast="integer"),
    in_process=True,
    note="the smallest whole type that holds every value. " + UPDATE_IN_PROCESS,
)


def _read_inside_context(pd):
    """`display.max_rows` read inside an `option_context` and again after it."""
    with pd.option_context("display.max_rows", 7, "max_colwidth", 20):
        inside = [pd.get_option("display.max_rows"), pd.options.display.max_colwidth]
    return pd.Series([*inside, pd.get_option("display.max_rows")])


case(
    "basics/get-option",
    "pandas.get_option",
    frames=("single",),
    expr=lambda pd, df: pd.Series(
        [
            pd.get_option("display.max_rows"),
            pd.get_option("display.precision"),
            pd.options.display.width,
        ]
    ),
    in_process=True,
    note="three defaults read by full name and by attribute. In process because "
    "the options are process wide",
)
case(
    "basics/option-context",
    "pandas.option_context",
    frames=("single",),
    expr=lambda pd, df: _read_inside_context(pd),
    in_process=True,
    note="two options set for a block and put back after it. In process because the options "
    "are process wide",
)
case(
    "basics/crosstab",
    "pandas.crosstab",
    covers=("margins",),
    frames=("keys_two_column",),
    expr=lambda pd, df: pd.crosstab(df["left"], df["right"].astype(str), margins=True),
    in_process=True,
    note="counts of each pair of keys with totals. " + UPDATE_IN_PROCESS,
)
case(
    "basics/crosstab-normalize",
    "pandas.crosstab",
    covers=("normalize",),
    frames=("keys_two_column",),
    expr=lambda pd, df: pd.crosstab(df["left"], df["right"].astype(str), normalize="index"),
    in_process=True,
    note="each row of counts over its total. " + UPDATE_IN_PROCESS,
)
case(
    "basics/from-dummies",
    "pandas.from_dummies",
    frames=("keys_two_column",),
    expr=lambda pd, df: pd.from_dummies(pd.get_dummies(df[["left"]]), sep="_"),
    in_process=True,
    note="get_dummies read back into the text it came from. " + UPDATE_IN_PROCESS,
)
case(
    "basics/lreshape",
    "pandas.lreshape",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.lreshape(df.assign(other=df["value"] * 2), {"both": ["value", "other"]}),
    in_process=True,
    note="two columns stacked into one. " + UPDATE_IN_PROCESS,
)
case(
    "basics/cut-codes",
    "pandas.cut",
    covers=("labels",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.cut(df["value"], 4, labels=False),
    in_process=True,
    note="the position of each value's bin among four even bins. " + UPDATE_IN_PROCESS,
)
case(
    "basics/cut-text-labels",
    "pandas.cut",
    covers=("labels",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.cut(df["value"], 3, labels=["low", "middle", "high"]).astype(str),
    in_process=True,
    note="each value's bin named by a label of its own. " + UPDATE_IN_PROCESS,
)
case(
    "basics/qcut-codes",
    "pandas.qcut",
    covers=("labels",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.qcut(df["value"], 4, labels=False, duplicates="drop"),
    in_process=True,
    note="the quartile each value falls in. " + UPDATE_IN_PROCESS,
)
case(
    "basics/to-csv",
    "DataFrame.to_csv",
    frames=("strings_ascii",),
    expr=lambda pd, df: df.to_csv(),
    in_process=True,
    note="the frame as comma separated text. " + UPDATE_IN_PROCESS,
)
case(
    "basics/to-csv-options",
    "DataFrame.to_csv",
    covers=("sep", "na_rep", "index", "float_format"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: df.assign(half=df["value"] / 2).to_csv(
        sep=";", na_rep="NA", index=False, float_format="%.2f"
    ),
    in_process=True,
    note="another separator, a missing marker, no labels and two float digits. "
    + UPDATE_IN_PROCESS,
)
case(
    "basics/series-to-csv",
    "Series.to_csv",
    frames=("strings_ascii",),
    expr=lambda pd, df: df["value"].to_csv(),
    in_process=True,
    note="one column as comma separated text. " + UPDATE_IN_PROCESS,
)
case(
    "basics/to-json",
    "DataFrame.to_json",
    frames=("float64_half_null",),
    expr=lambda pd, df: df.to_json(),
    in_process=True,
    note="the frame as JSON by column, floats in pandas' encoder digits and gaps as null. "
    + UPDATE_IN_PROCESS,
)
case(
    "basics/to-json-orients",
    "DataFrame.to_json",
    covers=("orient", "double_precision", "indent", "lines"),
    frames=("strings_ascii",),
    expr=lambda pd, df: (
        [
            df.assign(third=1 / 3).to_json(orient=orient, double_precision=4)
            for orient in ("index", "records", "split", "values")
        ]
        + [df.to_json(orient="records", lines=True), df.to_json(indent=2)]
    ),
    in_process=True,
    note="every orient but table, a shorter precision, one record per line and indenting. "
    + UPDATE_IN_PROCESS,
)
case(
    "basics/series-to-json",
    "Series.to_json",
    covers=("orient",),
    frames=("float64_no_nulls",),
    expr=lambda pd, df: [df["value"].to_json(), df["value"].to_json(orient="split")],
    in_process=True,
    note="one column as JSON by label and split into name, labels and data. " + UPDATE_IN_PROCESS,
)
case(
    "basics/read-json",
    "pandas.read_json",
    frames=("float64_half_null",),
    expr=lambda pd, df: pd.read_json(io.StringIO(df.to_json())),
    in_process=True,
    note="what to_json writes by column reads back with pandas' float decoder. "
    + UPDATE_IN_PROCESS,
)
case(
    "basics/read-json-split",
    "pandas.read_json",
    covers=("orient",),
    frames=("strings_ascii",),
    expr=lambda pd, df: pd.read_json(io.StringIO(df.to_json(orient="split")), orient="split"),
    in_process=True,
    note="the split orient to_json writes reads back with its labels. " + UPDATE_IN_PROCESS,
)
case(
    "basics/read-json-lines",
    "pandas.read_json",
    covers=("lines",),
    frames=("strings_ascii",),
    expr=lambda pd, df: pd.read_json(
        io.StringIO(df.to_json(orient="records", lines=True)), lines=True
    ),
    in_process=True,
    note="one record per line reads back. " + UPDATE_IN_PROCESS,
)
case(
    "basics/read-json-inference",
    "pandas.read_json",
    covers=("convert_dates",),
    frames=("strings_ascii",),
    expr=lambda pd, df: pd.read_json(
        io.StringIO(
            '{"n":{"0":"1","1":"2"},"f":{"0":1.0,"1":2.0},'
            '"date":{"0":1577836800000,"1":null},"t":{"0":"x","1":null}}'
        )
    ),
    in_process=True,
    note="numeric text becomes numbers, whole floats integers, a date column instants. "
    + UPDATE_IN_PROCESS,
)
case(
    "basics/read-json-series",
    "pandas.read_json",
    covers=("typ",),
    frames=("strings_ascii",),
    expr=lambda pd, df: pd.read_json(io.StringIO('{"x":1.5,"y":2}'), typ="series"),
    in_process=True,
    note="a mapping read as a column labelled by its keys. " + UPDATE_IN_PROCESS,
)
case(
    "basics/to-timedelta",
    "pandas.to_timedelta",
    covers=("arg",),
    frames=("strings_ascii",),
    expr=lambda pd, df: pd.to_timedelta(pd.Series(["1 day", "2h", None, "00:00:03"], name="t")),
    in_process=True,
    note="text becomes spans at microsecond resolution, a gap stays missing. " + UPDATE_IN_PROCESS,
)
case(
    "basics/to-timedelta-unit",
    "pandas.to_timedelta",
    covers=("unit",),
    frames=("strings_ascii",),
    expr=lambda pd, df: pd.to_timedelta(pd.Series([1, 2, 3]), unit="s"),
    in_process=True,
    note="whole numbers with a unit keep that unit. " + UPDATE_IN_PROCESS,
)
case(
    "basics/to-timedelta-coerce",
    "pandas.to_timedelta",
    covers=("errors",),
    frames=("strings_ascii",),
    expr=lambda pd, df: pd.to_timedelta(pd.Series(["1s", "oops", "3ms"]), errors="coerce"),
    in_process=True,
    note="text that is not a span becomes missing under coerce. " + UPDATE_IN_PROCESS,
)
case(
    "basics/to-string",
    "DataFrame.to_string",
    frames=("float64_half_null",),
    expr=lambda pd, df: df.to_string(),
    in_process=True,
    note="the frame as a table of text, floats trimmed across each column and gaps as NaN. "
    + UPDATE_IN_PROCESS,
)
case(
    "basics/to-string-options",
    "DataFrame.to_string",
    covers=("index", "header", "na_rep", "float_format", "justify", "show_dimensions"),
    frames=("strings_ascii",),
    expr=lambda pd, df: [
        df.assign(third=1 / 3).to_string(index=False, float_format="%.2f"),
        df.to_string(header=False, na_rep="-"),
        df.to_string(justify="left", show_dimensions=True),
    ],
    in_process=True,
    note="no labels, no header, a float format, a gap marker, left headers and the size line. "
    + UPDATE_IN_PROCESS,
)
case(
    "basics/series-to-string",
    "Series.to_string",
    covers=("name", "dtype", "length"),
    frames=("float64_no_nulls",),
    expr=lambda pd, df: [
        df["value"].to_string(),
        df["value"].rename("v").to_string(name=True, dtype=True, length=True),
    ],
    in_process=True,
    note="one column as text, with and without its name, type and length line. "
    + UPDATE_IN_PROCESS,
)
case(
    "basics/to-html",
    "DataFrame.to_html",
    frames=("float64_half_null", "strings_ascii", "tall"),
    expr=lambda pd, df: df.to_html(),
    in_process=True,
    note="the frame as pandas' HTML table, with the cells to_string writes. " + UPDATE_IN_PROCESS,
)
case(
    "basics/to-html-options",
    "DataFrame.to_html",
    covers=(
        "index",
        "header",
        "na_rep",
        "float_format",
        "max_rows",
        "max_cols",
        "show_dimensions",
        "bold_rows",
        "classes",
        "escape",
        "notebook",
        "border",
        "table_id",
        "col_space",
    ),
    frames=("strings_ascii",),
    expr=lambda pd, df: [
        df.assign(third=1 / 3).to_html(index=False, float_format="%.2f", border=0),
        df.to_html(header=False, na_rep="-", bold_rows=False, classes="wide"),
        df.to_html(max_rows=2, max_cols=1, show_dimensions=True, table_id="t", col_space=40),
        df.to_html(notebook=True, escape=False),
    ],
    in_process=True,
    note="cut rows and columns, the size line, the table's attributes and the notebook form. "
    + UPDATE_IN_PROCESS,
)
case(
    "basics/repr-html",
    "DataFrame.to_html",
    frames=("float64_half_null", "tall"),
    expr=lambda pd, df: df._repr_html_(),
    in_process=True,
    note="the table a notebook shows, under the display options. " + UPDATE_IN_PROCESS,
)
case(
    "basics/to-xml",
    "DataFrame.to_xml",
    level="L3",
    covers=("parser",),
    frames=("float64_half_null", "strings_ascii", "tall"),
    expr=lambda pd, df: df.to_xml(parser="etree"),
    in_process=True,
    note="the frame as pandas' XML document, one element per row, with the standard library's "
    "parser, since lxml is not installed here. " + XML_IN_PROCESS,
)
case(
    "basics/to-xml-options",
    "DataFrame.to_xml",
    level="L3",
    covers=(
        "index",
        "root_name",
        "row_name",
        "na_rep",
        "attr_cols",
        "elem_cols",
        "namespaces",
        "prefix",
        "encoding",
        "xml_declaration",
        "pretty_print",
        "parser",
    ),
    frames=("float64_half_null",),
    expr=lambda pd, df: [
        df.to_xml(parser="etree", index=False, pretty_print=False, xml_declaration=False),
        df.to_xml(parser="etree", na_rep="-", attr_cols=list(df.columns)[:1]),
        df.to_xml(parser="etree", elem_cols=list(df.columns)[-1:], root_name="r", row_name="w"),
        df.to_xml(parser="etree", namespaces={"doc": "https://example.com"}, prefix="doc"),
        df.to_xml(parser="etree", encoding="latin-1", pretty_print=False),
    ],
    in_process=True,
    note="columns as attributes, a stand-in for missing values, namespaces and the encoding. "
    + XML_IN_PROCESS,
)


XML_DOC = """<?xml version='1.0' encoding='utf-8'?>
<data>
  <row id="1"><shape>square</shape><deg>360</deg><sides>4.0</sides><d>2020-01-02</d></row>
  <row id="2"><shape>circle</shape><deg>360</deg><sides/><d>2021-03-04</d></row>
  <row id="3"><shape>triangle</shape><deg>180</deg><sides>3.0</sides><d>2022-05-06</d></row>
</data>"""
XML_SPACED = """<?xml version='1.0'?>
<s:data xmlns:s="https://example.com/s"><s:row><s:a>1</s:a><s:b>x</s:b></s:row>\
<s:row><s:a>2</s:a><s:b>y</s:b></s:row></s:data>"""
XML_READ = "firepanda reads XML in the Python layer with the standard library's parser"


def _xml_read(pd, **options):
    """The document read by the engine's `read_xml` with the standard library's parser."""
    return pd.read_xml(io.StringIO(XML_DOC), parser="etree", **options)


def _xml_doubled(value):
    """A cell's text read as an integer and doubled, the converter the cases hand over."""
    return int(value) * 2


def _xml_on_disk(pd, **options):
    """The document read back from a file, the only source `iterparse` and `compression` take."""
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "doc.xml")
        if options.get("compression"):
            _xml_read(pd).to_xml(path, parser="etree", compression=options["compression"])
        else:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(XML_DOC)
        return pd.read_xml(path, parser="etree", **options)


XML_READS = {
    "plain": ({}, ("path_or_buffer", "parser")),
    "xpath": ({"xpath": ".//row[@id='2']"}, ("xpath",)),
    "elements": ({"elems_only": True}, ("elems_only",)),
    "attributes": ({"attrs_only": True}, ("attrs_only",)),
    "names": ({"names": ["i", "s", "g", "n", "t"]}, ("names",)),
    "dtype": ({"dtype": {"deg": "float64", "shape": "string"}}, ("dtype",)),
    "converters": ({"converters": {"deg": _xml_doubled}}, ("converters",)),
    "dates": ({"parse_dates": ["d"]}, ("parse_dates",)),
    "arrow-backend": ({"dtype_backend": "pyarrow"}, ("dtype_backend",)),
    "nullable-backend": ({"dtype_backend": "numpy_nullable"}, ("dtype_backend",)),
}
for name, (options, covers) in XML_READS.items():
    case(
        f"basics/read-xml-{name}",
        "pandas.read_xml",
        level="L3",
        covers=covers,
        frames=("single",),
        expr=lambda pd, df, options=options: _xml_read(pd, **options),
        in_process=True,
        note="a small document of rows, attributes and a gap. " + XML_READ,
    )
case(
    "basics/read-xml-namespaces",
    "pandas.read_xml",
    level="L3",
    covers=("namespaces", "xpath"),
    frames=("single",),
    expr=lambda pd, df: pd.read_xml(
        io.StringIO(XML_SPACED),
        xpath=".//s:row",
        namespaces={"s": "https://example.com/s"},
        parser="etree",
    ),
    in_process=True,
    note="rows found through a namespace prefix. " + XML_READ,
)
case(
    "basics/read-xml-encoding",
    "pandas.read_xml",
    level="L3",
    covers=("encoding",),
    frames=("single",),
    expr=lambda pd, df: pd.read_xml(
        io.BytesIO(XML_DOC.replace("utf-8", "latin-1").encode("latin-1")),
        parser="etree",
        encoding="latin-1",
    ),
    in_process=True,
    note="bytes decoded with the encoding given. " + XML_READ,
)
case(
    "basics/read-xml-iterparse",
    "pandas.read_xml",
    level="L3",
    covers=("iterparse",),
    frames=("single",),
    expr=lambda pd, df: _xml_on_disk(pd, iterparse={"row": ["id", "shape", "deg"]}),
    in_process=True,
    note="a file read one element at a time, keeping the fields named. " + XML_READ,
)
case(
    "basics/read-xml-compression",
    "pandas.read_xml",
    level="L3",
    covers=("compression",),
    frames=("single",),
    expr=lambda pd, df: _xml_on_disk(pd, compression="gzip"),
    in_process=True,
    note="a gzip file written by the engine's to_xml and read back. " + XML_READ,
)
case(
    "basics/read-xml-stylesheet",
    "pandas.read_xml",
    level="L4",
    raises=("ValueError", "To use stylesheet, you need lxml"),
    frames=("single",),
    expr=lambda pd, df: _xml_read(pd, stylesheet=io.StringIO("<x/>")),
    in_process=True,
    note="a stylesheet needs lxml, which the standard library's parser refuses. " + XML_READ,
)
case(
    "basics/read-xml-storage",
    "pandas.read_xml",
    level="L4",
    raises=("ValueError", "storage_options passed with file object"),
    frames=("single",),
    expr=lambda pd, df: _xml_read(pd, storage_options={"anon": True}),
    in_process=True,
    note="storage options only go with an fsspec path, not a buffer. " + XML_READ,
)
case(
    "basics/read-xml-iterparse-list",
    "pandas.read_xml",
    level="L4",
    raises=("TypeError", "list is not a valid type for iterparse"),
    frames=("single",),
    expr=lambda pd, df: _xml_on_disk(pd, iterparse=["row"]),
    in_process=True,
    note="iterparse takes a mapping of the row element to its fields, not a list. " + XML_READ,
)


case(
    "basics/frame-repr",
    "DataFrame.to_string",
    frames=("float64_half_null", "strings_ascii"),
    expr=lambda pd, df: repr(df),
    in_process=True,
    note="repr, which is to_string under the display options, prints rows and not a summary. "
    + UPDATE_IN_PROCESS,
)
case(
    "basics/frame-repr-cut",
    "DataFrame.to_string",
    covers=("max_rows", "min_rows", "max_cols", "line_width"),
    frames=("float64_no_nulls",),
    expr=lambda pd, df: [
        repr(pd.DataFrame({f"c{i}": df["value"].head(70) for i in range(20)})),
        repr(pd.DataFrame({f"c{i}": df["value"].head(2) for i in range(30)})),
    ],
    in_process=True,
    note="a long frame loses its middle rows to dots and a wide one its middle columns. "
    + UPDATE_IN_PROCESS,
)
case(
    "basics/to-string-limits",
    "DataFrame.to_string",
    covers=("max_rows", "min_rows", "max_cols", "line_width"),
    frames=("float64_no_nulls",),
    expr=lambda pd, df: [
        df.to_string(max_rows=6),
        df.to_string(max_rows=20, min_rows=4, show_dimensions=True),
        pd.DataFrame({f"c{i}": df["value"].head(3) for i in range(8)}).to_string(max_cols=4),
        pd.DataFrame({f"c{i}": df["value"].head(3) for i in range(8)}).to_string(line_width=40),
    ],
    in_process=True,
    note="the row and column limits cut the middle out, and a line width wraps into blocks. "
    + UPDATE_IN_PROCESS,
)


def _trades(pd):
    """Trades and quotes on whole number times for two tickers, both sorted by time."""
    trades = pd.DataFrame(
        {"time": [1, 3, 5, 5, 8, 12], "ticker": list("ABABAC"), "qty": [10, 20, 30, 40, 50, 60]}
    )
    quotes = pd.DataFrame(
        {
            "time": [0, 2, 3, 5, 7, 9],
            "ticker": list("ABABAB"),
            "bid": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            "size": [1, 2, 3, 4, 5, 6],
        }
    )
    return trades, quotes


case(
    "basics/merge-asof",
    "pandas.merge_asof",
    frames=("two",),
    expr=lambda pd, df: pd.merge_asof(*_trades(pd), on="time", by="ticker"),
    in_process=True,
    note="each trade takes the last quote at or before its time with the same ticker, and a "
    "trade with no quote keeps gaps, its integer column widened to float64. " + BUILT,
)
case(
    "basics/merge-asof-direction",
    "pandas.merge_asof",
    covers=("direction",),
    frames=("two",),
    expr=lambda pd, df: pd.merge_asof(*_trades(pd), on="time", direction="forward"),
    in_process=True,
    note="forward takes the first quote at or after the trade. " + BUILT,
)
case(
    "basics/merge-asof-nearest",
    "pandas.merge_asof",
    covers=("direction", "allow_exact_matches"),
    frames=("two",),
    expr=lambda pd, df: pd.merge_asof(
        *_trades(pd), on="time", direction="nearest", allow_exact_matches=False
    ),
    in_process=True,
    note="nearest takes the closer quote with ties going backward, and allow_exact_matches=False "
    "skips a quote at the same time. " + BUILT,
)
case(
    "basics/merge-asof-tolerance",
    "pandas.merge_asof",
    covers=("tolerance", "left_by", "right_by", "suffixes"),
    frames=("two",),
    expr=lambda pd, df: pd.merge_asof(
        _trades(pd)[0].rename(columns={"ticker": "sym"}),
        _trades(pd)[1],
        on="time",
        left_by="sym",
        right_by="ticker",
        tolerance=1,
    ),
    in_process=True,
    note="a quote further than the tolerance is no match, and by columns with different names "
    "on each side are both kept. " + BUILT,
)


def _groups(pd):
    """The pandas documentation's frames for an ordered merge: a left in two groups."""
    left = pd.DataFrame(
        {"key": list("aceace"), "lvalue": [1, 2, 3, 1, 2, 3], "group": list("aaabbb")}
    )
    return left, pd.DataFrame({"key": ["b", "c", "d"], "rvalue": [1, 2, 3]})


case(
    "basics/merge-ordered",
    "pandas.merge_ordered",
    covers=("left_by", "fill_method"),
    frames=("two",),
    expr=lambda pd, df: pd.merge_ordered(*_groups(pd), fill_method="ffill", left_by="group"),
    in_process=True,
    note="each group of the left is joined with the right in key order, the rows with no left "
    "match take the left row above, and the group column is filled in for every row. " + BUILT,
)
case(
    "basics/merge-ordered-how",
    "pandas.merge_ordered",
    covers=("on", "how", "fill_method"),
    frames=("two",),
    expr=lambda pd, df: pd.merge_ordered(
        pd.DataFrame({"k": [3, 1, 2], "v": [1, 2, 3]}),
        pd.DataFrame({"k": [2, 5, 0], "w": [1.5, 2.5, float("nan")]}),
        on="k",
        how="right",
        fill_method="ffill",
    ),
    in_process=True,
    note="a right join sorted by the key, where the fill carries the left row down and a "
    "gap the right side really has stays a gap. " + BUILT,
)
case(
    "basics/merge-outer-instants",
    "pandas.merge",
    covers=("how", "on"),
    frames=("two",),
    expr=lambda pd, df: pd.merge(
        pd.DataFrame({"t": pd.to_datetime(pd.Series(["2020-01-01", "2020-01-03"])), "a": [1, 2]}),
        pd.DataFrame({"t": pd.to_datetime(pd.Series(["2020-01-02", "2020-01-03"])), "b": [1, 2]}),
        on="t",
        how="outer",
    ),
    in_process=True,
    note="the key an outer join puts together from both sides stays instants. " + BUILT,
)


def _moments(pd, zone=None):
    """A frame labelled by four instants, the last in the next month."""
    instants = pd.to_datetime(
        pd.Series(["2020-01-01 00:00", "2020-01-01 12:00", "2020-01-02 06:00", "2020-02-01 00:00"])
    )
    if zone:
        instants = instants.dt.tz_localize(zone)
    return pd.DataFrame({"d": instants, "v": [1, 2, 3, 4]}).set_index("d")


case(
    "basics/loc-timestamp",
    "DataFrame.loc",
    frames=("two",),
    expr=lambda pd, df: _moments(pd).loc[pd.Timestamp("2020-01-02 06:00")],
    in_process=True,
    note="one instant in the labels picks its row as a column. " + BUILT,
)
case(
    "basics/loc-partial-string",
    "DataFrame.loc",
    frames=("two",),
    expr=lambda pd, df: _moments(pd).loc["2020-01"],
    in_process=True,
    note="text naming a month picks every row inside it. " + BUILT,
)
case(
    "basics/loc-slice-strings",
    "DataFrame.loc",
    frames=("two",),
    expr=lambda pd, df: _moments(pd).loc["2020-01-01 06:00":"2020-01-02"],
    in_process=True,
    note="a slice with text bounds runs from the first moment to the end of the day the upper "
    "bound names. " + BUILT,
)
case(
    "basics/loc-zoned-string",
    "DataFrame.loc",
    frames=("two",),
    expr=lambda pd, df: _moments(pd, "Asia/Tokyo").loc["2020-01-01"],
    in_process=True,
    note="zoned labels read naive text on their own clock. " + BUILT,
)
case(
    "basics/loc-timedelta",
    "DataFrame.loc",
    frames=("two",),
    expr=lambda pd, df: (
        pd.DataFrame({"d": pd.to_timedelta(pd.Series(["1D", "2D", "3D"])), "v": [1, 2, 3]})
        .set_index("d")
        .loc["2 days":]
    ),
    in_process=True,
    note="an index of spans answers the text of a span. " + BUILT,
)
NESTED_RECORDS = [
    {"id": 1, "name": {"first": "Coleen", "last": "Volk"}},
    {"name": {"given": "Mark", "family": "Regner"}},
    {"id": 2, "name": "Faye Raker"},
]
STATES = [
    {
        "state": "Florida",
        "info": {"governor": "Rick Scott"},
        "counties": [
            {"name": "Dade", "population": 12345},
            {"name": "Broward", "population": 40000},
        ],
    },
    {
        "state": "Ohio",
        "info": {"governor": "John Kasich"},
        "counties": [{"name": "Summit", "population": 1234}],
    },
]
case(
    "basics/json-normalize",
    "pandas.json_normalize",
    frames=("single",),
    expr=lambda pd, df: pd.json_normalize(NESTED_RECORDS),
    in_process=True,
    note="nested dicts flatten into dotted columns, with ragged keys filled with gaps",
)
case(
    "basics/json-normalize-sep",
    "pandas.json_normalize",
    covers=("sep", "max_level"),
    frames=("single",),
    expr=lambda pd, df: pd.json_normalize(NESTED_RECORDS, sep="_", max_level=1),
    in_process=True,
    note="sep joins the nested keys and max_level at the depth flattens all of them",
)
case(
    "basics/json-normalize-gap",
    "pandas.json_normalize",
    frames=("single",),
    expr=lambda pd, df: pd.json_normalize([{"a": 1}, None, {"a": 3}]),
    in_process=True,
    note="a missing record is an empty row",
)
case(
    "basics/json-normalize-records",
    "pandas.json_normalize",
    covers=("record_path", "meta"),
    frames=("single",),
    expr=lambda pd, df: pd.json_normalize(STATES, "counties", ["state", ["info", "governor"]]),
    in_process=True,
    note="the rows nested under record_path come out with the outer fields repeated on each",
)
case(
    "basics/json-normalize-prefixes",
    "pandas.json_normalize",
    covers=("record_path", "meta", "meta_prefix", "record_prefix"),
    frames=("single",),
    expr=lambda pd, df: pd.json_normalize(
        STATES, "counties", ["state"], record_prefix="c.", meta_prefix="m."
    ),
    in_process=True,
    note="the prefixes go in front of the record and metadata columns",
)
case(
    "basics/nat",
    "pandas.NaT",
    frames=("single",),
    expr=lambda pd, df: [
        repr(pd.NaT),
        pd.NaT == pd.NaT,
        pd.NaT != pd.NaT,
        pd.Timestamp(None) is pd.NaT,
        pd.Timedelta("nat") is pd.NaT,
        pd.isna(pd.NaT),
        repr(pd.NaT + pd.Timedelta("1D")),
        repr(pd.Timestamp("2020-01-01") - pd.NaT),
        repr(pd.NaT.day_name()),
        pd.NaT.year != pd.NaT.year,
    ],
    in_process=True,
    note="the missing moment and span, its equality, its arithmetic and its fields",
)
case(
    "basics/nat-refuses",
    "pandas.NaT",
    level="L4",
    frames=("single",),
    expr=lambda pd, df: pd.NaT.strftime("%Y"),
    raises=("ValueError", "NaTType does not support strftime"),
    in_process=True,
    note="NaT refuses to format itself with the same error pandas raises",
)
case(
    "basics/frame-plus-list",
    "DataFrame.add",
    level="L3",
    covers=("other", "axis"),
    frames=("single",),
    expr=lambda pd, df: pd.DataFrame({"a": [1, 2], "b": [3.5, 4.0]}).add([1, 10], axis=0),
    in_process=True,
    note="a list beside a frame lines up with the rows for the named form on the rows. "
    "firepanda #1242 reads the list as a series in the Python layer",
)

# Column labels that are integers. These three asserted `engine/integer-column-labels`
# until firepanda #1258 named columns with any value, and they score the ordinary way now.

case(
    "basics/integer-labels-partition",
    "str.partition",
    frames=("strings_pattern",),
    expr=lambda pd, df: [type(label).__name__ for label in df["value"].str.partition("o").columns],
    in_process=True,
    note="pandas labels the three columns `str.partition` hands back with the integers 0, "
    "1 and 2. The type name of each label is read rather than the label, because the text "
    "`0` and the integer 0 print the same way",
)
case(
    "basics/integer-labels-constructor",
    "pandas.DataFrame",
    milestone="M2",
    frames=("empty",),
    expr=lambda pd, df: [
        type(label).__name__ for label in pd.DataFrame({0: [1, 2], 1: [3, 4]}).columns
    ],
    in_process=True,
    note="a dictionary with integer keys builds a frame with integer labels, with no string "
    "method in the way. The corpus frame is ignored because the case builds its own",
)
case(
    "basics/integer-labels-values",
    "str.partition",
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.partition("o").iloc[:, 1].tolist(),
    in_process=True,
    note="the same three columns read by position rather than by label",
)

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# These five were the `divergences/plotting/*` cases, on the grounds that plotting is a
# different library's job and firepanda refused every plotting name. firepanda has
# pandas' matplotlib front end now, so each name resolves to what pandas hands back.
# They return an accessor or a bound method, neither of which this suite can compare,
# so the case asks for the type name. matplotlib is imported only when a plot is drawn,
# so the pinned environment does not need it.

case(
    "basics/plotting-frame-plot",
    "DataFrame.plot",
    level="L0",
    frames=("two",),
    expr=lambda pd, df: type(df.plot).__name__,
    note="pandas hands back a PlotAccessor, and so does firepanda",
)
case(
    "basics/plotting-series-plot",
    "Series.plot",
    level="L0",
    frames=("two",),
    expr=lambda pd, df: type(df["a"].plot).__name__,
)
case(
    "basics/plotting-frame-hist",
    "DataFrame.hist",
    level="L0",
    frames=("two",),
    expr=lambda pd, df: type(df.hist).__name__,
)
case(
    "basics/plotting-series-hist",
    "Series.hist",
    level="L0",
    frames=("two",),
    expr=lambda pd, df: type(df["a"].hist).__name__,
)
case(
    "basics/plotting-frame-boxplot",
    "DataFrame.boxplot",
    level="L0",
    frames=("two",),
    expr=lambda pd, df: type(df.boxplot).__name__,
)

# ---------------------------------------------------------------------------
# Parameters of names that already have a plain case
# ---------------------------------------------------------------------------

PARAMETER_IN_PROCESS = (
    "the driver has an entry for the plain spelling of this name and not for this "
    "parameter, so it is run in process"
)

for name in ("diff", "pct_change"):
    case(
        f"basics/{name.replace('_', '-')}-periods",
        f"Series.{name}",
        level="L3",
        covers=("periods",),
        frames=("float64_no_nulls", "float64_half_null", "int64_half_null"),
        expr=(lambda how: lambda pd, df: getattr(df["value"], how)(periods=-2))(name),
        in_process=True,
        note="a negative period looks forward rather than back, so the gaps are at the "
        "end. " + PARAMETER_IN_PROCESS,
    )
    case(
        f"basics/frame-{name.replace('_', '-')}",
        f"DataFrame.{name}",
        level="L3",
        covers=("periods",),
        frames=("float64_half_null", "int64_half_null"),
        expr=(lambda how: lambda pd, df: getattr(df[["value"]], how)(periods=2))(name),
        in_process=True,
        note="the frame spelling, one column at a time. " + PARAMETER_IN_PROCESS,
    )
case(
    "basics/frame-shift",
    "DataFrame.shift",
    level="L3",
    covers=("periods", "fill_value"),
    frames=("int64_no_nulls", "float64_half_null"),
    expr=lambda pd, df: df[["value"]].shift(periods=3, fill_value=0),
    in_process=True,
    note="with a fill value the integer column stays an integer one. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/cumsum-skipna",
    "Series.cumsum",
    level="L3",
    covers=("skipna",),
    frames=("float64_half_null", "float64_no_nulls"),
    expr=lambda pd, df: df["value"].cumsum(skipna=False),
    in_process=True,
    note="without skipping, every row after the first gap is a gap. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/frame-cumsum",
    "DataFrame.cumsum",
    level="L3",
    covers=("skipna",),
    frames=("float64_half_null", "int64_half_null"),
    expr=lambda pd, df: df[["value"]].cumsum(skipna=True),
    in_process=True,
    note="the running total of each column. " + PARAMETER_IN_PROCESS,
)
for name in ("head", "tail"):
    case(
        f"basics/series-{name}-n",
        f"Series.{name}",
        level="L3",
        covers=("n",),
        frames=("float64_half_null", "two", "empty"),
        expr=(lambda how: lambda pd, df: getattr(df[df.columns[0]], how)(n=-3))(name),
        in_process=True,
        note="a negative count keeps all but that many rows. " + PARAMETER_IN_PROCESS,
    )
for name in ("any", "all"):
    case(
        f"basics/series-{name}-skipna",
        f"Series.{name}",
        level="L3",
        covers=("skipna",),
        frames=("float64_half_null", "float64_all_null", "int64_no_nulls"),
        expr=(lambda how: lambda pd, df: getattr(df["value"], how)(skipna=False))(name),
        in_process=True,
        note="a gap counts as true when it is not skipped. " + PARAMETER_IN_PROCESS,
    )
    case(
        f"basics/frame-{name}-bool-only",
        f"DataFrame.{name}",
        level="L3",
        covers=("bool_only", "skipna"),
        frames=("tall", "two"),
        expr=(lambda how: lambda pd, df: getattr(df, how)(bool_only=True, skipna=True))(name),
        in_process=True,
        note="only the boolean columns answer. " + PARAMETER_IN_PROCESS,
    )
case(
    "basics/frame-idxmax-numeric-only",
    "DataFrame.idxmax",
    level="L3",
    covers=("skipna", "numeric_only"),
    frames=("two", "tall"),
    expr=lambda pd, df: df.idxmax(numeric_only=True),
    in_process=True,
    note="the label of each numeric column's largest row. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/frame-value-counts",
    "DataFrame.value_counts",
    level="L3",
    covers=("subset", "normalize", "sort", "ascending", "dropna"),
    frames=("tall",),
    expr=lambda pd, df: df.value_counts(subset=["key", "flag"], normalize=True, ascending=True),
    in_process=True,
    note="the share of each pair of values, smallest first. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/frame-value-counts-dropna",
    "DataFrame.value_counts",
    level="L3",
    covers=("dropna", "sort"),
    frames=("two",),
    expr=lambda pd, df: df.value_counts(dropna=False, sort=False),
    in_process=True,
    note="a row with a gap is counted when gaps are kept. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/sort-index-descending",
    "DataFrame.sort_index",
    level="L3",
    covers=("ascending", "ignore_index"),
    frames=("two", "tall"),
    expr=lambda pd, df: df.sort_index(ascending=False),
    in_process=True,
    note="the labels in reverse. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/sort-index-axis",
    "DataFrame.sort_index",
    level="L3",
    covers=("axis", "ascending"),
    frames=("tall", "wide"),
    expr=lambda pd, df: df.sort_index(axis=1, ascending=False),
    in_process=True,
    note="the columns rather than the rows. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/drop-duplicates-keep",
    "DataFrame.drop_duplicates",
    level="L3",
    covers=("subset", "keep", "ignore_index"),
    frames=("tall",),
    expr=lambda pd, df: df.drop_duplicates(subset=["key"], keep="last", ignore_index=True),
    in_process=True,
    note="the last row of each key survives, numbered again from zero. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/drop-duplicates-keep-false",
    "DataFrame.drop_duplicates",
    level="L3",
    covers=("subset", "keep"),
    frames=("tall",),
    expr=lambda pd, df: df.drop_duplicates(subset=["key", "flag"], keep=False),
    in_process=True,
    note="no row of a repeated pair survives. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/astype-errors-ignore",
    "Series.astype",
    level="L3",
    covers=("dtype", "errors"),
    frames=("strings_ascii",),
    expr=lambda pd, df: df["value"].astype("int64", errors="ignore"),
    in_process=True,
    note="a cast that fails hands the column back unchanged. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/replace-regex",
    "Series.replace",
    level="L3",
    covers=("to_replace", "value", "regex"),
    frames=("strings_ascii", "strings_null_heavy"),
    expr=lambda pd, df: df["value"].replace(r"^a", "A", regex=True),
    in_process=True,
    note="the pattern is matched inside each row rather than against the whole of it. "
    + PARAMETER_IN_PROCESS,
)
case(
    "basics/rank-descending",
    "Series.rank",
    level="L3",
    covers=("ascending",),
    frames=("float64_half_null", "int64_no_nulls"),
    expr=lambda pd, df: df["value"].rank(ascending=False),
    in_process=True,
    note="the largest row ranks first. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/frame-rank-pct",
    "DataFrame.rank",
    level="L3",
    covers=("na_option", "pct"),
    frames=("float64_half_null",),
    expr=lambda pd, df: df[["value"]].rank(na_option="bottom", pct=True),
    in_process=True,
    note="ranks as a share of the rows, with the gaps last. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/fillna-limit",
    "Series.fillna",
    level="L3",
    covers=("value", "limit"),
    frames=("float64_half_null",),
    expr=lambda pd, df: df["value"].fillna(0.0, limit=2),
    in_process=True,
    note="only the first two gaps are filled. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/round-negative",
    "Series.round",
    level="L3",
    covers=("decimals",),
    frames=("tall",),
    expr=lambda pd, df: df["value"].round(-3),
    in_process=True,
    note="a negative count rounds to thousands. " + PARAMETER_IN_PROCESS,
)

for name in ("std", "var", "sem"):
    case(
        f"basics/{name}-ddof",
        f"Series.{name}",
        level="L3",
        covers=("ddof", "skipna"),
        frames=("float64_half_null", "float64_no_nulls", "int64_no_nulls"),
        expr=(lambda how: lambda pd, df: getattr(df["value"], how)(ddof=0, skipna=True))(name),
        in_process=True,
        note="the population spread rather than the sample one. " + PARAMETER_IN_PROCESS,
    )
    case(
        f"basics/frame-{name}-ddof",
        f"DataFrame.{name}",
        level="L3",
        covers=("ddof", "numeric_only"),
        frames=("tall", "two"),
        expr=(lambda how: lambda pd, df: getattr(df, how)(ddof=0, numeric_only=True))(name),
        in_process=True,
        note="each numeric column's population spread. " + PARAMETER_IN_PROCESS,
    )
for owner in ("Series", "DataFrame"):
    case(
        f"basics/{owner.lower()}-prod-min-count",
        f"{owner}.prod",
        level="L3",
        covers=("skipna", "min_count"),
        frames=("float64_all_null", "float64_half_null"),
        expr=(
            (lambda pd, df: df["value"].prod(min_count=1))
            if owner == "Series"
            else (lambda pd, df: df[["value"]].prod(min_count=1))
        ),
        in_process=True,
        note="a product of nothing is missing once a count is asked for. " + PARAMETER_IN_PROCESS,
    )
case(
    "basics/cut-right-false",
    "pandas.cut",
    level="L3",
    covers=("x", "bins", "right", "labels"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.cut(df["value"], [-100, 0, 10, 100], right=False, labels=False),
    in_process=True,
    note="bins closed on the left. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/cut-include-lowest",
    "pandas.cut",
    level="L3",
    covers=("x", "bins", "include_lowest", "labels"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.cut(
        df["value"], [0, 10, 100, 1000], include_lowest=True, labels=["a", "b", "c"]
    ).astype(str),
    in_process=True,
    note="the lowest edge belongs to the first bin. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/get-dummies-prefix",
    "pandas.get_dummies",
    level="L3",
    covers=("data", "prefix", "prefix_sep", "drop_first", "dtype"),
    frames=("keys_two_column",),
    expr=lambda pd, df: pd.get_dummies(
        df[["left"]], prefix="k", prefix_sep="-", drop_first=True, dtype=int
    ),
    in_process=True,
    note="named columns of whole numbers with the first value dropped. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/get-dummies-dummy-na",
    "pandas.get_dummies",
    level="L3",
    covers=("data", "dummy_na", "columns"),
    frames=("keys_awkward",),
    expr=lambda pd, df: pd.get_dummies(df, columns=["key"], dummy_na=True),
    in_process=True,
    note="a column for the missing value too. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/melt-names",
    "pandas.melt",
    level="L3",
    covers=("frame", "id_vars", "value_vars", "var_name", "value_name"),
    frames=("keys_two_column",),
    expr=lambda pd, df: pd.melt(
        df, id_vars=["left"], value_vars=["right", "value"], var_name="what", value_name="n"
    ),
    in_process=True,
    note="the two new columns named by the caller. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/melt-ignore-index",
    "pandas.melt",
    level="L3",
    covers=("frame", "id_vars", "ignore_index"),
    frames=("keys_two_column",),
    expr=lambda pd, df: pd.melt(df, id_vars=["left"], value_name="n", ignore_index=False),
    in_process=True,
    note="the row labels repeat rather than being numbered again. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/to-datetime-unit",
    "pandas.to_datetime",
    level="L3",
    covers=("arg", "unit", "utc"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.to_datetime(df["value"].abs() % 1000, unit="D", utc=True),
    in_process=True,
    note="whole numbers read as days since the epoch, in UTC. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/to-datetime-errors-coerce",
    "pandas.to_datetime",
    level="L3",
    covers=("arg", "errors", "format"),
    frames=("strings_ascii",),
    expr=lambda pd, df: pd.to_datetime(df["value"], errors="coerce", format="%Y-%m-%d"),
    in_process=True,
    note="text that is not a date becomes a missing instant. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/interpolate-limit",
    "Series.interpolate",
    level="L3",
    covers=("method", "limit", "limit_direction"),
    frames=("float64_half_null",),
    expr=lambda pd, df: df["value"].interpolate(limit=1, limit_direction="both"),
    in_process=True,
    note="one gap filled on each side of a run. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/align-join",
    "Series.align",
    level="L3",
    covers=("other", "join", "fill_value"),
    frames=("float64_no_nulls",),
    expr=lambda pd, df: (
        lambda pair: pd.concat([pair[0].rename("l"), pair[1].rename("r")], axis=1)
    )(df["value"].head(5).align(df["value"].iloc[3:8], join="outer", fill_value=0.0)),
    in_process=True,
    note="both sides over the labels of either, with a fill. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/compare-keep-shape",
    "Series.compare",
    level="L3",
    covers=("other", "keep_shape", "keep_equal"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: df["value"].compare(df["value"].where(df["value"] > 0, 0), keep_shape=True),
    in_process=True,
    note="every row kept, with the equal ones missing. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/concat-sort",
    "pandas.concat",
    level="L3",
    covers=("objs", "sort", "names", "keys"),
    frames=("keys_two_column",),
    expr=lambda pd, df: pd.concat(
        [df[["value", "left"]], df[["right", "value"]]],
        sort=True,
        keys=["a", "b"],
        names=["k", "i"],
    ),
    in_process=True,
    note="the union of the columns in sorted order under named keys. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/crosstab-values",
    "pandas.crosstab",
    level="L3",
    covers=("index", "columns", "values", "aggfunc", "rownames", "colnames"),
    frames=("keys_two_column",),
    expr=lambda pd, df: pd.crosstab(
        df["left"], df["right"], values=df["value"], aggfunc="sum", rownames=["l"], colnames=["r"]
    ),
    in_process=True,
    note="a sum per pair rather than a count. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/ewm-com-adjust",
    "Series.ewm",
    level="L3",
    covers=("com", "min_periods", "adjust", "ignore_na"),
    frames=("float64_half_null",),
    expr=lambda pd, df: (
        df["value"]
        .mask(df["value"].abs() == float("inf"))
        .ewm(com=1.5, min_periods=2, adjust=False, ignore_na=True)
        .mean()
    ),
    in_process=True,
    note="a recursive mean that steps over the gaps, with the infinities left to the window"
    " cases. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/ewm-span-halflife-alpha",
    "Series.ewm",
    level="L3",
    covers=("span", "halflife", "alpha"),
    frames=("tall",),
    expr=lambda pd, df: pd.concat(
        [
            df["value"].ewm(span=4).mean().rename("s"),
            df["value"].ewm(halflife=2).mean().rename("h"),
            df["value"].ewm(alpha=0.3).mean().rename("a"),
        ],
        axis=1,
    ),
    in_process=True,
    note="the three other ways to say how fast the weights decay. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/rolling-center",
    "Series.rolling",
    level="L3",
    covers=("window", "min_periods", "center"),
    frames=("tall",),
    expr=lambda pd, df: df["value"].rolling(window=5, min_periods=2, center=True).sum(),
    in_process=True,
    note="a centred window that answers from two rows on. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/rolling-closed",
    "Series.rolling",
    level="L3",
    covers=("closed", "step"),
    frames=("tall",),
    expr=lambda pd, df: pd.concat(
        [
            df["value"].rolling(3, closed="left").mean().rename("left"),
            df["value"].rolling(3, step=2).mean().rename("step"),
        ],
        axis=1,
    ).dropna(how="all"),
    in_process=True,
    note="a window that leaves its own row out, and one that answers every other row. "
    + PARAMETER_IN_PROCESS,
)
case(
    "basics/series-reindex-fill",
    "Series.reindex",
    level="L3",
    covers=("index", "fill_value"),
    frames=("float64_no_nulls",),
    expr=lambda pd, df: df["value"].reindex([5, 3, 100, 1], fill_value=0.0),
    in_process=True,
    note="labels in a new order with a fill for the one that is not there. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/series-reindex-method",
    "Series.reindex",
    level="L3",
    covers=("method", "limit"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: df["value"].head(5).reindex(range(8), method="ffill", limit=1),
    in_process=True,
    note="new labels carried forward one step and no further. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/series-sort-index-ignore",
    "Series.sort_index",
    level="L3",
    covers=("ascending", "ignore_index"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: df["value"].iloc[::-3].sort_index(ascending=False, ignore_index=True),
    in_process=True,
    note="labels sorted high to low and then thrown away. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/frame-sort-index-columns",
    "DataFrame.sort_index",
    level="L3",
    covers=("axis", "ascending"),
    frames=("two", "tall"),
    expr=lambda pd, df: df.sort_index(axis=1, ascending=False),
    in_process=True,
    note="the column names sorted high to low. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/merge-left-on-right-on",
    "DataFrame.merge",
    level="L3",
    covers=("left_on", "right_on", "suffixes", "sort"),
    frames=("keys_two_column",),
    expr=lambda pd, df: df.merge(
        df.rename(columns={"left": "other"}),
        left_on="left",
        right_on="other",
        suffixes=("_l", "_r"),
        sort=True,
    ),
    in_process=True,
    note="keys under different names on each side, sorted. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/merge-index-indicator",
    "DataFrame.merge",
    level="L3",
    covers=("left_index", "right_index", "indicator", "validate"),
    frames=("keys_two_column",),
    expr=lambda pd, df: df.merge(
        df[["value"]].iloc[2:],
        how="left",
        left_index=True,
        right_index=True,
        suffixes=("", "_r"),
        indicator=True,
        validate="one_to_one",
    ),
    in_process=True,
    note="both sides on their labels, with the column that says where each row came from. "
    + PARAMETER_IN_PROCESS,
)
case(
    "basics/pivot-table-margins",
    "pandas.pivot_table",
    level="L3",
    covers=("data", "values", "index", "columns", "aggfunc", "fill_value", "margins"),
    frames=("keys_two_column",),
    expr=lambda pd, df: pd.pivot_table(
        df.assign(right=df["right"].astype(str)),
        values="value",
        index="left",
        columns="right",
        aggfunc="sum",
        fill_value=0,
        margins=True,
    ),
    in_process=True,
    note="a sum per pair with the totals on the edges. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/frame-pivot-table-margins-name",
    "DataFrame.pivot_table",
    level="L3",
    covers=("margins", "margins_name", "dropna", "sort"),
    frames=("keys_two_column",),
    expr=lambda pd, df: df.pivot_table(
        values="value", index="left", aggfunc="mean", margins=True, margins_name="all", sort=False
    ),
    in_process=True,
    note="a mean per key with the total named and the keys in the order they came. "
    + PARAMETER_IN_PROCESS,
)
case(
    "basics/convert-dtypes-text",
    "DataFrame.convert_dtypes",
    level="L3",
    covers=("convert_string",),
    frames=("strings_ascii", "strings_null_heavy"),
    expr=lambda pd, df: df.convert_dtypes().dtypes.astype(str),
    in_process=True,
    note="text columns, which become the masked `string` type when `convert_string` is left on. "
    + PARAMETER_IN_PROCESS,
)
case(
    "basics/string-dtype-accessor",
    "Series.str.len",
    level="L2",
    frames=("strings_null_heavy",),
    expr=lambda pd, df: df["value"].astype("string").str.len(),
    in_process=True,
    note="the `str` accessor on the masked `string` type, which answers `Int64` with a gap "
    "where the row was one. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/string-dtype-contains",
    "Series.str.contains",
    level="L3",
    covers=("pat",),
    frames=("strings_null_heavy",),
    expr=lambda pd, df: df["value"].astype("string").str.contains("1"),
    in_process=True,
    note="a test on the masked `string` type, which answers `boolean` with a gap where the "
    "row was one. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/convert-dtypes-flags",
    "DataFrame.convert_dtypes",
    level="L3",
    covers=("infer_objects", "convert_string", "convert_integer", "convert_floating"),
    frames=("two", "single"),
    expr=lambda pd, df: df.convert_dtypes(
        convert_string=False, convert_integer=False
    ).dtypes.astype(str),
    in_process=True,
    note="the nullable dtypes with the text and integer conversions turned off. "
    + PARAMETER_IN_PROCESS,
)
case(
    "basics/corrwith-method",
    "DataFrame.corrwith",
    level="L3",
    covers=("other", "method", "drop"),
    frames=("tall",),
    expr=lambda pd, df: df[["key", "value"]].corrwith(
        df[["key", "value"]].iloc[::-1].reset_index(drop=True), method="pearson", drop=True
    ),
    in_process=True,
    note="a correlation of each column with the same column reversed. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/frame-interpolate-area",
    "DataFrame.interpolate",
    level="L3",
    covers=("method", "limit_direction", "limit_area"),
    frames=("float64_half_null",),
    expr=lambda pd, df: df[["value"]].interpolate(
        method="linear", limit_direction="both", limit_area="inside"
    ),
    in_process=True,
    note="only the gaps with a value on both sides filled. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/get-dummies-columns",
    "pandas.get_dummies",
    level="L3",
    covers=("columns", "prefix_sep", "drop_first", "dtype"),
    frames=("keys_two_column",),
    expr=lambda pd, df: pd.get_dummies(
        df, columns=["left"], prefix_sep=":", drop_first=True, dtype=int
    ),
    in_process=True,
    note="one text column spread into integer columns with the first dropped. "
    + PARAMETER_IN_PROCESS,
)
case(
    "basics/timestamp-parts",
    "pandas.Timestamp",
    level="L3",
    covers=("year", "month", "day", "hour", "minute", "second"),
    frames=("single",),
    expr=lambda pd, df: pd.Series(
        [pd.Timestamp(year=2024, month=3, day=10, hour=5, minute=6, second=7)]
    ),
    in_process=True,
    note="an instant built from its parts. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/timestamp-replace",
    "Timestamp.replace",
    level="L3",
    covers=("year", "month", "day", "hour", "minute", "second"),
    frames=("single",),
    expr=lambda pd, df: pd.Series(
        [pd.Timestamp("2024-03-10 05:06:07").replace(year=2020, month=1, day=31, hour=0)]
    ),
    in_process=True,
    note="an instant with some of its parts swapped. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/timestamp-replace-fine",
    "Timestamp.replace",
    level="L3",
    covers=("microsecond", "nanosecond", "tzinfo", "fold"),
    frames=("single",),
    expr=lambda pd, df: [
        str(pd.Timestamp("2024-03-10 01:30:00.000001").replace(microsecond=5, nanosecond=7)),
        str(pd.Timestamp("2024-03-10 01:30:00").replace(tzinfo=__import__("datetime").UTC)),
        pd.Timestamp("2024-03-10 01:30:00").replace(fold=1).fold,
    ],
    in_process=True,
    note="the sub second parts, the zone and the fold swapped. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/series-to-csv-options",
    "Series.to_csv",
    level="L3",
    covers=("sep", "na_rep", "float_format", "header", "index_label"),
    frames=("float64_half_null",),
    expr=lambda pd, df: (
        df["value"]
        .head(8)
        .to_csv(sep=";", na_rep="-", float_format="%.3f", header=True, index_label="at")
    ),
    in_process=True,
    note="a column as separated text with a missing marker and three float digits. "
    + PARAMETER_IN_PROCESS,
)
case(
    "basics/series-to-csv-bare",
    "Series.to_csv",
    level="L3",
    covers=("index", "header", "lineterminator", "decimal"),
    frames=("float64_half_null",),
    expr=lambda pd, df: (
        df["value"].head(8).to_csv(index=False, header=False, lineterminator="|", decimal=",")
    ),
    in_process=True,
    note="a column with no labels and no header, ended by a bar, with a decimal comma. "
    + PARAMETER_IN_PROCESS,
)
case(
    "basics/to-csv-columns",
    "DataFrame.to_csv",
    level="L3",
    covers=("columns", "header", "index_label", "quoting", "quotechar", "decimal"),
    frames=("two",),
    expr=lambda pd, df: df.to_csv(
        columns=["b", "c"],
        header=["x", "y"],
        index_label="at",
        quoting=1,
        quotechar="'",
        decimal=",",
    ),
    in_process=True,
    note="two columns renamed in the header, every field quoted with a single quote. "
    + PARAMETER_IN_PROCESS,
)
case(
    "basics/to-csv-dates",
    "DataFrame.to_csv",
    level="L3",
    covers=("date_format", "lineterminator"),
    frames=("temporal_range",),
    expr=lambda pd, df: df.to_csv(date_format="%Y/%m/%d %H", lineterminator="\n"),
    in_process=True,
    note="instants written with a format of their own. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/to-csv-escape",
    "DataFrame.to_csv",
    level="L3",
    covers=("doublequote", "escapechar"),
    frames=("single",),
    expr=lambda pd, df: pd.DataFrame({"t": ['a"b', "c"]}).to_csv(
        doublequote=False, escapechar="\\"
    ),
    in_process=True,
    note="a quote inside a field escaped rather than doubled. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/series-to-string-options",
    "Series.to_string",
    level="L3",
    covers=("na_rep", "float_format", "header", "index", "max_rows", "min_rows"),
    frames=("float64_half_null",),
    expr=lambda pd, df: df["value"].to_string(
        na_rep="-", float_format="{:.2f}".format, header=True, index=False, max_rows=6, min_rows=4
    ),
    in_process=True,
    note="a column printed without labels, cut to a few rows. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/series-sample-options",
    "Series.sample",
    level="L3",
    covers=("n", "replace", "ignore_index", "axis"),
    frames=("tall",),
    expr=lambda pd, df: df["value"].sample(
        n=5, replace=True, random_state=1, ignore_index=True, axis=0
    ),
    in_process=True,
    note="five rows drawn with replacement and numbered again. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/series-sample-weights",
    "Series.sample",
    level="L3",
    covers=("frac", "weights"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: df["value"].sample(frac=0.25, weights=df["row"] + 1, random_state=3),
    in_process=True,
    note="a quarter of the rows drawn with weights. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/frame-sample-options",
    "DataFrame.sample",
    level="L3",
    covers=("n", "replace", "weights", "axis", "ignore_index"),
    frames=("two",),
    expr=lambda pd, df: (
        df.sample(n=2, axis=1, random_state=0).columns.tolist()
        + df.sample(n=3, replace=True, weights=[1, 2], random_state=0, ignore_index=True)[
            "a"
        ].tolist()
    ),
    in_process=True,
    note="columns drawn along the other axis and rows drawn with weights. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/series-sort-index-options",
    "Series.sort_index",
    level="L3",
    covers=("na_position", "kind", "key", "axis"),
    frames=("float64_half_null",),
    expr=lambda pd, df: (
        df.set_index("value")["row"]
        .sort_index(na_position="first", kind="stable", axis=0, key=lambda labels: -labels)
        .head(8)
    ),
    in_process=True,
    note="labels sorted through a key with the gaps first. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/frame-sort-index-level",
    "DataFrame.sort_index",
    level="L3",
    covers=("level", "sort_remaining", "kind", "na_position", "key"),
    frames=("keys_two_column",),
    expr=lambda pd, df: df.set_index(["left", "right"]).sort_index(
        level=1, sort_remaining=False, kind="stable", na_position="first", key=None
    ),
    in_process=True,
    note="two levels of labels sorted by the second only. " + PARAMETER_IN_PROCESS,
)
for name in ("cummax", "cummin", "cumprod"):
    case(
        f"basics/{name}-options",
        f"DataFrame.{name}",
        level="L3",
        covers=("axis", "skipna", "numeric_only"),
        frames=("two", "float64_half_null"),
        expr=(
            lambda method: (
                lambda pd, df: getattr(df.select_dtypes("number"), method)(
                    axis=0, skipna=False, numeric_only=True
                )
            )
        )(name),
        in_process=True,
        note="a running answer that stops at the first gap. " + PARAMETER_IN_PROCESS,
    )
case(
    "basics/product-options",
    "DataFrame.product",
    level="L3",
    covers=("axis", "skipna", "numeric_only", "min_count"),
    frames=("two", "float64_half_null"),
    expr=lambda pd, df: pd.concat(
        [
            df.product(numeric_only=True, min_count=3),
            df.product(axis=1, numeric_only=True).add_prefix("row"),
        ]
    ),
    in_process=True,
    note="a product that needs three values and a product across each row. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/reindex-like-options",
    "DataFrame.reindex_like",
    level="L3",
    covers=("other", "method", "limit", "tolerance"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.concat(
        [
            df.iloc[::3].reindex_like(df.iloc[:10], method="ffill", limit=1),
            df.iloc[::3].reindex_like(df.iloc[:10], method="nearest", tolerance=1),
        ]
    ),
    in_process=True,
    note="the labels of another frame, filled forward once or from the nearest label. "
    + PARAMETER_IN_PROCESS,
)
case(
    "basics/str-wrap-options",
    "str.wrap",
    level="L3",
    covers=(
        "expand_tabs",
        "tabsize",
        "replace_whitespace",
        "drop_whitespace",
        "initial_indent",
        "subsequent_indent",
        "fix_sentence_endings",
        "break_long_words",
        "break_on_hyphens",
        "max_lines",
        "placeholder",
    ),
    frames=("strings_ascii",),
    expr=lambda pd, df: df["value"].str.wrap(
        6,
        expand_tabs=False,
        tabsize=4,
        replace_whitespace=False,
        drop_whitespace=False,
        initial_indent=">",
        subsequent_indent=" ",
        fix_sentence_endings=True,
        break_long_words=False,
        break_on_hyphens=False,
        max_lines=2,
        placeholder="~",
    ),
    in_process=True,
    note="every textwrap option handed through. " + PARAMETER_IN_PROCESS,
)


def _plus(values, step, extra=0):
    """A function with a positional and a keyword argument, for the `args` and `kwargs` cases."""
    return values + step + extra


case(
    "basics/frame-apply-raw",
    "DataFrame.apply",
    level="L3",
    covers=("func", "raw", "args", "kwargs"),
    frames=("two", "float64_half_null"),
    expr=lambda pd, df: df.select_dtypes("number").apply(
        lambda values, step, extra: values.max() + step + extra, raw=True, args=(1,), extra=2
    ),
    in_process=True,
    note="each column handed over as a numpy array. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/frame-apply-expand",
    "DataFrame.apply",
    level="L3",
    covers=("result_type",),
    frames=("two", "float64_half_null"),
    expr=lambda pd, df: df.select_dtypes("number").apply(
        lambda row: [row.iloc[0], row.iloc[-1] * 2], axis=1, result_type="expand"
    ),
    in_process=True,
    note="a list for each row made into columns. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/frame-apply-broadcast",
    "DataFrame.apply",
    level="L3",
    covers=("result_type",),
    frames=("two", "float64_half_null"),
    expr=lambda pd, df: df.select_dtypes("number").apply(
        lambda column: column.sum(), result_type="broadcast"
    ),
    in_process=True,
    note="each column's total spread down the column. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/frame-apply-engine",
    "DataFrame.apply",
    level="L3",
    covers=("by_row", "engine", "engine_kwargs"),
    frames=("two",),
    expr=lambda pd, df: df[["a", "b"]].apply(
        lambda column: column * 2, by_row=False, engine="python", engine_kwargs=None
    ),
    in_process=True,
    note="the Python engine named, on whole columns. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/frame-transform-options",
    "DataFrame.transform",
    level="L3",
    covers=("func", "axis", "args", "kwargs"),
    frames=("two", "float64_half_null"),
    expr=lambda pd, df: df.select_dtypes("number").transform(_plus, 1, 1, extra=2),
    in_process=True,
    note="a transform across each row with arguments. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/series-transform-options",
    "Series.transform",
    level="L3",
    covers=("func", "axis", "args", "kwargs"),
    frames=("two",),
    expr=lambda pd, df: df["a"].transform(
        {"plus": _plus, "twice": lambda v, step: v * 2 + step}, 0, 1
    ),
    in_process=True,
    note="a dict of functions, which answers a frame. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/frame-agg-rows",
    "DataFrame.agg",
    level="L3",
    covers=("axis", "args", "kwargs"),
    frames=("two", "float64_half_null"),
    expr=lambda pd, df: (
        df.select_dtypes("number")
        .agg(["sum", "max"], axis=1)
        .assign(
            plus=df.select_dtypes("number").agg(
                lambda row, step, extra: row.sum() + step + extra, 1, 1, extra=2
            )
        )
    ),
    in_process=True,
    note="a list of reductions across each row, and a function with an argument. "
    + PARAMETER_IN_PROCESS,
)
case(
    "basics/frame-map-options",
    "DataFrame.map",
    level="L3",
    covers=("func", "na_action", "kwargs"),
    frames=("two", "float64_half_null"),
    expr=lambda pd, df: df.select_dtypes("number").map(_plus, na_action="ignore", step=1),
    in_process=True,
    note="each cell through a function with a keyword, gaps left alone. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/frame-pipe-options",
    "DataFrame.pipe",
    level="L3",
    covers=("func", "args", "kwargs"),
    frames=("two",),
    expr=lambda pd, df: df[["a", "b"]].pipe(_plus, 1, extra=2),
    in_process=True,
    note="the frame through a function with arguments. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/series-pipe-options",
    "Series.pipe",
    level="L3",
    covers=("func", "args", "kwargs"),
    frames=("two",),
    expr=lambda pd, df: df["b"].pipe((lambda step, values: values * step, "values"), 3),
    in_process=True,
    note="a column piped in as a named argument. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/series-agg-options",
    "Series.agg",
    level="L3",
    covers=("axis", "args", "kwargs"),
    frames=("two", "float64_half_null"),
    expr=lambda pd, df: df.iloc[:, 1].agg(
        lambda values, step, extra: values.sum() + step + extra, 0, 1, extra=2
    ),
    in_process=True,
    note="a reduction with arguments. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/series-apply-options",
    "Series.apply",
    level="L3",
    covers=("func", "by_row", "kwargs"),
    frames=("two",),
    expr=lambda pd, df: df["a"].apply(_plus, by_row=False, step=1, extra=2),
    in_process=True,
    note="the whole column through a function with keywords. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/series-map-options",
    "Series.map",
    level="L3",
    covers=("engine", "kwargs"),
    frames=("two",),
    expr=lambda pd, df: df["a"].map(_plus, engine=None, step=1),
    in_process=True,
    note="each value through a function with a keyword. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/index-dtype",
    "pandas.Index",
    covers=("dtype",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Index(df["value"].head(5).tolist(), dtype="float64"),
    in_process=True,
    note="the labels are read as a series of the asked type reads them. Since firepanda #1363",
)
case(
    "basics/index-dtype-dates",
    "pandas.Index",
    covers=("dtype",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Index(["2026-01-01", "2026-03-01"], dtype="datetime64[s]"),
    in_process=True,
    note="text asked for as instants is parsed, and the index is a DatetimeIndex. Since "
    "firepanda #1363",
)
case(
    "basics/series-dtype-dates",
    "pandas.Series",
    covers=("dtype",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series(["2026-01-01", None, "2026-01-03 10:00"], dtype="datetime64[ms]"),
    in_process=True,
    note="each value parsed on its own and given the unit, not cast from counts. Since "
    "firepanda #1364",
)
case(
    "basics/series-dtype-spans",
    "pandas.Series",
    covers=("dtype",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series(["1D", None, "90s"], dtype="timedelta64[ns]"),
    in_process=True,
    note="parsed as to_timedelta parses. Since firepanda #1363",
)
case(
    "basics/series-dtype-fraction",
    "pandas.Series",
    level="L4",
    covers=("dtype",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series([1.5, 2.0], dtype="int64"),
    raises=("ValueError", "Trying to coerce float values to integers"),
    in_process=True,
    note="a list is refused rather than truncated when it does not fit a whole number type. "
    "Since firepanda #1363",
)
case(
    "basics/series-dtype-too-big",
    "pandas.Series",
    level="L4",
    covers=("dtype",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series([300], dtype="uint8"),
    raises=("OverflowError", "cannot all be casted to the dtype uint8"),
    in_process=True,
    note="a number outside the type's range is refused rather than wrapped. Since firepanda #1363",
)
case(
    "basics/index-objects",
    "pandas.Index",
    covers=("dtype",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Index(["a", 1, None, 2.5]),
    in_process=True,
    note="labels of no one type are an index of objects, each label as written. Since "
    "firepanda #1366",
)
case(
    "basics/index-objects-lookup",
    "pandas.Series.loc",
    covers=(),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series([10, 20, 30], index=pd.Index(["a", 1, 2.5])).loc[[1, "a"]],
    in_process=True,
    note="an object label is found by its value. Since firepanda #1366",
)
case(
    "basics/index-objects-sort",
    "pandas.Series.sort_index",
    covers=(),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series(
        [1, 2, 3], index=pd.Index([3, 1, 2.5], dtype=object)
    ).sort_index(),
    in_process=True,
    note="object labels sort the way Python sorts them. Since firepanda #1366",
)
case(
    "basics/index-objects-unorderable",
    "pandas.Index.sort_values",
    level="L4",
    covers=(),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Index([3, "b"]).sort_values(),
    raises=("TypeError", "'<' not supported between instances of"),
    in_process=True,
    note="a mix Python cannot order raises as pandas raises. Since firepanda #1366",
)
case(
    "basics/series-object-keeps-ints",
    "pandas.Series",
    covers=("dtype",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series([3, 1, 2.5], dtype=object),
    in_process=True,
    note="an integer asked for as an object stays an integer. Since firepanda #1366",
)
case(
    "basics/concat-label-kinds",
    "pandas.concat",
    covers=("objs",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.concat([pd.Series([1], index=["a"]), pd.Series([2], index=[1])]),
    in_process=True,
    note="labels of two kinds meet as an index of objects. Since firepanda #1366",
)
case(
    "basics/index-masked",
    "pandas.Index",
    covers=("dtype",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Index([3, None, 1], dtype="Int64"),
    in_process=True,
    note="a masked index keeps its type and its NA. Since firepanda #1367",
)
case(
    "basics/index-masked-text",
    "pandas.Index",
    covers=("dtype",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Index(["a", None], dtype="string"),
    in_process=True,
    note="text with NA is the string index. Since firepanda #1367",
)
case(
    "basics/index-masked-lookup",
    "pandas.Series.loc",
    covers=(),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series([1, 2, 3], index=pd.Index([3, None, 1], dtype="Int64")).loc[
        [1, 3]
    ],
    in_process=True,
    note="a masked label is found by its value. Since firepanda #1367",
)
case(
    "basics/index-masked-set-index",
    "pandas.DataFrame.set_index",
    covers=("keys",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame(
        {"k": pd.Series([3, None, 1], dtype="Int64"), "v": [1, 2, 3]}
    ).set_index("k"),
    in_process=True,
    note="a masked column becomes a masked index. Since firepanda #1367",
)
case(
    "basics/index-masked-sort",
    "pandas.Series.sort_index",
    covers=(),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series(
        [1, 2, 3], index=pd.Index([3, None, 1], dtype="Int64")
    ).sort_index(),
    in_process=True,
    note="masked labels sort by value with NA last. Since firepanda #1367",
)
case(
    "basics/astype-text-dates",
    "Series.astype",
    covers=("dtype",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series(["2026-01-01", None, "2026-03-04 10:00"]).astype(
        "datetime64[ms]"
    ),
    in_process=True,
    note="text parses to instants at the unit asked for. Since firepanda #1371 and #1372",
)
case(
    "basics/astype-text-spans",
    "Series.astype",
    covers=("dtype",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series(["1D", "2h", None]).astype("timedelta64[s]"),
    in_process=True,
    note="text parses to spans at the unit asked for. Since firepanda #1371",
)
case(
    "basics/frame-dtype-dates",
    "pandas.DataFrame",
    covers=("dtype",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame(
        {"a": ["2026-01-01", "2026-01-02"], "b": ["2026-02-01", None]}, dtype="datetime64[ns]"
    ),
    in_process=True,
    note="dtype= with an instant type parses each column. Since firepanda #1371",
)
case(
    "basics/index-masked-map",
    "Index.map",
    covers=("mapper",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series(
        [1, 2, 3], index=pd.Index([1, None, 3], dtype="Int32").map(lambda x: x * 1.5)
    ),
    in_process=True,
    note="a masked index maps to a masked type. Since firepanda #1371",
)
case(
    "basics/mask-gap-refused",
    "Series.mask",
    level="L4",
    covers=("cond",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series([1, 2, 3]).mask([True, None, False]),
    raises=("TypeError", "bad operand type for unary ~"),
    in_process=True,
    note="a gap in a list condition cannot be turned over. Since firepanda #1371",
)


def _assigned(owner, attribute, value):
    """The object after `owner.attribute = value`, which a lambda cannot write."""
    setattr(owner, attribute, value)
    return owner


def _index_named(owner, value):
    """The object after its index is named through `owner.index.name = value`."""
    owner.index.name = value
    return owner


case(
    "basics/index-tuples-levels",
    "pandas.Index",
    covers=("data",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series([1, 2], index=pd.Index([("a", 1), ("b", 2)])),
    in_process=True,
    note="a list of tuples is read as levels. Since firepanda #1373",
)
case(
    "basics/index-tuples-kept",
    "pandas.Index",
    covers=("tupleize_cols",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series(
        [1, 2], index=pd.Index([("a", 1), ("b", 2)], tupleize_cols=False)
    ).reset_index(drop=True),
    in_process=True,
    note="turning tupleize_cols off keeps the tuples as labels. Since firepanda #1373",
)
case(
    "basics/series-tuple-keys",
    "pandas.Series",
    covers=("data",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series({("a", 1): 1, ("b", 2): 2, ("b", 3): 3}),
    in_process=True,
    note="a mapping with tuple keys is labelled by levels. Since firepanda #1373",
)
case(
    "basics/columns-assigned",
    "DataFrame.columns",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: _assigned(df.head(3), "columns", [f"c{i}" for i in range(df.shape[1])]),
    in_process=True,
    note="assigning to the columns relabels them. Since firepanda #1374",
)
case(
    "basics/index-assigned",
    "DataFrame.index",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: _assigned(df.head(3), "index", ["x", "y", "z"]),
    in_process=True,
    note="assigning to the index relabels the rows. Since firepanda #1374",
)
case(
    "basics/name-assigned",
    "Series.name",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: _assigned(df["value"].head(3), "name", "renamed"),
    in_process=True,
    note="assigning a name renames the column. Since firepanda #1374",
)
case(
    "basics/index-name-assigned",
    "DataFrame.index",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: _index_named(df.head(3), "rows"),
    in_process=True,
    note="naming an index read off a frame names the frame's rows. Since firepanda #1374",
)
case(
    "basics/columns-assigned-short",
    "DataFrame.columns",
    level="L4",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: _assigned(df.head(3), "columns", ["only"]),
    raises=("ValueError", "Length mismatch"),
    in_process=True,
    note="labels of another length are refused. Since firepanda #1374",
)
case(
    "basics/number-plus-text-refused",
    "Series.add",
    level="L4",
    frames=("strings_ascii",),
    expr=lambda pd, df: pd.Series([1, 2, 3], index=df.head(3).index) + df.head(3)["value"],
    raises=("TypeError", "operation 'radd' not supported for dtype 'str'"),
    in_process=True,
    note="a number on the left hands the text its reflected operator. Since firepanda #1375",
)
case(
    "basics/frame-plus-number-text-refused",
    "DataFrame.add",
    level="L4",
    frames=("strings_ascii",),
    expr=lambda pd, df: df.head(3) + 1,
    raises=("TypeError", "operation 'add' not supported for dtype 'str'"),
    in_process=True,
    note="the text column refuses the number as it would alone. Since firepanda #1375",
)
case(
    "basics/attribute-writes-column",
    "DataFrame.columns",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: _assigned(df.head(3), "value", [7, 8, 9]),
    in_process=True,
    note="an attribute that names a column writes the column. Since firepanda #1375",
)
case(
    "basics/text-sum-joins",
    "Series.sum",
    frames=("strings_ascii", "strings_null_heavy"),
    expr=lambda pd, df: df["value"].head(20).sum(),
    in_process=True,
    note="a sum of text joins the rows in order and passes over gaps. Since firepanda #1376",
)
case(
    "basics/text-cumsum-joins",
    "Series.cumsum",
    frames=("strings_ascii", "strings_null_heavy"),
    expr=lambda pd, df: df["value"].head(8).cumsum(),
    in_process=True,
    note="a running sum of text is a running join. Since firepanda #1376",
)
case(
    "basics/text-group-sum-joins",
    "GroupBy.sum",
    frames=("strings_ascii", "strings_null_heavy"),
    expr=lambda pd, df: df.head(6).assign(k=["a", "b"] * 3).groupby("k")["value"].sum(),
    in_process=True,
    note="each group's text is joined in row order. Since firepanda #1376",
)
case(
    "basics/text-mean-refused",
    "Series.mean",
    level="L4",
    frames=("strings_ascii",),
    expr=lambda pd, df: df["value"].mean(),
    raises=("TypeError", "Cannot perform reduction 'mean' with string dtype"),
    in_process=True,
    note="a mean has no meaning for text. Since firepanda #1376",
)
case(
    "basics/rename-rows-and-columns",
    "DataFrame.rename",
    level="L3",
    covers=("index", "columns"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: df.head(4).rename(index=lambda label: label * 10, columns=str.upper),
    in_process=True,
    note="both axes in one call, the row labels through a function and built into an index again",
)
case(
    "basics/rename-one-level",
    "DataFrame.rename",
    level="L3",
    covers=("index", "level"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame(
        {"v": [1, 2, 3]},
        index=pd.MultiIndex.from_arrays([["p", "p", "q"], ["x", "y", "z"]], names=["o", "i"]),
    ).rename(index={"p": "P", "x": "X"}, level="o"),
    in_process=True,
    note="level names the one level of a MultiIndex the mapping applies to",
)
case(
    "basics/series-rename-labels",
    "Series.rename",
    level="L3",
    covers=("index",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: df["value"].head(3).rename({0: "zero", 2: "two"}),
    in_process=True,
    note="a mapping renames the row labels rather than the column, a label with no key "
    "kept as it was",
)
case(
    "basics/rename-missing-label",
    "DataFrame.rename",
    level="L4",
    covers=("index", "errors"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: df.rename(index={"not_a_label": 1}, errors="raise"),
    raises=("KeyError", "['not_a_label'] not found in axis"),
    in_process=True,
    note="errors='raise' lists the keys that are not labels",
)
for whole in ("mean", "median", "std", "var", "sem", "skew", "kurt"):
    case(
        f"basics/frame-whole-{whole}",
        f"DataFrame.{whole}",
        level="L3",
        covers=("axis",),
        frames=("wide", "int64_no_nulls"),
        expr=(lambda method: lambda pd, df: getattr(df, method)(axis=None))(whole),
        rules=Rules(
            tolerance=Tolerance.STATISTICAL,
            reason="every cell is read in one pass, and the spreads and shapes are sums "
            "of powers that each library adds up in its own order",
        ),
        in_process=True,
        note="axis=None reads every cell of the frame at once, since a mean of means is "
        "not a mean. Since firepanda #1403",
    )
case(
    "basics/series-add-on-level",
    "Series.add",
    level="L3",
    covers=("other", "level"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series(
        [1.0, 2.0, 4.0, 8.0],
        index=pd.MultiIndex.from_arrays(
            [["p", "p", "q", "r"], ["x", "y", "z", "x"]], names=["o", "i"]
        ),
    ).add(pd.Series([10.0, 20.0], index=["p", "q"]), level="o"),
    in_process=True,
    note="level reads the flat side once for every row by its label on that level",
)
case(
    "basics/series-mul-on-level-filled",
    "Series.mul",
    level="L3",
    covers=("other", "level", "fill_value"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series(
        [1.0, 2.0, 4.0, 8.0],
        index=pd.MultiIndex.from_arrays(
            [["p", "p", "q", "r"], ["x", "y", "z", "x"]], names=["o", "i"]
        ),
    ).mul(pd.Series([10.0, 20.0], index=["p", "q"]), level=0, fill_value=1),
    in_process=True,
    note="a row whose label the flat side lacks takes fill_value",
)
case(
    "basics/frame-mul-on-level",
    "DataFrame.mul",
    level="L3",
    covers=("other", "axis", "level"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame(
        {"a": [1.0, 2.0, 4.0, 8.0], "b": [1, 2, 3, 4]},
        index=pd.MultiIndex.from_arrays(
            [["p", "p", "q", "r"], ["x", "y", "z", "x"]], names=["o", "i"]
        ),
    ).mul(pd.Series([100, 200, 300], index=["x", "y", "z"]), axis=0, level="i"),
    in_process=True,
    note="a frame's rows read a Series on flat labels by the inner level",
)
case(
    "basics/frame-mul-multiindex-columns-int",
    "DataFrame.mul",
    level="L3",
    covers=("other", "axis"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame(
        [[1, 2, 3]], columns=pd.MultiIndex.from_tuples([("A", "x"), ("A", "y"), ("B", "x")])
    ).mul(
        pd.Series(
            [10, 10, 100], index=pd.MultiIndex.from_tuples([("A", "x"), ("A", "y"), ("B", "x")])
        ),
        axis=1,
    ),
    in_process=True,
    note="an int Series with a label for every column keeps the answer int64",
)
case(
    "basics/frame-sum-flags-and-ints",
    "DataFrame.sum",
    level="L2",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame({"a": [1, 2], "b": [True, False]}).sum(),
    in_process=True,
    note="flags add up as whole numbers, so the answer stays int64",
)
case(
    "basics/frame-max-flags-and-ints",
    "DataFrame.max",
    level="L2",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame({"a": [1, 2], "b": [True, False]}).max(),
    in_process=True,
    note="answers that share no type are held as they are in an object answer",
)
case(
    "basics/frame-min-text-and-ints",
    "DataFrame.min",
    level="L2",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).min(),
    in_process=True,
    note="a number and a string side by side in an object answer",
)
case(
    "basics/frame-row-sum-flags-and-numbers",
    "DataFrame.sum",
    level="L3",
    covers=("axis",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame({"a": [1, 2], "b": [True, False], "c": [1.5, 2.5]}).sum(
        axis=1
    ),
    in_process=True,
    note="a row of flags beside numbers is read as objects and added in Python",
)
case(
    "basics/frame-row-mean-flags-and-gaps",
    "DataFrame.mean",
    level="L3",
    covers=("axis", "skipna"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame({"a": [1.0, None, 3.0], "b": [True, False, True]}).mean(
        axis=1, skipna=True
    ),
    in_process=True,
    note="the mean of an object row skips the gap",
)
case(
    "basics/frame-cumsum-across-rows",
    "DataFrame.cumsum",
    level="L3",
    covers=("axis",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame(
        {"a": [1, 2], "b": [3, 4], "c": [5, 6]}, index=["x", "y"]
    ).cumsum(axis=1),
    in_process=True,
    note="a scan across the rows is the scan down the frame turned on its side",
)
case(
    "basics/frame-cummax-across-rows-with-gaps",
    "DataFrame.cummax",
    level="L3",
    covers=("axis", "skipna"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame({"a": [1.0, None], "b": [3.0, 4.0], "c": [None, 6.0]}).cummax(
        axis=1, skipna=True
    ),
    in_process=True,
    note="a gap across a row is passed over",
)
case(
    "basics/frame-cumsum-flags-across-rows",
    "DataFrame.cumsum",
    level="L3",
    covers=("axis",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame({"a": [1.0, None], "b": [True, False], "c": [2, 3]}).cumsum(
        axis=1
    ),
    in_process=True,
    note="flags beside numbers turn into objects scanned in Python",
)
case(
    "basics/frame-rank-across-rows",
    "DataFrame.rank",
    level="L3",
    covers=("axis", "pct"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame({"a": [1, 2], "b": [3, 1], "c": [5, 6]}).rank(
        axis=1, pct=True
    ),
    in_process=True,
    note="ranked across each row, as pandas ranks the frame turned on its side",
)
case(
    "basics/frame-shift-columns",
    "DataFrame.shift",
    level="L3",
    covers=("periods", "axis"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame({"a": [1, 2], "b": [3, 1], "c": [5, 6]}).shift(1, axis=1),
    in_process=True,
    note="whole columns move along, each keeping its type",
)
case(
    "basics/frame-diff-across-rows",
    "DataFrame.diff",
    level="L3",
    covers=("periods", "axis"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame({"a": [1, 2], "b": [3, 1], "c": [5, 6]}).diff(-1, axis=1),
    in_process=True,
    note="the frame less its columns shifted along keeps whole numbers whole",
)
case(
    "basics/frame-pct-change-across-rows",
    "DataFrame.pct_change",
    level="L3",
    covers=("periods",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame({"a": [1, 2], "b": [3, 1], "c": [5, 6]}).pct_change(
        periods=1, axis=1
    ),
    in_process=True,
    note="each column over the one before it, less one",
)
case(
    "basics/frame-ffill-across-rows",
    "DataFrame.ffill",
    level="L3",
    covers=("axis",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame({"a": [1.0, None], "b": [3.0, 4.0], "c": [None, 6.0]}).ffill(
        axis=1
    ),
    in_process=True,
    note="a gap takes the value to its left",
)
case(
    "basics/frame-interpolate-across-rows",
    "DataFrame.interpolate",
    level="L3",
    covers=("axis",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame(
        {"a": [1.0, None], "b": [None, 4.0], "c": [5.0, 6.0]}
    ).interpolate(axis=1),
    in_process=True,
    note="a gap across a row is on the line between its neighbours",
)
case(
    "basics/series-value-counts-bins",
    "Series.value_counts",
    level="L3",
    covers=("bins",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series([1, 2, 2, 5, 9, 10]).value_counts(bins=3),
    in_process=True,
    note="values counted in equal-width bins, the bins an unnamed interval index",
)
case(
    "basics/frame-set-index-verify-integrity",
    "DataFrame.set_index",
    level="L3",
    covers=("keys", "verify_integrity"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame({"k": [1, 2, 3], "v": [4, 5, 6]}).set_index(
        "k", verify_integrity=True
    ),
    in_process=True,
    note="labels each seen once pass the check",
)
case(
    "basics/frame-set-index-verify-integrity-repeated",
    "DataFrame.set_index",
    level="L4",
    covers=("keys", "verify_integrity"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame({"k": [1, 1, 3], "v": [4, 5, 6]}).set_index(
        "k", verify_integrity=True
    ),
    raises=("ValueError", "Index has duplicate keys"),
    in_process=True,
    note="a repeated label is refused",
)
case(
    "basics/groupby-quantile-list",
    "GroupBy.quantile",
    level="L3",
    covers=("q",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: (
        pd.DataFrame({"k": ["b", "a", "b", "a", "b"], "v": [1.0, 2.0, 3.0, 4.0, 5.0]})
        .groupby("k")
        .quantile([0.25, 0.75])
    ),
    in_process=True,
    note="a list of quantiles is a level after the keys",
)
case(
    "basics/groupby-quantile-picked",
    "GroupBy.quantile",
    level="L3",
    covers=("q", "interpolation"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: (
        pd.DataFrame({"k": ["b", "a", "b", "a", "b"], "w": [5, 4, 3, 2, 1]})
        .groupby("k")["w"]
        .quantile([0.3, 0.6], interpolation="nearest")
    ),
    in_process=True,
    note="a picking rule keeps whole numbers whole",
)
case(
    "basics/groupby-nth-slice",
    "GroupBy.nth",
    level="L2",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: (
        pd.DataFrame({"k": ["a", "b", "a", "a", "b", "a"], "v": [1, 2, 3, 4, 5, 6]})
        .groupby("k")
        .nth(slice(-3, None, 2))
    ),
    in_process=True,
    note="a slice from the back with a step keeps the rows pandas' mask keeps",
)
case(
    "basics/frame-sort-index-directions",
    "DataFrame.sort_index",
    level="L3",
    covers=("ascending",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame(
        {"v": [1, 2, 3, 4]},
        index=pd.MultiIndex.from_tuples([("b", 1), ("a", 2), ("b", 3), ("a", 1)]),
    ).sort_index(ascending=[True, False]),
    in_process=True,
    note="each level sorted in the direction given for it",
)
case(
    "basics/frame-sort-index-level-directions",
    "DataFrame.sort_index",
    level="L3",
    covers=("level", "ascending"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame(
        {"v": [1, 2, 3, 4]},
        index=pd.MultiIndex.from_tuples([("b", 1), ("a", 2), ("b", 3), ("a", 1)]),
    ).sort_index(level=1, ascending=[False]),
    in_process=True,
    note="a direction for the level asked for leaves the other level in the order it came in",
)
case(
    "basics/frame-shift-period-list",
    "DataFrame.shift",
    level="L3",
    covers=("periods", "suffix"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame({"v": [1, 2, 3, 4], "w": [1.5, 2.5, 3.5, 4.5]}).shift(
        [0, 1, -1], suffix="_lag"
    ),
    in_process=True,
    note="a shift by each period side by side, named after the column, suffix and period",
)
case(
    "basics/groupby-shift-period-list",
    "GroupBy.shift",
    level="L3",
    covers=("periods",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: (
        pd.DataFrame({"k": ["a", "b", "a", "b"], "v": [1, 2, 3, 4]}).groupby("k")["v"].shift([0, 1])
    ),
    in_process=True,
    note="a group by shifts within each group by each period and answers a frame",
)
case(
    "basics/frame-quantile-across-rows",
    "DataFrame.quantile",
    level="L3",
    covers=("q", "axis", "interpolation"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame(
        {"a": [1, 2, 3], "b": [4.5, 0.5, 6.0], "c": [7, 8, 1]}, index=["x", "y", "z"]
    ).quantile([0.1, 0.9], axis=1, interpolation="nearest"),
    in_process=True,
    note="a quantile across each row is one down the frame turned on its side",
)
case(
    "basics/series-unstack-level-list",
    "Series.unstack",
    level="L3",
    covers=("level", "fill_value"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series(
        [1, 2, 3, 4, 5],
        index=pd.MultiIndex.from_tuples(
            [
                ("r1", "b", "x"),
                ("r1", "a", "y"),
                ("r2", "b", "y"),
                ("r2", "a", "x"),
                ("r1", "b", "y"),
            ],
            names=["r", "p", "q"],
        ),
    ).unstack(["p", "q"], fill_value=0),
    in_process=True,
    note="several levels unstacked at once spread their pairs across the columns",
)
case(
    "basics/frame-unstack-level-list",
    "DataFrame.unstack",
    level="L3",
    covers=("level",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame(
        {"v": [1, 2, 3, 4]},
        index=pd.MultiIndex.from_tuples(
            [("r1", "b", "x"), ("r1", "a", "y"), ("r2", "b", "y"), ("r2", "a", "x")],
            names=["r", "p", "q"],
        ),
    ).unstack([1, 2]),
    in_process=True,
    note="a frame unstacks several levels under its own column names",
)
case(
    "basics/frame-at-repeated-label",
    "DataFrame.at",
    level="L2",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame({"v": [1, 2, 3]}, index=["x", "y", "x"]).at["x", "v"],
    in_process=True,
    note="at on a label there twice answers every value under it, as loc does",
)
case(
    "basics/series-at-repeated-label",
    "Series.at",
    level="L2",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series([1.5, 2.5, 3.5], index=["x", "y", "x"]).at["x"],
    in_process=True,
    note="a series answers a label there twice with both values",
)
case(
    "basics/series-compare-align-rows",
    "Series.compare",
    level="L3",
    covers=("align_axis", "keep_shape"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series([1, 2, 3, 4], index=["w", "x", "y", "z"]).compare(
        pd.Series([1, 5, 3, 7], index=["w", "x", "y", "z"]), align_axis=0, keep_shape=True
    ),
    in_process=True,
    note="the two sides take turns down the rows under the label and the side",
)
case(
    "basics/frame-mode-along-rows",
    "DataFrame.mode",
    level="L3",
    covers=("axis", "dropna"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame(
        {"a": [1.5, None, 2.0], "b": [1.5, None, 3.0], "c": [None, None, 3.0]}
    ).mode(axis=1, dropna=False),
    in_process=True,
    note="the most common values of each row, a column per place",
)
case(
    "basics/series-getitem-nullable-mask",
    "Series.__getitem__",
    level="L2",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series([1, 2, 3], index=["a", "b", "c"])[
        pd.Series([True, None, False], index=["a", "b", "c"], dtype="boolean")
    ],
    in_process=True,
    note="a mask of flags that can be missing reads a missing flag as false",
)
case(
    "basics/frame-from-records-index-levels",
    "DataFrame.from_records",
    level="L3",
    covers=("index", "exclude"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame.from_records(
        [{"k": "a", "n": 1, "v": 2.5, "w": 0}, {"k": "b", "n": 2, "v": 3.5, "w": 1}],
        index=["k", "n"],
        exclude=["w"],
    ),
    in_process=True,
    note="several index columns label the rows by a level each",
)
case(
    "basics/groupby-idxmax-keys-as-columns",
    "GroupBy.idxmax",
    level="L2",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: (
        pd.DataFrame({"k": ["a", "b", "a", "b"], "v": [3, 1, 5, 2]}, index=["p", "q", "r", "s"])
        .groupby("k", as_index=False)
        .idxmax()
    ),
    in_process=True,
    note="with as_index=False the keys come back as columns before the labels found",
)
case(
    "basics/groupby-multiindex-levels",
    "DataFrame.groupby",
    level="L3",
    covers=("level", "sort"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: (
        pd.DataFrame(
            {"v": [1, 2, 3, 4]},
            index=pd.MultiIndex.from_tuples(
                [("x", 1), ("y", 1), ("x", 2), ("y", 2)], names=["p", "q"]
            ),
        )
        .groupby(level=["q", "p"], sort=False)
        .sum()
    ),
    in_process=True,
    note="the levels of a MultiIndex named in level= are the keys",
)
case(
    "basics/groupby-grouper-freq-no-key",
    "DataFrame.groupby",
    level="L3",
    covers=("by",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: (
        pd.DataFrame(
            {"v": [1, 2, 3, 4, 5]}, index=pd.date_range("2024-01-01", periods=5, freq="12h")
        )
        .groupby(pd.Grouper(freq="D"))
        .sum()
    ),
    in_process=True,
    note="a Grouper with a frequency and no key bins the row labels",
)

case(
    "basics/groupby-grouper-freq-in-list",
    "DataFrame.groupby",
    level="L3",
    covers=("by",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: (
        pd.DataFrame(
            {
                "t": pd.to_datetime(
                    ["2024-01-01 03:00", "2024-01-01 20:00", "2024-01-02 05:00", "2024-01-03 01:00"]
                ),
                "k": ["a", "b", "a", "b"],
                "v": [1, 2, 3, 4],
            }
        )
        .groupby([pd.Grouper(key="t", freq="D"), "k"])
        .sum()
    ),
    in_process=True,
    note="a Grouper with a frequency beside a column bins each row",
)

case(
    "basics/groupby-level-name-in-list",
    "DataFrame.groupby",
    level="L3",
    covers=("by",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: (
        pd.DataFrame(
            {"v": [1, 2, 3, 4], "k": ["a", "b", "a", "b"]},
            index=pd.MultiIndex.from_tuples(
                [("x", 1), ("y", 1), ("x", 2), ("y", 2)], names=["p", "q"]
            ),
        )
        .groupby(["p", "k"])
        .sum()
    ),
    in_process=True,
    note="a level name in a list of keys groups by that level",
)

case(
    "basics/corr-callable-method",
    "DataFrame.corr",
    level="L3",
    covers=("method", "min_periods"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame(
        {"a": [1, 2, 3, 4, 7], "b": [2.0, None, 1.0, 8.0, 3.0], "c": [3, 3, 3, 3, 3]}
    ).corr(method=lambda x, y: float(sum(x * y)), min_periods=4),
    in_process=True,
    note="a callable is handed the rows each pair shares, 1 on the diagonal",
)

case(
    "basics/multiindex-whole-level-gap",
    "DataFrame.sort_index",
    level="L3",
    covers=("ascending",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame(
        {"v": [1, 2, 3, 4, 5]},
        index=pd.MultiIndex.from_tuples(
            [("b", 2), ("a", 1), ("b", 1), ("a", 3), ("a", None)], names=["p", "q"]
        ),
    ).sort_index(ascending=[True, False]),
    in_process=True,
    note="a level of whole numbers with a gap stays whole",
)

case(
    "basics/multiindex-levels-frozen",
    "MultiIndex.levels",
    level="L2",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: repr(
        pd.MultiIndex.from_tuples([("b", 2), ("a", 1)], names=["p", "q"]).levels
    ),
    in_process=True,
    note="levels is a FrozenList printing each level's values",
)

case(
    "basics/multiindex-names-frozen",
    "MultiIndex.names",
    level="L2",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: repr(
        pd.MultiIndex.from_tuples([("b", 2), ("a", 1)], names=["p", "q"]).names
    ),
    in_process=True,
    note="names is a FrozenList",
)

case(
    "basics/groupby-key-selected-as-index-false",
    "DataFrame.groupby",
    level="L3",
    covers=("by", "as_index"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: (
        pd.DataFrame({"k": ["a", "b", "a"], "j": [1, 1, 2], "v": [1.0, 2.0, 3.0]})
        .groupby(["k", "j"], as_index=False)[["j", "v"]]
        .max()
    ),
    in_process=True,
    note="a reduced key shows its value alone with as_index=False",
)

case(
    "basics/groupby-apply-scalar-as-index-false",
    "GroupBy.apply",
    level="L3",
    covers=("func",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: repr(
        pd.DataFrame({"k": ["a", "b", "a"], "v": [1.0, 2.0, 3.0]})
        .groupby("k", as_index=False)
        .apply(lambda g: g["v"].sum())
    ),
    in_process=True,
    note="one value a group goes beside the keys in a column named None",
)

case(
    "basics/groupby-transform-name-with-args",
    "GroupBy.transform",
    level="L3",
    covers=("func",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: (
        pd.DataFrame({"k": ["a", "b", "a", "b"], "v": [1.0, 2.0, 3.0, 5.0]})
        .groupby("k")["v"]
        .transform("quantile", 0.25)
    ),
    in_process=True,
    note="a reduction by name takes its arguments and spreads over the group",
)

case(
    "basics/resample-aggregate-function",
    "Resampler.aggregate",
    level="L3",
    covers=("func",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: (
        pd.DataFrame(
            {"a": [1, 2, 3, 4], "b": [1.5, 2.5, 3.5, 4.5]},
            index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-07", "2024-01-08"]),
        )
        .resample("2D")
        .agg(lambda x: x.sum() if len(x) else -1)
    ),
    in_process=True,
    note="a function runs on every bin, the empty ones too, and on each column",
)

case(
    "basics/resample-ohlc-frame",
    "Resampler.ohlc",
    level="L2",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: (
        pd.DataFrame(
            {"a": [1, 2, 3, 4], "b": [1.5, 2.5, 3.5, 4.5]},
            index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-07", "2024-01-08"]),
        )
        .resample("2D")
        .ohlc()
    ),
    in_process=True,
    note="over a frame each column gets its four under two levels of column labels",
)

case(
    "basics/resample-zoned-day",
    "Series.resample",
    level="L3",
    covers=("rule",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: (
        pd.Series(
            range(8),
            index=pd.date_range("2024-11-02 22:00", periods=8, freq="2h", tz="US/Eastern"),
        )
        .resample("D")
        .sum()
    ),
    in_process=True,
    note="days on the zone's own clock, one of them 25 hours long",
)

case(
    "basics/multiindex-to-frame-labelled",
    "MultiIndex.to_frame",
    level="L3",
    covers=("index", "name"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.MultiIndex.from_arrays(
        [["a", "a", "b"], [1, 2, 1]], names=["k", "n"]
    ).to_frame(index=True, name=["x", "y"]),
    in_process=True,
    note="the frame's rows are labelled by the index itself",
)

case(
    "basics/multiindex-value-counts",
    "MultiIndex.value_counts",
    level="L2",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.MultiIndex.from_arrays(
        [["a", "b", "a"], [1, 2, 1]], names=["k", "n"]
    ).value_counts(),
    in_process=True,
    note="each row counted, labelled by the rows",
)

case(
    "basics/tz-localize-infer",
    "DatetimeIndex.tz_localize",
    level="L3",
    covers=("ambiguous",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DatetimeIndex(
        pd.to_datetime(
            [
                "2024-11-03 00:30",
                "2024-11-03 01:00",
                "2024-11-03 01:30",
                "2024-11-03 01:00",
                "2024-11-03 01:30",
                "2024-11-03 02:30",
            ]
        )
    ).tz_localize("US/Eastern", ambiguous="infer"),
    in_process=True,
    note="the repeated hour told apart by where the readings go back",
)

case(
    "basics/align-by-level",
    "DataFrame.align",
    level="L3",
    covers=("join", "level"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame(
        {"v": [1, 2, 3, 4]},
        index=pd.MultiIndex.from_arrays([["b", "a", "b", "c"], [1, 2, 2, 1]], names=["k", "n"]),
    ).align(
        pd.DataFrame({"w": [10, 20, 30]}, index=pd.Index(["a", "b", "z"], name="k")),
        join="outer",
        level="k",
    )[1],
    in_process=True,
    note="a flat index lined up with one level of a MultiIndex",
)

case(
    "basics/rolling-numeric-only",
    "Rolling.sum",
    level="L3",
    covers=("numeric_only",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: (
        pd.DataFrame({"a": [1, 2, 3], "s": ["x", "y", "z"]}).rolling(2).sum(numeric_only=True)
    ),
    in_process=True,
    note="the text column dropped rather than refused",
)

case(
    "basics/groupby-filter-blanked",
    "GroupBy.filter",
    level="L3",
    covers=("dropna",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: (
        pd.DataFrame({"k": ["a", "b", "a"], "v": [1, 2, 3]})
        .groupby("k")
        .filter(lambda g: len(g) > 1, dropna=False)
    ),
    in_process=True,
    note="the rows of a dropped group kept as blanks",
)

case(
    "basics/groupby-nth-dropna",
    "GroupBy.nth",
    level="L2",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: (
        pd.DataFrame({"k": ["a", "a", "b"], "v": [None, 1.0, 2.0]})
        .groupby("k")
        .nth(0, dropna="any")
    ),
    in_process=True,
    note="only the rows without a gap counted",
)

case(
    "basics/crosstab-several-keys",
    "pandas.crosstab",
    level="L3",
    covers=("index", "columns"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.crosstab(
        [pd.Series(["x", "x", "y"], name="a"), pd.Series(["p", "q", "p"], name="b")],
        [pd.Series(["u", "v", "u"], name="c"), pd.Series(["m", "m", "n"], name="d")],
    ),
    in_process=True,
    note="both axes labelled by the pairs of keys seen",
)


case(
    "basics/ffill-limit-area",
    "Series.ffill",
    level="L3",
    covers=("limit_area",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series([None, 1.0, None, 3.0, None]).ffill(limit_area="inside"),
    in_process=True,
    note="only the gap between two present values is filled",
)


case(
    "basics/shift-freq-columns",
    "DataFrame.shift",
    level="L3",
    covers=("freq", "axis"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame(
        [[1, 2]], columns=pd.date_range("2024-01-01", periods=2)
    ).shift(1, freq="D", axis=1),
    in_process=True,
    note="the column labels move a day on and the values stay put",
)


case(
    "basics/groupby-cumsum-numeric-only",
    "GroupBy.cumsum",
    level="L3",
    covers=("numeric_only",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: (
        pd.DataFrame(
            {"k": ["a", "b", "a"], "v": [1, 5, 3], "s": ["x", "y", "z"], "b": [True, False, True]}
        )
        .groupby("k")
        .cumsum(numeric_only=True)
    ),
    in_process=True,
    note="the text column is left out and the flags are summed as whole numbers",
)


case(
    "basics/groupby-shift-fill",
    "GroupBy.shift",
    level="L3",
    covers=("periods", "fill_value"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: (
        pd.DataFrame({"k": ["a", "b", "a", "a", "b"], "i": [1, 2, 3, 4, 5]})
        .groupby("k")
        .shift(1, fill_value=0)
    ),
    in_process=True,
    note="each group's first row takes the fill and the column stays whole numbers",
)


case(
    "basics/groupby-cumcount-descending",
    "GroupBy.cumcount",
    level="L3",
    covers=("ascending",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: (
        pd.DataFrame({"k": ["a", "b", "a", "a", "b"]}).groupby("k").cumcount(ascending=False)
    ),
    in_process=True,
    note="each group's rows are numbered from its last row",
)
case(
    "basics/groupby-any-keep-gaps",
    "GroupBy.any",
    level="L3",
    covers=("skipna",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: (
        pd.DataFrame({"k": ["a", "a", "b", "b"], "v": [0.0, None, 0.0, 0.0]})
        .groupby("k")
        .any(skipna=False)
    ),
    in_process=True,
    note="a gap that is not skipped counts as true, so the group with one is true",
)
case(
    "basics/pct-change-freq",
    "Series.pct_change",
    level="L3",
    covers=("freq",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series(
        [1.0, 2.0, 4.0, 5.0, 10.0], index=pd.date_range("2024-01-01", periods=5, freq="D")
    ).pct_change(freq="2D"),
    in_process=True,
    note="each value over the one two days earlier, matched by label",
)
case(
    "basics/groupby-describe-include-object",
    "GroupBy.describe",
    level="L3",
    covers=("include",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: (
        pd.DataFrame(
            {"k": ["a", "b", "a", "b"], "v": [1.0, 2.0, 3.0, 4.0], "s": ["x", "y", "x", "z"]}
        )
        .groupby("k")
        .describe(include="object")
    ),
    in_process=True,
    note="only the text column is described, by count, unique, top and freq",
)
case(
    "basics/groupby-value-counts-bins",
    "GroupBy.value_counts",
    level="L2",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: (
        pd.DataFrame({"k": ["a", "b", "a", "b", "a"], "v": [1.0, 2.0, 3.0, 10.0, 9.0]})
        .groupby("k")["v"]
        .value_counts(bins=[0, 2, 5, 10])
    ),
    in_process=True,
    note="every group lists every bin, the empty ones too, most common first",
)
case(
    "basics/index-get-indexer-nearest",
    "Index.get_indexer",
    level="L3",
    covers=("method", "tolerance"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: [
        int(at)
        for at in pd.Index([10, 20, 30, 40]).get_indexer(
            [5, 10, 14, 26, 35, 50], method="nearest", tolerance=6
        )
    ],
    in_process=True,
    note="a missing label reads from the nearest label within the tolerance, -1 past it",
)
case(
    "basics/sort-index-columns-key",
    "DataFrame.sort_index",
    level="L3",
    covers=("axis", "key"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame({"b": [1, 2], "A": [3.0, 4.0], "c": ["x", "y"]}).sort_index(
        axis=1, key=lambda labels: labels.str.lower()
    ),
    in_process=True,
    note="the column labels sort through the key, so A goes before b",
)
case(
    "basics/melt-col-level",
    "DataFrame.melt",
    level="L3",
    covers=("col_level",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame(
        [[1, 2, 3], [4, 5, 6]],
        columns=pd.MultiIndex.from_tuples(
            [("A", "a"), ("B", "b"), ("C", "c")], names=["up", "low"]
        ),
    ).melt(col_level="low", id_vars=["a"]),
    in_process=True,
    note="the lower level of columns is melted as if it were the only one",
)
case(
    "basics/reset-index-names-flat",
    "DataFrame.reset_index",
    level="L3",
    covers=("names",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame({"v": [1, 2]}, index=pd.Index([10, 20], name="k")).reset_index(
        names="x"
    ),
    in_process=True,
    note="the old row labels land in a column named from names rather than the index name",
)
case(
    "basics/dt-as-unit-round-ok-false",
    "dt.as_unit",
    level="L4",
    covers=("round_ok",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series(
        pd.to_datetime(["2024-01-01 00:00:01", "2024-01-01 00:00:02.25"], format="ISO8601")
    ).dt.as_unit("s", round_ok=False),
    raises=("ValueError", "Cannot losslessly cast '1704067202250000 us' to s"),
    in_process=True,
    note="the first instant a coarser unit would round is named by its count in microseconds",
)
case(
    "basics/to-datetime-origin-julian",
    "pandas.to_datetime",
    level="L3",
    covers=("origin", "unit"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.to_datetime(
        pd.Series([2451544.5, 2451545.0]), unit="D", origin="julian"
    ),
    in_process=True,
    note="Julian days are moved to days from 1970 before they are read",
)

case(
    "basics/to-datetime-exact-false",
    "pandas.to_datetime",
    level="L3",
    covers=("exact", "format"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.to_datetime(
        pd.Series(["on 2020-01-15 ok", "x 2021-02-16", "none"]),
        format="%Y-%m-%d",
        exact=False,
        errors="coerce",
    ),
    in_process=True,
    note="exact=False searches each row for the format and passes over the rest",
)

case(
    "basics/reset-index-col-level",
    "DataFrame.reset_index",
    level="L3",
    covers=("col_level", "col_fill"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame(
        [[1, 2], [3, 4]],
        index=pd.MultiIndex.from_tuples([("p", 1), ("q", 2)], names=["k", "n"]),
        columns=pd.MultiIndex.from_tuples([("a", "x"), ("a", "y")], names=["u", "w"]),
    ).reset_index(col_level=1, col_fill="f"),
    in_process=True,
    note="the labels land under a tuple with the name at col_level and col_fill elsewhere",
)

case(
    "basics/quantile-method-table",
    "DataFrame.quantile",
    level="L3",
    covers=("method", "interpolation"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame(
        {"a": [3, 1, 2, 1, 5], "b": [1.5, 9.0, 4.0, 2.0, 0.5]}
    ).quantile([0.1, 0.5, 0.9], method="table", interpolation="nearest"),
    in_process=True,
    note="whole rows are taken from the frame sorted by every column at once",
)

case(
    "basics/groupby-cummax-skipna-false",
    "GroupBy.cummax",
    level="L2",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: (
        pd.DataFrame({"k": ["a", "a", "b", "a", "b"], "v": [1.0, float("nan"), 2.0, 5.0, 3.0]})
        .groupby("k")
        .cummax(skipna=False)
    ),
    in_process=True,
    note="a gap that may not be skipped leaves the rest of its group missing",
)

case(
    "basics/rank-axis-none",
    "DataFrame.rank",
    level="L4",
    covers=("axis",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: df.rank(axis=None),
    raises=("ValueError", "No axis named None for object type DataFrame"),
    in_process=True,
    note="pandas reads axis=None as no axis here rather than as the default",
)

case(
    "basics/category-quantile-nearest",
    "Series.quantile",
    level="L3",
    covers=("interpolation",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series(
        pd.Categorical(["b", "a", "c", "b", "a"], categories=["c", "b", "a"], ordered=True)
    ).quantile([0.1, 0.5, 0.9], interpolation="nearest"),
    in_process=True,
    note="an ordered categorical's quantile picks categories through their codes",
)

case(
    "basics/reset-index-allow-duplicates",
    "DataFrame.reset_index",
    level="L3",
    covers=("allow_duplicates",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame({"index": [1, 2], "v": [3, 4]}).reset_index(
        allow_duplicates=True
    ),
    in_process=True,
    note="with no clash the flag changes nothing, and index falls back to level_0",
)

case(
    "basics/index-view-cls",
    "Index.view",
    level="L3",
    covers=("cls",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series(pd.Index([1.5, 2.0, -3.0]).view("int64").tolist()),
    in_process=True,
    note="the labels' bytes read as another type",
)

case(
    "basics/pct-change-fill-method",
    "Series.pct_change",
    level="L4",
    covers=("fill_method",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series([1.0, float("nan"), 3.0]).pct_change(fill_method="ffill"),
    raises=("ValueError", "fill_method must be None"),
    in_process=True,
    note="pandas 3 took the fill away and refuses any method",
)

case(
    "basics/join-list-of-frames",
    "DataFrame.join",
    level="L3",
    covers=("other", "how"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame({"z": [6.0, 7.0]}, index=["c", "a"]).join(
        [
            pd.DataFrame({"y": [4, 5]}, index=["b", "d"]),
            pd.DataFrame({"x": [1, 2, 3]}, index=["a", "b", "c"]),
        ],
        how="right",
    ),
    in_process=True,
    note="a list of frames with unique labels is laid side by side in the last frame's order",
)

case(
    "basics/astype-category-datetime",
    "Series.astype",
    level="L3",
    covers=("dtype",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series(
        pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-02"])
    ).astype("category"),
    in_process=True,
    note="categories of instants stay instants and print as dates",
)

case(
    "basics/isin-datetime",
    "Series.isin",
    level="L3",
    covers=("values",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series(pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-03"])).isin(
        [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03")]
    ),
    in_process=True,
    note="a datetime column finds a row by its instant",
)

case(
    "basics/nlargest-two-columns-keep-all",
    "DataFrame.nlargest",
    level="L3",
    covers=("n", "columns", "keep"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame(
        {"a": [1, 1, 2, 2, 3, 3], "b": [5, 4, 3, 9, 1, 1]}, index=list("pqrstu")
    ).nlargest(3, ["a", "b"], keep="all"),
    in_process=True,
    note="several columns are ranked one at a time, and keep='all' keeps every tie at the edge",
)

case(
    "basics/str-repeat-per-row",
    "str.repeat",
    level="L3",
    covers=("repeats",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series(["a", "bc", "d"], index=[5, 6, 7]).str.repeat([2, 0, 3]),
    in_process=True,
    note="one count per row, read by position",
)

case(
    "basics/isin-aligned-frame",
    "DataFrame.isin",
    level="L3",
    covers=("values",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=[10, 11, 12]).isin(
        pd.DataFrame({"a": [1, 0, 3], "c": [4, 5, 6]}, index=[10, 11, 13])
    ),
    in_process=True,
    note="a frame is lined up by label and compared cell against cell",
)

case(
    "basics/str-get-dummies-float",
    "str.get_dummies",
    level="L3",
    covers=("sep", "dtype"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series(["a|b", "b", "c"]).str.get_dummies(sep="|", dtype=float),
    in_process=True,
    note="the flags come back in the numeric type asked for",
)

case(
    "basics/concat-keys-verify-integrity",
    "pandas.concat",
    level="L3",
    covers=("objs", "keys", "verify_integrity"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.concat(
        [pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [3, 4]})],
        keys=["x", "y"],
        verify_integrity=True,
    ),
    in_process=True,
    note="keys keep repeated labels apart, so the check passes",
)

case(
    "basics/squeeze-row-axis",
    "DataFrame.squeeze",
    level="L3",
    covers=("axis",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame({"a": [1], "b": [2]}, index=["r"]).squeeze(axis=0),
    in_process=True,
    note="dropping the row axis answers the one row as a series named by its label",
)

case(
    "basics/bitwise-and-integers",
    "Series.__and__",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series([12, 10, 7]) & pd.Series([10, 6, 1]),
    in_process=True,
    note="two integer columns answer the bitwise and",
)

case(
    "basics/str-replace-callable",
    "str.replace",
    level="L3",
    covers=("pat", "repl", "regex"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series(["foo 12", "bar 3"]).str.replace(
        r"\d+", lambda m: str(int(m.group(0)) * 2), regex=True
    ),
    in_process=True,
    note="a callable replacement is handed each match",
)

case(
    "basics/factorize-categorical-sort",
    "Series.factorize",
    level="L3",
    covers=("sort",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series(["b", "a", "b", "c"], dtype="category").factorize(sort=True)[1],
    in_process=True,
    note="the uniques of a categorical column are a categorical index in category order",
)

case(
    "basics/replace-regex-list",
    "Series.replace",
    level="L3",
    covers=("to_replace", "value", "regex"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series(["a", "ab", "cd"]).replace(["a", "b"], ["b", "c"], regex=True),
    in_process=True,
    note="each pattern rewrites only the rows it matched in the column as it arrived",
)

case(
    "basics/replace-regex-by-column",
    "DataFrame.replace",
    level="L3",
    covers=("to_replace", "regex"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame({"x": ["a", "ab"], "y": ["cd", "b"]}).replace(
        {"x": {"a": "X"}, "y": {"b": "Y", "c": "C"}}, regex=True
    ),
    in_process=True,
    note="a mapping of column names to mappings gives each column its own patterns",
)

case(
    "basics/align-repeated-labels",
    "Series.align",
    level="L3",
    covers=("other", "join"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series([1.0, 2.0], index=["a", "a"]).align(
        pd.Series([5.0, 6.0], index=["a", "b"]), join="outer"
    )[1],
    in_process=True,
    note="a repeated label on one side meets every row of it on the other",
)

case(
    "basics/combine-first-repeated-labels",
    "DataFrame.combine_first",
    level="L3",
    covers=("other",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame({"v": [1.0, float("nan")]}, index=["a", "a"]).combine_first(
        pd.DataFrame({"v": [5.0, 6.0]}, index=["a", "b"])
    ),
    in_process=True,
    note="a frame aligns with an outer join first, so a repeated label is joined",
)

case(
    "basics/combine-first-series-repeated-raises",
    "Series.combine_first",
    level="L4",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series([1.0, 2.0], index=["a", "a"]).combine_first(
        pd.Series([5.0], index=["b"])
    ),
    raises=("ValueError", "cannot reindex on an axis with duplicate labels"),
    in_process=True,
    note="a column reindexes each side onto the labels it keeps, which a repeat refuses",
)

case(
    "basics/pivot-two-column-keys",
    "DataFrame.pivot",
    level="L3",
    covers=("columns", "index", "values"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame(
        {
            "a": ["x", "x", "y", "y"],
            "b": ["p", "q", "p", "q"],
            "c": ["u", "u", "v", "v"],
            "v": [1, 2, 3, 4],
        }
    ).pivot(index="a", columns=["b", "c"], values="v"),
    in_process=True,
    note="several columns keys label the answer's columns with a MultiIndex",
)

case(
    "basics/pivot-table-two-column-keys",
    "DataFrame.pivot_table",
    level="L3",
    covers=("columns", "fill_value", "aggfunc"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame(
        {
            "a": ["x", "x", "y", "y"],
            "b": ["p", "q", "p", "q"],
            "c": ["u", "u", "v", "v"],
            "v": [1, 2, 3, 4],
        }
    ).pivot_table(index="a", columns=["b", "c"], values="v", aggfunc="sum", fill_value=0),
    in_process=True,
    note="aggregates over every key are unstacked by the columns keys and filled",
)

case(
    "basics/nunique-float-nan",
    "Series.nunique",
    level="L3",
    covers=("dropna",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series([1.0, float("nan"), 2.0, 1.0]).nunique(),
    in_process=True,
    note="a NaN is a missing entry, never a distinct value of its own",
)

case(
    "basics/nunique-frame-float-nan-kept",
    "DataFrame.nunique",
    level="L3",
    covers=("dropna",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame(
        {"p": [1.0, float("nan"), float("nan")], "q": [1, 2, 2]}
    ).nunique(dropna=False),
    in_process=True,
    note="every NaN counts once as a single missing value when kept",
)

case(
    "basics/reduction-keeps-column-level-names",
    "DataFrame.sum",
    level="L2",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: (
        pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"], "v": [1, 2]})
        .pivot(index="a", columns="b", values=["v"])
        .sum()
    ),
    in_process=True,
    note="the answer's index keeps the names of the column levels",
)

case(
    "basics/idxmax-column-levels",
    "DataFrame.idxmax",
    level="L3",
    covers=("axis",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: (
        pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"], "v": [1, 2]})
        .pivot(index="a", columns="b", values=["v"])
        .idxmax(axis=0)
    ),
    in_process=True,
    note="labels of several levels answer a MultiIndex that keeps the level names",
)

case(
    "basics/cut-instants-count",
    "pandas.cut",
    level="L3",
    covers=("x", "bins"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.cut(
        pd.Series(pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-10"]), name="d"), 3
    ),
    in_process=True,
    note="bins of instants are intervals of instants, widened by a thousandth of the range",
)

case(
    "basics/cut-spans-edges",
    "pandas.cut",
    level="L3",
    covers=("bins", "retbins"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.cut(
        pd.Series(pd.to_timedelta(["1h", "3h", "10h"])),
        pd.to_timedelta(["0h", "2h", "12h"]),
        retbins=True,
    )[0],
    in_process=True,
    note="edges of spans bin spans into intervals of spans",
)

case(
    "basics/qcut-instants",
    "pandas.qcut",
    level="L3",
    covers=("x", "q", "labels"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.qcut(
        pd.Series(pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-10", "2024-01-20"])),
        2,
        labels=False,
    ),
    in_process=True,
    note="quantiles of instants are taken over their counts",
)

case(
    "basics/cut-list-categorical",
    "pandas.cut",
    level="L3",
    covers=("x", "labels"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series(pd.cut([1, 5, 9, 3], 2, labels=["lo", "hi"])),
    in_process=True,
    note="a list is binned into a Categorical",
)

case(
    "basics/interval-range-days",
    "pandas.interval_range",
    level="L3",
    covers=("start", "periods"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series(pd.interval_range(pd.Timestamp("2024-01-01"), periods=3)),
    in_process=True,
    note="instants step by a day when no freq is given",
)

case(
    "basics/interval-range-spans-freq",
    "pandas.interval_range",
    level="L3",
    covers=("start", "end", "freq"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series(
        pd.interval_range(pd.Timedelta("1h"), pd.Timedelta("3h"), freq="30min")
    ),
    in_process=True,
    note="spans step by freq between the ends",
)

case(
    "basics/interpolate-index-unsorted",
    "Series.interpolate",
    level="L3",
    covers=("method",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series(
        [1.0, float("nan"), 3.0, float("nan"), 10.0], index=[0, 4, 1, 3, 2]
    ).interpolate(method="index"),
    in_process=True,
    note="gaps are filled along the labels sorted, not along the rows",
)

case(
    "basics/interpolate-index-unsorted-limit",
    "Series.interpolate",
    level="L3",
    covers=("method", "limit_direction", "limit_area"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series(
        [float("nan"), 1.0, float("nan"), 4.0, float("nan")], index=[5.0, 4.0, 3.0, 1.0, 0.5]
    ).interpolate(method="index", limit_direction="both", limit_area="inside"),
    in_process=True,
    note="the limits are still counted by position",
)


case(
    "basics/interpolate-zoned-instants",
    "Series.interpolate",
    level="L2",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: (
        pd.Series(
            pd.to_datetime(
                ["2024-01-01", "NaT", "2024-01-04", "NaT", "NaT", "2024-01-10"], utc=True
            )
        )
        .dt.tz_convert("Asia/Tokyo")
        .interpolate()
    ),
    in_process=True,
    note="the line runs through the UTC instants and the answer keeps the zone",
)

case(
    "basics/interpolate-zoned-instants-limit",
    "Series.interpolate",
    level="L3",
    covers=("limit", "limit_direction"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series(
        pd.to_datetime(["2024-01-01", "NaT", "2024-01-04", "NaT", "NaT", "2024-01-10"], utc=True)
    ).interpolate(limit=1, limit_direction="backward"),
    in_process=True,
    note="the limit is counted by position on the zoned column",
)

case(
    "basics/describe-intervals",
    "Series.describe",
    level="L2",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series(
        pd.IntervalIndex.from_arrays([0.0, float("nan"), 2.0, 0.0], [1.0, float("nan"), 3.0, 1.0])
    ).describe(),
    in_process=True,
    note="an interval column is counted the way text is",
)


case(
    "basics/corr-frame-instants",
    "DataFrame.corr",
    level="L2",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame(
        {
            "d": pd.to_datetime(["2024-01-01", "2024-01-03", "NaT", "2024-01-08", "2024-01-02"]),
            "v": [1.0, 3.0, 2.0, 9.0, 4.0],
        }
    ).corr(),
    in_process=True,
    note="a frame reads the instants as counts and NaT as missing",
)

case(
    "basics/corr-series-instants-nat",
    "Series.corr",
    level="L2",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.Series(
        pd.to_datetime(["2024-01-01", "2024-01-03", "NaT", "2024-01-08", "2024-01-02"])
    ).corr(pd.Series([1.0, 3.0, 2.0, 9.0, 4.0])),
    in_process=True,
    note="a column against a column counts NaT as the smallest int64",
)

case(
    "basics/cov-frame-instants-refused",
    "DataFrame.cov",
    level="L4",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.DataFrame(
        {"d": pd.to_datetime(["2024-01-01", "2024-01-03"]), "v": [1.0, 3.0]}
    ).cov(),
    raises=("TypeError", "not supported for cov"),
    in_process=True,
    note="a frame's covariance refuses instants and spans",
)


def _layered_columns(pd):
    """A frame whose columns have two named levels."""
    return pd.DataFrame(
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        columns=pd.MultiIndex.from_tuples(
            [("a", "x"), ("a", "y"), ("b", "x"), ("b", "z")], names=["one", "two"]
        ),
    )


case(
    "basics/xs-column-level",
    "DataFrame.xs",
    level="L3",
    covers=("key", "axis", "level"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: _layered_columns(pd).xs("x", axis=1, level="two"),
    in_process=True,
    note="the column labels are crossed the way row labels are",
)

case(
    "basics/xs-column-level-kept",
    "DataFrame.xs",
    level="L3",
    covers=("key", "axis", "level", "drop_level"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: _layered_columns(pd).xs("x", axis=1, level=1, drop_level=False),
    in_process=True,
    note="the crossed level stays when drop_level is False",
)

case(
    "basics/loc-first-column-label",
    "DataFrame.loc",
    level="L2",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: _layered_columns(pd).loc[[1], "a"],
    in_process=True,
    note="a first level label names the columns under it and keeps the level names left",
)


def _monthly(pd):
    """Three months with March missing, the labels of the period resample cases."""
    index = pd.PeriodIndex(["2024-01", "2024-02", "2024-04"], freq="M")
    return pd.Series([1, 2, 4], index=index)


case(
    "basics/resample-periods-down",
    "Series.resample",
    level="L3",
    covers=("rule",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: _monthly(pd).resample("Q").sum(),
    in_process=True,
    note="months gathered into their quarters, labelled by the quarters",
)

case(
    "basics/resample-periods-convention",
    "Series.resample",
    level="L3",
    covers=("convention",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: _monthly(pd).resample("W", convention="end").asfreq(),
    in_process=True,
    note="each month lands on the week holding its last day",
)

case(
    "basics/resample-periods-ffill",
    "Series.resample",
    level="L3",
    covers=("rule",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: _monthly(pd).resample("W").ffill(limit=2),
    in_process=True,
    note="months spread over weeks from their first day, filled two weeks on",
)

case(
    "basics/resample-periods-incompatible",
    "Series.resample",
    level="L4",
    raises=("IncompatibleFrequency", "not sub or super periods"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: _monthly(pd).resample("W").sum(),
    in_process=True,
    note="weeks do not divide months, so a reduction between them is refused",
)


def _quarter_hours(pd):
    """Six rows twenty five minutes apart, the rows of the resample apply cases."""
    index = pd.date_range("2024-01-01", periods=6, freq="25min")
    return pd.DataFrame({"v": [0, 1, 2, 3, 4, 5], "w": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=index)


def _first_two(x):
    return x.head(2)


def _doubled(x):
    return x * 2


case(
    "basics/resample-apply-keyed",
    "Series.resample",
    level="L3",
    covers=("group_keys",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: _quarter_hours(pd)["v"].resample("h", group_keys=True).apply(_first_two),
    in_process=True,
    note="an apply answering pieces puts each under the label of its bin",
)

case(
    "basics/resample-apply-transform-keyed",
    "DataFrame.resample",
    level="L3",
    covers=("group_keys",),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: _quarter_hours(pd).resample("h", group_keys=True).apply(_doubled),
    in_process=True,
    note="a piece the size of its bin is still keyed when group_keys is set",
)

case(
    "basics/resample-apply-columns-first",
    "DataFrame.resample",
    level="L2",
    frames=("int64_no_nulls",),
    expr=lambda pd, df: _quarter_hours(pd).resample("h").apply(lambda x: x.head(1)),
    in_process=True,
    note="a function answering one row a bin is an aggregate on each column",
)


def _repeated_hour():
    """Half past one on the night New York leaves summer time, which happens twice."""
    import datetime

    return datetime.datetime(2024, 11, 3, 1, 30)


case(
    "basics/timestamp-fold-second",
    "pandas.Timestamp",
    level="L3",
    covers=("fold", "tz"),
    frames=("single",),
    expr=lambda pd, df: pd.Series(
        [pd.Timestamp(_repeated_hour(), tz="US/Eastern", fold=1).isoformat()]
    ),
    in_process=True,
    note="a fold of one puts a repeated wall clock on its second, winter, side",
)

case(
    "basics/timestamp-fold-unambiguous",
    "pandas.Timestamp",
    level="L4",
    raises=("ValueError", "Cannot pass fold with possibly unambiguous input"),
    frames=("single",),
    expr=lambda pd, df: pd.Timestamp("2024-11-03 01:30", tz="US/Eastern", fold=1),
    in_process=True,
    note="text says its own moment, so a fold beside it is refused",
)

case(
    "basics/multiindex-sortorder-kept",
    "MultiIndex.from_arrays",
    level="L3",
    covers=("sortorder",),
    frames=("single",),
    expr=lambda pd, df: pd.Series(
        [pd.MultiIndex.from_arrays([["a", "a", "b"], [2, 1, 3]], sortorder=1)[:2].sortorder]
    ),
    in_process=True,
    note="the sortorder given is kept, and a forward slice carries it",
)

case(
    "basics/multiindex-sortorder-too-deep",
    "MultiIndex.from_product",
    level="L4",
    raises=("ValueError", "must be inferior or equal to actual lexsort_depth"),
    frames=("single",),
    expr=lambda pd, df: pd.MultiIndex.from_product([["b", "a"], [1, 2]], sortorder=2),
    in_process=True,
    note="a sortorder deeper than the sorted levels is refused",
)


def _two_level(pd):
    """Four rows under two levels, the labels of the level name and reindex cases."""
    return pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1), ("c", 3)], names=["k", "n"])


def _small_text_frame(pd):
    """A float column and a text column, the frame of the text layout cases."""
    return pd.DataFrame({"a": [1.5, 22.25, 3.0], "b": ["x", "yyyyyy", "z"]})


def _angled(v):
    return f"<{v}>"


def _info_text(pd):
    import io

    buf = io.StringIO()
    pd.Series([1, 2], name="x").info(verbose=True, buf=buf, memory_usage=False, show_counts=True)
    return buf.getvalue().replace(pd.__name__, "")


case(
    "basics/to-string-unsparsified",
    "DataFrame.to_string",
    level="L3",
    covers=("sparsify",),
    frames=("single",),
    expr=lambda pd, df: pd.Series(
        [pd.DataFrame({"v": [1, 2, 3, 4]}, index=_two_level(pd)).to_string(sparsify=False)]
    ),
    in_process=True,
    note="with sparsify off every level label prints whole",
)

case(
    "basics/to-string-layout",
    "DataFrame.to_string",
    level="L3",
    covers=("col_space", "formatters", "justify"),
    frames=("single",),
    expr=lambda pd, df: pd.Series(
        [
            _small_text_frame(pd).to_string(
                col_space={"a": 9}, formatters={"b": _angled}, justify="left"
            )
        ]
    ),
    in_process=True,
    note="a width for one column, a formatter for another and left headers",
)

case(
    "basics/to-string-decimal-width",
    "DataFrame.to_string",
    level="L3",
    covers=("decimal", "max_colwidth", "index_names"),
    frames=("single",),
    expr=lambda pd, df: pd.Series(
        [
            _small_text_frame(pd)
            .rename_axis("i")
            .to_string(decimal=",", max_colwidth=4, index_names=False)
        ]
    ),
    in_process=True,
    note="a comma for the point, text cut to four and the index name left off",
)

case(
    "basics/stack-named-level",
    "DataFrame.stack",
    level="L3",
    covers=("level",),
    frames=("single",),
    expr=lambda pd, df: pd.DataFrame([[1, 5]], columns=_two_level(pd)[:2]).stack(level="n"),
    in_process=True,
    note="a column level asked for by name moves with its name to the rows",
)

case(
    "basics/rename-axis-mapping",
    "Series.rename_axis",
    level="L3",
    covers=("index",),
    frames=("single",),
    expr=lambda pd, df: pd.Series([1, 2, 3, 4], index=_two_level(pd)).rename_axis(index={"k": "K"}),
    in_process=True,
    note="a mapping given as index renames the level names it has keys for",
)

case(
    "basics/reindex-onto-level",
    "Series.reindex",
    level="L3",
    covers=("level", "fill_value"),
    frames=("single",),
    expr=lambda pd, df: pd.Series([1, 2], index=["a", "b"]).reindex(
        _two_level(pd), level="k", fill_value=0
    ),
    in_process=True,
    note="flat labels are read against one level, a label missing there filled",
)

case(
    "basics/reindex-from-level",
    "Series.reindex",
    level="L3",
    covers=("level",),
    frames=("single",),
    expr=lambda pd, df: pd.Series([1.0, 2.0, 3.0, 4.0], index=_two_level(pd)).reindex(
        ["b", "a", "z"], level=0
    ),
    in_process=True,
    note="rows whose label on the level is asked for are kept in the order asked",
)

case(
    "basics/asfreq-filled",
    "Series.asfreq",
    level="L3",
    covers=("method",),
    frames=("single",),
    expr=lambda pd, df: pd.Series(
        [0, 1, 2, 3], index=pd.date_range("2024-01-01 00:07", periods=4, freq="17min")
    ).asfreq("10min", method="ffill"),
    in_process=True,
    note="a forward filled asfreq keeps the step of its range",
)

case(
    "basics/asfreq-periods-start",
    "Series.asfreq",
    level="L3",
    covers=("how",),
    frames=("single",),
    expr=lambda pd, df: pd.Series([1, 2], index=pd.PeriodIndex(["2024", "2025"], freq="Y")).asfreq(
        "M", how="start"
    ),
    in_process=True,
    note="each year moves to its first month",
)

case(
    "basics/info-without-memory",
    "Series.info",
    level="L3",
    covers=("buf", "memory_usage", "show_counts", "verbose"),
    frames=("single",),
    expr=lambda pd, df: pd.Series([_info_text(pd)]),
    in_process=True,
    note="the report without its memory line ends on its last line",
)

case(
    "basics/period-fields",
    "pandas.Period",
    level="L3",
    covers=("year", "month", "day", "hour", "minute", "second", "freq"),
    frames=("single",),
    expr=lambda pd, df: pd.Series(
        [str(pd.Period(year=2024, month=1, day=1, hour=5, minute=7, second=9, freq="s"))]
    ),
    in_process=True,
    note="a period of a second built from its fields",
)

case(
    "basics/period-quarter-ordinal",
    "pandas.Period",
    level="L3",
    covers=("quarter", "ordinal", "value"),
    frames=("single",),
    expr=lambda pd, df: pd.Series(
        [
            str(pd.Period(year=2024, quarter=2, freq="Q")),
            str(pd.Period(ordinal=100, freq="M")),
            str(pd.Period(value="2024-05", freq="M")),
        ]
    ),
    in_process=True,
    note="a period by quarter, by ordinal and by text",
)

case(
    "basics/eval-dicts",
    "pandas.eval",
    level="L3",
    covers=("local_dict", "global_dict", "parser", "engine", "resolvers"),
    frames=("single",),
    expr=lambda pd, df: pd.Series(
        [
            int(pd.eval("x + y", global_dict={"x": 2}, local_dict={"y": 3})),
            int(pd.eval("1 + 2", parser="python", engine="python")),
            int(pd.eval("a + 1", resolvers=[{"a": 5}])),
        ]
    ),
    in_process=True,
    note="names read from the dicts and resolvers given, and the python parser and engine",
)

case(
    "basics/eval-target",
    "pandas.eval",
    level="L3",
    covers=("target",),
    frames=("single",),
    expr=lambda pd, df: pd.eval("c = 2 * 3", target=pd.DataFrame({"a": [1, 2]})),
    in_process=True,
    note="an assignment into the target frame",
)

case(
    "basics/wide-to-long-sep-suffix",
    "pandas.wide_to_long",
    level="L3",
    covers=("sep", "suffix"),
    frames=("single",),
    expr=lambda pd, df: pd.wide_to_long(
        pd.DataFrame({"id": [1, 2], "A-x": [1, 2], "A-y": [3, 4]}),
        ["A"],
        i="id",
        j="n",
        sep="-",
        suffix=r"\w",
    ),
    in_process=True,
    note="stubs split from their suffix by a dash, the suffix a letter",
)

case(
    "basics/category-index-built",
    "pandas.CategoricalIndex",
    level="L3",
    covers=("categories", "ordered", "name", "copy"),
    frames=("single",),
    expr=lambda pd, df: pd.Series(
        [1, 2],
        index=pd.CategoricalIndex(
            ["a", "b"], categories=["b", "a", "c"], ordered=True, name="n", copy=True
        ),
    ),
    in_process=True,
    note="an ordered category index with categories of its own",
)

case(
    "basics/merge-asof-by",
    "pandas.merge_asof",
    level="L3",
    covers=("left_on", "right_on", "by"),
    frames=("single",),
    expr=lambda pd, df: pd.merge_asof(
        pd.DataFrame({"t": [1, 5, 10], "k": ["a", "b", "a"], "l": [1, 2, 3]}),
        pd.DataFrame({"u": [2, 6, 9], "k": ["a", "a", "b"], "r": [7, 8, 9]}),
        left_on="t",
        right_on="u",
        by="k",
    ),
    in_process=True,
    note="each row takes the last earlier row of its own key",
)

case(
    "basics/merge-asof-index",
    "pandas.merge_asof",
    level="L3",
    covers=("left_index", "right_index"),
    frames=("single",),
    expr=lambda pd, df: pd.merge_asof(
        pd.DataFrame({"l": [1, 2, 3]}, index=[1, 5, 10]),
        pd.DataFrame({"r": [7, 8, 9]}, index=[2, 6, 9]),
        left_index=True,
        right_index=True,
    ),
    in_process=True,
    note="both sides matched on their row labels",
)


def _hourly(pd):
    return pd.Series(range(1, 7), index=pd.date_range("2024-01-01", periods=6, freq="h"))


def _pair_keyed(pd):
    labels = pd.MultiIndex.from_arrays([list("abab"), [1, 1, 2, 2]], names=["x", "y"])
    return pd.Series([1, 2, 3, 4], index=labels)


def _pivot_source(pd):
    return pd.DataFrame(
        {
            "k": pd.Categorical(["x", "x", "y", "y"], categories=["x", "y", "z"]),
            "c": ["p", "q", "p", "p"],
            "v": [1.0, 2.0, 3.0, float("nan")],
        }
    )


case(
    "basics/resample-closed-label",
    "Series.resample",
    level="L3",
    covers=("closed", "label"),
    frames=("single",),
    expr=lambda pd, df: _hourly(pd).resample("2h", closed="right", label="right").sum(),
    in_process=True,
    note="bins closed and labelled on their right edge",
)

case(
    "basics/resample-origin-offset",
    "Series.resample",
    level="L3",
    covers=("origin", "offset"),
    frames=("single",),
    expr=lambda pd, df: pd.concat(
        [
            _hourly(pd).resample("4h", origin="2023-12-31 23:00").sum(),
            _hourly(pd).resample("4h", offset="1h").sum(),
        ]
    ),
    in_process=True,
    note="bins laid from a given origin and moved by an offset",
)

case(
    "basics/resample-by-level",
    "Series.resample",
    level="L3",
    covers=("level",),
    frames=("single",),
    expr=lambda pd, df: (
        pd.Series(
            range(6),
            index=pd.MultiIndex.from_arrays(
                [pd.date_range("2024-01-01", periods=6, freq="h"), list("aabbcc")], names=["t", "k"]
            ),
        )
        .resample("2h", level="t")
        .sum()
    ),
    in_process=True,
    note="the instants read off one level of the row labels",
)

case(
    "basics/series-group-level",
    "Series.groupby",
    level="L3",
    covers=("level", "as_index"),
    frames=("single",),
    expr=lambda pd, df: _pair_keyed(pd).groupby(level="x", as_index=True).sum(),
    in_process=True,
    note="groups read off a named level of the row labels",
)

case(
    "basics/series-group-sort-dropna",
    "Series.groupby",
    level="L3",
    covers=("sort", "dropna"),
    frames=("single",),
    expr=lambda pd, df: pd.concat(
        [
            pd.Series([1, 2, 3]).groupby(["b", "a", "b"], sort=False).sum(),
            pd.Series([1.0, 2, 3]).groupby(pd.Series(["a", float("nan"), "a"]), dropna=False).sum(),
        ]
    ),
    in_process=True,
    note="groups in first seen order, and a missing key kept as a group",
)

case(
    "basics/series-group-keys-observed",
    "Series.groupby",
    level="L3",
    covers=("group_keys", "observed"),
    frames=("single",),
    expr=lambda pd, df: pd.concat(
        [
            pd.Series([1, 2, 3]).groupby(["b", "a", "b"], group_keys=False).apply(_doubled),
            pd.Series([1, 2])
            .groupby(pd.Series(pd.Categorical(["a", "a"], categories=["a", "b"])), observed=False)
            .sum(),
        ]
    ),
    in_process=True,
    note="apply without the group keys, and an unobserved category kept",
)

case(
    "basics/query-parser-dicts",
    "DataFrame.query",
    level="L3",
    covers=("parser", "engine", "local_dict", "global_dict", "resolvers", "level"),
    frames=("single",),
    expr=lambda pd, df: pd.concat(
        [
            pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]}).query(
                "a > b", parser="python", engine="python"
            ),
            pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]}).query("a > @x", local_dict={"x": 1}),
            pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]}).query("a > @x", global_dict={"x": 1}),
            pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]}).query("a > z", resolvers=[{"z": 2}]),
            pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]}).query("a > 1", level=0),
        ]
    ),
    in_process=True,
    note="the parser, engine and names given to a query",
)

case(
    "basics/pivot-table-margins-name",
    "pandas.pivot_table",
    level="L3",
    covers=("margins_name", "dropna"),
    frames=("single",),
    expr=lambda pd, df: pd.concat(
        [
            pd.pivot_table(
                _pivot_source(pd).astype({"k": str}),
                values="v",
                index="k",
                columns="c",
                aggfunc="sum",
                margins=True,
                margins_name="Total",
            ),
            pd.pivot_table(
                _pivot_source(pd).astype({"k": str}),
                values="v",
                index="k",
                columns="c",
                aggfunc="count",
                dropna=False,
            ),
        ]
    ),
    in_process=True,
    note="a named margin, and a count keeping the empty cells",
)

case(
    "basics/pivot-table-unobserved",
    "pandas.pivot_table",
    level="L3",
    covers=("observed", "sort"),
    frames=("single",),
    expr=lambda pd, df: pd.concat(
        [
            pd.pivot_table(_pivot_source(pd), values="v", index="k", aggfunc="sum", observed=False),
            pd.pivot_table(
                _pivot_source(pd).iloc[::-1].astype({"k": str}),
                values="v",
                index="k",
                aggfunc="sum",
                sort=False,
            ),
        ]
    ),
    in_process=True,
    note="an unobserved category kept, and keys in first seen order",
)

case(
    "basics/pivot-table-unobserved-columns",
    "DataFrame.pivot_table",
    level="L3",
    covers=("observed",),
    frames=("single",),
    expr=lambda pd, df: (
        _pivot_source(pd)
        .fillna({"v": 4.0})
        .pivot_table(values="v", index="k", columns="c", aggfunc="sum", observed=False)
    ),
    in_process=True,
    note="every category of the row key a row, an empty cell holding nought",
)

case(
    "basics/merge-ordered-sides",
    "pandas.merge_ordered",
    level="L3",
    covers=("left_on", "right_on", "suffixes", "right_by"),
    frames=("single",),
    expr=lambda pd, df: pd.concat(
        [
            pd.merge_ordered(
                pd.DataFrame({"lk": [1, 3, 5], "g": ["a", "a", "b"], "lv": [1, 2, 3]}),
                pd.DataFrame({"rk": [2, 3], "g": ["a", "a"], "rv": [9, 8]}),
                left_on="lk",
                right_on="rk",
                suffixes=("_l", "_r"),
            ),
            pd.merge_ordered(
                pd.DataFrame({"rk": [2, 3], "g": ["a", "a"], "rv": [9, 8]}),
                pd.DataFrame({"lk": [1, 3, 5], "g": ["a", "a", "b"], "lv": [1, 2, 3]}),
                left_on="rk",
                right_on="lk",
                right_by="g",
            ),
        ]
    ),
    in_process=True,
    note="keys named on each side, suffixes, and groups of the right side",
)

case(
    "basics/bdate-range-zoned",
    "pandas.bdate_range",
    level="L3",
    covers=("tz", "normalize", "name", "inclusive"),
    frames=("single",),
    expr=lambda pd, df: pd.Series(
        [
            str(pd.bdate_range("2024-01-05 10:00", periods=3, tz="UTC", normalize=True, name="d")),
            str(pd.bdate_range("2024-01-01", "2024-01-05", inclusive="neither")),
        ]
    ),
    in_process=True,
    note="business days in a zone at midnight, and the ends left out",
)

case(
    "basics/series-xs-level",
    "Series.xs",
    level="L3",
    covers=("key", "axis", "level", "drop_level"),
    frames=("single",),
    expr=lambda pd, df: pd.concat(
        [
            _pair_keyed(pd).xs(1, level="y", drop_level=False),
            _pair_keyed(pd)
            .xs("a", axis=0)
            .set_axis(pd.MultiIndex.from_tuples([("a", 1), ("a", 2)])),
        ]
    ),
    in_process=True,
    note="a cross section on a named level, its level kept or dropped",
)

case(
    "basics/tz-localize-ambiguous",
    "Series.tz_localize",
    level="L3",
    covers=("ambiguous", "nonexistent", "axis"),
    frames=("single",),
    expr=lambda pd, df: pd.Series(
        [
            str(
                pd.Series([1], index=pd.DatetimeIndex(["2024-11-03 01:30"]))
                .tz_localize("America/New_York", ambiguous=[True])
                .index[0]
            ),
            str(
                pd.Series([2], index=pd.DatetimeIndex(["2024-03-10 02:30"]))
                .tz_localize("America/New_York", nonexistent="shift_forward", axis=0)
                .index[0]
            ),
        ]
    ),
    in_process=True,
    note="an ambiguous wall time read as daylight time, and a missing one moved forward",
)

case(
    "basics/convert-dtypes-float-kept",
    "Series.convert_dtypes",
    level="L3",
    covers=(
        "infer_objects",
        "convert_string",
        "convert_integer",
        "convert_floating",
        "dtype_backend",
    ),
    frames=("single",),
    expr=lambda pd, df: pd.Series(
        [
            str(
                pd.Series([1.0, 2.0])
                .convert_dtypes(
                    infer_objects=True,
                    convert_string=True,
                    convert_integer=False,
                    convert_floating=True,
                    dtype_backend="numpy_nullable",
                )
                .dtype
            ),
            str(pd.Series(["a", "b"]).convert_dtypes(convert_string=False).dtype),
        ]
    ),
    in_process=True,
    note="whole floats kept as floats, and text left as it was",
)

case(
    "basics/reindex-like-filled",
    "Series.reindex_like",
    level="L3",
    covers=("other", "method", "limit", "tolerance"),
    frames=("single",),
    expr=lambda pd, df: pd.Series([1, 2, 3], index=[1, 3, 5]).reindex_like(
        pd.Series([0, 0, 0, 0], index=[1, 2, 4, 6]), method="ffill", limit=1, tolerance=2
    ),
    in_process=True,
    note="labels taken from another series and filled forward within bounds",
)

case(
    "basics/category-built-ordered",
    "pandas.Categorical",
    level="L3",
    covers=("values", "categories", "ordered", "dtype"),
    frames=("single",),
    expr=lambda pd, df: pd.concat(
        [
            pd.Series(pd.Categorical(["b", "a"], categories=["b", "a"], ordered=True)),
            pd.Series(pd.Categorical(["b", "a"], dtype=pd.CategoricalDtype(["a", "b", "c"]))),
        ]
    ),
    in_process=True,
    note="categories in a given order, and a dtype handed over whole",
)

case(
    "basics/period-range-end",
    "pandas.period_range",
    level="L3",
    covers=("end", "periods", "freq", "name"),
    frames=("single",),
    expr=lambda pd, df: pd.Series(
        [str(pd.period_range(end="2024-03", periods=3, freq="M", name="p"))]
    ),
    in_process=True,
    note="months counted back from the last",
)

case(
    "basics/timestamp-parts-unit",
    "pandas.Timestamp",
    level="L3",
    covers=("ts_input", "microsecond", "nanosecond", "tzinfo", "unit"),
    frames=("single",),
    expr=lambda pd, df: pd.Series(
        [
            str(pd.Timestamp(2024, 1, 2, 3, 4, 5, microsecond=6, nanosecond=7, tzinfo=None)),
            str(pd.Timestamp(1_700_000_000, unit="s")),
        ]
    ),
    in_process=True,
    note="an instant from its parts down to the nanosecond, and from a count of seconds",
)

case(
    "basics/period-index-built",
    "pandas.PeriodIndex",
    level="L3",
    covers=("data", "freq", "dtype", "name"),
    frames=("single",),
    expr=lambda pd, df: pd.Series(
        [
            str(pd.PeriodIndex(["2024-01", "2024-02"], freq="M", name="p")),
            str(pd.PeriodIndex(["2024-01"], dtype="period[M]")),
        ]
    ),
    in_process=True,
    note="periods from text with a frequency or a dtype",
)

case(
    "basics/datetime-index-dayfirst",
    "pandas.DatetimeIndex",
    level="L3",
    covers=("data", "dayfirst", "yearfirst", "name"),
    frames=("single",),
    expr=lambda pd, df: pd.Series(
        [
            str(pd.DatetimeIndex(["01/02/2024"], dayfirst=True, name="d")),
            str(pd.DatetimeIndex(["24/01/02"], yearfirst=True)),
        ]
    ),
    in_process=True,
    note="dates read day first and year first",
)

case(
    "basics/interval-index-built",
    "pandas.IntervalIndex",
    level="L3",
    covers=("data", "closed", "name", "verify_integrity"),
    frames=("single",),
    expr=lambda pd, df: pd.Series(
        [
            str(
                pd.IntervalIndex(
                    [pd.Interval(0, 1), pd.Interval(1, 2)],
                    closed="right",
                    name="i",
                    verify_integrity=True,
                )
            )
        ]
    ),
    in_process=True,
    note="intervals gathered into an index with a name",
)

case(
    "basics/multiindex-built-codes",
    "pandas.MultiIndex",
    level="L3",
    covers=("levels", "codes", "names", "verify_integrity", "sortorder"),
    frames=("single",),
    expr=lambda pd, df: pd.Series(
        [1, 2],
        index=pd.MultiIndex(
            levels=[["a", "b"], [1, 2]],
            codes=[[0, 1], [1, 0]],
            names=["x", "y"],
            verify_integrity=True,
            sortorder=None,
        ),
    ),
    in_process=True,
    note="labels built from levels and the codes into them",
)

case(
    "basics/pivot-wide",
    "pandas.pivot",
    level="L3",
    covers=("data", "index", "columns", "values"),
    frames=("single",),
    expr=lambda pd, df: pd.pivot(
        pd.DataFrame({"k": ["x", "x", "y"], "c": ["p", "q", "p"], "v": [1.0, 2.0, 3.0]}),
        index="k",
        columns="c",
        values="v",
    ),
    in_process=True,
    note="one value per pair of keys spread into columns",
)


def _level_pair(pd):
    labels = pd.MultiIndex.from_arrays([list("aabb"), [1, 2, 1, 2]], names=["x", "y"])
    return (
        pd.Series([1.0, 2.0, 3.0, 4.0], index=labels),
        pd.Series([10.0, 20.0], index=pd.Index(["a", "b"], name="x")),
    )


def _series_level_fill(pd, op):
    layered, flat = _level_pair(pd)
    left = pd.Series([1.0, float("nan"), 3.0], index=[0, 1, 2])
    right = pd.Series([4.0, 5.0, float("nan")], index=[1, 2, 3])
    answers = [
        getattr(layered, op)(flat, level="x", axis=0),
        getattr(left, op)(right, fill_value=2.0),
    ]
    return pd.concat([part for answer in answers for part in _parts(answer)])


def _parts(answer):
    return list(answer) if isinstance(answer, tuple) else [answer]


def _frame_level_fill(pd, op):
    labels = pd.MultiIndex.from_arrays([list("aabb"), [1, 2, 1, 2]], names=["x", "y"])
    layered = pd.DataFrame(
        {"p": [1.0, 2.0, 3.0, 4.0], "q": [5.0, float("nan"), 7.0, 8.0]}, index=labels
    )
    flat = pd.DataFrame(
        {"p": [1.0, float("nan"), 9.0], "r": [2.0, 3.0, 4.0]},
        index=pd.Index(["a", "c", "b"], name="x"),
    )
    return pd.concat(
        [
            getattr(layered, op)(flat, level="x", fill_value=1.0),
            getattr(layered, op)(pd.Series({"p": 2.0, "q": 3.0}), axis="columns"),
        ]
    )


for _op in (
    "add",
    "sub",
    "subtract",
    "mul",
    "multiply",
    "truediv",
    "div",
    "divide",
    "floordiv",
    "mod",
    "pow",
    "radd",
    "rsub",
    "rmul",
    "rtruediv",
    "rdiv",
    "rfloordiv",
    "rmod",
    "rpow",
    "divmod",
    "rdivmod",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
):
    case(
        f"basics/series-{_op}-level-fill",
        f"Series.{_op}",
        level="L3",
        covers=("other", "level", "fill_value", "axis"),
        frames=("single",),
        expr=(lambda op: lambda pd, df: _series_level_fill(pd, op))(_op),
        in_process=True,
        note="a flat operand read by one level of the labels, and a gap on one side filled",
    )

for _op in (
    "add",
    "sub",
    "subtract",
    "mul",
    "multiply",
    "truediv",
    "div",
    "divide",
    "floordiv",
    "mod",
    "pow",
    "radd",
    "rsub",
    "rmul",
    "rtruediv",
    "rdiv",
    "rfloordiv",
    "rmod",
    "rpow",
):
    case(
        f"basics/frame-{_op}-level-fill",
        f"DataFrame.{_op}",
        level="L3",
        covers=("other", "axis", "level", "fill_value"),
        frames=("single",),
        expr=(lambda op: lambda pd, df: _frame_level_fill(pd, op))(_op),
        in_process=True,
        note="a flat frame read by one level of the rows with gaps filled, and a row across",
    )


def _hourly_six(pd):
    return pd.date_range("2024-01-01", periods=6, freq="4h")


def _utc_pair(pd):
    return pd.date_range("2024-01-01", periods=2, tz="UTC")


def _xy_pairs(pd):
    return pd.MultiIndex.from_tuples([("a", 1), ("b", 2)], names=["x", "y"])


def _uneven_rows(pd):
    pairs = [("b", 1), ("a", 2), ("b", 2), ("a", 1), ("c", 3)]
    return pd.MultiIndex.from_tuples(pairs, names=["x", "y"])


def _update_pair(pd):
    left = pd.DataFrame({"a": [1.0, 2.0, float("nan")], "b": [4.0, 5.0, 6.0]})
    right = pd.DataFrame({"a": [10.0, float("nan"), 30.0]})
    return left, right


def _updated_left_only(pd, df):
    left, right = _update_pair(pd)
    left.update(right, join="left", overwrite=False)
    return left


def _updated_filtered(pd, df):
    left, right = _update_pair(pd)
    left.update(right, filter_func=_above_one)
    return left


def _above_one(values):
    return values > 1


def _updated_overlap(pd, df):
    left, right = _update_pair(pd)
    left.update(right, errors="raise")
    return left


def _inserted_beside(pd, df):
    made = pd.DataFrame({"a": [1, 2]})
    made.insert(1, "c", [3, 4], allow_duplicates=True)
    return made


def _compare_pair(pd):
    left = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    right = pd.DataFrame({"a": [1, 9, 3], "b": [4, 5, 7]})
    return left, right


def _negated(values):
    return -values


case(
    "basics/frame-from-mapping-columns",
    "pandas.DataFrame",
    level="L3",
    covers=("data",),
    frames=("single",),
    expr=lambda pd, df: pd.DataFrame({"p": {"y": 1, "x": 3}, "q": {"z": 2}}),
    in_process=True,
    note="each inner mapping read by its keys, the rows every key in the order first seen",
)
case(
    "basics/frame-from-mapping-columns-named-rows",
    "pandas.DataFrame",
    level="L3",
    covers=("data", "index"),
    frames=("single",),
    expr=lambda pd, df: pd.DataFrame({"p": {"y": 1}, "q": 7}, index=["y", "w"]),
    in_process=True,
    note="with rows named, a mapping column is read by those labels and a scalar spread",
)
case(
    "basics/frame-mapping-beside-list",
    "pandas.DataFrame",
    level="L4",
    covers=("data",),
    frames=("single",),
    expr=lambda pd, df: pd.DataFrame({"p": {"y": 1, "x": 3}, "q": [5, 6]}),
    raises=("ValueError", "Mixing dicts with non-Series"),
    in_process=True,
    note="a list beside a mapping has no labels to line up by, so pandas refuses it",
)
case(
    "basics/from-dict-uneven-mappings",
    "DataFrame.from_dict",
    level="L3",
    covers=("data", "orient", "dtype"),
    frames=("single",),
    expr=lambda pd, df: pd.DataFrame.from_dict(
        {"p": {"x": 1, "y": 3}, "q": {"x": 2}}, orient="columns", dtype="float64"
    ),
    in_process=True,
    note="uneven inner mappings, a missing key a gap, cast to the dtype asked for",
)
case(
    "basics/from-dict-rows-named-columns",
    "DataFrame.from_dict",
    level="L3",
    covers=("data", "orient", "columns"),
    frames=("single",),
    expr=lambda pd, df: pd.DataFrame.from_dict(
        {"r1": [1, 2], "r2": [3, 4]}, orient="index", columns=["p", "q"]
    ),
    in_process=True,
    note="each key a row, the lists read across the column names given",
)
case(
    "basics/series-tz-convert-step",
    "Series.tz_convert",
    level="L3",
    covers=("tz", "axis"),
    frames=("single",),
    expr=lambda pd, df: pd.Series([1, 2], index=_utc_pair(pd)).tz_convert("US/Eastern", axis=0),
    in_process=True,
    note="a daily step moved to a clock with daylight saving is dropped from the labels",
)
case(
    "basics/frame-tz-convert-level",
    "DataFrame.tz_convert",
    level="L3",
    covers=("tz", "level"),
    frames=("single",),
    expr=lambda pd, df: pd.DataFrame(
        {"v": [1, 2]}, index=pd.MultiIndex.from_arrays([_utc_pair(pd), [1, 2]])
    ).tz_convert("Asia/Tokyo", level=0),
    in_process=True,
    note="the instants of one level of the rows moved to another clock, the other kept",
)
case(
    "basics/frame-tz-localize-nonexistent",
    "DataFrame.tz_localize",
    level="L3",
    covers=("tz", "nonexistent"),
    frames=("single",),
    expr=lambda pd, df: pd.DataFrame(
        {"v": [1, 2]}, index=pd.date_range("2024-03-10 01:30", periods=2, freq="h")
    ).tz_localize("US/Eastern", nonexistent="shift_forward"),
    in_process=True,
    note="a wall time skipped by the spring change is moved to the first one that exists",
)
case(
    "basics/frame-tz-localize-ambiguous",
    "DataFrame.tz_localize",
    level="L3",
    covers=("tz", "ambiguous"),
    frames=("single",),
    expr=lambda pd, df: pd.DataFrame(
        {"v": [1, 2]}, index=pd.DatetimeIndex(["2024-11-03 01:30", "2024-11-03 01:30"])
    ).tz_localize("US/Eastern", ambiguous=[True, False]),
    in_process=True,
    note="a wall time the autumn change repeats, read as summer time and then winter time",
)
case(
    "basics/series-between-time-inclusive",
    "Series.between_time",
    level="L3",
    covers=("start_time", "end_time", "inclusive"),
    frames=("single",),
    expr=lambda pd, df: pd.Series(range(6), index=_hourly_six(pd)).between_time(
        "20:00", "04:00", inclusive="right"
    ),
    in_process=True,
    note="a window across midnight that takes its end and leaves its start out",
)
case(
    "basics/frame-between-time-neither",
    "DataFrame.between_time",
    level="L3",
    covers=("start_time", "end_time", "inclusive", "axis"),
    frames=("single",),
    expr=lambda pd, df: pd.DataFrame({"x": range(6)}, index=_hourly_six(pd)).between_time(
        "04:00", "12:00", inclusive="neither", axis=0
    ),
    in_process=True,
    note="a window that leaves both ends out, read along the rows",
)
case(
    "basics/frame-update-left-kept",
    "DataFrame.update",
    level="L3",
    covers=("other", "join", "overwrite"),
    frames=("single",),
    expr=_updated_left_only,
    in_process=True,
    note="only gaps in the frame are filled from the other one when overwrite is off",
)
case(
    "basics/frame-update-filtered",
    "DataFrame.update",
    level="L3",
    covers=("other", "filter_func"),
    frames=("single",),
    expr=_updated_filtered,
    in_process=True,
    note="only cells the filter passes are written from the other frame",
)
case(
    "basics/frame-update-overlap-refused",
    "DataFrame.update",
    level="L4",
    covers=("other", "errors"),
    frames=("single",),
    expr=_updated_overlap,
    raises=("ValueError", "Data overlaps"),
    in_process=True,
    note="errors='raise' refuses to write over a value both frames hold",
)
case(
    "basics/series-product-flags",
    "Series.product",
    level="L3",
    covers=("axis", "skipna", "numeric_only", "min_count"),
    frames=("single",),
    expr=lambda pd, df: pd.Series(
        [
            pd.Series([1.0, float("nan"), 3.0]).product(axis=0, numeric_only=False),
            pd.Series([1.0, float("nan"), 3.0]).product(skipna=False),
            pd.Series([1.0, float("nan"), 3.0]).product(min_count=5),
        ]
    ),
    in_process=True,
    note="a gap skipped, a gap kept, and too few values for min_count",
)
case(
    "basics/series-rename-on-level",
    "Series.rename",
    level="L3",
    covers=("index", "axis", "level"),
    frames=("single",),
    expr=lambda pd, df: pd.Series([1, 2], index=_xy_pairs(pd)).rename({1: 9}, axis=0, level=1),
    in_process=True,
    note="labels renamed on the inner level of the rows only",
)
case(
    "basics/series-rename-missing-refused",
    "Series.rename",
    level="L4",
    covers=("index", "errors"),
    frames=("single",),
    expr=lambda pd, df: pd.Series([1, 2]).rename({5: 6}, errors="raise"),
    raises=("KeyError", "not found in axis"),
    in_process=True,
    note="errors='raise' refuses a label the series does not have",
)
case(
    "basics/frame-insert-allow-duplicates",
    "DataFrame.insert",
    level="L3",
    covers=("loc", "column", "value", "allow_duplicates"),
    frames=("single",),
    expr=_inserted_beside,
    in_process=True,
    note="allow_duplicates given for a new label, which goes in where loc says",
)
case(
    "basics/frame-compare-stacked",
    "DataFrame.compare",
    level="L3",
    covers=("other", "align_axis"),
    frames=("single",),
    expr=lambda pd, df: _compare_pair(pd)[0].compare(_compare_pair(pd)[1], align_axis=0),
    in_process=True,
    note="the two sides of each difference stacked as rows rather than set side by side",
)
case(
    "basics/frame-compare-kept-shape",
    "DataFrame.compare",
    level="L3",
    covers=("other", "keep_shape", "keep_equal"),
    frames=("single",),
    expr=lambda pd, df: _compare_pair(pd)[0].compare(
        _compare_pair(pd)[1], keep_shape=True, keep_equal=True
    ),
    in_process=True,
    note="every row and column kept, the equal values shown rather than blanked",
)
case(
    "basics/frame-compare-result-names",
    "DataFrame.compare",
    level="L3",
    covers=("other", "result_names"),
    frames=("single",),
    expr=lambda pd, df: _compare_pair(pd)[0].compare(_compare_pair(pd)[1], result_names=("L", "R")),
    in_process=True,
    note="the two sides named as asked for in the inner column level",
)
case(
    "basics/frame-to-timestamp-day-end",
    "DataFrame.to_timestamp",
    level="L3",
    covers=("freq", "how", "axis"),
    frames=("single",),
    expr=lambda pd, df: pd.DataFrame(
        {"v": [1, 2]}, index=pd.period_range("2024-01", periods=2, freq="M")
    ).to_timestamp(freq="D", how="end", axis=0),
    in_process=True,
    note="monthly periods read as the last day of each month",
)
case(
    "basics/frame-reindex-inner-level",
    "DataFrame.reindex",
    level="L3",
    covers=("labels", "axis", "level"),
    frames=("single",),
    expr=lambda pd, df: pd.DataFrame({"v": range(5)}, index=_uneven_rows(pd)).reindex(
        labels=[3, 2], axis="index", level="y"
    ),
    in_process=True,
    note="rows kept by the inner level in the order they stand, not the order asked for",
)
case(
    "basics/series-reindex-outer-level",
    "Series.reindex",
    level="L3",
    covers=("index", "level"),
    frames=("single",),
    expr=lambda pd, df: pd.Series(range(5), index=_uneven_rows(pd)).reindex(["c", "b"], level=0),
    in_process=True,
    note="rows taken by the outer level in the order the labels are asked for",
)
case(
    "basics/series-swaplevel-named",
    "Series.swaplevel",
    level="L3",
    covers=("i", "j"),
    frames=("single",),
    expr=lambda pd, df: pd.Series([1, 2], index=_xy_pairs(pd)).swaplevel("x", "y"),
    in_process=True,
    note="the two levels of the rows swapped by name",
)
case(
    "basics/series-reset-index-level-dropped",
    "Series.reset_index",
    level="L3",
    covers=("level", "drop"),
    frames=("single",),
    expr=lambda pd, df: pd.Series([1, 2], index=_xy_pairs(pd)).reset_index(level="y", drop=True),
    in_process=True,
    note="one level of the rows taken away and not kept as a column",
)
case(
    "basics/series-reset-index-named",
    "Series.reset_index",
    level="L3",
    covers=("name", "allow_duplicates"),
    frames=("single",),
    expr=lambda pd, df: pd.Series([1, 2], index=_xy_pairs(pd)).reset_index(
        name="q", allow_duplicates=False
    ),
    in_process=True,
    note="every level made a column and the values named as asked",
)
case(
    "basics/series-rename-axis-mapping",
    "Series.rename_axis",
    level="L3",
    covers=("index",),
    frames=("single",),
    expr=lambda pd, df: pd.Series([1, 2], index=_xy_pairs(pd)).rename_axis(index={"x": "z"}),
    in_process=True,
    note="one level name changed through a mapping, the other kept",
)
case(
    "basics/series-rename-axis-list",
    "Series.rename_axis",
    level="L3",
    covers=("mapper",),
    frames=("single",),
    expr=lambda pd, df: pd.Series([1, 2], index=_xy_pairs(pd)).rename_axis(["p", "q"]),
    in_process=True,
    note="every level of the rows given a new name in order",
)
case(
    "basics/series-sort-values-keyed",
    "Series.sort_values",
    level="L3",
    covers=("kind", "key"),
    frames=("single",),
    expr=lambda pd, df: pd.Series([3, 1, 2], index=list("abc")).sort_values(
        kind="stable", key=_negated
    ),
    in_process=True,
    note="sorted by a key that turns the order around, with a stable sort",
)
case(
    "basics/series-sort-values-renumbered",
    "Series.sort_values",
    level="L3",
    covers=("ascending", "na_position", "ignore_index"),
    frames=("single",),
    expr=lambda pd, df: pd.Series([3.0, float("nan"), 2.0]).sort_values(
        ascending=False, na_position="first", ignore_index=True
    ),
    in_process=True,
    note="largest first with the gap leading, and the labels numbered again",
)
case(
    "basics/series-skew-flags",
    "Series.skew",
    level="L3",
    covers=("axis", "skipna", "numeric_only"),
    frames=("single",),
    expr=lambda pd, df: pd.Series(
        [
            pd.Series([1.0, 2.0, 5.0, 9.0]).skew(axis=0, numeric_only=False),
            pd.Series([1.0, float("nan"), 5.0, 9.0]).skew(skipna=False),
        ]
    ),
    in_process=True,
    note="the skew of four values, and a gap kept so the answer is missing",
)


def _leveled_rows(pd):
    """A frame on two levels of row labels, the outer one repeating down the rows."""
    index = pd.MultiIndex.from_tuples([(1, "x"), (1, "y"), (2, "z")], names=["a", "b"])
    return pd.DataFrame({"c": [1.5, float("nan"), 2.0], "d": ["p", "q", "r"]}, index=index)


def _into_buffer(write):
    """What a writer puts in a text buffer it is handed."""
    buffer = io.StringIO()
    write(buffer)
    return buffer.getvalue()


def _stamped_pair(pd):
    """Two dates beside a float, a text and a whole number."""
    return pd.DataFrame(
        {
            "n": [1, 2],
            "t": ["é", "y<z"],
            "f": [1.123456, float("nan")],
            "d": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        }
    )


def _odd_cell(value):
    """The text default_handler writes for an object JSON cannot hold."""
    return "odd"


case(
    "basics/frame-to-html-levels",
    "DataFrame.to_html",
    frames=("single",),
    expr=lambda pd, df: [_leveled_rows(pd).to_html(), _leveled_rows(pd)._repr_html_()],
    in_process=True,
    note="each level of the row labels in its own cell, a repeat joined into one tall cell",
)
case(
    "basics/frame-to-html-levels-unsparse",
    "DataFrame.to_html",
    level="L3",
    covers=("sparsify", "index_names"),
    frames=("single",),
    expr=lambda pd, df: [
        _leveled_rows(pd).to_html(sparsify=False),
        _leveled_rows(pd).to_html(index_names=False),
        _leveled_rows(pd).to_html(index=False),
    ],
    in_process=True,
    note="every label repeated, the row of level names left out, and no labels at all",
)
case(
    "basics/frame-to-html-levels-cut",
    "DataFrame.to_html",
    level="L3",
    covers=("max_rows",),
    frames=("single",),
    expr=lambda pd, df: pd.DataFrame(
        {"v": range(8)},
        index=pd.MultiIndex.from_tuples([("a", n) for n in range(8)], names=["k", "n"]),
    ).to_html(max_rows=4),
    in_process=True,
    note="the row of dots for cut rows breaks the tall cell of a repeated label",
)
case(
    "basics/frame-to-html-picked",
    "DataFrame.to_html",
    level="L3",
    covers=("columns", "formatters", "justify", "decimal", "buf"),
    frames=("single",),
    expr=lambda pd, df: [
        _stamped_pair(pd).to_html(columns=["n", "t"], justify="left"),
        _stamped_pair(pd).to_html(formatters={"n": lambda v: f"<{v}>"}),
        _stamped_pair(pd).to_html(decimal=","),
        _into_buffer(lambda buffer: _stamped_pair(pd).to_html(buf=buffer)),
    ],
    in_process=True,
    note="some columns left aligned, a formatter, a comma for the point, and a buffer",
)
case(
    "basics/frame-to-html-links",
    "DataFrame.to_html",
    level="L3",
    covers=("render_links",),
    frames=("single",),
    expr=lambda pd, df: pd.DataFrame({"u": ["http://x.org", "plain"]}).to_html(render_links=True),
    in_process=True,
    note="a cell that reads as a link becomes an anchor",
)
case(
    "basics/series-to-csv-quoted",
    "Series.to_csv",
    level="L3",
    covers=("quoting", "quotechar", "chunksize", "columns"),
    frames=("single",),
    expr=lambda pd, df: [
        pd.Series([1.5, float("nan")], name="v").to_csv(quoting=1),
        pd.Series(["a", "b"], name="v").to_csv(quotechar="'", quoting=2),
        pd.Series([1, 2, 3], name="v").to_csv(chunksize=1),
        pd.Series([1, 2], name="v").to_csv(columns=["v"]),
    ],
    in_process=True,
    note="every cell quoted, text quoted with a chosen mark, rows in chunks, and the name",
)
case(
    "basics/series-to-csv-target",
    "Series.to_csv",
    level="L3",
    covers=("path_or_buf", "mode", "encoding", "compression"),
    frames=("single",),
    expr=lambda pd, df: [
        _into_buffer(lambda buffer: pd.Series([1, 2], name="v").to_csv(buffer, mode="w")),
        pd.Series(["é"], name="v").to_csv(encoding="utf-8", compression=None),
    ],
    in_process=True,
    note="written into a buffer, and handed back with the encoding and compression given",
)
case(
    "basics/frame-to-csv-target",
    "DataFrame.to_csv",
    level="L3",
    covers=("path_or_buf", "mode", "encoding", "compression", "chunksize", "errors"),
    frames=("single",),
    expr=lambda pd, df: [
        _into_buffer(lambda buffer: _stamped_pair(pd).to_csv(buffer, mode="w", chunksize=1)),
        _stamped_pair(pd).to_csv(encoding="utf-8", errors="strict", compression=None),
    ],
    in_process=True,
    note="written into a buffer one row at a time, and handed back under the encoding options",
)
case(
    "basics/series-to-json-options",
    "Series.to_json",
    level="L3",
    covers=(
        "path_or_buf",
        "date_format",
        "double_precision",
        "force_ascii",
        "date_unit",
        "default_handler",
        "lines",
        "compression",
    ),
    frames=("single",),
    expr=lambda pd, df: [
        pd.Series(pd.to_datetime(["2024-01-02"])).to_json(date_format="iso", date_unit="s"),
        pd.Series([1.23456, 2.5]).to_json(double_precision=2),
        pd.Series(["é"]).to_json(force_ascii=False),
        pd.Series([object()]).to_json(default_handler=_odd_cell),
        pd.Series([1, 2]).to_json(orient="records", lines=True, compression=None),
        _into_buffer(lambda buffer: pd.Series([1, 2]).to_json(buffer)),
    ],
    in_process=True,
    note="ISO dates in seconds, fewer digits, raw text, a handler, lines, and a buffer",
)
case(
    "basics/frame-to-json-options",
    "DataFrame.to_json",
    level="L3",
    covers=(
        "path_or_buf",
        "date_format",
        "force_ascii",
        "date_unit",
        "default_handler",
        "compression",
        "index",
    ),
    frames=("single",),
    expr=lambda pd, df: [
        _stamped_pair(pd).to_json(date_format="iso", date_unit="ms"),
        _stamped_pair(pd).to_json(force_ascii=False, orient="split", index=False),
        pd.DataFrame({"a": [object()]}).to_json(default_handler=_odd_cell, compression=None),
        _into_buffer(lambda buffer: _stamped_pair(pd).to_json(buffer, date_unit="s")),
    ],
    in_process=True,
    note="ISO dates, raw text without the labels, a handler, and a buffer",
)
case(
    "basics/frame-to-dict-shapes",
    "DataFrame.to_dict",
    level="L3",
    covers=("orient", "into", "index"),
    frames=("single",),
    expr=lambda pd, df: [
        _stamped_pair(pd)[["n"]].to_dict(orient="split", index=False),
        _stamped_pair(pd)[["n"]].to_dict(orient="tight", index=False),
        type(_stamped_pair(pd)[["n"]].to_dict(into=dict)).__name__,
    ],
    in_process=True,
    note="the split and tight shapes without labels, and the mapping type asked for",
)
case(
    "basics/frame-to-string-target",
    "DataFrame.to_string",
    level="L3",
    covers=("buf", "columns"),
    frames=("single",),
    expr=lambda pd, df: [
        _stamped_pair(pd).to_string(columns=["n", "f"]),
        _into_buffer(lambda buffer: _stamped_pair(pd).to_string(buffer)),
    ],
    in_process=True,
    note="some of the columns, and the text written into a buffer",
)
case(
    "basics/series-to-string-target",
    "Series.to_string",
    level="L3",
    covers=("buf",),
    frames=("single",),
    expr=lambda pd, df: _into_buffer(lambda buffer: pd.Series([1.5, 2.0]).to_string(buffer)),
    in_process=True,
    note="a column's text written into a buffer",
)
case(
    "basics/series-to-dict-into",
    "Series.to_dict",
    level="L3",
    covers=("into",),
    frames=("single",),
    expr=lambda pd, df: [
        pd.Series([1, 2], index=["a", "b"]).to_dict(into=dict),
        type(pd.Series([1, 2]).to_dict(into=dict)).__name__,
    ],
    in_process=True,
    note="a column as a plain mapping of label to value",
)

case(
    "basics/wide-to-long-named",
    "pandas.wide_to_long",
    level="L3",
    covers=("df", "stubnames", "i", "j"),
    frames=("single",),
    expr=lambda pd, df: pd.wide_to_long(
        df=pd.DataFrame(
            {
                "A1970": [1, 2],
                "A1980": [3, 4],
                "B1970": [5.0, 6.0],
                "B1980": [7.0, 8.0],
                "id": [0, 1],
            }
        ),
        stubnames=["A", "B"],
        i="id",
        j="year",
    ),
    in_process=True,
    note="two stubs, each given by keyword",
)
case(
    "basics/series-to-csv-dates",
    "Series.to_csv",
    level="L3",
    covers=("date_format",),
    frames=("single",),
    expr=lambda pd, df: pd.Series(
        pd.to_datetime(["2024-01-02 03:04", "2024-02-03 00:00"]), name="t"
    ).to_csv(date_format="%Y/%m/%d"),
    in_process=True,
    note="instants written in the given format",
)
case(
    "basics/series-to-csv-escaped",
    "Series.to_csv",
    level="L3",
    covers=("doublequote", "escapechar", "errors"),
    frames=("single",),
    expr=lambda pd, df: pd.Series(['a"b', "é"], name="v").to_csv(
        doublequote=False, escapechar="\\", errors="strict"
    ),
    in_process=True,
    note="a quote escaped rather than doubled",
)
case(
    "basics/series-to-json-shaped",
    "Series.to_json",
    level="L3",
    covers=("index", "indent"),
    frames=("single",),
    expr=lambda pd, df: pd.Series([1, 2], index=["a", "b"]).to_json(
        orient="split", index=False, indent=2
    ),
    in_process=True,
    note="split without labels, indented",
)
case(
    "basics/series-to-json-mode",
    "Series.to_json",
    level="L3",
    covers=("mode", "lines", "orient"),
    frames=("single",),
    expr=lambda pd, df: pd.Series([1, 2]).to_json(orient="records", lines=True, mode="w"),
    in_process=True,
    note="JSON lines written in the default mode",
)


def _sql_filled(pd, **options):
    """A sqlite3 database in memory with table `t` written by the engine's own `to_sql`."""
    import sqlite3

    con = sqlite3.connect(":memory:")
    frame = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
            "t": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "c": [1.5, float("nan"), 2.5],
        }
    )
    frame.to_sql(name="t", con=con, **({"index": False} | options))
    con.execute("create table g (i integer, f real, s text)")
    con.execute("insert into g values (1, 1.5, 'x'), (null, null, null), (3, 2.0, 'z')")
    return con


def _sql_back(pd, query="select * from t", **options):
    """Table `t` written with `options` and read back whole."""
    return pd.read_sql_query(query, _sql_filled(pd, **options))


def _sql_append(pd):
    """Table `t` replaced, then appended to in chunks with multi-row inserts."""
    con = _sql_filled(pd)
    frame = pd.DataFrame({"a": [7, 8, 9]})
    frame.to_sql("t", con, if_exists="replace", index=False)
    frame.to_sql("t", con, if_exists="append", index=False, chunksize=2, method="multi")
    return pd.read_sql_query("select * from t", con)


def _sql_read_types(pd):
    """The types of table `g` under each backend, and with a column cast."""
    read = [
        pd.read_sql("select * from g", _sql_filled(pd), dtype_backend=backend).dtypes
        for backend in ("numpy_nullable", "pyarrow")
    ]
    read.append(pd.read_sql("select * from g", _sql_filled(pd), dtype={"f": "float32"}).dtypes)
    return [types.astype(str).tolist() for types in read]


def _sql_chunk_types(pd):
    """The Arrow types of the first chunk, and the labels of every chunk joined."""
    chunks = pd.read_sql_query(
        "select * from g", _sql_filled(pd), chunksize=2, dtype_backend="pyarrow"
    )
    return next(iter(chunks)).dtypes.astype(str).tolist(), _sql_chunks(
        pd, pd.read_sql_query
    ).index.tolist()


def _sql_chunks(pd, read, **options):
    """Every chunk a chunked read hands back, joined in order."""
    return pd.concat(list(read("select * from t", _sql_filled(pd), chunksize=2, **options)))


SQL_NOTE = (
    "the SQL functions live in firepanda's Python layer and talk to sqlite3 there, so a "
    "driver entry could only emit the frame it was handed"
)


def _series_sql(pd):
    """A column written with every option, its count, then appended to and read back raw."""
    import sqlite3

    con = sqlite3.connect(":memory:")
    written = pd.Series([1, 2, 3], name="v").to_sql(
        name="t",
        con=con,
        schema=None,
        if_exists="replace",
        index=True,
        index_label="i",
        chunksize=2,
        dtype={"v": "INTEGER"},
        method="multi",
    )
    pd.Series([4], name="v").to_sql("t", con, if_exists="append", index=False)
    return written, con.execute("select * from t").fetchall()


case(
    "basics/series-sql-options",
    "Series.to_sql",
    level="L3",
    covers=(
        "name",
        "con",
        "schema",
        "if_exists",
        "index",
        "index_label",
        "chunksize",
        "dtype",
        "method",
    ),
    frames=("single",),
    expr=lambda pd, df: _series_sql(pd),
    in_process=True,
    note="a column becomes a table with its labels, and the rows written are counted. " + SQL_NOTE,
)
case(
    "basics/sql-roundtrip",
    "DataFrame.to_sql",
    level="L3",
    covers=("name", "con", "index"),
    frames=("single",),
    expr=lambda pd, df: _sql_back(pd),
    in_process=True,
    note="a frame written and read back. " + SQL_NOTE,
)
case(
    "basics/sql-write-labels",
    "DataFrame.to_sql",
    level="L3",
    covers=("index", "index_label"),
    frames=("single",),
    expr=lambda pd, df: _sql_back(pd, index=True, index_label="ix"),
    in_process=True,
    note="the row labels written as a named column. " + SQL_NOTE,
)
case(
    "basics/sql-write-replace",
    "DataFrame.to_sql",
    level="L3",
    covers=("if_exists", "chunksize", "method"),
    frames=("single",),
    expr=lambda pd, df: _sql_append(pd),
    in_process=True,
    note="a table replaced, then appended to in chunks with one statement each. " + SQL_NOTE,
)
case(
    "basics/sql-write-types",
    "DataFrame.to_sql",
    level="L3",
    covers=("dtype", "schema"),
    frames=("single",),
    expr=lambda pd, df: _sql_back(pd, dtype={"a": "REAL"}, schema="main"),
    in_process=True,
    note="a column's SQL type named, in the main schema. " + SQL_NOTE,
)
case(
    "basics/sql-write-count",
    "DataFrame.to_sql",
    frames=("single",),
    expr=lambda pd, df: pd.DataFrame({"a": [1, 2]}).to_sql(
        "n", __import__("sqlite3").connect(":memory:")
    ),
    in_process=True,
    note="the number of rows written. " + SQL_NOTE,
)
case(
    "basics/sql-write-taken",
    "DataFrame.to_sql",
    level="L4",
    frames=("single",),
    expr=lambda pd, df: pd.DataFrame({"a": [1]}).to_sql("t", _sql_filled(pd)),
    in_process=True,
    note=SQL_NOTE,
    raises=("ValueError", "Table 't' already exists."),
)
case(
    "basics/sql-read",
    "pandas.read_sql",
    level="L3",
    covers=("sql", "con", "index_col", "parse_dates"),
    frames=("single",),
    expr=lambda pd, df: pd.read_sql(
        sql="select * from t", con=_sql_filled(pd), index_col="a", parse_dates=["t"]
    ),
    in_process=True,
    note="a column moved to the labels and one read as instants. " + SQL_NOTE,
)
case(
    "basics/sql-read-params",
    "pandas.read_sql",
    level="L3",
    covers=("params", "columns", "coerce_float"),
    frames=("single",),
    expr=lambda pd, df: pd.read_sql(
        "select a, b from t where a > :low",
        _sql_filled(pd),
        params={"low": 1},
        columns=["a"],
        coerce_float=False,
    ),
    in_process=True,
    note="a named parameter; columns is unused over sqlite3, as in pandas. " + SQL_NOTE,
)
case(
    "basics/sql-read-types",
    "pandas.read_sql",
    level="L3",
    covers=("dtype", "dtype_backend"),
    frames=("single",),
    expr=lambda pd, df: _sql_read_types(pd),
    in_process=True,
    note="whole numbers with a gap stay whole under both backends. " + SQL_NOTE,
)
case(
    "basics/sql-read-nullable",
    "pandas.read_sql",
    level="L3",
    covers=("dtype_backend",),
    frames=("single",),
    expr=lambda pd, df: pd.read_sql(
        "select * from g", _sql_filled(pd), dtype_backend="numpy_nullable"
    ),
    in_process=True,
    note="the nullable values themselves. " + SQL_NOTE,
)
case(
    "basics/sql-read-chunks",
    "pandas.read_sql",
    level="L3",
    covers=("chunksize",),
    frames=("single",),
    expr=lambda pd, df: _sql_chunks(pd, pd.read_sql),
    in_process=True,
    note="chunks of two rows, joined. " + SQL_NOTE,
)
case(
    "basics/sql-query",
    "pandas.read_sql_query",
    level="L3",
    covers=("sql", "con", "params", "index_col"),
    frames=("single",),
    expr=lambda pd, df: pd.read_sql_query(
        "select * from t where a > ?", _sql_filled(pd), params=(1,), index_col="b"
    ),
    in_process=True,
    note="a positional parameter. " + SQL_NOTE,
)
case(
    "basics/sql-query-dates",
    "pandas.read_sql_query",
    level="L3",
    covers=("parse_dates", "coerce_float", "dtype"),
    frames=("single",),
    expr=lambda pd, df: pd.read_sql_query(
        "select * from t",
        _sql_filled(pd),
        parse_dates={"t": "%Y-%m-%d"},
        coerce_float=False,
        dtype={"a": "float64"},
    ),
    in_process=True,
    note="instants read with a format and a column cast. " + SQL_NOTE,
)
case(
    "basics/sql-query-chunks",
    "pandas.read_sql_query",
    level="L3",
    covers=("chunksize", "dtype_backend"),
    frames=("single",),
    expr=lambda pd, df: _sql_chunk_types(pd),
    in_process=True,
    note="each chunk in Arrow types, and the labels of the joined chunks. " + SQL_NOTE,
)
case(
    "basics/sql-query-missing",
    "pandas.read_sql_query",
    level="L4",
    frames=("single",),
    expr=lambda pd, df: pd.read_sql_query("select * from nothere", _sql_filled(pd)),
    in_process=True,
    note=SQL_NOTE,
    raises=("DatabaseError", "no such table: nothere"),
)
case(
    "basics/sql-table-missing",
    "pandas.read_sql_table",
    level="L4",
    frames=("single",),
    expr=lambda pd, df: pd.read_sql_table("nothere", _sql_filled(pd)),
    in_process=True,
    note="over sqlite3 a table is read by name only through SQLAlchemy. " + SQL_NOTE,
    raises=("ValueError", "Table nothere not found"),
)


def _stata_written(pd, **options):
    """A small frame written to Stata in memory by the engine's own `to_stata`."""
    import datetime

    frame = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
            "c": [1.5, float("nan"), 2.5],
            "d": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        }
    )
    buffer = io.BytesIO()
    frame.to_stata(buffer, time_stamp=datetime.datetime(2024, 5, 6, 7, 8), **options)
    buffer.seek(0)
    return buffer


def _stata_back(pd, read=None, **options):
    """A frame written to Stata with `options` and read back with `read`."""
    return pd.read_stata(_stata_written(pd, **options), **(read or {}))


def _stata_labels(pd):
    """The labels a Stata file carries, read through the reader object."""
    written = _stata_written(
        pd,
        data_label="hello",
        variable_labels={"a": "A label"},
        value_labels={"a": {1: "one", 2: "two", 3: "three"}},
    )
    with pd.read_stata(written, iterator=True) as reader:
        return reader.data_label, reader.variable_labels(), reader.value_labels()


STATA_NOTE = (
    "the Stata reader and writer live in firepanda's Python layer, which reads and "
    "writes the bytes itself, so a driver entry could only emit the frame it was handed"
)
NAMED = {"a": {1: "one", 2: "two", 3: "three"}}

STATAS = {
    "roundtrip": ({}, {}, ("path",), ()),
    "no-labels": ({"write_index": False}, {}, ("write_index",), ()),
    "day-dates": ({"convert_dates": {"d": "td"}, "write_index": False}, {}, ("convert_dates",), ()),
    "raw-dates": (
        {"convert_dates": {"d": "tc"}},
        {"convert_dates": False},
        ("convert_dates",),
        ("convert_dates",),
    ),
    "version-117": ({"version": 117}, {}, ("version",), ()),
    "long-text": ({"version": 118, "convert_strl": ["b"]}, {}, ("version", "convert_strl"), ()),
    "version-119": ({"version": 119}, {}, ("version",), ()),
    "big-endian": ({"version": 114, "byteorder": "big"}, {}, ("version", "byteorder"), ()),
    "value-labels": ({"value_labels": NAMED}, {}, ("value_labels",), ()),
    "codes": (
        {"value_labels": NAMED},
        {"convert_categoricals": False},
        ("value_labels",),
        ("convert_categoricals",),
    ),
    "unordered": (
        {"value_labels": NAMED},
        {"order_categoricals": False},
        ("value_labels",),
        ("order_categoricals",),
    ),
    "index-col": ({}, {"index_col": "index"}, (), ("index_col",)),
    "columns": ({}, {"columns": ["b", "a"]}, (), ("columns",)),
    "widened": ({}, {"preserve_dtypes": False}, (), ("preserve_dtypes",)),
    "plain-read": ({}, {"compression": None}, (), ("compression",)),
}

for name, (write, read, writes, reads) in STATAS.items():
    if writes:
        case(
            f"basics/stata-write-{name}",
            "DataFrame.to_stata",
            level="L3",
            covers=("time_stamp", *writes),
            frames=("single",),
            expr=lambda pd, df, write=write, read=read: _stata_back(pd, read, **write),
            in_process=True,
            note="written and read back by the same engine. " + STATA_NOTE,
        )
    if reads:
        case(
            f"basics/stata-read-{name}",
            "pandas.read_stata",
            level="L3",
            covers=("filepath_or_buffer", *reads),
            frames=("single",),
            expr=lambda pd, df, write=write, read=read: _stata_back(pd, read, **write),
            in_process=True,
            note="written and read back by the same engine. " + STATA_NOTE,
        )

case(
    "basics/stata-write-labels",
    "DataFrame.to_stata",
    level="L3",
    covers=("data_label", "variable_labels", "value_labels"),
    frames=("single",),
    expr=lambda pd, df: _stata_labels(pd),
    in_process=True,
    note="the labels read back through the reader object. " + STATA_NOTE,
)
case(
    "basics/stata-read-iterator",
    "pandas.read_stata",
    level="L3",
    covers=("iterator",),
    frames=("single",),
    expr=lambda pd, df: pd.read_stata(_stata_written(pd), iterator=True).read(2),
    in_process=True,
    note="a reader handed back and asked for two rows. " + STATA_NOTE,
)
case(
    "basics/stata-read-chunks",
    "pandas.read_stata",
    level="L3",
    covers=("chunksize",),
    frames=("single",),
    expr=lambda pd, df: pd.concat(list(pd.read_stata(_stata_written(pd), chunksize=2))),
    in_process=True,
    note="chunks of two rows, joined. " + STATA_NOTE,
)
case(
    "basics/stata-read-missing-kept",
    "pandas.read_stata",
    level="L3",
    covers=("convert_missing",),
    frames=("single",),
    expr=lambda pd, df: [
        (type(value).__name__, str(value))
        for value in _stata_back(pd, {"convert_missing": True})["c"].tolist()
    ],
    in_process=True,
    note="a gap kept as Stata's missing value, compared by class name and text, since "
    "the class lives in each library's own module. " + STATA_NOTE,
)


def _plain_answer(out):
    """A tuple or array answer as plain lists, with NaN spelled so two of them compare."""
    if isinstance(out, tuple):
        return tuple(_plain_answer(x) for x in out)
    if hasattr(out, "tolist") and not hasattr(out, "to_frame"):
        out = out.tolist()
    if isinstance(out, list):
        return ["nan" if isinstance(x, float) and x != x else x for x in out]
    return out


def _zoned_steps(pd):
    """Steps onto a repeated hour and onto a missing one, under each policy."""
    fold = pd.Timestamp("2024-11-03 00:31", tz="US/Eastern")
    gap = pd.Timestamp("2024-03-10 03:30", tz="US/Eastern")
    return [
        str(fold.ceil("h", ambiguous=True)),
        str(fold.ceil("h", ambiguous=False)),
        str(fold.ceil("h", ambiguous="NaT")),
        str(gap.floor("2h", nonexistent="shift_forward")),
        str(gap.floor("2h", nonexistent="shift_backward")),
        str(gap.floor("2h", nonexistent="NaT")),
        str(pd.Timestamp("2024-07-01 13:45:07", tz="Europe/Paris").round("min")),
    ]


ZONED_NOTE = (
    "a zoned moment steps on its wall clock and takes its zone back through tz_localize, so "
    "the two policies settle a step onto a repeated or missing hour. "
)
for _name in ("round", "floor", "ceil"):
    case(
        f"basics/timestamp-{_name}-zone-policies",
        f"Timestamp.{_name}",
        level="L3",
        covers=("freq", "ambiguous", "nonexistent"),
        frames=("single",),
        expr=lambda pd, df, _name=_name: [
            str(
                getattr(pd.Timestamp("2024-11-03 00:31", tz="US/Eastern"), _name)(
                    "h", ambiguous=False, nonexistent="shift_forward"
                )
            ),
            str(
                getattr(pd.Timestamp("2024-03-10 01:40", tz="US/Eastern"), _name)(
                    "h", ambiguous=True, nonexistent="shift_backward"
                )
            ),
        ],
        in_process=True,
        note=ZONED_NOTE + PARAMETER_IN_PROCESS,
    )
case(
    "basics/timestamp-zone-steps",
    "Timestamp.ceil",
    level="L3",
    covers=("ambiguous", "nonexistent"),
    frames=("single",),
    expr=lambda pd, df: _zoned_steps(pd),
    in_process=True,
    note=ZONED_NOTE + PARAMETER_IN_PROCESS,
)
case(
    "basics/timestamp-ceil-repeated-hour",
    "Timestamp.ceil",
    level="L4",
    frames=("single",),
    expr=lambda pd, df: pd.Timestamp("2024-11-03 00:31", tz="US/Eastern").ceil("h"),
    raises=("ValueError", "Cannot infer dst time"),
    in_process=True,
    note="a step onto a repeated hour raises under the default policy. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/timestamp-floor-missing-hour",
    "Timestamp.floor",
    level="L4",
    frames=("single",),
    expr=lambda pd, df: pd.Timestamp("2024-03-10 03:30", tz="US/Eastern").floor("2h"),
    raises=("ValueError", "is a nonexistent time due to daylight savings time"),
    in_process=True,
    note="a step onto a missing hour raises under the default policy. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/series-drop-level",
    "Series.drop",
    level="L3",
    covers=("labels", "index", "level", "errors"),
    frames=("single",),
    expr=lambda pd, df: pd.concat(
        [
            pd.Series(
                [1, 2, 3],
                index=pd.MultiIndex.from_tuples([("a", 1), ("b", 2), ("a", 3)], names=["k", "n"]),
            ).drop("a", level=0),
            pd.Series(
                [1, 2, 3],
                index=pd.MultiIndex.from_tuples([("a", 1), ("b", 2), ("a", 3)], names=["k", "n"]),
            ).drop(index=[2, 3], level="n", errors="ignore"),
            pd.Series(
                [1, 2, 3], index=pd.MultiIndex.from_tuples([("a", 1), ("a", 1), ("b", 2)])
            ).drop("a", level=0),
        ]
    ),
    in_process=True,
    note="rows dropped by one level's values, repeated rows included. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/series-drop-level-missing",
    "Series.drop",
    level="L4",
    frames=("single",),
    expr=lambda pd, df: pd.Series(
        [1, 2], index=pd.MultiIndex.from_tuples([("a", 1), ("b", 2)])
    ).drop(["a", "z"], level=0),
    raises=("KeyError", "not found in level"),
    in_process=True,
    note="a label the level lacks is refused even beside one it has. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/series-drop-level-flat",
    "Series.drop",
    level="L4",
    frames=("single",),
    expr=lambda pd, df: pd.Series([1, 2]).drop(0, level=0),
    raises=("AssertionError", "axis must be a MultiIndex"),
    in_process=True,
    note="a level on flat labels is pandas' assertion. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/frame-drop-level",
    "DataFrame.drop",
    level="L3",
    covers=("labels", "axis", "columns", "level"),
    frames=("single",),
    expr=lambda pd, df: [
        pd.DataFrame(
            [[1, 2, 3]], columns=pd.MultiIndex.from_tuples([("a", "p"), ("a", "q"), ("b", "p")])
        )
        .drop("p", axis=1, level=1)
        .columns.tolist(),
        pd.DataFrame(
            [[1, 2, 3]], columns=pd.MultiIndex.from_tuples([("a", "p"), ("a", "q"), ("b", "p")])
        )
        .drop(columns="a", level=0)
        .columns.tolist(),
    ],
    in_process=True,
    note="columns dropped by one level of their labels. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/series-reindex-multi-fill",
    "Series.reindex",
    level="L3",
    covers=("index", "method", "limit"),
    frames=("single",),
    expr=lambda pd, df: pd.Series(
        [1, 2, 3], index=pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1)])
    ).reindex(pd.MultiIndex.from_tuples([("a", 3), ("a", 4), ("b", 5)]), method="ffill", limit=1),
    in_process=True,
    note="rows filled along a MultiIndex in tuple order, one inexact row per source. "
    + PARAMETER_IN_PROCESS,
)
case(
    "basics/series-reindex-tolerance",
    "Series.reindex",
    level="L3",
    covers=("index", "axis", "method", "tolerance"),
    frames=("single",),
    expr=lambda pd, df: pd.Series([1, 2], index=[0, 10]).reindex(
        [1, 8], method="nearest", tolerance=2, axis=0
    ),
    in_process=True,
    note="the nearest label within the tolerance. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/series-reductions-axis",
    "Series.min",
    level="L3",
    covers=("axis", "skipna", "numeric_only"),
    frames=("single",),
    expr=lambda pd, df: _plain_answer(
        [
            pd.Series([3.0, float("nan"), 1.0]).min(axis=0, skipna=False, numeric_only=False),
            pd.Series([3.0, float("nan"), 1.0]).min(axis=0, skipna=True),
        ]
    ),
    in_process=True,
    note="the smallest value, NaN when a gap is not skipped. " + PARAMETER_IN_PROCESS,
)
for _name, _data in (
    ("max", [3.0, float("nan"), 1.0]),
    ("median", [3.0, float("nan"), 1.0, 8.0]),
    ("kurt", [3.0, 1.0, 2.0, 8.0, float("nan")]),
    ("kurtosis", [3.0, 1.0, 2.0, 8.0, 5.0]),
):
    case(
        f"basics/series-{_name}-axis",
        f"Series.{_name}",
        level="L3",
        covers=("axis", "skipna", "numeric_only"),
        frames=("single",),
        expr=lambda pd, df, _name=_name, _data=_data: _plain_answer(
            [
                getattr(pd.Series(_data), _name)(axis=0, skipna=True, numeric_only=True),
                getattr(pd.Series(_data), _name)(axis=0, skipna=False, numeric_only=False),
            ]
        ),
        in_process=True,
        note="the reduction with gaps skipped and not. " + PARAMETER_IN_PROCESS,
        rules=Rules(tolerance=Tolerance.STATISTICAL, reason="a fourth moment squares twice"),
    )
case(
    "basics/series-filter-options",
    "Series.filter",
    level="L3",
    covers=("items", "like", "regex", "axis"),
    frames=("single",),
    expr=lambda pd, df: pd.concat(
        [
            pd.Series([1, 2, 3], index=["ab", "bc", "cd"]).filter(regex="^b", axis=0),
            pd.Series([1, 2, 3], index=["ab", "bc", "cd"]).filter(items=["cd", "zz"]),
            pd.Series([1, 2, 3], index=["ab", "bc", "cd"]).filter(like="a"),
        ]
    ),
    in_process=True,
    note="rows picked by a pattern, a list or a fragment of their label. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/series-dropna-options",
    "Series.dropna",
    level="L3",
    covers=("axis", "how", "ignore_index"),
    frames=("single",),
    expr=lambda pd, df: pd.Series([1.0, float("nan"), 3.0]).dropna(
        axis=0, how="any", ignore_index=True
    ),
    in_process=True,
    note="gaps dropped and the labels counted again. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/series-bfill-options",
    "Series.bfill",
    level="L3",
    covers=("axis", "limit", "limit_area"),
    frames=("single",),
    expr=lambda pd, df: pd.Series(
        [float("nan"), 1.0, float("nan"), float("nan"), 3.0, float("nan")]
    ).bfill(axis=0, limit=1, limit_area="inside"),
    in_process=True,
    note="gaps between values filled from below, one at a time. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/series-at-time-options",
    "Series.at_time",
    level="L3",
    covers=("time", "asof", "axis"),
    frames=("single",),
    expr=lambda pd, df: pd.Series(
        [1, 2, 3], index=pd.date_range("2024-01-01", periods=3, freq="12h")
    ).at_time("12:00", asof=False, axis=0),
    in_process=True,
    note="the rows at a time of day. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/series-asfreq-options",
    "Series.asfreq",
    level="L3",
    covers=("freq", "normalize", "fill_value"),
    frames=("single",),
    expr=lambda pd, df: pd.Series(
        [1.0, 2.0], index=pd.DatetimeIndex(["2024-01-01 06:00", "2024-01-03 06:00"])
    ).asfreq("D", normalize=True, fill_value=0.0),
    in_process=True,
    note="a daily grid on midnight with the new rows filled. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/series-align-level",
    "Series.align",
    level="L3",
    covers=("other", "join", "axis", "level", "copy"),
    frames=("single",),
    expr=lambda pd, df: [
        pd.Series([1, 2], index=pd.MultiIndex.from_tuples([("a", 1), ("b", 2)]))
        .align(pd.Series([10, 20], index=["a", "b"]), level=0, axis=0, copy=None)[1]
        .tolist(),
        pd.Series([1, 2], index=pd.MultiIndex.from_tuples([("a", 1), ("b", 2)]))
        .align(pd.Series([10], index=["b"]), join="inner", level=0)[1]
        .tolist(),
    ],
    in_process=True,
    note="a flat column spread over the rows of a level. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/series-to-timestamp-options",
    "Series.to_timestamp",
    level="L3",
    covers=("freq", "how", "copy"),
    frames=("single",),
    expr=lambda pd, df: [
        pd.Series([1, 2], index=pd.period_range("2024-01", periods=2, freq="M"))
        .to_timestamp(freq="D", how="end", copy=False)
        .index.astype(str)
        .tolist(),
        pd.Series([1, 2], index=pd.period_range("2024-01", periods=2, freq="M"))
        .to_timestamp(how="start")
        .index.astype(str)
        .tolist(),
    ],
    in_process=True,
    note="periods read at their end or their start. " + PARAMETER_IN_PROCESS,
)
for _name in ("first", "last"):
    case(
        f"basics/resampler-{_name}-options",
        f"Resampler.{_name}",
        level="L3",
        covers=("numeric_only", "min_count", "skipna"),
        frames=("single",),
        expr=lambda pd, df, _name=_name: pd.concat(
            [
                getattr(
                    pd.Series(
                        [1.0, float("nan"), 3.0, 4.0],
                        index=pd.date_range("2024-01-01", periods=4, freq="12h"),
                    ).resample("D"),
                    _name,
                )(numeric_only=False, min_count=2, skipna=True),
                getattr(
                    pd.Series(
                        [1.0, float("nan"), 3.0, 4.0],
                        index=pd.date_range("2024-01-01", periods=4, freq="12h"),
                    ).resample("D"),
                    _name,
                )(numeric_only=True, min_count=1, skipna=False),
            ],
            axis=1,
        ),
        in_process=True,
        note="the first or last value of each day, gaps skipped or not. " + PARAMETER_IN_PROCESS,
    )
case(
    "basics/str-index-options",
    "str.index",
    level="L3",
    covers=("sub", "start", "end"),
    frames=("single",),
    expr=lambda pd, df: pd.Series(["abcab", "xbz", "bb"]).str.index("b", start=1, end=4),
    in_process=True,
    note="where the text first appears inside the bounds. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/str-rindex-options",
    "str.rindex",
    level="L3",
    covers=("sub", "start", "end"),
    frames=("single",),
    expr=lambda pd, df: pd.Series(["abcab", "xbz", "bb"]).str.rindex("b", start=0, end=3),
    in_process=True,
    note="where the text last appears inside the bounds. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/str-decode-options",
    "str.decode",
    level="L3",
    covers=("encoding", "errors", "dtype"),
    frames=("single",),
    expr=lambda pd, df: (
        pd.Series([b"ab", b"\xff"]).str.decode("utf-8", errors="replace", dtype=object).tolist()
    ),
    in_process=True,
    note="bytes read as text with a bad byte replaced. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/qcut-options",
    "pandas.qcut",
    level="L3",
    covers=("x", "q", "retbins", "precision", "duplicates"),
    frames=("single",),
    expr=lambda pd, df: (
        pd.qcut([1, 2, 3, 4, 5, 5, 5], 3, retbins=True, precision=2, duplicates="drop")[0]
        .astype(str)
        .tolist(),
        _plain_answer(
            pd.qcut([1, 2, 3, 4, 5, 5, 5], 3, retbins=True, precision=2, duplicates="drop")[1]
        ),
    ),
    in_process=True,
    note="quantile bins with a repeated edge dropped, and the edges handed back. "
    + PARAMETER_IN_PROCESS,
)
case(
    "basics/cut-options",
    "pandas.cut",
    level="L3",
    covers=("x", "bins", "labels", "precision", "duplicates", "ordered"),
    frames=("single",),
    expr=lambda pd, df: [
        pd.cut([1, 5, 9], [0, 5, 5, 10], precision=1, duplicates="drop", ordered=True)
        .astype(str)
        .tolist(),
        list(pd.cut([1, 5, 9], 3, labels=["lo", "mid", "hi"], ordered=False)),
    ],
    in_process=True,
    note="bins with a repeated edge dropped, and labels in no order. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/factorize-options",
    "pandas.factorize",
    level="L3",
    covers=("values", "sort", "use_na_sentinel", "size_hint"),
    frames=("single",),
    expr=lambda pd, df: _plain_answer(
        pd.factorize(
            pd.Series(["b", "c", "a", "b"]), sort=True, use_na_sentinel=False, size_hint=10
        )
    ),
    in_process=True,
    note="codes against the sorted distinct values. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/lreshape-options",
    "pandas.lreshape",
    level="L3",
    covers=("data", "groups", "dropna"),
    frames=("single",),
    expr=lambda pd, df: pd.lreshape(
        pd.DataFrame({"id": [1, 2], "a1": [1.0, float("nan")], "a2": [3.0, 4.0]}),
        {"a": ["a1", "a2"]},
        dropna=False,
    ),
    in_process=True,
    note="wide columns stacked long with the gaps kept. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/from-dummies-options",
    "pandas.from_dummies",
    level="L3",
    covers=("data", "sep", "default_category"),
    frames=("single",),
    expr=lambda pd, df: pd.from_dummies(
        pd.DataFrame({"c_x": [1, 0, 0], "c_y": [0, 1, 0]}), sep="_", default_category="z"
    ),
    in_process=True,
    note="indicator columns read back into a category, the default for a row of zeros. "
    + PARAMETER_IN_PROCESS,
)
case(
    "basics/array-options",
    "pandas.array",
    level="L3",
    covers=("data", "dtype", "copy"),
    frames=("single",),
    expr=lambda pd, df: [
        str(pd.array([1, 2, 3], dtype="Int32", copy=True).dtype),
        list(pd.array([1, 2, 3], dtype="Int32", copy=True)),
    ],
    in_process=True,
    note="a masked array of the named type. " + PARAMETER_IN_PROCESS,
)


def _tail_frame(pd):
    """Floats with a gap, whole numbers and text, for the parameter cases below."""
    return pd.DataFrame(
        {"a": [1.0, float("nan"), 3.0, 4.0], "b": [4, 3, 2, 1], "s": ["x", "y", "x", "z"]}
    )


def _tail_numbers(pd):
    """The two number columns of `_tail_frame`."""
    return _tail_frame(pd)[["a", "b"]]


def _tail_pairs(pd):
    """A frame on three key pairs."""
    return pd.DataFrame(
        {"x": [1, 2, 3]},
        index=pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1)], names=["k", "n"]),
    )


def _tail_days(pd):
    """Three rows twelve hours apart."""
    return pd.DataFrame(
        {"x": [1.0, 2.0, 3.0]}, index=pd.date_range("2024-01-01", periods=3, freq="12h")
    )


def _bigger(x, y):
    """The larger of two columns, row by row."""
    return x.where(x > y, y)


TAIL_CASES = (
    (
        "frame-to-records-options",
        "DataFrame.to_records",
        ("index", "column_dtypes", "index_dtypes"),
        lambda pd: [
            _tail_frame(pd).to_records(index=False, column_dtypes={"b": "int32"}).dtype.descr,
            _tail_numbers(pd).to_records(index_dtypes="<U2").dtype.descr,
        ],
    ),
    (
        "frame-to-period-options",
        "DataFrame.to_period",
        ("freq", "axis", "copy"),
        lambda pd: (
            pd.DataFrame({"x": [1]}, index=pd.DatetimeIndex(["2024-03-05"]))
            .to_period(freq="M", axis=0, copy=False)
            .index.astype(str)
            .tolist()
        ),
    ),
    (
        "frame-to-numpy-options",
        "DataFrame.to_numpy",
        ("dtype", "copy", "na_value"),
        lambda pd: _tail_numbers(pd).to_numpy(dtype="float32", copy=True, na_value=-1).tolist(),
    ),
    (
        "frame-swaplevel-options",
        "DataFrame.swaplevel",
        ("i", "j", "axis"),
        lambda pd: _tail_pairs(pd).swaplevel(i="k", j="n", axis=0),
    ),
    (
        "frame-stack-future",
        "DataFrame.stack",
        ("future_stack",),
        lambda pd: pd.DataFrame({"x": [1], "y": [2]}).stack(future_stack=True),
    ),
    (
        "frame-ne-level",
        "DataFrame.ne",
        ("other", "axis", "level"),
        lambda pd: _tail_pairs(pd).ne(pd.Series([1, 9], index=["a", "b"]), axis=0, level=0),
    ),
    (
        "frame-lt-across",
        "DataFrame.lt",
        ("other", "axis"),
        lambda pd: _tail_numbers(pd).lt(pd.Series([2, 2], index=["a", "b"]), axis=1),
    ),
    (
        "frame-ge-down",
        "DataFrame.ge",
        ("other", "axis"),
        lambda pd: _tail_numbers(pd).ge([2, 2, 2, 2], axis=0),
    ),
    (
        "frame-kurtosis-options",
        "DataFrame.kurtosis",
        ("axis", "skipna", "numeric_only"),
        lambda pd: _tail_numbers(pd).kurtosis(axis=0, skipna=False, numeric_only=True),
    ),
    (
        "frame-from-records-options",
        "DataFrame.from_records",
        ("data", "columns", "coerce_float", "nrows"),
        lambda pd: pd.DataFrame.from_records(
            [(1, "1.5"), (2, "2.5"), (3, "x")], columns=["i", "f"], coerce_float=True, nrows=2
        ),
    ),
    (
        "frame-corrwith-options",
        "DataFrame.corrwith",
        ("other", "axis", "numeric_only", "min_periods"),
        lambda pd: pd.concat(
            [
                _tail_numbers(pd).corrwith(
                    pd.Series([1.0, 2.0, 3.0, 5.0]), axis=0, numeric_only=True, min_periods=3
                ),
                _tail_numbers(pd).corrwith(_tail_numbers(pd) * 2, axis=1),
            ]
        ),
    ),
    (
        "frame-combine-options",
        "DataFrame.combine",
        ("other", "func", "overwrite"),
        lambda pd: _tail_numbers(pd).combine(
            pd.DataFrame({"a": [9.0, 9.0, float("nan"), float("nan")]}), _bigger, overwrite=False
        ),
    ),
    (
        "frame-bfill-options",
        "DataFrame.bfill",
        ("axis", "limit", "limit_area"),
        lambda pd: pd.concat(
            [
                _tail_numbers(pd).bfill(axis=0, limit=1, limit_area="inside"),
                _tail_numbers(pd).bfill(axis=1),
            ],
            keys=["down", "across"],
        ),
    ),
    (
        "frame-at-time-options",
        "DataFrame.at_time",
        ("time", "asof", "axis"),
        lambda pd: _tail_days(pd).at_time("12:00", asof=False, axis=0),
    ),
    (
        "frame-astype-options",
        "DataFrame.astype",
        ("dtype", "copy", "errors"),
        lambda pd: [
            _tail_frame(pd).astype({"b": "float32"}, copy=False).dtypes.astype(str).tolist(),
            _tail_frame(pd).astype("int64", errors="ignore").dtypes.astype(str).tolist(),
        ],
    ),
    (
        "frame-asfreq-options",
        "DataFrame.asfreq",
        ("freq", "how", "normalize", "fill_value"),
        lambda pd: pd.DataFrame(
            {"x": [1.0, 2.0]}, index=pd.DatetimeIndex(["2024-01-01 06:00", "2024-01-03 06:00"])
        ).asfreq("D", how="start", normalize=True, fill_value=0.0),
    ),
    (
        "frame-align-column-fill",
        "DataFrame.align",
        ("other", "axis", "copy", "fill_value"),
        lambda pd: [
            part.to_dict()
            for part in _tail_numbers(pd).align(
                pd.Series([1, 2], index=["a", "z"]), axis=1, copy=None, fill_value=0
            )
        ],
    ),
    (
        "str-rfind-bounds",
        "str.rfind",
        ("sub", "start", "end"),
        lambda pd: pd.Series(["abcab", "xb"]).str.rfind("b", start=1, end=4),
    ),
    (
        "str-find-bounds",
        "str.find",
        ("sub", "start", "end"),
        lambda pd: pd.Series(["abcab", "xb"]).str.find("b", start=2, end=5),
    ),
    (
        "str-match-flags",
        "str.match",
        ("pat", "flags", "na"),
        lambda pd: pd.Series(["Ab", "ab", "b"]).str.match("a", flags=2, na=False),
    ),
    (
        "str-fullmatch-flags",
        "str.fullmatch",
        ("pat", "flags", "na"),
        lambda pd: pd.Series(["AB", "ab", "b"]).str.fullmatch("ab", flags=2, na=True),
    ),
    (
        "str-encode-replace",
        "str.encode",
        ("encoding", "errors"),
        lambda pd: pd.Series(["aé"]).str.encode("ascii", errors="replace").tolist(),
    ),
    (
        "to-numeric-backends",
        "pandas.to_numeric",
        ("arg", "errors", "dtype_backend"),
        lambda pd: [
            str(pd.to_numeric(pd.Series(["1", "2"]), dtype_backend="numpy_nullable").dtype),
            str(pd.to_numeric(pd.Series(["1", "2.5"]), dtype_backend="pyarrow").dtype),
            str(
                pd.to_numeric(
                    pd.Series(["1", "x"]), errors="coerce", dtype_backend="numpy_nullable"
                ).dtype
            ),
            str(pd.to_numeric(pd.Series([1, 2], dtype="Int64"), dtype_backend="pyarrow").dtype),
        ],
    ),
    (
        "json-normalize-ignore",
        "pandas.json_normalize",
        ("data", "record_path", "meta", "errors"),
        lambda pd: pd.json_normalize(
            [{"a": {"b": 1}}, {"a": {"c": 2}}], meta=[["a", "b"]], errors="ignore", record_path=None
        ),
    ),
    (
        "interval-range-named",
        "pandas.interval_range",
        ("start", "end", "name", "closed"),
        lambda pd: pd.interval_range(0, 3, name="r", closed="left").astype(str).tolist(),
    ),
    (
        "date-range-normalize-unit",
        "pandas.date_range",
        ("start", "periods", "normalize", "unit"),
        lambda pd: [
            pd.date_range("2024-01-01 05:00", periods=2, normalize=True, unit="s")
            .astype(str)
            .tolist(),
            str(pd.date_range("2024-01-01", periods=2, unit="ms").dtype),
        ],
    ),
    (
        "crosstab-margins-named",
        "pandas.crosstab",
        ("index", "columns", "margins", "margins_name", "dropna"),
        lambda pd: pd.crosstab(
            _tail_frame(pd)["s"],
            _tail_frame(pd)["s"].str.upper(),
            margins=True,
            margins_name="Tot",
            dropna=False,
        ),
    ),
    (
        "concat-levels-keys",
        "pandas.concat",
        ("objs", "keys", "levels", "copy"),
        lambda pd: pd.concat(
            [pd.Series([1, 2]), pd.Series([3])], keys=["p", "q"], levels=[["q", "p"]], copy=False
        ),
    ),
    (
        "timedelta-index-freq",
        "pandas.TimedeltaIndex",
        ("data", "freq", "copy"),
        lambda pd: str(pd.TimedeltaIndex(["1D", "2D"], freq="D", copy=True).freq),
    ),
    (
        "string-dtype-options",
        "pandas.StringDtype",
        ("storage", "na_value"),
        lambda pd: [
            str(pd.StringDtype(storage="python", na_value=float("nan"))),
            repr(pd.StringDtype("pyarrow")),
        ],
    ),
    (
        "series-built-named",
        "pandas.Series",
        ("data", "name", "copy"),
        lambda pd: pd.Series([1, 2], name="z", copy=True),
    ),
    (
        "range-index-dtype",
        "pandas.RangeIndex",
        ("start", "dtype", "copy"),
        lambda pd: repr(pd.RangeIndex(3, dtype="int64", copy=False)),
    ),
    (
        "named-agg-fields",
        "pandas.NamedAgg",
        ("column", "aggfunc"),
        lambda pd: _tail_frame(pd).groupby("s").agg(t=pd.NamedAgg(column="b", aggfunc="sum")),
    ),
    (
        "index-built-named",
        "pandas.Index",
        ("data", "copy", "name"),
        lambda pd: repr(pd.Index([1, 2], copy=True, name="w")),
    ),
    (
        "multi-index-built",
        "pandas.MultiIndex",
        ("levels", "codes", "copy", "name"),
        lambda pd: (
            pd.MultiIndex(levels=[["a"], [1]], codes=[[0], [0]], copy=True, name=["k", "n"]).names
        ),
    ),
    (
        "interval-dtype-options",
        "pandas.IntervalDtype",
        ("subtype", "closed"),
        lambda pd: str(pd.IntervalDtype(subtype="int64", closed="left")),
    ),
    (
        "zoned-dtype-options",
        "pandas.DatetimeTZDtype",
        ("unit", "tz"),
        lambda pd: str(pd.DatetimeTZDtype(unit="ms", tz="UTC")),
    ),
    (
        "category-index-dtype",
        "pandas.CategoricalIndex",
        ("data", "dtype"),
        lambda pd: repr(
            pd.CategoricalIndex(data=["a", "b"], dtype=pd.CategoricalDtype(["b", "a"]))
        ),
    ),
    (
        "category-dtype-options",
        "pandas.CategoricalDtype",
        ("categories", "ordered"),
        lambda pd: str(pd.CategoricalDtype(categories=["a", "b"], ordered=True)),
    ),
    (
        "set-categories-options",
        "cat.set_categories",
        ("new_categories", "ordered", "rename"),
        lambda pd: [
            pd.Series(["a", "b"], dtype="category")
            .cat.set_categories(["b", "a", "c"], ordered=True, rename=False)
            .cat.categories.tolist(),
            pd.Series(["a", "b"], dtype="category")
            .cat.set_categories(["x", "y"], rename=True)
            .tolist(),
        ],
    ),
    (
        "is-list-like-sets",
        "api.types.is_list_like",
        ("obj", "allow_sets"),
        lambda pd: [
            pd.api.types.is_list_like({1}, allow_sets=False),
            pd.api.types.is_list_like(obj=[1]),
        ],
    ),
    (
        "is-hashable-obj",
        "api.types.is_hashable",
        ("obj",),
        lambda pd: [pd.api.types.is_hashable(obj=(1,)), pd.api.types.is_hashable([1])],
    ),
    (
        "is-dtype-equal-named",
        "api.types.is_dtype_equal",
        ("source", "target"),
        lambda pd: pd.api.types.is_dtype_equal(source="int64", target="int64"),
    ),
    (
        "infer-dtype-gaps",
        "api.types.infer_dtype",
        ("value", "skipna"),
        lambda pd: [
            pd.api.types.infer_dtype(value=[1, float("nan")], skipna=False),
            pd.api.types.infer_dtype([1, float("nan")], skipna=True),
        ],
    ),
    (
        "timestamp-to-numpy-plain",
        "Timestamp.to_numpy",
        ("dtype", "copy"),
        lambda pd: str(pd.Timestamp("2024-01-01").to_numpy(dtype=None, copy=False)),
    ),
    (
        "timestamp-isoformat-options",
        "Timestamp.isoformat",
        ("sep", "timespec"),
        lambda pd: pd.Timestamp("2024-01-02 03:04:05.123").isoformat(sep=" ", timespec="seconds"),
    ),
    (
        "timestamp-fromtimestamp-zone",
        "Timestamp.fromtimestamp",
        ("ts", "tz"),
        lambda pd: str(pd.Timestamp.fromtimestamp(ts=0, tz="UTC")),
    ),
    (
        "timestamp-fromordinal-zone",
        "Timestamp.fromordinal",
        ("ordinal", "tz"),
        lambda pd: str(pd.Timestamp.fromordinal(ordinal=738000, tz="UTC")),
    ),
    (
        "timestamp-as-unit-round",
        "Timestamp.as_unit",
        ("unit", "round_ok"),
        lambda pd: str(pd.Timestamp("2024-01-01 00:00:00.5").as_unit("s", round_ok=True)),
    ),
    (
        "timedelta-as-unit-round",
        "Timedelta.as_unit",
        ("unit", "round_ok"),
        lambda pd: str(pd.Timedelta("1.5s").as_unit("s", round_ok=True)),
    ),
    (
        "flags-built",
        "pandas.Flags",
        ("obj", "allows_duplicate_labels"),
        lambda pd: (
            pd.Flags(obj=_tail_frame(pd), allows_duplicate_labels=False).allows_duplicate_labels
        ),
    ),
    (
        "sparse-dtype-options",
        "pandas.SparseDtype",
        ("dtype", "fill_value"),
        lambda pd: str(pd.SparseDtype(dtype="float64", fill_value=0.0)),
    ),
    (
        "dt-ceil-zone-policies",
        "dt.ceil",
        ("freq", "ambiguous", "nonexistent"),
        lambda pd: (
            pd.Series(pd.to_datetime(["2024-11-03 00:31"]))
            .dt.tz_localize("US/Eastern")
            .dt.ceil("h", ambiguous=[False], nonexistent="raise")
            .astype(str)
            .tolist()
        ),
    ),
    (
        "dt-floor-zone-policies",
        "dt.floor",
        ("freq", "ambiguous", "nonexistent"),
        lambda pd: (
            pd.Series(pd.to_datetime(["2024-03-10 03:30"]))
            .dt.tz_localize("US/Eastern")
            .dt.floor("2h", nonexistent="shift_forward", ambiguous="raise")
            .astype(str)
            .tolist()
        ),
    ),
    (
        "datetime-index-round-policies",
        "DatetimeIndex.round",
        ("freq", "ambiguous", "nonexistent"),
        lambda pd: (
            pd.DatetimeIndex(["2024-11-03 00:31"], tz="US/Eastern")
            .round("h", ambiguous="NaT", nonexistent="raise")
            .astype(str)
            .tolist()
        ),
    ),
    (
        "eval-level",
        "pandas.eval",
        ("expr", "level"),
        lambda pd: pd.eval("1 + 2", level=0),
    ),
)


def _tail_groups(pd):
    """Two groups of two float columns."""
    return pd.DataFrame(
        {
            "g": ["a", "b", "a", "b", "a", "b"],
            "x": [1.0, 2.0, 4.0, 3.0, 5.0, 7.0],
            "y": [2.0, 1.0, 3.0, 5.0, 4.0, 6.0],
        }
    )


def _tail_multi(pd):
    """A named two-level index of text and whole numbers."""
    return pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1), ("c", 3)], names=["k", "n"])


def _tail_instants(pd):
    """Three unsorted instants with times of day."""
    return pd.DatetimeIndex(["2024-01-03 10:20", "2024-01-01 05:50", "2024-01-02 12:00"], name="t")


def _tail_slice(found):
    """A slice as plain whole numbers, which numpy ones print differently from."""
    return [int(found.start), int(found.stop), int(found.step)]


def _tail_wide(pd):
    """Two float columns for the window cases."""
    return pd.DataFrame({"x": [1.0, 2.0, 4.0, 3.0, 5.0], "y": [2.0, 1.0, 3.0, 5.0, 4.0]})


def _tail_levels(index):
    """A MultiIndex answer as the frame of its levels, which the comparator reads."""
    return index.to_frame(index=False)


TAIL2_CASES = (
    (
        "grouped-corrwith-options",
        "GroupBy.corrwith",
        ("other", "drop", "method", "numeric_only"),
        lambda pd: (
            _tail_groups(pd)
            .groupby("g")
            .corrwith(_tail_groups(pd)[["x"]], drop=True, method="pearson", numeric_only=True)
        ),
    ),
    (
        "expanding-cov-options",
        "Expanding.cov",
        ("other", "pairwise", "ddof", "numeric_only"),
        lambda pd: (
            _tail_wide(pd)
            .expanding(2)
            .cov(_tail_wide(pd)["y"], pairwise=False, ddof=0, numeric_only=True)
        ),
    ),
    (
        "expanding-corr-options",
        "Expanding.corr",
        ("other", "pairwise", "ddof", "numeric_only"),
        lambda pd: (
            _tail_wide(pd)
            .expanding(2)
            .corr(_tail_wide(pd)["y"], pairwise=False, ddof=1, numeric_only=True)
        ),
    ),
    (
        "rolling-cov-options",
        "Rolling.cov",
        ("pairwise", "ddof", "numeric_only"),
        lambda pd: _tail_wide(pd).rolling(3).cov(pairwise=True, ddof=0, numeric_only=True),
    ),
    (
        "rolling-corr-options",
        "Rolling.corr",
        ("ddof", "numeric_only"),
        lambda pd: _tail_wide(pd).rolling(3).corr(_tail_wide(pd)["x"], ddof=1, numeric_only=True),
    ),
    (
        "rolling-sem-options",
        "Rolling.sem",
        ("ddof", "numeric_only"),
        lambda pd: _tail_wide(pd).rolling(3).sem(ddof=0, numeric_only=True),
    ),
    (
        "rolling-quantile-options",
        "Rolling.quantile",
        ("interpolation", "numeric_only"),
        lambda pd: (
            _tail_wide(pd).rolling(3).quantile(0.4, interpolation="lower", numeric_only=True)
        ),
    ),
    (
        "rolling-mean-engine",
        "Rolling.mean",
        ("numeric_only", "engine", "engine_kwargs"),
        lambda pd: (
            _tail_wide(pd).rolling(2).mean(numeric_only=True, engine="cython", engine_kwargs=None)
        ),
    ),
    (
        "rolling-max-engine",
        "Rolling.max",
        ("numeric_only", "engine", "engine_kwargs"),
        lambda pd: (
            _tail_wide(pd).rolling(2).max(numeric_only=True, engine="cython", engine_kwargs=None)
        ),
    ),
    (
        "rolling-min-engine",
        "Rolling.min",
        ("numeric_only", "engine", "engine_kwargs"),
        lambda pd: (
            _tail_wide(pd).rolling(2).min(numeric_only=True, engine="cython", engine_kwargs=None)
        ),
    ),
    (
        "rolling-median-engine",
        "Rolling.median",
        ("numeric_only", "engine", "engine_kwargs"),
        lambda pd: (
            _tail_wide(pd).rolling(3).median(numeric_only=True, engine="cython", engine_kwargs=None)
        ),
    ),
    (
        "rolling-sum-engine",
        "Rolling.sum",
        ("engine", "engine_kwargs"),
        lambda pd: _tail_wide(pd).rolling(2).sum(engine="cython", engine_kwargs=None),
    ),
    (
        "rolling-std-engine",
        "Rolling.std",
        ("engine", "engine_kwargs"),
        lambda pd: _tail_wide(pd).rolling(3).std(engine="cython", engine_kwargs=None),
    ),
    (
        "rolling-var-engine",
        "Rolling.var",
        ("engine", "engine_kwargs"),
        lambda pd: _tail_wide(pd).rolling(3).var(engine="cython", engine_kwargs=None),
    ),
    (
        "rolling-apply-engine",
        "Rolling.apply",
        ("raw", "engine", "engine_kwargs"),
        lambda pd: (
            _tail_wide(pd).rolling(2).apply(sum, raw=True, engine="cython", engine_kwargs=None)
        ),
    ),
    (
        "expanding-mean-engine",
        "Expanding.mean",
        ("numeric_only", "engine", "engine_kwargs"),
        lambda pd: (
            _tail_wide(pd).expanding().mean(numeric_only=True, engine="cython", engine_kwargs=None)
        ),
    ),
    (
        "expanding-sum-engine",
        "Expanding.sum",
        ("numeric_only", "engine", "engine_kwargs"),
        lambda pd: (
            _tail_wide(pd).expanding().sum(numeric_only=True, engine="cython", engine_kwargs=None)
        ),
    ),
    (
        "expanding-max-engine",
        "Expanding.max",
        ("numeric_only", "engine", "engine_kwargs"),
        lambda pd: (
            _tail_wide(pd).expanding().max(numeric_only=True, engine="cython", engine_kwargs=None)
        ),
    ),
    (
        "expanding-min-engine",
        "Expanding.min",
        ("numeric_only", "engine", "engine_kwargs"),
        lambda pd: (
            _tail_wide(pd).expanding().min(numeric_only=True, engine="cython", engine_kwargs=None)
        ),
    ),
    (
        "expanding-median-engine",
        "Expanding.median",
        ("numeric_only", "engine", "engine_kwargs"),
        lambda pd: (
            _tail_wide(pd)
            .expanding()
            .median(numeric_only=True, engine="cython", engine_kwargs=None)
        ),
    ),
    (
        "expanding-var-engine",
        "Expanding.var",
        ("engine", "engine_kwargs"),
        lambda pd: _tail_wide(pd).expanding(2).var(engine="cython", engine_kwargs=None),
    ),
    (
        "expanding-apply-engine",
        "Expanding.apply",
        ("raw", "engine", "engine_kwargs"),
        lambda pd: (
            _tail_wide(pd).expanding().apply(max, raw=False, engine="cython", engine_kwargs=None)
        ),
    ),
    (
        "ewm-mean-engine",
        "ExponentialMovingWindow.mean",
        ("numeric_only", "engine", "engine_kwargs"),
        lambda pd: (
            _tail_wide(pd).ewm(com=1).mean(numeric_only=True, engine="cython", engine_kwargs=None)
        ),
    ),
    (
        "ewm-sum-engine",
        "ExponentialMovingWindow.sum",
        ("numeric_only", "engine", "engine_kwargs"),
        lambda pd: (
            _tail_wide(pd).ewm(com=1).sum(numeric_only=True, engine="cython", engine_kwargs=None)
        ),
    ),
    (
        "ewm-var-bias",
        "ExponentialMovingWindow.var",
        ("bias", "numeric_only"),
        lambda pd: _tail_wide(pd).ewm(com=1).var(bias=True, numeric_only=True),
    ),
    (
        "ewm-std-bias",
        "ExponentialMovingWindow.std",
        ("bias", "numeric_only"),
        lambda pd: _tail_wide(pd).ewm(com=1).std(bias=False, numeric_only=True),
    ),
    (
        "ewm-cov-bias",
        "ExponentialMovingWindow.cov",
        ("bias", "numeric_only"),
        lambda pd: _tail_wide(pd).ewm(com=1).cov(_tail_wide(pd)["x"], bias=True, numeric_only=True),
    ),
    (
        "ewm-corr-pairwise",
        "ExponentialMovingWindow.corr",
        ("pairwise", "numeric_only"),
        lambda pd: _tail_wide(pd).ewm(com=1).corr(pairwise=True, numeric_only=True),
    ),
    (
        "grouped-cov-options",
        "GroupBy.cov",
        ("min_periods", "ddof", "numeric_only"),
        lambda pd: _tail_groups(pd).groupby("g").cov(min_periods=2, ddof=0, numeric_only=True),
    ),
    (
        "grouped-corr-options",
        "GroupBy.corr",
        ("method", "min_periods", "numeric_only"),
        lambda pd: (
            _tail_groups(pd).groupby("g").corr(method="pearson", min_periods=2, numeric_only=True)
        ),
    ),
    (
        "grouped-mean-engine",
        "GroupBy.mean",
        ("engine", "engine_kwargs"),
        lambda pd: _tail_groups(pd).groupby("g").mean(engine="cython", engine_kwargs=None),
    ),
    (
        "grouped-sum-engine",
        "GroupBy.sum",
        ("engine", "engine_kwargs"),
        lambda pd: _tail_groups(pd).groupby("g").sum(engine="cython", engine_kwargs=None),
    ),
    (
        "grouped-min-engine",
        "GroupBy.min",
        ("engine", "engine_kwargs"),
        lambda pd: _tail_groups(pd).groupby("g").min(engine="cython", engine_kwargs=None),
    ),
    (
        "grouped-max-engine",
        "GroupBy.max",
        ("engine", "engine_kwargs"),
        lambda pd: _tail_groups(pd).groupby("g").max(engine="cython", engine_kwargs=None),
    ),
    (
        "grouped-std-engine",
        "GroupBy.std",
        ("engine", "engine_kwargs"),
        lambda pd: _tail_groups(pd).groupby("g").std(engine="cython", engine_kwargs=None),
    ),
    (
        "grouped-var-engine",
        "GroupBy.var",
        ("engine", "engine_kwargs"),
        lambda pd: _tail_groups(pd).groupby("g").var(engine="cython", engine_kwargs=None),
    ),
    (
        "grouped-transform-engine",
        "GroupBy.transform",
        ("engine", "engine_kwargs"),
        lambda pd: (
            _tail_groups(pd).groupby("g").transform("sum", engine="cython", engine_kwargs=None)
        ),
    ),
    (
        "grouped-agg-engine",
        "GroupBy.agg",
        ("engine", "engine_kwargs"),
        lambda pd: _tail_groups(pd).groupby("g").agg("max", engine="cython", engine_kwargs=None),
    ),
    (
        "grouped-aggregate-engine",
        "GroupBy.aggregate",
        ("func", "engine", "engine_kwargs"),
        lambda pd: (
            _tail_groups(pd).groupby("g").aggregate(func="min", engine="cython", engine_kwargs=None)
        ),
    ),
    (
        "grouped-first-min-count",
        "GroupBy.first",
        ("numeric_only", "min_count"),
        lambda pd: _tail_groups(pd).groupby("g").first(numeric_only=True, min_count=4),
    ),
    (
        "grouped-last-min-count",
        "GroupBy.last",
        ("numeric_only", "min_count"),
        lambda pd: _tail_groups(pd).groupby("g").last(numeric_only=True, min_count=2),
    ),
    (
        "grouped-kurt-options",
        "GroupBy.kurt",
        ("skipna", "numeric_only"),
        lambda pd: _tail_groups(pd).groupby("g").kurt(skipna=True, numeric_only=True),
    ),
    (
        "grouped-idxmin-options",
        "GroupBy.idxmin",
        ("skipna", "numeric_only"),
        lambda pd: _tail_groups(pd).groupby("g").idxmin(skipna=True, numeric_only=True),
    ),
    (
        "grouped-idxmax-options",
        "GroupBy.idxmax",
        ("skipna", "numeric_only"),
        lambda pd: _tail_groups(pd).groupby("g").idxmax(skipna=True, numeric_only=True),
    ),
    (
        "grouped-describe-options",
        "GroupBy.describe",
        ("percentiles", "exclude"),
        lambda pd: _tail_groups(pd).groupby("g").describe(percentiles=[0.5], exclude=None),
    ),
    (
        "grouped-value-counts-options",
        "GroupBy.value_counts",
        ("subset", "ascending"),
        lambda pd: (
            _tail_groups(pd)
            .assign(x=[1.0, 2.0, 1.0, 3.0, 5.0, 2.0])
            .groupby("g")
            .value_counts(subset=["x"], ascending=True)
        ),
    ),
    (
        "grouped-pct-change-options",
        "GroupBy.pct_change",
        ("fill_method", "freq"),
        lambda pd: _tail_groups(pd).groupby("g").pct_change(fill_method=None, freq=None),
    ),
    (
        "grouped-expanding-options",
        "GroupBy.expanding",
        ("min_periods", "method"),
        lambda pd: _tail_groups(pd).groupby("g").expanding(min_periods=2, method="single").sum(),
    ),
    (
        "multi-index-from-frame-options",
        "MultiIndex.from_frame",
        ("df", "sortorder", "names"),
        lambda pd: _tail_levels(
            pd.MultiIndex.from_frame(
                pd.DataFrame({"a": ["x", "y"], "b": [1, 2]}), sortorder=None, names=["p", "q"]
            )
        ),
    ),
    (
        "multi-index-drop-options",
        "MultiIndex.drop",
        ("codes", "level", "errors"),
        lambda pd: _tail_levels(_tail_multi(pd).drop(codes=["a", "z"], level="k", errors="ignore")),
    ),
    (
        "multi-index-copy-options",
        "MultiIndex.copy",
        ("names", "deep", "name"),
        lambda pd: _tail_levels(_tail_multi(pd).copy(names=["p", "q"], deep=True, name=None)),
    ),
    (
        "multi-index-union-sort",
        "MultiIndex.union",
        ("other", "sort"),
        lambda pd: _tail_levels(
            _tail_multi(pd).union(pd.MultiIndex.from_tuples([("a", 0)]), sort=False)
        ),
    ),
    (
        "multi-index-intersection-sort",
        "MultiIndex.intersection",
        ("other", "sort"),
        lambda pd: _tail_levels(_tail_multi(pd).intersection(_tail_multi(pd)[1:], sort=True)),
    ),
    (
        "multi-index-difference-sort",
        "MultiIndex.difference",
        ("other", "sort"),
        lambda pd: _tail_levels(_tail_multi(pd).difference(_tail_multi(pd)[:1], sort=False)),
    ),
    (
        "multi-index-swaplevel-options",
        "MultiIndex.swaplevel",
        ("i", "j"),
        lambda pd: _tail_levels(_tail_multi(pd).swaplevel(i="n", j="k")),
    ),
    (
        "multi-index-set-names-level",
        "MultiIndex.set_names",
        ("names", "level"),
        lambda pd: _tail_levels(_tail_multi(pd).set_names(names="m", level=1)),
    ),
    (
        "multi-index-rename-level",
        "MultiIndex.rename",
        ("names", "level"),
        lambda pd: _tail_levels(_tail_multi(pd).rename(names=["m"], level=[0])),
    ),
    (
        "multi-index-repeat-axis",
        "MultiIndex.repeat",
        ("repeats", "axis"),
        lambda pd: _tail_levels(_tail_multi(pd).repeat(repeats=2, axis=None)),
    ),
    (
        "multi-index-isin-level",
        "MultiIndex.isin",
        ("values", "level"),
        lambda pd: _tail_multi(pd).isin(values=[1], level="n").tolist(),
    ),
    (
        "multi-index-insert-options",
        "MultiIndex.insert",
        ("loc", "item"),
        lambda pd: _tail_levels(_tail_multi(pd).insert(loc=1, item=("z", 9))),
    ),
    (
        "multi-index-slice-bound",
        "MultiIndex.get_slice_bound",
        ("label", "side"),
        lambda pd: int(_tail_multi(pd).get_slice_bound(label=("b", 1), side="right")),
    ),
    (
        "multi-index-from-arrays-named",
        "MultiIndex.from_arrays",
        ("arrays", "names"),
        lambda pd: _tail_levels(
            pd.MultiIndex.from_arrays(arrays=[["a", "b"], [1, 2]], names=["p", "q"])
        ),
    ),
    (
        "multi-index-astype-copy",
        "MultiIndex.astype",
        ("dtype", "copy"),
        lambda pd: _tail_levels(_tail_multi(pd).astype(dtype="object", copy=True)),
    ),
    (
        "multi-index-min-options",
        "MultiIndex.min",
        ("axis", "skipna"),
        lambda pd: [str(part) for part in _tail_multi(pd).min(axis=None, skipna=True)],
    ),
    (
        "multi-index-max-options",
        "MultiIndex.max",
        ("axis", "skipna"),
        lambda pd: [str(part) for part in _tail_multi(pd).max(axis=None, skipna=True)],
    ),
    (
        "multi-index-argmin-options",
        "MultiIndex.argmin",
        ("axis", "skipna"),
        lambda pd: int(_tail_multi(pd).argmin(axis=None, skipna=True)),
    ),
    (
        "multi-index-argmax-options",
        "MultiIndex.argmax",
        ("axis", "skipna"),
        lambda pd: int(_tail_multi(pd).argmax(axis=None, skipna=True)),
    ),
    (
        "multi-index-map-options",
        "MultiIndex.map",
        ("mapper", "na_action"),
        lambda pd: _tail_multi(pd).map(mapper=str, na_action=None),
    ),
    (
        "multi-index-take-fill",
        "MultiIndex.take",
        ("allow_fill", "fill_value"),
        lambda pd: _tail_levels(_tail_multi(pd).take([2, 0], allow_fill=True, fill_value=None)),
    ),
    (
        "datetime-index-sort-options",
        "DatetimeIndex.sort_values",
        ("return_indexer", "na_position", "key"),
        lambda pd: _tail_instants(pd).sort_values(
            return_indexer=False, na_position="first", key=None
        ),
    ),
    (
        "datetime-index-symmetric-difference",
        "DatetimeIndex.symmetric_difference",
        ("other", "result_name", "sort"),
        lambda pd: _tail_instants(pd).symmetric_difference(
            _tail_instants(pd)[:1], result_name="u", sort=None
        ),
    ),
    (
        "datetime-index-slice-indexer",
        "DatetimeIndex.slice_indexer",
        ("start", "end", "step"),
        lambda pd: _tail_slice(
            _tail_instants(pd)
            .sort_values()
            .slice_indexer(start="2024-01-02", end="2024-01-03 23:00", step=1)
        ),
    ),
    (
        "datetime-index-floor-options",
        "DatetimeIndex.floor",
        ("freq", "ambiguous", "nonexistent"),
        lambda pd: _tail_instants(pd).floor(freq="h", ambiguous="raise", nonexistent="raise"),
    ),
    (
        "datetime-index-ceil-options",
        "DatetimeIndex.ceil",
        ("freq", "ambiguous", "nonexistent"),
        lambda pd: _tail_instants(pd).ceil(freq="D", ambiguous="raise", nonexistent="raise"),
    ),
    (
        "offsets-year-end-month",
        "offsets.YearEnd",
        ("n", "normalize", "month"),
        lambda pd: (
            pd.Timestamp("2024-03-15 10:00") + pd.offsets.YearEnd(n=1, normalize=True, month=6)
        ),
    ),
)


def _tail_series(pd):
    """Five floats with a gap under text labels."""
    return pd.Series([3.0, 1.0, float("nan"), 4.0, 2.0], index=list("abcde"), name="v")


def _tail_labels(pd):
    """Six whole numbers with a repeat."""
    return pd.Index([5, 2, 9, 2, 7, 1], name="i")


def _tail_stamps(pd):
    """Four daily instants."""
    return pd.date_range("2024-01-01 09:00", periods=4, freq="D", name="t")


def _tail_mixed(pd):
    """Floats, whole numbers and text, for the frame cases."""
    return pd.DataFrame(
        {"a": [1.0, float("nan"), 3.0, 4.0], "b": [4, 3, 2, 1], "s": ["x", "y", "x", "z"]},
        index=list("pqrs"),
    )


def _tail_hourly(pd):
    """Six hourly readings."""
    return pd.DataFrame(
        {"v": [1.0, 2.0, 4.0, 3.0, 5.0, 7.0], "w": [1, 2, 3, 4, 5, 6]},
        index=pd.date_range("2024-01-01", periods=6, freq="h"),
    )


TAIL3_CASES = (
    (
        "series-var-axis",
        "Series.var",
        ("axis", "numeric_only"),
        lambda pd: float(_tail_series(pd).var(axis=0, numeric_only=False)),
    ),
    (
        "series-std-axis",
        "Series.std",
        ("axis", "numeric_only"),
        lambda pd: float(_tail_series(pd).std(axis=0, numeric_only=False)),
    ),
    (
        "series-sem-axis",
        "Series.sem",
        ("axis", "numeric_only"),
        lambda pd: float(_tail_series(pd).sem(axis="index", numeric_only=False)),
    ),
    (
        "series-sum-axis",
        "Series.sum",
        ("axis", "numeric_only"),
        lambda pd: float(_tail_series(pd).sum(axis=0, numeric_only=False)),
    ),
    (
        "series-prod-axis",
        "Series.prod",
        ("axis", "numeric_only"),
        lambda pd: float(_tail_series(pd).prod(axis=0, numeric_only=False)),
    ),
    (
        "series-mean-axis",
        "Series.mean",
        ("axis", "numeric_only"),
        lambda pd: float(_tail_series(pd).mean(axis=None, numeric_only=False)),
    ),
    (
        "series-rank-axis",
        "Series.rank",
        ("axis", "numeric_only"),
        lambda pd: _tail_series(pd).rank(axis=0, numeric_only=False),
    ),
    (
        "series-idxmin-axis",
        "Series.idxmin",
        ("axis", "skipna"),
        lambda pd: _tail_series(pd).idxmin(axis=0, skipna=True),
    ),
    (
        "series-argmin-axis",
        "Series.argmin",
        ("axis", "skipna"),
        lambda pd: int(_tail_series(pd).argmin(axis=None, skipna=True)),
    ),
    (
        "series-argmax-axis",
        "Series.argmax",
        ("axis", "skipna"),
        lambda pd: int(_tail_series(pd).argmax(axis=None, skipna=True)),
    ),
    (
        "series-cumprod-axis",
        "Series.cumprod",
        ("axis", "skipna"),
        lambda pd: _tail_series(pd).cumprod(axis=0, skipna=True),
    ),
    (
        "series-cummin-axis",
        "Series.cummin",
        ("axis", "skipna"),
        lambda pd: _tail_series(pd).cummin(axis=0, skipna=False),
    ),
    (
        "series-cummax-axis",
        "Series.cummax",
        ("axis", "skipna"),
        lambda pd: _tail_series(pd).cummax(axis="index", skipna=True),
    ),
    (
        "series-all-axis",
        "Series.all",
        ("axis", "bool_only"),
        lambda pd: bool((_tail_series(pd) > 0).all(axis=0, bool_only=False)),
    ),
    (
        "series-aggregate-axis",
        "Series.aggregate",
        ("func", "axis"),
        lambda pd: _tail_series(pd).aggregate(func=["min", "max"], axis=0),
    ),
    (
        "series-add-suffix-axis",
        "Series.add_suffix",
        ("suffix", "axis"),
        lambda pd: _tail_series(pd).add_suffix(suffix="_x", axis=0),
    ),
    (
        "series-add-prefix-axis",
        "Series.add_prefix",
        ("prefix", "axis"),
        lambda pd: _tail_series(pd).add_prefix(prefix="x_", axis="index"),
    ),
    (
        "series-take-axis",
        "Series.take",
        ("indices", "axis"),
        lambda pd: _tail_series(pd).take(indices=[3, 0], axis=0),
    ),
    (
        "series-shift-axis",
        "Series.shift",
        ("axis", "suffix"),
        lambda pd: _tail_series(pd).shift(1, axis=0, suffix=None),
    ),
    (
        "series-get-default",
        "Series.get",
        ("key", "default"),
        lambda pd: _tail_series(pd).get(key="zz", default=-1.0),
    ),
    (
        "series-ffill-axis",
        "Series.ffill",
        ("axis", "limit"),
        lambda pd: _tail_series(pd).ffill(axis=0, limit=1),
    ),
    (
        "series-droplevel-axis",
        "Series.droplevel",
        ("level", "axis"),
        lambda pd: pd.Series(
            [1, 2], index=pd.MultiIndex.from_tuples([("a", 1), ("b", 2)])
        ).droplevel(level=0, axis=0),
    ),
    (
        "series-describe-include",
        "Series.describe",
        ("include", "exclude"),
        lambda pd: _tail_series(pd).describe(include=None, exclude=None),
    ),
    (
        "series-cov-options",
        "Series.cov",
        ("min_periods", "ddof"),
        lambda pd: float(
            _tail_series(pd).cov(
                _tail_series(pd)[::-1].reset_index(drop=True).set_axis(list("abcde")),
                min_periods=2,
                ddof=0,
            )
        ),
    ),
    (
        "series-combine-func",
        "Series.combine",
        ("other", "func"),
        lambda pd: _tail_series(pd).combine(other=2.5, func=max),
    ),
    (
        "series-asof-where",
        "Series.asof",
        ("where", "subset"),
        lambda pd: pd.Series([1.0, float("nan"), 3.0], index=[10, 20, 30]).asof(
            where=[15, 25, 35], subset=None
        ),
    ),
    (
        "series-argsort-order",
        "Series.argsort",
        ("order", "stable"),
        lambda pd: _tail_series(pd).fillna(0).argsort(order=None, stable=None).tolist(),
    ),
    (
        "series-sort-index-level",
        "Series.sort_index",
        ("level", "sort_remaining"),
        lambda pd: pd.Series(
            [1, 2, 3], index=pd.MultiIndex.from_tuples([("b", 2), ("a", 9), ("b", 1)])
        ).sort_index(level=1, sort_remaining=False),
    ),
    (
        "series-set-axis-copy",
        "Series.set_axis",
        ("axis", "copy"),
        lambda pd: _tail_series(pd).set_axis(list("vwxyz"), axis=0, copy=False),
    ),
    (
        "series-rename-axis-copy",
        "Series.rename_axis",
        ("axis", "copy"),
        lambda pd: _tail_series(pd).rename_axis("k", axis=0, copy=False),
    ),
    (
        "series-truncate-axis",
        "Series.truncate",
        ("axis", "copy"),
        lambda pd: _tail_series(pd).truncate("b", "d", axis=0, copy=False),
    ),
    (
        "series-to-numpy-dtype",
        "Series.to_numpy",
        ("dtype", "copy"),
        lambda pd: _tail_series(pd).fillna(0).to_numpy(dtype="int64", copy=True).tolist(),
    ),
    (
        "series-to-period-copy",
        "Series.to_period",
        ("freq", "copy"),
        lambda pd: pd.Series(
            [1, 2], index=pd.date_range("2024-01-31", periods=2, freq="ME")
        ).to_period(freq="M", copy=False),
    ),
    (
        "series-tz-convert-level",
        "Series.tz_convert",
        ("level", "copy"),
        lambda pd: pd.Series(
            [1, 2], index=pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
        ).tz_convert("Asia/Tokyo", level=None, copy=False),
    ),
    (
        "series-where-axis",
        "Series.where",
        ("axis", "level"),
        lambda pd: _tail_series(pd).where(_tail_series(pd) > 2, 0.0, axis=0, level=None),
    ),
    (
        "series-mask-axis",
        "Series.mask",
        ("axis", "level"),
        lambda pd: _tail_series(pd).mask(_tail_series(pd) > 2, -1.0, axis=0, level=None),
    ),
    (
        "series-expanding-method",
        "Series.expanding",
        ("min_periods", "method"),
        lambda pd: _tail_series(pd).expanding(min_periods=2, method="single").sum(),
    ),
    (
        "series-ewm-times",
        "Series.ewm",
        ("times", "method"),
        lambda pd: (
            pd.Series([1.0, 2.0, 4.0])
            .ewm(
                halflife="1D",
                times=pd.date_range("2024-01-01", periods=3, freq="D"),
                method="single",
            )
            .mean()
        ),
    ),
    (
        "index-union-sort",
        "Index.union",
        ("other", "sort"),
        lambda pd: _tail_labels(pd).unique().union(pd.Index([3, 2]), sort=False),
    ),
    (
        "index-intersection-sort",
        "Index.intersection",
        ("other", "sort"),
        lambda pd: _tail_labels(pd).intersection(pd.Index([9, 1, 4]), sort=True),
    ),
    (
        "index-difference-sort",
        "Index.difference",
        ("other", "sort"),
        lambda pd: _tail_labels(pd).difference(pd.Index([2]), sort=False),
    ),
    (
        "index-repeat-axis",
        "Index.repeat",
        ("repeats", "axis"),
        lambda pd: _tail_labels(pd).repeat(repeats=2, axis=None),
    ),
    (
        "index-putmask-value",
        "Index.putmask",
        ("mask", "value"),
        lambda pd: _tail_labels(pd).putmask(mask=[True, False, False, True, False, False], value=0),
    ),
    (
        "index-min-axis",
        "Index.min",
        ("axis", "skipna"),
        lambda pd: int(_tail_labels(pd).min(axis=None, skipna=True)),
    ),
    (
        "index-max-axis",
        "Index.max",
        ("axis", "skipna"),
        lambda pd: int(_tail_labels(pd).max(axis=None, skipna=True)),
    ),
    (
        "index-argmin-axis",
        "Index.argmin",
        ("axis", "skipna"),
        lambda pd: int(_tail_labels(pd).argmin(axis=None, skipna=True)),
    ),
    (
        "index-argmax-axis",
        "Index.argmax",
        ("axis", "skipna"),
        lambda pd: int(_tail_labels(pd).argmax(axis=None, skipna=True)),
    ),
    (
        "index-insert-item",
        "Index.insert",
        ("loc", "item"),
        lambda pd: _tail_labels(pd).insert(loc=2, item=11),
    ),
    (
        "index-slice-bound-side",
        "Index.get_slice_bound",
        ("label", "side"),
        lambda pd: int(pd.Index([1, 3, 5, 7]).get_slice_bound(label=5, side="left")),
    ),
    (
        "index-factorize-options",
        "Index.factorize",
        ("sort", "use_na_sentinel"),
        lambda pd: [
            part.tolist() for part in _tail_labels(pd).factorize(sort=True, use_na_sentinel=True)
        ],
    ),
    (
        "index-drop-errors",
        "Index.drop",
        ("labels", "errors"),
        lambda pd: _tail_labels(pd).drop(labels=[9, 99], errors="ignore"),
    ),
    (
        "index-astype-copy",
        "Index.astype",
        ("dtype", "copy"),
        lambda pd: _tail_labels(pd).astype(dtype="float64", copy=True),
    ),
    (
        "index-shift-periods",
        "Index.shift",
        ("periods", "freq"),
        lambda pd: pd.date_range("2024-01-01", periods=3, freq="D").shift(periods=2, freq="h"),
    ),
    (
        "datetime-index-max-axis",
        "DatetimeIndex.max",
        ("axis", "skipna"),
        lambda pd: _tail_stamps(pd).max(axis=None, skipna=True),
    ),
    (
        "datetime-index-min-axis",
        "DatetimeIndex.min",
        ("axis", "skipna"),
        lambda pd: _tail_stamps(pd).min(axis=None, skipna=True),
    ),
    (
        "datetime-index-argmin-axis",
        "DatetimeIndex.argmin",
        ("axis", "skipna"),
        lambda pd: int(_tail_stamps(pd).argmin(axis=None, skipna=True)),
    ),
    (
        "datetime-index-argmax-axis",
        "DatetimeIndex.argmax",
        ("axis", "skipna"),
        lambda pd: int(_tail_stamps(pd).argmax(axis=None, skipna=True)),
    ),
    (
        "datetime-index-map-action",
        "DatetimeIndex.map",
        ("mapper", "na_action"),
        lambda pd: _tail_stamps(pd).map(mapper=lambda t: t.day, na_action=None),
    ),
    (
        "datetime-index-isin-level",
        "DatetimeIndex.isin",
        ("values", "level"),
        lambda pd: [
            bool(x) for x in _tail_stamps(pd).isin(values=_tail_stamps(pd)[:2], level=None)
        ],
    ),
    (
        "datetime-index-intersection-sort",
        "DatetimeIndex.intersection",
        ("other", "sort"),
        lambda pd: _tail_stamps(pd).intersection(_tail_stamps(pd)[1:], sort=False),
    ),
    (
        "datetime-index-union-sort",
        "DatetimeIndex.union",
        ("other", "sort"),
        lambda pd: _tail_stamps(pd)[2:].union(_tail_stamps(pd)[:1], sort=None),
    ),
    (
        "datetime-index-difference-sort",
        "DatetimeIndex.difference",
        ("other", "sort"),
        lambda pd: _tail_stamps(pd).difference(_tail_stamps(pd)[1:2], sort=False),
    ),
    (
        "datetime-index-insert-item",
        "DatetimeIndex.insert",
        ("loc", "item"),
        lambda pd: _tail_stamps(pd).insert(loc=1, item=pd.Timestamp("2023-12-31")),
    ),
    (
        "datetime-index-at-time-asof",
        "DatetimeIndex.indexer_at_time",
        ("time", "asof"),
        lambda pd: [int(x) for x in _tail_stamps(pd).indexer_at_time(time="09:00", asof=False)],
    ),
    (
        "datetime-index-slice-bound",
        "DatetimeIndex.get_slice_bound",
        ("label", "side"),
        lambda pd: int(_tail_stamps(pd).get_slice_bound(label="2024-01-02", side="right")),
    ),
    (
        "datetime-index-factorize-options",
        "DatetimeIndex.factorize",
        ("sort", "use_na_sentinel"),
        lambda pd: _tail_stamps(pd).factorize(sort=True, use_na_sentinel=True)[0].tolist(),
    ),
    (
        "datetime-index-astype-copy",
        "DatetimeIndex.astype",
        ("dtype", "copy"),
        lambda pd: _tail_stamps(pd).astype(dtype="datetime64[s]", copy=True),
    ),
    (
        "datetime-index-as-unit-round",
        "DatetimeIndex.as_unit",
        ("unit", "round_ok"),
        lambda pd: _tail_stamps(pd).as_unit(unit="s", round_ok=True),
    ),
    (
        "datetime-index-shift-freq",
        "DatetimeIndex.shift",
        ("periods", "freq"),
        lambda pd: _tail_stamps(pd).shift(periods=1, freq="2h"),
    ),
    (
        "datetime-index-set-names-level",
        "DatetimeIndex.set_names",
        ("names", "level"),
        lambda pd: _tail_stamps(pd).set_names(names="u", level=None),
    ),
    (
        "datetime-index-repeat-axis",
        "DatetimeIndex.repeat",
        ("repeats", "axis"),
        lambda pd: _tail_stamps(pd).repeat(repeats=2, axis=None),
    ),
    (
        "datetime-index-putmask-value",
        "DatetimeIndex.putmask",
        ("mask", "value"),
        lambda pd: _tail_stamps(pd).putmask(
            mask=[False, True, False, False], value=pd.Timestamp("2020-01-01 09:00")
        ),
    ),
    (
        "datetime-index-where-other",
        "DatetimeIndex.where",
        ("cond", "other"),
        lambda pd: _tail_stamps(pd).where(
            cond=[True, False, True, True], other=pd.Timestamp("2020-01-01")
        ),
    ),
    (
        "datetime-index-to-numpy-options",
        "DatetimeIndex.to_numpy",
        ("dtype", "copy", "na_value"),
        lambda pd: [
            str(x)
            for x in _tail_stamps(pd).to_numpy(dtype="datetime64[D]", copy=True, na_value=None)
        ],
    ),
    (
        "frame-le-axis",
        "DataFrame.le",
        ("other", "axis", "level"),
        lambda pd: _tail_mixed(pd)[["a", "b"]].le(other=[2, 3], axis=1, level=None),
    ),
    (
        "frame-gt-axis",
        "DataFrame.gt",
        ("other", "axis", "level"),
        lambda pd: _tail_mixed(pd)[["a", "b"]].gt(other=_tail_mixed(pd)["b"], axis=0, level=None),
    ),
    (
        "frame-eq-axis",
        "DataFrame.eq",
        ("other", "axis", "level"),
        lambda pd: _tail_mixed(pd)[["a", "b"]].eq(other=3, axis="columns", level=None),
    ),
    (
        "frame-where-axis",
        "DataFrame.where",
        ("axis", "level"),
        lambda pd: _tail_mixed(pd)[["a", "b"]].where(
            _tail_mixed(pd)[["a", "b"]] > 2, 0, axis=None, level=None
        ),
    ),
    (
        "frame-mask-axis",
        "DataFrame.mask",
        ("axis", "level"),
        lambda pd: _tail_mixed(pd)[["a", "b"]].mask(
            _tail_mixed(pd)[["a", "b"]] > 2, -1, axis=None, level=None
        ),
    ),
    (
        "frame-tz-localize-axis",
        "DataFrame.tz_localize",
        ("axis", "copy"),
        lambda pd: _tail_hourly(pd).tz_localize("UTC", axis=0, copy=False),
    ),
    (
        "frame-tz-convert-axis",
        "DataFrame.tz_convert",
        ("axis", "copy"),
        lambda pd: (
            _tail_hourly(pd).tz_localize("UTC").tz_convert("Europe/Paris", axis=0, copy=False)
        ),
    ),
    (
        "frame-truncate-axis",
        "DataFrame.truncate",
        ("axis", "copy"),
        lambda pd: _tail_mixed(pd).truncate("a", "b", axis=1, copy=False),
    ),
    (
        "frame-stack-dropna",
        "DataFrame.stack",
        ("dropna", "sort"),
        lambda pd: _tail_mixed(pd)[["a", "b"]].stack(future_stack=False, dropna=True, sort=True),
    ),
    (
        "frame-sort-values-key",
        "DataFrame.sort_values",
        ("axis", "key"),
        lambda pd: _tail_mixed(pd).sort_values("s", axis=0, key=lambda c: c.str.upper()),
    ),
    (
        "frame-skew-options",
        "DataFrame.skew",
        ("skipna", "numeric_only"),
        lambda pd: _tail_mixed(pd).skew(skipna=True, numeric_only=True),
    ),
    (
        "frame-kurt-options",
        "DataFrame.kurt",
        ("skipna", "numeric_only"),
        lambda pd: _tail_mixed(pd).kurt(skipna=True, numeric_only=True),
    ),
    (
        "frame-min-options",
        "DataFrame.min",
        ("skipna", "numeric_only"),
        lambda pd: _tail_mixed(pd).min(skipna=False, numeric_only=True),
    ),
    (
        "frame-median-options",
        "DataFrame.median",
        ("skipna", "numeric_only"),
        lambda pd: _tail_mixed(pd).median(skipna=True, numeric_only=True),
    ),
    (
        "frame-idxmin-options",
        "DataFrame.idxmin",
        ("skipna", "numeric_only"),
        lambda pd: _tail_mixed(pd).idxmin(skipna=True, numeric_only=True),
    ),
    (
        "frame-count-axis",
        "DataFrame.count",
        ("axis", "numeric_only"),
        lambda pd: _tail_mixed(pd).count(axis=1, numeric_only=True),
    ),
    (
        "frame-cov-options",
        "DataFrame.cov",
        ("min_periods", "ddof"),
        lambda pd: _tail_mixed(pd).cov(min_periods=2, ddof=0, numeric_only=True),
    ),
    (
        "frame-fillna-axis",
        "DataFrame.fillna",
        ("axis", "limit"),
        lambda pd: _tail_mixed(pd)[["a", "b"]].fillna(0.5, axis=0, limit=1),
    ),
    (
        "frame-ffill-area",
        "DataFrame.ffill",
        ("limit", "limit_area"),
        lambda pd: _tail_mixed(pd)[["a", "b"]].ffill(limit=1, limit_area="inside"),
    ),
    (
        "frame-pct-change-options",
        "DataFrame.pct_change",
        ("fill_method", "freq"),
        lambda pd: _tail_mixed(pd)[["a", "b"]].pct_change(fill_method=None, freq=None),
    ),
    (
        "frame-reorder-levels-axis",
        "DataFrame.reorder_levels",
        ("order", "axis"),
        lambda pd: pd.DataFrame(
            {"v": [1, 2]}, index=pd.MultiIndex.from_tuples([("a", 1), ("b", 2)], names=["k", "n"])
        ).reorder_levels(order=["n", "k"], axis=0),
    ),
    ("frame-isetitem-loc", "DataFrame.isetitem", ("loc", "value"), lambda pd: _tail_isetitem(pd)),
    (
        "frame-asof-subset",
        "DataFrame.asof",
        ("where", "subset"),
        lambda pd: pd.DataFrame(
            {"a": [1.0, float("nan"), 3.0], "b": [1.0, 2.0, 3.0]}, index=[10, 20, 30]
        ).asof(where=[25, 35], subset=["a"]),
    ),
    (
        "frame-aggregate-axis",
        "DataFrame.aggregate",
        ("func", "axis"),
        lambda pd: _tail_mixed(pd)[["a", "b"]].aggregate(func=["sum", "max"], axis=0),
    ),
    (
        "frame-add-suffix-axis",
        "DataFrame.add_suffix",
        ("suffix", "axis"),
        lambda pd: _tail_mixed(pd).add_suffix(suffix="_r", axis=0),
    ),
    (
        "frame-add-prefix-axis",
        "DataFrame.add_prefix",
        ("prefix", "axis"),
        lambda pd: _tail_mixed(pd).add_prefix(prefix="c_", axis=1),
    ),
    (
        "frame-convert-dtypes-options",
        "DataFrame.convert_dtypes",
        ("convert_boolean", "dtype_backend"),
        lambda pd: _tail_mixed(pd).convert_dtypes(
            convert_boolean=False, dtype_backend="numpy_nullable"
        ),
    ),
    (
        "resampler-var-options",
        "Resampler.var",
        ("ddof", "numeric_only"),
        lambda pd: _tail_hourly(pd).resample("3h").var(ddof=0, numeric_only=True),
    ),
    (
        "resampler-std-options",
        "Resampler.std",
        ("ddof", "numeric_only"),
        lambda pd: _tail_hourly(pd).resample("3h").std(ddof=1, numeric_only=True),
    ),
    (
        "resampler-sem-options",
        "Resampler.sem",
        ("ddof", "numeric_only"),
        lambda pd: _tail_hourly(pd).resample("3h").sem(ddof=1, numeric_only=True),
    ),
    (
        "resampler-sum-min-count",
        "Resampler.sum",
        ("numeric_only", "min_count"),
        lambda pd: _tail_hourly(pd).resample("4h").sum(numeric_only=True, min_count=3),
    ),
    (
        "resampler-prod-min-count",
        "Resampler.prod",
        ("numeric_only", "min_count"),
        lambda pd: _tail_hourly(pd).resample("4h").prod(numeric_only=True, min_count=3),
    ),
    (
        "resampler-min-min-count",
        "Resampler.min",
        ("numeric_only", "min_count"),
        lambda pd: _tail_hourly(pd).resample("4h").min(numeric_only=True, min_count=3),
    ),
    (
        "resampler-max-min-count",
        "Resampler.max",
        ("numeric_only", "min_count"),
        lambda pd: _tail_hourly(pd).resample("4h").max(numeric_only=True, min_count=1),
    ),
    (
        "expanding-std-engine",
        "Expanding.std",
        ("engine", "engine_kwargs"),
        lambda pd: _tail_hourly(pd).expanding(2).std(engine="cython", engine_kwargs=None),
    ),
    (
        "expanding-sem-options",
        "Expanding.sem",
        ("ddof", "numeric_only"),
        lambda pd: _tail_hourly(pd).expanding(2).sem(ddof=0, numeric_only=True),
    ),
    (
        "expanding-quantile-options",
        "Expanding.quantile",
        ("interpolation", "numeric_only"),
        lambda pd: (
            _tail_hourly(pd).expanding().quantile(0.25, interpolation="higher", numeric_only=True)
        ),
    ),
    (
        "grouped-shift-freq",
        "GroupBy.shift",
        ("freq", "suffix"),
        lambda pd: (
            _tail_hourly(pd)
            .assign(g=[1, 2, 1, 2, 1, 2])
            .groupby("g")
            .shift(1, freq="h", suffix=None)
        ),
    ),
    (
        "dt-round-policies",
        "dt.round",
        ("ambiguous", "nonexistent"),
        lambda pd: pd.Series(_tail_stamps(pd)).dt.round(
            "h", ambiguous="raise", nonexistent="raise"
        ),
    ),
    (
        "interval-dtype-subtype",
        "api.types.IntervalDtype",
        ("subtype", "closed"),
        lambda pd: str(pd.api.types.IntervalDtype(subtype="int64", closed="left")),
    ),
    (
        "zoned-dtype-unit",
        "api.types.DatetimeTZDtype",
        ("unit", "tz"),
        lambda pd: str(pd.api.types.DatetimeTZDtype(unit="ms", tz="UTC")),
    ),
    (
        "category-dtype-ordered",
        "api.types.CategoricalDtype",
        ("categories", "ordered"),
        lambda pd: pd.api.types.CategoricalDtype(
            categories=["lo", "hi"], ordered=True
        ).categories.tolist(),
    ),
    (
        "timestamp-strptime-format",
        "Timestamp.strptime",
        ("date_string", "format"),
        lambda pd: _tail_strptime(pd),
    ),
    (
        "timestamp-combine-parts",
        "Timestamp.combine",
        ("date", "time"),
        lambda pd: _tail_combine(pd),
    ),
    (
        "interval-index-dtype",
        "pandas.IntervalIndex",
        ("dtype", "copy"),
        lambda pd: pd.IntervalIndex(
            pd.IntervalIndex.from_breaks([0, 1, 2]), dtype="interval[float64, right]", copy=True
        ),
    ),
)


def _tail_isetitem(pd):
    """A frame with a column put in by position."""
    frame = _tail_mixed(pd).copy()
    frame.isetitem(loc=1, value=[9, 8, 7, 6])
    return frame


def _tail_strptime(pd):
    """strptime, which pandas refuses, as the error it raises."""
    try:
        return str(pd.Timestamp.strptime(date_string="2024-01-02", format="%Y-%m-%d"))
    except NotImplementedError as error:
        return str(error)


def _tail_combine(pd):
    """One instant from a date and a time of day."""
    import datetime

    return pd.Timestamp.combine(date=datetime.date(2024, 1, 2), time=datetime.time(3, 4))


def _tail_floats(pd):
    """Five floats with a gap and a repeat."""
    return pd.Series([3.0, 1.0, float("nan"), 3.0, 2.0], index=list("abcde"), name="v")


def _tail_pairs_index(pd):
    """A two-level index with a repeat."""
    return pd.MultiIndex.from_tuples([("a", 1), ("b", 2), ("a", 1), ("c", 3)], names=["k", "n"])


def _tail_table(pd):
    """Floats, whole numbers and text."""
    return pd.DataFrame(
        {"a": [1.0, float("nan"), 3.0, 4.0], "b": [4, 3, 2, 1], "s": ["x", "y", "x", "z"]}
    )


def _tail_readings(pd):
    """Six hourly readings."""
    return pd.DataFrame(
        {"v": [1.0, 2.0, 4.0, 3.0, 5.0, 7.0]},
        index=pd.date_range("2024-01-01", periods=6, freq="h"),
    )


TAIL4_CASES = (
    (
        "is-bool-dtype-named",
        "api.types.is_bool_dtype",
        ("arr_or_dtype",),
        lambda pd: pd.api.types.is_bool_dtype(arr_or_dtype="bool"),
    ),
    ("is-bool-named", "api.types.is_bool", ("obj",), lambda pd: pd.api.types.is_bool(obj=True)),
    (
        "is-array-like-named",
        "api.types.is_array_like",
        ("obj",),
        lambda pd: pd.api.types.is_array_like(obj=[1, 2]),
    ),
    (
        "is-any-real-numeric-named",
        "api.types.is_any_real_numeric_dtype",
        ("arr_or_dtype",),
        lambda pd: pd.api.types.is_any_real_numeric_dtype(arr_or_dtype="float64"),
    ),
    (
        "is-unsigned-named",
        "api.types.is_unsigned_integer_dtype",
        ("arr_or_dtype",),
        lambda pd: pd.api.types.is_unsigned_integer_dtype(arr_or_dtype="uint8"),
    ),
    (
        "is-timedelta-ns-named",
        "api.types.is_timedelta64_ns_dtype",
        ("arr_or_dtype",),
        lambda pd: pd.api.types.is_timedelta64_ns_dtype(arr_or_dtype="timedelta64[ns]"),
    ),
    (
        "is-timedelta-named",
        "api.types.is_timedelta64_dtype",
        ("arr_or_dtype",),
        lambda pd: pd.api.types.is_timedelta64_dtype(arr_or_dtype="timedelta64[s]"),
    ),
    (
        "is-string-dtype-named",
        "api.types.is_string_dtype",
        ("arr_or_dtype",),
        lambda pd: pd.api.types.is_string_dtype(arr_or_dtype="str"),
    ),
    (
        "is-signed-named",
        "api.types.is_signed_integer_dtype",
        ("arr_or_dtype",),
        lambda pd: pd.api.types.is_signed_integer_dtype(arr_or_dtype="int16"),
    ),
    (
        "is-scalar-named",
        "api.types.is_scalar",
        ("val",),
        lambda pd: pd.api.types.is_scalar(val=1.5),
    ),
    (
        "is-re-compilable-named",
        "api.types.is_re_compilable",
        ("obj",),
        lambda pd: pd.api.types.is_re_compilable(obj="a+"),
    ),
    ("is-re-named", "api.types.is_re", ("obj",), lambda pd: pd.api.types.is_re(obj="a+")),
    (
        "is-object-dtype-named",
        "api.types.is_object_dtype",
        ("arr_or_dtype",),
        lambda pd: pd.api.types.is_object_dtype(arr_or_dtype="object"),
    ),
    (
        "is-numeric-dtype-named",
        "api.types.is_numeric_dtype",
        ("arr_or_dtype",),
        lambda pd: pd.api.types.is_numeric_dtype(arr_or_dtype="int8"),
    ),
    ("is-number-named", "api.types.is_number", ("obj",), lambda pd: pd.api.types.is_number(obj=3)),
    (
        "is-named-tuple-named",
        "api.types.is_named_tuple",
        ("obj",),
        lambda pd: pd.api.types.is_named_tuple(obj=(1, 2)),
    ),
    (
        "is-iterator-named",
        "api.types.is_iterator",
        ("obj",),
        lambda pd: pd.api.types.is_iterator(obj=iter([1])),
    ),
    (
        "is-integer-dtype-named",
        "api.types.is_integer_dtype",
        ("arr_or_dtype",),
        lambda pd: pd.api.types.is_integer_dtype(arr_or_dtype="int32"),
    ),
    (
        "is-integer-named",
        "api.types.is_integer",
        ("obj",),
        lambda pd: pd.api.types.is_integer(obj=7),
    ),
    (
        "is-hashable-slice",
        "api.types.is_hashable",
        ("allow_slice",),
        lambda pd: pd.api.types.is_hashable(slice(1, 2), allow_slice=False),
    ),
    (
        "is-float-dtype-named",
        "api.types.is_float_dtype",
        ("arr_or_dtype",),
        lambda pd: pd.api.types.is_float_dtype(arr_or_dtype="float32"),
    ),
    ("is-float-named", "api.types.is_float", ("obj",), lambda pd: pd.api.types.is_float(obj=1.0)),
    (
        "is-file-like-named",
        "api.types.is_file_like",
        ("obj",),
        lambda pd: pd.api.types.is_file_like(obj="path"),
    ),
    (
        "is-extension-dtype-named",
        "api.types.is_extension_array_dtype",
        ("arr_or_dtype",),
        lambda pd: pd.api.types.is_extension_array_dtype(arr_or_dtype="Int64"),
    ),
    (
        "is-dict-like-named",
        "api.types.is_dict_like",
        ("obj",),
        lambda pd: pd.api.types.is_dict_like(obj={"a": 1}),
    ),
    (
        "is-zoned-dtype-named",
        "api.types.is_datetime64tz_dtype",
        ("arr_or_dtype",),
        lambda pd: pd.api.types.is_datetime64tz_dtype(arr_or_dtype="datetime64[ns, UTC]"),
    ),
    (
        "is-instant-ns-named",
        "api.types.is_datetime64_ns_dtype",
        ("arr_or_dtype",),
        lambda pd: pd.api.types.is_datetime64_ns_dtype(arr_or_dtype="datetime64[ns]"),
    ),
    (
        "is-instant-dtype-named",
        "api.types.is_datetime64_dtype",
        ("arr_or_dtype",),
        lambda pd: pd.api.types.is_datetime64_dtype(arr_or_dtype="datetime64[s]"),
    ),
    (
        "is-instant-any-named",
        "api.types.is_datetime64_any_dtype",
        ("arr_or_dtype",),
        lambda pd: pd.api.types.is_datetime64_any_dtype(arr_or_dtype="datetime64[ms]"),
    ),
    (
        "is-complex-dtype-named",
        "api.types.is_complex_dtype",
        ("arr_or_dtype",),
        lambda pd: pd.api.types.is_complex_dtype(arr_or_dtype="float64"),
    ),
    (
        "is-complex-named",
        "api.types.is_complex",
        ("obj",),
        lambda pd: pd.api.types.is_complex(obj=1j),
    ),
    (
        "is-category-dtype-named",
        "api.types.is_categorical_dtype",
        ("arr_or_dtype",),
        lambda pd: pd.api.types.is_categorical_dtype(arr_or_dtype="category"),
    ),
    (
        "is-interval-dtype-named",
        "api.types.is_interval_dtype",
        ("arr_or_dtype",),
        lambda pd: pd.api.types.is_interval_dtype(arr_or_dtype="interval[int64, right]"),
    ),
    (
        "is-int64-dtype-named",
        "api.types.is_int64_dtype",
        ("arr_or_dtype",),
        lambda pd: pd.api.types.is_int64_dtype(arr_or_dtype="int64"),
    ),
    (
        "is-period-dtype-named",
        "api.types.is_period_dtype",
        ("arr_or_dtype",),
        lambda pd: pd.api.types.is_period_dtype(arr_or_dtype="period[D]"),
    ),
    (
        "pandas-dtype-named",
        "api.types.pandas_dtype",
        ("dtype",),
        lambda pd: str(pd.api.types.pandas_dtype(dtype="int32")),
    ),
    (
        "period-dtype-freq",
        "api.types.PeriodDtype",
        ("freq",),
        lambda pd: str(pd.api.types.PeriodDtype(freq="M")),
    ),
    (
        "top-period-dtype-freq",
        "pandas.PeriodDtype",
        ("freq",),
        lambda pd: str(pd.PeriodDtype(freq="D")),
    ),
    (
        "top-unique-values",
        "pandas.unique",
        ("values",),
        lambda pd: pd.unique(values=pd.Series([3, 1, 3])).tolist(),
    ),
    ("top-notnull-obj", "pandas.notnull", ("obj",), lambda pd: pd.notnull(obj=_tail_floats(pd))),
    ("top-notna-obj", "pandas.notna", ("obj",), lambda pd: pd.notna(obj=_tail_floats(pd))),
    ("top-isnull-obj", "pandas.isnull", ("obj",), lambda pd: pd.isnull(obj=_tail_floats(pd))),
    ("top-isna-obj", "pandas.isna", ("obj",), lambda pd: pd.isna(obj=_tail_floats(pd))),
    (
        "top-get-option-pat",
        "pandas.get_option",
        ("pat",),
        lambda pd: pd.get_option(pat="display.max_rows"),
    ),
    (
        "top-period-range-start",
        "pandas.period_range",
        ("start",),
        lambda pd: pd.period_range(start="2024-01", periods=3, freq="M"),
    ),
    (
        "top-to-datetime-cache",
        "pandas.to_datetime",
        ("cache",),
        lambda pd: pd.to_datetime(pd.Series(["2024-01-02", "2024-03-04"]), cache=False),
    ),
    (
        "top-merge-copy",
        "pandas.merge",
        ("copy",),
        lambda pd: pd.merge(
            _tail_table(pd), _tail_table(pd)[["s"]].drop_duplicates(), on="s", copy=False
        ),
    ),
    (
        "top-melt-col-level",
        "pandas.melt",
        ("col_level",),
        lambda pd: pd.melt(_tail_table(pd)[["b"]], col_level=None),
    ),
    (
        "top-get-dummies-sparse",
        "pandas.get_dummies",
        ("sparse",),
        lambda pd: pd.get_dummies(_tail_table(pd)["s"], sparse=False),
    ),
    (
        "top-col-name",
        "pandas.col",
        ("col_name",),
        lambda pd: _tail_table(pd).assign(c=pd.col(col_name="b") * 2),
    ),
    (
        "top-frame-copy",
        "pandas.DataFrame",
        ("copy",),
        lambda pd: pd.DataFrame({"a": [1, 2]}, copy=True),
    ),
    (
        "top-datetime-index-copy",
        "pandas.DatetimeIndex",
        ("copy",),
        lambda pd: pd.DatetimeIndex(["2024-01-01"], copy=True),
    ),
    (
        "top-period-index-copy",
        "pandas.PeriodIndex",
        ("copy",),
        lambda pd: pd.PeriodIndex(["2024-01"], freq="M", copy=True),
    ),
    (
        "top-category-copy",
        "pandas.Categorical",
        ("copy",),
        lambda pd: pd.Categorical(["a", "b", "a"], copy=True).tolist(),
    ),
    (
        "timestamp-tz-convert-named",
        "Timestamp.tz_convert",
        ("tz",),
        lambda pd: pd.Timestamp("2024-01-01", tz="UTC").tz_convert(tz="Asia/Tokyo"),
    ),
    (
        "timestamp-to-period-named",
        "Timestamp.to_period",
        ("freq",),
        lambda pd: str(pd.Timestamp("2024-05-06").to_period(freq="M")),
    ),
    (
        "timestamp-strftime-named",
        "Timestamp.strftime",
        ("format",),
        lambda pd: pd.Timestamp("2024-05-06").strftime(format="%d/%m/%Y"),
    ),
    (
        "timestamp-month-name-locale",
        "Timestamp.month_name",
        ("locale",),
        lambda pd: pd.Timestamp("2024-05-06").month_name(locale=None),
    ),
    (
        "timestamp-day-name-locale",
        "Timestamp.day_name",
        ("locale",),
        lambda pd: pd.Timestamp("2024-05-06").day_name(locale=None),
    ),
    (
        "timestamp-fromisoformat-named",
        "Timestamp.fromisoformat",
        ("object",),
        lambda pd: pd.Timestamp.fromisoformat("2024-05-06T07:08"),
    ),
    (
        "timestamp-astimezone-named",
        "Timestamp.astimezone",
        ("tz",),
        lambda pd: pd.Timestamp("2024-01-01", tz="UTC").astimezone(tz="Europe/Paris"),
    ),
    (
        "timestamp-to-pydatetime-warn",
        "Timestamp.to_pydatetime",
        ("warn",),
        lambda pd: str(pd.Timestamp("2024-01-01 01:02").to_pydatetime(warn=False)),
    ),
    (
        "timedelta-round-named",
        "Timedelta.round",
        ("freq",),
        lambda pd: pd.Timedelta("1h31min").round(freq="h"),
    ),
    (
        "timedelta-floor-named",
        "Timedelta.floor",
        ("freq",),
        lambda pd: pd.Timedelta("1h31min").floor(freq="h"),
    ),
    (
        "timedelta-ceil-named",
        "Timedelta.ceil",
        ("freq",),
        lambda pd: pd.Timedelta("1h31min").ceil(freq="h"),
    ),
    ("series-update-other", "Series.update", ("other",), lambda pd: _tail_update(pd)),
    (
        "series-unstack-sort",
        "Series.unstack",
        ("sort",),
        lambda pd: pd.Series(
            [1, 2, 3, 4],
            index=pd.MultiIndex.from_tuples([("b", "y"), ("b", "x"), ("a", "y"), ("a", "x")]),
        ).unstack(sort=True),
    ),
    (
        "series-tz-localize-copy",
        "Series.tz_localize",
        ("copy",),
        lambda pd: pd.Series([1], index=pd.DatetimeIndex(["2024-01-01"])).tz_localize(
            "UTC", copy=False
        ),
    ),
    (
        "series-swaplevel-copy",
        "Series.swaplevel",
        ("copy",),
        lambda pd: pd.Series([1, 2], index=_tail_pairs_index(pd)[:2]).swaplevel(copy=False),
    ),
    (
        "series-squeeze-axis",
        "Series.squeeze",
        ("axis",),
        lambda pd: float(pd.Series([2.5]).squeeze(axis=None)),
    ),
    (
        "series-sort-values-axis",
        "Series.sort_values",
        ("axis",),
        lambda pd: _tail_floats(pd).sort_values(axis=0),
    ),
    (
        "series-resample-on",
        "Series.resample",
        ("on",),
        lambda pd: _tail_readings(pd)["v"].resample("2h", on=None).sum(),
    ),
    (
        "series-repeat-axis",
        "Series.repeat",
        ("axis",),
        lambda pd: _tail_floats(pd).repeat(2, axis=None),
    ),
    (
        "series-reorder-levels-order",
        "Series.reorder_levels",
        ("order",),
        lambda pd: pd.Series([1, 2], index=_tail_pairs_index(pd)[:2]).reorder_levels(order=[1, 0]),
    ),
    (
        "series-rename-copy",
        "Series.rename",
        ("copy",),
        lambda pd: _tail_floats(pd).rename("w", copy=False),
    ),
    (
        "series-reindex-like-copy",
        "Series.reindex_like",
        ("copy",),
        lambda pd: _tail_floats(pd).reindex_like(_tail_floats(pd)[::-1], copy=False),
    ),
    (
        "series-reindex-copy",
        "Series.reindex",
        ("copy",),
        lambda pd: _tail_floats(pd).reindex(list("eca"), copy=False),
    ),
    ("series-pop-item", "Series.pop", ("item",), lambda pd: float(_tail_floats(pd).pop(item="a"))),
    (
        "series-mode-dropna",
        "Series.mode",
        ("dropna",),
        lambda pd: _tail_floats(pd).mode(dropna=True),
    ),
    (
        "series-interpolate-axis",
        "Series.interpolate",
        ("axis",),
        lambda pd: _tail_floats(pd).interpolate(axis=0),
    ),
    (
        "series-infer-objects-copy",
        "Series.infer_objects",
        ("copy",),
        lambda pd: _tail_floats(pd).infer_objects(copy=False),
    ),
    ("series-idxmax-axis", "Series.idxmax", ("axis",), lambda pd: _tail_floats(pd).idxmax(axis=0)),
    (
        "series-factorize-sentinel",
        "Series.factorize",
        ("use_na_sentinel",),
        lambda pd: _tail_floats(pd).factorize(use_na_sentinel=True)[0].tolist(),
    ),
    (
        "series-explode-ignore-index",
        "Series.explode",
        ("ignore_index",),
        lambda pd: pd.Series([[1, 2], [3]]).explode(ignore_index=True).tolist(),
    ),
    (
        "series-equals-other",
        "Series.equals",
        ("other",),
        lambda pd: _tail_floats(pd).equals(other=_tail_floats(pd)),
    ),
    (
        "series-drop-columns",
        "Series.drop",
        ("columns",),
        lambda pd: _tail_floats(pd).drop(index="a", columns=None),
    ),
    (
        "series-dot-other",
        "Series.dot",
        ("other",),
        lambda pd: float(pd.Series([1.0, 2.0]).dot(other=pd.Series([3.0, 4.0]))),
    ),
    ("series-cumsum-axis", "Series.cumsum", ("axis",), lambda pd: _tail_floats(pd).cumsum(axis=0)),
    (
        "series-corr-min-periods",
        "Series.corr",
        ("min_periods",),
        lambda pd: float(
            _tail_floats(pd).corr(_tail_floats(pd)[::-1].set_axis(list("abcde")), min_periods=2)
        ),
    ),
    (
        "series-compare-result-names",
        "Series.compare",
        ("result_names",),
        lambda pd: pd.Series([1, 2, 3]).compare(
            pd.Series([1, 5, 3]), result_names=("mine", "theirs")
        ),
    ),
    (
        "series-combine-first-other",
        "Series.combine_first",
        ("other",),
        lambda pd: _tail_floats(pd).combine_first(other=pd.Series([9.0], index=["c"])),
    ),
    (
        "series-clip-axis",
        "Series.clip",
        ("axis",),
        lambda pd: _tail_floats(pd).clip(1.5, 2.5, axis=0),
    ),
    (
        "series-case-when-list",
        "Series.case_when",
        ("caselist",),
        lambda pd: _tail_floats(pd).case_when(caselist=[(_tail_floats(pd) > 2, 0.0)]),
    ),
    (
        "series-between-time-axis",
        "Series.between_time",
        ("axis",),
        lambda pd: _tail_readings(pd)["v"].between_time("01:00", "03:00", axis=0),
    ),
    (
        "series-astype-copy",
        "Series.astype",
        ("copy",),
        lambda pd: _tail_floats(pd).astype("float32", copy=True),
    ),
    (
        "series-any-axis",
        "Series.any",
        ("axis",),
        lambda pd: bool((_tail_floats(pd) > 2).any(axis=0)),
    ),
    (
        "series-memory-deep",
        "Series.memory_usage",
        ("deep",),
        lambda pd: _tail_floats(pd).memory_usage(deep=False) > 0,
    ),
    (
        "rolling-skew-numeric",
        "Rolling.skew",
        ("numeric_only",),
        lambda pd: (
            _tail_readings(pd)
            .assign(v=[1.0, 2.0, 7.0, 3.0, 12.0, 4.0])
            .rolling(3)
            .skew(numeric_only=True)
        ),
    ),
    (
        "rolling-kurt-numeric",
        "Rolling.kurt",
        ("numeric_only",),
        lambda pd: _tail_readings(pd).rolling(4).kurt(numeric_only=True),
    ),
    (
        "rolling-count-numeric",
        "Rolling.count",
        ("numeric_only",),
        lambda pd: _tail_readings(pd).rolling(2).count(numeric_only=True),
    ),
    (
        "rolling-nunique-numeric",
        "Rolling.nunique",
        ("numeric_only",),
        lambda pd: _tail_readings(pd).rolling(2).nunique(numeric_only=True),
    ),
    (
        "rolling-first-numeric",
        "Rolling.first",
        ("numeric_only",),
        lambda pd: _tail_readings(pd).rolling(2).first(numeric_only=True),
    ),
    (
        "rolling-last-numeric",
        "Rolling.last",
        ("numeric_only",),
        lambda pd: _tail_readings(pd).rolling(2).last(numeric_only=True),
    ),
    (
        "rolling-agg-func",
        "Rolling.agg",
        ("func",),
        lambda pd: _tail_readings(pd).rolling(2).agg(func="max"),
    ),
    (
        "rolling-pipe-func",
        "Rolling.pipe",
        ("func",),
        lambda pd: _tail_readings(pd).rolling(2).pipe(func=lambda r: r.sum()),
    ),
    (
        "expanding-skew-numeric",
        "Expanding.skew",
        ("numeric_only",),
        lambda pd: (
            _tail_readings(pd)
            .assign(v=[1.0, 2.0, 7.0, 3.0, 12.0, 4.0])
            .expanding(3)
            .skew(numeric_only=True)
        ),
    ),
    (
        "expanding-kurt-numeric",
        "Expanding.kurt",
        ("numeric_only",),
        lambda pd: _tail_readings(pd).expanding(4).kurt(numeric_only=True),
    ),
    (
        "expanding-count-numeric",
        "Expanding.count",
        ("numeric_only",),
        lambda pd: _tail_readings(pd).expanding().count(numeric_only=True),
    ),
    (
        "expanding-nunique-numeric",
        "Expanding.nunique",
        ("numeric_only",),
        lambda pd: _tail_readings(pd).expanding().nunique(numeric_only=True),
    ),
    (
        "expanding-first-numeric",
        "Expanding.first",
        ("numeric_only",),
        lambda pd: _tail_readings(pd).expanding().first(numeric_only=True),
    ),
    (
        "expanding-last-numeric",
        "Expanding.last",
        ("numeric_only",),
        lambda pd: _tail_readings(pd).expanding().last(numeric_only=True),
    ),
    (
        "expanding-agg-func",
        "Expanding.agg",
        ("func",),
        lambda pd: _tail_readings(pd).expanding().agg(func="sum"),
    ),
    (
        "expanding-pipe-func",
        "Expanding.pipe",
        ("func",),
        lambda pd: _tail_readings(pd).expanding().pipe(func=lambda r: r.max()),
    ),
    (
        "ewm-agg-func",
        "ExponentialMovingWindow.agg",
        ("func",),
        lambda pd: _tail_readings(pd).ewm(com=1).agg(func="mean"),
    ),
    (
        "resampler-quantile-q",
        "Resampler.quantile",
        ("q",),
        lambda pd: _tail_readings(pd).resample("3h").quantile(q=0.5),
    ),
    (
        "resampler-median-numeric",
        "Resampler.median",
        ("numeric_only",),
        lambda pd: _tail_readings(pd).resample("3h").median(numeric_only=True),
    ),
    (
        "resampler-mean-numeric",
        "Resampler.mean",
        ("numeric_only",),
        lambda pd: _tail_readings(pd).resample("3h").mean(numeric_only=True),
    ),
    (
        "resampler-transform-arg",
        "Resampler.transform",
        ("arg",),
        lambda pd: _tail_readings(pd).resample("3h").transform(arg="sum"),
    ),
    (
        "resampler-pipe-func",
        "Resampler.pipe",
        ("func",),
        lambda pd: _tail_readings(pd).resample("3h").pipe(func=lambda r: r.max()),
    ),
    (
        "resampler-nearest-limit",
        "Resampler.nearest",
        ("limit",),
        lambda pd: _tail_readings(pd).resample("30min").nearest(limit=1),
    ),
    (
        "resampler-ffill-limit",
        "Resampler.ffill",
        ("limit",),
        lambda pd: _tail_readings(pd).resample("30min").ffill(limit=1),
    ),
    (
        "resampler-bfill-limit",
        "Resampler.bfill",
        ("limit",),
        lambda pd: _tail_readings(pd).resample("30min").bfill(limit=1),
    ),
    (
        "resampler-asfreq-fill",
        "Resampler.asfreq",
        ("fill_value",),
        lambda pd: _tail_readings(pd).resample("30min").asfreq(fill_value=0.0),
    ),
    (
        "resampler-apply-func",
        "Resampler.apply",
        ("func",),
        lambda pd: _tail_readings(pd).resample("3h").apply(func="min"),
    ),
    (
        "resampler-agg-func",
        "Resampler.agg",
        ("func",),
        lambda pd: _tail_readings(pd).resample("3h").agg(func=["min", "max"]),
    ),
    (
        "resampler-get-group-name",
        "Resampler.get_group",
        ("name",),
        lambda pd: (
            _tail_readings(pd).resample("3h").get_group(name=pd.Timestamp("2024-01-01 03:00"))
        ),
    ),
    (
        "grouped-take-indices",
        "GroupBy.take",
        ("indices",),
        lambda pd: _tail_table(pd).groupby("s")[["b"]].take(indices=[0]),
    ),
    (
        "grouped-quantile-numeric",
        "GroupBy.quantile",
        ("numeric_only",),
        lambda pd: _tail_table(pd).groupby("s").quantile(0.5, numeric_only=True),
    ),
    (
        "grouped-pipe-func",
        "GroupBy.pipe",
        ("func",),
        lambda pd: _tail_table(pd).groupby("s").pipe(func=lambda g: g.b.sum()),
    ),
    (
        "grouped-nunique-dropna",
        "GroupBy.nunique",
        ("dropna",),
        lambda pd: _tail_table(pd).groupby("s").nunique(dropna=False),
    ),
    (
        "grouped-ngroup-ascending",
        "GroupBy.ngroup",
        ("ascending",),
        lambda pd: _tail_table(pd).groupby("s").ngroup(ascending=False),
    ),
    (
        "grouped-get-group-name",
        "GroupBy.get_group",
        ("name",),
        lambda pd: _tail_table(pd).groupby("s").get_group(name="x"),
    ),
    (
        "grouped-ffill-limit",
        "GroupBy.ffill",
        ("limit",),
        lambda pd: _tail_table(pd).groupby("s").ffill(limit=1),
    ),
    (
        "grouped-ewm-method",
        "GroupBy.ewm",
        ("method",),
        lambda pd: _tail_table(pd).groupby("s")[["b"]].ewm(com=1, method="single").mean(),
    ),
    (
        "grouped-diff-periods",
        "GroupBy.diff",
        ("periods",),
        lambda pd: _tail_table(pd).groupby("s")[["b"]].diff(periods=1),
    ),
    (
        "grouped-cumprod-numeric",
        "GroupBy.cumprod",
        ("numeric_only",),
        lambda pd: _tail_table(pd).groupby("s").cumprod(numeric_only=True),
    ),
    (
        "grouped-cummin-numeric",
        "GroupBy.cummin",
        ("numeric_only",),
        lambda pd: _tail_table(pd).groupby("s").cummin(numeric_only=True),
    ),
    (
        "grouped-cummax-numeric",
        "GroupBy.cummax",
        ("numeric_only",),
        lambda pd: _tail_table(pd).groupby("s").cummax(numeric_only=True),
    ),
    (
        "grouped-apply-include-groups",
        "GroupBy.apply",
        ("include_groups",),
        lambda pd: _tail_table(pd).groupby("s").apply(lambda g: g.b.sum(), include_groups=False),
    ),
    (
        "grouped-all-skipna",
        "GroupBy.all",
        ("skipna",),
        lambda pd: _tail_table(pd).groupby("s")[["b"]].all(skipna=True),
    ),
)


def _tail_update(pd):
    """A column updated in place from another."""
    column = _tail_floats(pd).copy()
    column.update(other=pd.Series([9.0, 8.0], index=["a", "c"]))
    return column


def _tail_sorted(pd):
    """Ascending floats with labels."""
    return pd.Series([1.0, 2.0, 4.0, 8.0], index=list("wxyz"), name="v")


def _tail_shiftable(pd):
    """A two-level index of three rows."""
    return pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1)], names=["k", "n"])


def _tail_daily(pd):
    """Six daily readings with a group key."""
    return pd.DataFrame(
        {"g": list("ababab"), "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
        index=pd.date_range("2024-01-01", periods=6, freq="D"),
    )


TAIL5_CASES = (
    (
        "series-searchsorted-side",
        "Series.searchsorted",
        ("side", "sorter"),
        lambda pd: [
            int(x) for x in _tail_sorted(pd).searchsorted([2.0, 5.0], side="right", sorter=None)
        ],
    ),
    (
        "index-searchsorted-side",
        "Index.searchsorted",
        ("side", "sorter"),
        lambda pd: [
            int(x) for x in pd.Index([1, 3, 5]).searchsorted([3, 4], side="left", sorter=None)
        ],
    ),
    (
        "series-set-flags-dupes",
        "Series.set_flags",
        ("copy", "allows_duplicate_labels"),
        lambda pd: (
            _tail_sorted(pd)
            .set_flags(copy=False, allows_duplicate_labels=True)
            .flags.allows_duplicate_labels
        ),
    ),
    (
        "frame-set-flags-dupes",
        "DataFrame.set_flags",
        ("copy", "allows_duplicate_labels"),
        lambda pd: (
            pd.DataFrame({"a": [1]})
            .set_flags(copy=False, allows_duplicate_labels=False)
            .flags.allows_duplicate_labels
        ),
    ),
    (
        "series-rolling-method",
        "Series.rolling",
        ("win_type", "on", "method"),
        lambda pd: _tail_sorted(pd).rolling(2, win_type=None, on=None, method="single").sum(),
    ),
    (
        "frame-rolling-method",
        "DataFrame.rolling",
        ("win_type", "method"),
        lambda pd: _tail_daily(pd)[["v"]].rolling(2, win_type=None, method="single").sum(),
    ),
    (
        "multi-putmask-value",
        "MultiIndex.putmask",
        ("mask", "value"),
        lambda pd: _tail_levels(
            _tail_shiftable(pd).putmask(
                mask=[True, False, False], value=pd.MultiIndex.from_tuples([("z", 9)] * 3)
            )
        ),
    ),
    (
        "multi-factorize-sort",
        "MultiIndex.factorize",
        ("sort", "use_na_sentinel"),
        lambda pd: _tail_shiftable(pd).factorize(sort=True, use_na_sentinel=True)[0].tolist(),
    ),
    (
        "index-asof-locs-mask",
        "Index.asof_locs",
        ("where", "mask"),
        lambda pd: [
            int(x)
            for x in pd.Index([1, 3, 5]).asof_locs(
                where=pd.Index([2, 6]), mask=pd.Series([True, True, True]).to_numpy()
            )
        ],
    ),
    (
        "instant-asof-locs-mask",
        "DatetimeIndex.asof_locs",
        ("where", "mask"),
        lambda pd: [
            int(x)
            for x in _tail_daily(pd).index.asof_locs(
                where=pd.DatetimeIndex(["2024-01-03 12:00"]), mask=pd.Series([True] * 6).to_numpy()
            )
        ],
    ),
    (
        "grouped-resample-rule",
        "GroupBy.resample",
        ("rule", "include_groups"),
        lambda pd: _tail_daily(pd).groupby("g").resample(rule="2D", include_groups=False).sum(),
    ),
    (
        "frame-resample-level",
        "DataFrame.resample",
        ("convention", "level"),
        lambda pd: _tail_daily(pd)[["v"]].resample("2D", convention="start", level=None).sum(),
    ),
    (
        "instant-std-keepdims",
        "DatetimeIndex.std",
        ("dtype", "out", "keepdims"),
        lambda pd: str(_tail_daily(pd).index.std(dtype=None, out=None, keepdims=False)),
    ),
    (
        "pandas-reset-option-pat",
        "pandas.reset_option",
        ("pat",),
        lambda pd: pd.reset_option(pat="display.max_rows"),
    ),
    (
        "frame-describe-percentiles",
        "DataFrame.describe",
        ("percentiles",),
        lambda pd: _tail_daily(pd)[["v"]].describe(percentiles=[0.1, 0.9]),
    ),
    (
        "frame-var-skipna",
        "DataFrame.var",
        ("skipna",),
        lambda pd: _tail_daily(pd)[["v"]].var(skipna=True),
    ),
    (
        "frame-std-skipna",
        "DataFrame.std",
        ("skipna",),
        lambda pd: _tail_daily(pd)[["v"]].std(skipna=False),
    ),
    (
        "frame-sem-skipna",
        "DataFrame.sem",
        ("skipna",),
        lambda pd: _tail_daily(pd)[["v"]].sem(skipna=True),
    ),
    (
        "frame-unstack-sort",
        "DataFrame.unstack",
        ("sort",),
        lambda pd: _tail_daily(pd).set_index("g", append=True)[["v"]].unstack(sort=False),
    ),
    (
        "frame-take-axis",
        "DataFrame.take",
        ("axis",),
        lambda pd: _tail_daily(pd).take([1, 0], axis=1),
    ),
    (
        "frame-select-exclude",
        "DataFrame.select_dtypes",
        ("exclude",),
        lambda pd: _tail_daily(pd).select_dtypes(exclude="number"),
    ),
    (
        "frame-quantile-numeric",
        "DataFrame.quantile",
        ("numeric_only",),
        lambda pd: _tail_daily(pd).quantile(0.5, numeric_only=True),
    ),
    (
        "frame-prod-numeric",
        "DataFrame.prod",
        ("numeric_only",),
        lambda pd: _tail_daily(pd).prod(numeric_only=True),
    ),
    (
        "frame-mean-numeric",
        "DataFrame.mean",
        ("numeric_only",),
        lambda pd: _tail_daily(pd).mean(numeric_only=True),
    ),
    (
        "frame-mode-numeric",
        "DataFrame.mode",
        ("numeric_only",),
        lambda pd: _tail_daily(pd).mode(numeric_only=True),
    ),
    (
        "frame-cumsum-numeric",
        "DataFrame.cumsum",
        ("numeric_only",),
        lambda pd: _tail_daily(pd)[["v"]].cumsum(numeric_only=True),
    ),
    ("frame-pop-item", "DataFrame.pop", ("item",), lambda pd: _tail_daily(pd).pop(item="v")),
    (
        "frame-nunique-axis",
        "DataFrame.nunique",
        ("axis",),
        lambda pd: _tail_daily(pd).nunique(axis=0),
    ),
    (
        "frame-nsmallest-keep",
        "DataFrame.nsmallest",
        ("keep",),
        lambda pd: _tail_daily(pd).nsmallest(2, "v", keep="last"),
    ),
    (
        "frame-lt-level",
        "DataFrame.lt",
        ("level",),
        lambda pd: _tail_daily(pd)[["v"]].lt(3.0, level=None),
    ),
    (
        "frame-ge-level",
        "DataFrame.ge",
        ("level",),
        lambda pd: _tail_daily(pd)[["v"]].ge(3.0, level=None),
    ),
    (
        "frame-join-validate",
        "DataFrame.join",
        ("validate",),
        lambda pd: _tail_daily(pd)[["v"]].join(_tail_daily(pd)[["g"]], validate="one_to_one"),
    ),
    (
        "frame-interpolate-limit",
        "DataFrame.interpolate",
        ("limit",),
        lambda pd: pd.DataFrame({"a": [1.0, float("nan"), float("nan"), 4.0]}).interpolate(limit=1),
    ),
    (
        "frame-group-keys",
        "DataFrame.groupby",
        ("group_keys",),
        lambda pd: (
            _tail_daily(pd)
            .reset_index(drop=True)
            .groupby("g", group_keys=False)[["v"]]
            .apply(lambda f: f.head(1))
        ),
    ),
    (
        "frame-filter-axis",
        "DataFrame.filter",
        ("axis",),
        lambda pd: _tail_daily(pd).filter(items=["v"], axis=1),
    ),
    (
        "frame-expanding-method",
        "DataFrame.expanding",
        ("method",),
        lambda pd: _tail_daily(pd)[["v"]].expanding(method="single").sum(),
    ),
    (
        "frame-ewm-method",
        "DataFrame.ewm",
        ("method",),
        lambda pd: _tail_daily(pd)[["v"]].ewm(com=1, method="single").mean(),
    ),
    ("frame-eval-expr", "DataFrame.eval", ("expr",), lambda pd: _tail_daily(pd).eval(expr="v * 2")),
    (
        "frame-equals-other",
        "DataFrame.equals",
        ("other",),
        lambda pd: _tail_daily(pd).equals(other=_tail_daily(pd)),
    ),
    (
        "frame-droplevel-axis",
        "DataFrame.droplevel",
        ("axis",),
        lambda pd: _tail_daily(pd).set_index("g", append=True).droplevel("g", axis=0),
    ),
    (
        "frame-dot-other",
        "DataFrame.dot",
        ("other",),
        lambda pd: pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}).dot(
            other=pd.Series([1.0, 1.0], index=["a", "b"])
        ),
    ),
    (
        "union-cats-sorted",
        "api.types.union_categoricals",
        ("to_union", "sort_categories"),
        lambda pd: pd.Series(
            pd.api.types.union_categoricals(
                to_union=[pd.Categorical(["b", "a"]), pd.Categorical(["c"])], sort_categories=True
            )
        ),
    ),
    (
        "union-cats-ignore-order",
        "api.types.union_categoricals",
        ("ignore_order",),
        lambda pd: pd.Series(
            pd.api.types.union_categoricals(
                [pd.Categorical(["a"], ordered=True), pd.Categorical(["b"], ordered=True)],
                ignore_order=True,
            )
        ),
    ),
)


def _tail_pairs(pd):
    """A two-level index of four rows with a repeat."""
    return pd.MultiIndex.from_tuples([("a", 1), ("b", 2), ("a", 1), ("c", 3)], names=["k", "n"])


def _tail_counts(pd):
    """Whole-number labels with a repeat."""
    return pd.Index([4, 1, 4, 2], name="i")


def _tail_halves(pd):
    """Float labels with a gap."""
    return pd.Index([1.25, float("nan"), 3.5], name="f")


def _tail_days(pd):
    """Four daily instants with a repeat."""
    return pd.DatetimeIndex(["2024-01-03", "2024-01-01", "2024-01-03", "2024-01-02"], name="t")


def _tail_run(pd):
    """Four ascending daily instants."""
    return pd.date_range("2024-01-01", periods=4, freq="D", name="t")


def _tail_grid(pd):
    """Two numeric columns over text labels."""
    return pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}, index=list("xyz"))


TAIL6_CASES = (
    (
        "multi-value-counts-bins",
        "MultiIndex.value_counts",
        ("bins",),
        lambda pd: _tail_levels(_tail_pairs(pd).value_counts(bins=None).index),
    ),
    (
        "multi-unique-level",
        "MultiIndex.unique",
        ("level",),
        lambda pd: _tail_pairs(pd).unique(level="k").tolist(),
    ),
    (
        "multi-to-frame-dupes",
        "MultiIndex.to_frame",
        ("allow_duplicates",),
        lambda pd: _tail_pairs(pd).to_frame(index=False, allow_duplicates=False),
    ),
    (
        "multi-slice-locs-step",
        "MultiIndex.slice_locs",
        ("step",),
        lambda pd: [
            int(x) for x in _tail_pairs(pd).sortlevel()[0].slice_locs(("a", 1), ("b", 2), step=None)
        ],
    ),
    (
        "multi-reorder-levels-order",
        "MultiIndex.reorder_levels",
        ("order",),
        lambda pd: _tail_levels(_tail_pairs(pd).reorder_levels(order=["n", "k"])),
    ),
    (
        "multi-reindex-tolerance",
        "MultiIndex.reindex",
        ("tolerance",),
        lambda pd: [
            int(x)
            for x in _tail_pairs(pd).unique().reindex([("b", 2), ("a", 1)], tolerance=None)[1]
        ],
    ),
    (
        "multi-ravel-order",
        "MultiIndex.ravel",
        ("order",),
        lambda pd: [tuple(x) for x in _tail_pairs(pd).ravel(order="C").tolist()],
    ),
    (
        "multi-nunique-dropna",
        "MultiIndex.nunique",
        ("dropna",),
        lambda pd: int(_tail_pairs(pd).nunique(dropna=False)),
    ),
    (
        "multi-memory-deep",
        "MultiIndex.memory_usage",
        ("deep",),
        lambda pd: _tail_pairs(pd).memory_usage(deep=False) > 0,
    ),
    (
        "multi-is-other",
        "MultiIndex.is_",
        ("other",),
        lambda pd: _tail_pairs(pd).is_(other=_tail_pairs(pd)),
    ),
    (
        "multi-identical-other",
        "MultiIndex.identical",
        ("other",),
        lambda pd: _tail_pairs(pd).identical(other=_tail_pairs(pd)),
    ),
    (
        "multi-get-locs-seq",
        "MultiIndex.get_locs",
        ("seq",),
        lambda pd: [int(x) for x in _tail_pairs(pd).sortlevel()[0].get_locs(seq=["a"])],
    ),
    (
        "multi-get-loc-key",
        "MultiIndex.get_loc",
        ("key",),
        lambda pd: _tail_pairs(pd).unique().get_loc(key=("b", 2)),
    ),
    (
        "multi-level-values-named",
        "MultiIndex.get_level_values",
        ("level",),
        lambda pd: _tail_pairs(pd).get_level_values(level="n"),
    ),
    (
        "multi-indexer-non-unique",
        "MultiIndex.get_indexer_non_unique",
        ("target",),
        lambda pd: [int(x) for x in _tail_pairs(pd).get_indexer_non_unique(target=[("a", 1)])[0]],
    ),
    (
        "multi-indexer-for-target",
        "MultiIndex.get_indexer_for",
        ("target",),
        lambda pd: [int(x) for x in _tail_pairs(pd).get_indexer_for(target=[("c", 3)])],
    ),
    (
        "multi-indexer-tolerance",
        "MultiIndex.get_indexer",
        ("tolerance",),
        lambda pd: [
            int(x) for x in _tail_pairs(pd).unique().get_indexer([("c", 3)], tolerance=None)
        ],
    ),
    (
        "multi-equals-other",
        "MultiIndex.equals",
        ("other",),
        lambda pd: _tail_pairs(pd).equals(other=_tail_pairs(pd)[::-1]),
    ),
    (
        "multi-equal-levels-other",
        "MultiIndex.equal_levels",
        ("other",),
        lambda pd: _tail_pairs(pd).equal_levels(other=_tail_pairs(pd)),
    ),
    (
        "multi-duplicated-keep",
        "MultiIndex.duplicated",
        ("keep",),
        lambda pd: [bool(x) for x in _tail_pairs(pd).duplicated(keep="last")],
    ),
    (
        "multi-dropna-how",
        "MultiIndex.dropna",
        ("how",),
        lambda pd: _tail_levels(_tail_pairs(pd).dropna(how="all")),
    ),
    (
        "multi-droplevel-named",
        "MultiIndex.droplevel",
        ("level",),
        lambda pd: _tail_pairs(pd).droplevel(level="k"),
    ),
    (
        "multi-drop-duplicates-keep",
        "MultiIndex.drop_duplicates",
        ("keep",),
        lambda pd: _tail_levels(_tail_pairs(pd).drop_duplicates(keep="last")),
    ),
    (
        "multi-delete-loc",
        "MultiIndex.delete",
        ("loc",),
        lambda pd: _tail_levels(_tail_pairs(pd).delete(loc=0)),
    ),
    (
        "multi-argsort-na-position",
        "MultiIndex.argsort",
        ("na_position",),
        lambda pd: [int(x) for x in _tail_pairs(pd).argsort(na_position="last")],
    ),
    (
        "multi-append-other",
        "MultiIndex.append",
        ("other",),
        lambda pd: _tail_levels(_tail_pairs(pd).append(other=_tail_pairs(pd)[:1])),
    ),
    (
        "index-unique-level",
        "Index.unique",
        ("level",),
        lambda pd: _tail_counts(pd).unique(level=None),
    ),
    (
        "index-to-numpy-na-value",
        "Index.to_numpy",
        ("na_value",),
        lambda pd: _tail_halves(pd).to_numpy(na_value=0.0).tolist(),
    ),
    (
        "index-set-names-level",
        "Index.set_names",
        ("level",),
        lambda pd: _tail_counts(pd).set_names("j", level=None),
    ),
    (
        "index-round-decimals",
        "Index.round",
        ("decimals",),
        lambda pd: _tail_halves(pd).round(decimals=1),
    ),
    ("index-rename-named", "Index.rename", ("name",), lambda pd: _tail_counts(pd).rename(name="j")),
    (
        "index-reindex-level",
        "Index.reindex",
        ("level",),
        lambda pd: _tail_counts(pd).unique().reindex([2, 4], level=None)[0],
    ),
    (
        "index-ravel-order",
        "Index.ravel",
        ("order",),
        lambda pd: _tail_counts(pd).ravel(order="C").tolist(),
    ),
    (
        "index-memory-deep",
        "Index.memory_usage",
        ("deep",),
        lambda pd: _tail_counts(pd).memory_usage(deep=True) > 0,
    ),
    (
        "index-map-na-action",
        "Index.map",
        ("na_action",),
        lambda pd: _tail_halves(pd).map(lambda x: x * 2, na_action="ignore"),
    ),
    (
        "index-isin-level",
        "Index.isin",
        ("level",),
        lambda pd: [bool(x) for x in _tail_counts(pd).isin([4], level=None)],
    ),
    (
        "index-is-other",
        "Index.is_",
        ("other",),
        lambda pd: _tail_counts(pd).is_(other=_tail_counts(pd)),
    ),
    (
        "index-infer-objects-copy",
        "Index.infer_objects",
        ("copy",),
        lambda pd: _tail_counts(pd).infer_objects(copy=False),
    ),
    (
        "index-identical-other",
        "Index.identical",
        ("other",),
        lambda pd: _tail_counts(pd).identical(other=_tail_counts(pd).rename("j")),
    ),
    (
        "index-level-values-named",
        "Index.get_level_values",
        ("level",),
        lambda pd: _tail_counts(pd).get_level_values(level=0),
    ),
    (
        "index-indexer-non-unique",
        "Index.get_indexer_non_unique",
        ("target",),
        lambda pd: [int(x) for x in _tail_counts(pd).get_indexer_non_unique(target=[4, 9])[0]],
    ),
    (
        "index-indexer-for-target",
        "Index.get_indexer_for",
        ("target",),
        lambda pd: [int(x) for x in _tail_counts(pd).get_indexer_for(target=[1, 2])],
    ),
    (
        "index-fillna-value",
        "Index.fillna",
        ("value",),
        lambda pd: _tail_halves(pd).fillna(value=0.0),
    ),
    (
        "index-equals-other",
        "Index.equals",
        ("other",),
        lambda pd: _tail_counts(pd).equals(other=pd.Index([4, 1, 4, 2])),
    ),
    (
        "index-droplevel-named",
        "Index.droplevel",
        ("level",),
        lambda pd: _tail_counts(pd).droplevel(level=[]),
    ),
    ("index-diff-periods", "Index.diff", ("periods",), lambda pd: _tail_counts(pd).diff(periods=2)),
    ("index-delete-loc", "Index.delete", ("loc",), lambda pd: _tail_counts(pd).delete(loc=[0, 2])),
    (
        "index-asof-label",
        "Index.asof",
        ("label",),
        lambda pd: int(pd.Index([1, 3, 5]).asof(label=4)),
    ),
    (
        "index-append-other",
        "Index.append",
        ("other",),
        lambda pd: _tail_counts(pd).append(other=pd.Index([7], name="i")),
    ),
    (
        "instant-value-counts-bins",
        "DatetimeIndex.value_counts",
        ("bins",),
        lambda pd: _tail_days(pd).value_counts(bins=None),
    ),
    (
        "instant-unique-level",
        "DatetimeIndex.unique",
        ("level",),
        lambda pd: _tail_days(pd).unique(level=None),
    ),
    (
        "instant-tz-localize-nonexistent",
        "DatetimeIndex.tz_localize",
        ("nonexistent",),
        lambda pd: _tail_run(pd).tz_localize("UTC", nonexistent="raise"),
    ),
    (
        "instant-tz-convert-named",
        "DatetimeIndex.tz_convert",
        ("tz",),
        lambda pd: _tail_run(pd).tz_localize("UTC").tz_convert(tz="Asia/Tokyo"),
    ),
    (
        "instant-to-period-named",
        "DatetimeIndex.to_period",
        ("freq",),
        lambda pd: [str(x) for x in _tail_run(pd).to_period(freq="M")],
    ),
    (
        "instant-strftime-named",
        "DatetimeIndex.strftime",
        ("date_format",),
        lambda pd: _tail_run(pd).strftime(date_format="%d/%m").tolist(),
    ),
    (
        "instant-snap-freq",
        "DatetimeIndex.snap",
        ("freq",),
        lambda pd: pd.DatetimeIndex(["2024-01-01 10:00", "2024-01-02 14:00"]).snap(freq="D"),
    ),
    (
        "instant-slice-locs-step",
        "DatetimeIndex.slice_locs",
        ("step",),
        lambda pd: [
            int(x) for x in _tail_run(pd).slice_locs("2024-01-02", "2024-01-03", step=None)
        ],
    ),
    (
        "instant-rename-named",
        "DatetimeIndex.rename",
        ("name",),
        lambda pd: _tail_run(pd).rename(name="u"),
    ),
    (
        "instant-reindex-level",
        "DatetimeIndex.reindex",
        ("level",),
        lambda pd: _tail_run(pd).reindex(_tail_run(pd)[::2], level=None)[0],
    ),
    (
        "instant-ravel-order",
        "DatetimeIndex.ravel",
        ("order",),
        lambda pd: [str(x) for x in _tail_run(pd).ravel(order="C")],
    ),
    (
        "instant-month-name-locale",
        "DatetimeIndex.month_name",
        ("locale",),
        lambda pd: _tail_run(pd).month_name(locale=None).tolist(),
    ),
    (
        "instant-day-name-locale",
        "DatetimeIndex.day_name",
        ("locale",),
        lambda pd: _tail_run(pd).day_name(locale=None).tolist(),
    ),
    (
        "instant-memory-deep",
        "DatetimeIndex.memory_usage",
        ("deep",),
        lambda pd: _tail_run(pd).memory_usage(deep=False) > 0,
    ),
    ("instant-mean-axis", "DatetimeIndex.mean", ("axis",), lambda pd: _tail_run(pd).mean(axis=0)),
    (
        "instant-join-level",
        "DatetimeIndex.join",
        ("level",),
        lambda pd: _tail_run(pd).join(_tail_run(pd)[1:], how="inner", level=None),
    ),
    (
        "instant-is-other",
        "DatetimeIndex.is_",
        ("other",),
        lambda pd: _tail_run(pd).is_(other=_tail_run(pd)),
    ),
    (
        "instant-infer-objects-copy",
        "DatetimeIndex.infer_objects",
        ("copy",),
        lambda pd: _tail_run(pd).infer_objects(copy=False),
    ),
    (
        "instant-identical-other",
        "DatetimeIndex.identical",
        ("other",),
        lambda pd: _tail_run(pd).identical(other=_tail_run(pd)),
    ),
    (
        "instant-get-loc-key",
        "DatetimeIndex.get_loc",
        ("key",),
        lambda pd: _tail_run(pd).get_loc(key="2024-01-03"),
    ),
    (
        "instant-level-values-named",
        "DatetimeIndex.get_level_values",
        ("level",),
        lambda pd: _tail_run(pd).get_level_values(level="t"),
    ),
    (
        "instant-indexer-non-unique",
        "DatetimeIndex.get_indexer_non_unique",
        ("target",),
        lambda pd: [
            int(x)
            for x in _tail_days(pd).get_indexer_non_unique(target=pd.DatetimeIndex(["2024-01-03"]))[
                0
            ]
        ],
    ),
    (
        "instant-indexer-for-target",
        "DatetimeIndex.get_indexer_for",
        ("target",),
        lambda pd: [
            int(x)
            for x in _tail_run(pd).get_indexer_for(
                target=pd.DatetimeIndex(["2024-01-02", "2025-01-01"])
            )
        ],
    ),
    (
        "instant-fillna-value",
        "DatetimeIndex.fillna",
        ("value",),
        lambda pd: pd.DatetimeIndex(["2024-01-01", None]).fillna(value=pd.Timestamp("2024-02-02")),
    ),
    (
        "instant-equals-other",
        "DatetimeIndex.equals",
        ("other",),
        lambda pd: _tail_run(pd).equals(other=_tail_run(pd)),
    ),
    (
        "instant-duplicated-keep",
        "DatetimeIndex.duplicated",
        ("keep",),
        lambda pd: [bool(x) for x in _tail_days(pd).duplicated(keep=False)],
    ),
    (
        "instant-droplevel-named",
        "DatetimeIndex.droplevel",
        ("level",),
        lambda pd: _tail_run(pd).droplevel(level=[]),
    ),
    (
        "instant-diff-periods",
        "DatetimeIndex.diff",
        ("periods",),
        lambda pd: _tail_run(pd).diff(periods=2),
    ),
    (
        "instant-delete-loc",
        "DatetimeIndex.delete",
        ("loc",),
        lambda pd: _tail_run(pd).delete(loc=1),
    ),
    (
        "instant-asof-label",
        "DatetimeIndex.asof",
        ("label",),
        lambda pd: _tail_run(pd).asof(label="2024-01-02 12:00"),
    ),
    (
        "instant-append-other",
        "DatetimeIndex.append",
        ("other",),
        lambda pd: _tail_run(pd).append(other=_tail_run(pd)[:1]),
    ),
    (
        "frame-transpose-copy",
        "DataFrame.transpose",
        ("copy",),
        lambda pd: _tail_grid(pd).transpose(copy=False),
    ),
    (
        "frame-to-timestamp-copy",
        "DataFrame.to_timestamp",
        ("copy",),
        lambda pd: pd.DataFrame(
            {"a": [1, 2]}, index=pd.period_range("2024-01", periods=2, freq="M")
        ).to_timestamp(copy=False),
    ),
    (
        "frame-to-string-encoding",
        "DataFrame.to_string",
        ("encoding",),
        lambda pd: _tail_grid(pd).to_string(encoding=None),
    ),
    (
        "frame-set-axis-copy",
        "DataFrame.set_axis",
        ("copy",),
        lambda pd: _tail_grid(pd).set_axis(["p", "q"], axis=1, copy=False),
    ),
    (
        "frame-rename-axis-copy",
        "DataFrame.rename_axis",
        ("copy",),
        lambda pd: _tail_grid(pd).rename_axis("row", copy=False),
    ),
    (
        "frame-rename-copy",
        "DataFrame.rename",
        ("copy",),
        lambda pd: _tail_grid(pd).rename(columns={"a": "c"}, copy=False),
    ),
    (
        "frame-reindex-like-copy",
        "DataFrame.reindex_like",
        ("copy",),
        lambda pd: _tail_grid(pd).reindex_like(_tail_grid(pd)[::-1], copy=False),
    ),
    (
        "frame-reindex-copy",
        "DataFrame.reindex",
        ("copy",),
        lambda pd: _tail_grid(pd).reindex(["z", "x"], copy=False),
    ),
    (
        "frame-merge-copy",
        "DataFrame.merge",
        ("copy",),
        lambda pd: _tail_grid(pd).merge(_tail_grid(pd), on="a", copy=False),
    ),
    (
        "frame-infer-objects-copy",
        "DataFrame.infer_objects",
        ("copy",),
        lambda pd: _tail_grid(pd).infer_objects(copy=False),
    ),
    (
        "cat-reorder-ordered",
        "cat.reorder_categories",
        ("ordered",),
        lambda pd: pd.Series(["a", "b"], dtype="category").cat.reorder_categories(
            ["b", "a"], ordered=True
        ),
    ),
    (
        "dt-to-period-named",
        "dt.to_period",
        ("freq",),
        lambda pd: pd.Series(_tail_run(pd)).dt.to_period(freq="M").astype(str),
    ),
    (
        "dt-month-name-locale",
        "dt.month_name",
        ("locale",),
        lambda pd: pd.Series(_tail_run(pd)).dt.month_name(locale=None),
    ),
    (
        "dt-day-name-locale",
        "dt.day_name",
        ("locale",),
        lambda pd: pd.Series(_tail_run(pd)).dt.day_name(locale=None),
    ),
    (
        "timestamp-utcfromtimestamp-named",
        "Timestamp.utcfromtimestamp",
        ("ts",),
        lambda pd: pd.Timestamp.utcfromtimestamp(ts=0),
    ),
    (
        "timestamp-today-tz",
        "Timestamp.today",
        ("tz",),
        lambda pd: str(pd.Timestamp.today(tz="UTC").tz),
    ),
    (
        "timestamp-now-tz",
        "Timestamp.now",
        ("tz",),
        lambda pd: str(pd.Timestamp.now(tz="Asia/Tokyo").tz),
    ),
    (
        "timedelta-view-dtype",
        "Timedelta.view",
        ("dtype",),
        lambda pd: int(pd.Timedelta("1s").view(dtype="i8")),
    ),
    (
        "is-sparse-arr",
        "api.types.is_sparse",
        ("arr",),
        lambda pd: pd.api.types.is_sparse(arr=pd.Series([1])),
    ),
    (
        "arrow-dtype-named",
        "pandas.ArrowDtype",
        ("pyarrow_dtype",),
        lambda pd: str(pd.ArrowDtype(pyarrow_dtype=__import__("pyarrow").int64())),
    ),
    ("series-info-max-cols", "Series.info", ("max_cols",), lambda pd: _tail_info(pd)),
    (
        "grouped-sample-weights",
        "GroupBy.sample",
        ("weights",),
        lambda pd: (
            pd.DataFrame({"g": list("aabb"), "v": [1, 2, 3, 4]})
            .groupby("g")
            .sample(n=1, weights=[0.0, 1.0, 1.0, 0.0])
        ),
    ),
    (
        "undefined-variable-local",
        "errors.UndefinedVariableError",
        ("name", "is_local"),
        lambda pd: str(pd.errors.UndefinedVariableError(name="x", is_local=True)),
    ),
)


def _tail_info(pd):
    """The lines info writes for a column, without the memory line."""
    import io

    buffer = io.StringIO()
    pd.Series([1, 2], name="v").info(buf=buffer, max_cols=None, memory_usage=False)
    return buffer.getvalue().splitlines()[1:]


def _tail7_ranked(pd):
    """Five values over text labels, grouped two ways."""
    return pd.Series([3, 1, 2, 5, 4], index=list("abcde"), name="v").groupby([1, 1, 1, 2, 2])


def _tail7_pairs(pd):
    """One column over a two-level index."""
    index = pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1)], names=["k", "n"])
    return pd.DataFrame({"v": [1, 2, 3]}, index=index)


def _tail7_spans(pd):
    """Three durations."""
    return pd.to_timedelta(["1s", "2s", "6s"])


TAIL7_CASES = (
    (
        "grouped-series-nlargest",
        "Series.groupby",
        ("by",),
        lambda pd: [
            (int(k), str(n), int(v)) for (k, n), v in _tail7_ranked(pd).nlargest(2).items()
        ],
    ),
    (
        "grouped-series-nsmallest",
        "Series.groupby",
        ("by",),
        lambda pd: _tail7_ranked(pd).nsmallest(1).tolist(),
    ),
    (
        "grouped-series-unique",
        "Series.groupby",
        ("by",),
        lambda pd: [sorted(int(x) for x in part) for part in _tail7_ranked(pd).unique().tolist()],
    ),
    (
        "grouped-series-rising",
        "Series.groupby",
        ("by",),
        lambda pd: [bool(x) for x in _tail7_ranked(pd).is_monotonic_increasing.tolist()],
    ),
    (
        "grouped-series-falling",
        "Series.groupby",
        ("by",),
        lambda pd: [bool(x) for x in _tail7_ranked(pd).is_monotonic_decreasing.tolist()],
    ),
    (
        "timedelta-index-sum",
        "pandas.to_timedelta",
        ("arg",),
        lambda pd: str(_tail7_spans(pd).sum()),
    ),
    (
        "timedelta-index-mean",
        "pandas.to_timedelta",
        ("arg",),
        lambda pd: str(_tail7_spans(pd).mean()),
    ),
    (
        "timedelta-index-median",
        "pandas.to_timedelta",
        ("arg",),
        lambda pd: str(_tail7_spans(pd).median()),
    ),
    (
        "timedelta-index-std",
        "pandas.to_timedelta",
        ("arg",),
        lambda pd: str(_tail7_spans(pd).std()),
    ),
    (
        "frame-drop-tuple-key",
        "DataFrame.drop",
        ("labels",),
        lambda pd: _tail7_pairs(pd).drop(("a", 2)).reset_index(),
    ),
    (
        "frame-drop-tuple-ignore",
        "DataFrame.drop",
        ("errors",),
        lambda pd: _tail7_pairs(pd).drop(("z", 9), errors="ignore").reset_index(),
    ),
    (
        "series-drop-tuple-key",
        "Series.drop",
        ("labels",),
        lambda pd: _tail7_pairs(pd)["v"].drop(("b", 1)).tolist(),
    ),
    (
        "frame-equals-label-kinds",
        "DataFrame.equals",
        ("other",),
        lambda pd: pd.DataFrame({1: [10], 2: [20]}).equals(pd.DataFrame({1.0: [10], 2.0: [20]})),
    ),
    (
        "frame-combine-ufunc",
        "DataFrame.combine",
        ("func",),
        lambda pd: pd.DataFrame({"A": [5, 0], "B": [2, 4]}).combine(
            pd.DataFrame({"A": [1, 1], "B": [3, 3]}), __import__("numpy").minimum
        ),
    ),
)


def _tail8_numbers(pd):
    """Masked whole numbers with a gap and a repeat."""
    return pd.array([3, 1, None, 4, 1], dtype="Int64")


def _tail8_letters(pd):
    """An ordered categorical with a gap."""
    return pd.Categorical(["b", "b", "a", "c", None], categories=["c", "b", "a"], ordered=True)


def _tail8_spans(pd):
    """Three intervals, two of them overlapping."""
    return pd.arrays.IntervalArray.from_tuples([(0, 1), (1, 3), (2, 4)])


def _tail8_set(pd):
    """A categorical with values put in place."""
    made = pd.Categorical(["a", "b", "c"])
    made[2] = "a"
    made[0:2] = ["c", "c"]
    return made.tolist()


TAIL8_CASES = (
    ("array-argmax", "pandas.array", ("data",), lambda pd: int(_tail8_numbers(pd).argmax())),
    ("array-argmin", "pandas.array", ("data",), lambda pd: int(_tail8_numbers(pd).argmin())),
    ("array-argsort", "pandas.array", ("data",), lambda pd: _tail8_numbers(pd).argsort().tolist()),
    (
        "array-argsort-falling",
        "pandas.array",
        ("data",),
        lambda pd: _tail8_numbers(pd).argsort(ascending=False).tolist(),
    ),
    (
        "array-argsort-gaps-first",
        "pandas.array",
        ("data",),
        lambda pd: _tail8_numbers(pd).argsort(na_position="first").tolist(),
    ),
    (
        "array-duplicated",
        "pandas.array",
        ("data",),
        lambda pd: _tail8_numbers(pd).duplicated().tolist(),
    ),
    (
        "array-duplicated-all",
        "pandas.array",
        ("data",),
        lambda pd: _tail8_numbers(pd).duplicated(keep=False).tolist(),
    ),
    (
        "array-isin",
        "pandas.array",
        ("data",),
        lambda pd: [bool(x) for x in _tail8_numbers(pd).isin([1, 4])],
    ),
    ("array-item", "pandas.array", ("data",), lambda pd: int(pd.array([7], dtype="Int64").item())),
    ("array-nbytes", "pandas.array", ("data",), lambda pd: _tail8_numbers(pd).nbytes),
    (
        "array-searchsorted",
        "pandas.array",
        ("data",),
        lambda pd: pd.array([1, 2, 3, 5], dtype="Int64").searchsorted([4]).tolist(),
    ),
    (
        "array-shift",
        "pandas.array",
        ("data",),
        lambda pd: str(_tail8_numbers(pd).shift(-1, fill_value=0).dtype),
    ),
    (
        "array-shift-values",
        "pandas.array",
        ("data",),
        lambda pd: [int(x) for x in _tail8_numbers(pd).shift(-1, fill_value=0)[2:]],
    ),
    (
        "array-map-text",
        "pandas.array",
        ("data",),
        lambda pd: pd.array(["x", "y"], dtype="string").map(str.upper).tolist(),
    ),
    (
        "array-numpy-interpolate",
        "pandas.array",
        ("data",),
        lambda pd: [
            float(x)
            for x in pd.arrays.NumpyExtensionArray(
                __import__("numpy").array([0, 1, float("nan"), 3])
            ).interpolate(
                method="linear",
                axis=0,
                index=pd.Index([1, 2, 3, 4]),
                limit=3,
                limit_direction="forward",
                limit_area="inside",
                copy=False,
            )
        ],
    ),
    (
        "categorical-argsort",
        "pandas.Categorical",
        ("values",),
        lambda pd: _tail8_letters(pd).argsort().tolist(),
    ),
    (
        "categorical-isin",
        "pandas.Categorical",
        ("values",),
        lambda pd: _tail8_letters(pd).isin(["a", "c"]).tolist(),
    ),
    (
        "categorical-sort-values",
        "pandas.Categorical",
        ("values",),
        lambda pd: _tail8_letters(pd).sort_values().tolist()[:4],
    ),
    (
        "categorical-sort-falling",
        "pandas.Categorical",
        ("values",),
        lambda pd: (
            _tail8_letters(pd).sort_values(ascending=False, na_position="first").tolist()[1:]
        ),
    ),
    (
        "categorical-map",
        "pandas.Categorical",
        ("values",),
        lambda pd: (
            _tail8_letters(pd).map(lambda v: v.upper(), na_action="ignore").categories.tolist()
        ),
    ),
    ("categorical-setitem", "pandas.Categorical", ("values",), _tail8_set),
    (
        "categorical-shift",
        "pandas.Categorical",
        ("values",),
        lambda pd: _tail8_letters(pd).shift(1).tolist()[1:],
    ),
    (
        "interval-array-contains",
        "pandas.IntervalIndex",
        ("data",),
        lambda pd: _tail8_spans(pd).contains(2).tolist(),
    ),
    (
        "interval-array-overlaps",
        "pandas.IntervalIndex",
        ("data",),
        lambda pd: _tail8_spans(pd).overlaps(pd.Interval(0.5, 1.5)).tolist(),
    ),
    (
        "interval-array-breaks-left",
        "pandas.IntervalIndex",
        ("data",),
        lambda pd: pd.arrays.IntervalArray.from_breaks(range(4)).left.tolist(),
    ),
    (
        "interval-array-set-closed",
        "pandas.IntervalIndex",
        ("closed",),
        lambda pd: pd.arrays.IntervalArray.from_breaks(range(4)).set_closed("both").closed,
    ),
    (
        "interval-array-from-list",
        "pandas.IntervalIndex",
        ("data",),
        lambda pd: pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)]).right.tolist(),
    ),
    (
        "interval-index-overlapping",
        "pandas.IntervalIndex",
        ("data",),
        lambda pd: pd.IntervalIndex.from_tuples([(0, 10), (1, 2), (3, 4)]).is_overlapping,
    ),
    (
        "interval-index-overlapping-closed",
        "pandas.IntervalIndex",
        ("closed",),
        lambda pd: [
            pd.IntervalIndex.from_breaks([0, 1, 2], closed=c).is_overlapping
            for c in ("right", "both")
        ],
    ),
)


def _tail9_gappy(pd):
    """Five floats with a gap."""
    return pd.Series([1.0, 2.0, float("nan"), 4.0, 5.0], name="v")


def _tail9_ahead(pd, size=2):
    """A window of the row and the rows after it."""
    return pd.api.indexers.FixedForwardWindowIndexer(window_size=size)


def _tail9_options(pd):
    """Two options set from one dict, read back, and put back."""
    pd.set_option({"display.max_columns": 4, "display.precision": 1})
    try:
        return [pd.get_option("display.max_columns"), pd.get_option("display.precision")]
    finally:
        pd.reset_option("display.max_columns")
        pd.reset_option("display.precision")


def _tail9_scores(pd):
    """Three numeric columns."""
    return pd.DataFrame({"A": [1, 2, 3], "B": [4.0, 5.0, 6.0], "C": [7, 8, 10]})


def _tail9_days(pd):
    """Six days of rising floats."""
    return pd.Series(
        range(6), index=pd.date_range("2020-01-01", periods=6, freq="D"), dtype="float64"
    )


TAIL9_CASES = (
    (
        "rolling-forward-sum",
        "Series.rolling",
        ("window",),
        lambda pd: _tail9_gappy(pd).rolling(_tail9_ahead(pd)).sum(),
    ),
    (
        "rolling-forward-sum-periods",
        "Series.rolling",
        ("min_periods",),
        lambda pd: _tail9_gappy(pd).rolling(_tail9_ahead(pd), min_periods=1).sum(),
    ),
    (
        "rolling-forward-count",
        "Series.rolling",
        ("window",),
        lambda pd: _tail9_gappy(pd).rolling(_tail9_ahead(pd)).count(),
    ),
    (
        "rolling-forward-max",
        "Series.rolling",
        ("window",),
        lambda pd: _tail9_gappy(pd).rolling(_tail9_ahead(pd), min_periods=1).max(),
    ),
    (
        "rolling-forward-mean",
        "Series.rolling",
        ("window",),
        lambda pd: _tail9_gappy(pd).rolling(_tail9_ahead(pd), min_periods=1).mean(),
    ),
    (
        "rolling-forward-std",
        "Series.rolling",
        ("window",),
        lambda pd: _tail9_gappy(pd).rolling(_tail9_ahead(pd, 3), min_periods=2).std(),
    ),
    (
        "rolling-forward-apply",
        "Series.rolling",
        ("window",),
        lambda pd: (
            _tail9_gappy(pd)
            .rolling(_tail9_ahead(pd), min_periods=1)
            .apply(lambda w: w.sum() * 2, raw=True)
        ),
    ),
    (
        "rolling-forward-frame",
        "DataFrame.rolling",
        ("window",),
        lambda pd: (
            pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            .rolling(_tail9_ahead(pd), min_periods=1)
            .sum()
        ),
    ),
    (
        "rolling-offset-business",
        "Series.rolling",
        ("window",),
        lambda pd: (
            _tail9_days(pd)
            .rolling(
                pd.api.indexers.VariableOffsetWindowIndexer(
                    index=_tail9_days(pd).index, offset=pd.offsets.BDay(1)
                ),
                min_periods=1,
            )
            .sum()
        ),
    ),
    (
        "rolling-offset-closed",
        "Series.rolling",
        ("closed",),
        lambda pd: (
            _tail9_days(pd)
            .rolling(
                pd.api.indexers.VariableOffsetWindowIndexer(
                    index=_tail9_days(pd).index, offset=pd.offsets.BDay(1)
                ),
                min_periods=1,
                closed="both",
            )
            .sum()
        ),
    ),
    (
        "check-array-indexer-flags",
        "pandas.array",
        ("data",),
        lambda pd: [
            bool(x)
            for x in pd.api.indexers.check_array_indexer(
                pd.array([1, 2, 3]), pd.array([True, None, True], dtype="boolean")
            )
        ],
    ),
    (
        "frame-agg-named",
        "DataFrame.agg",
        ("kwargs",),
        lambda pd: _tail9_scores(pd).agg(y=("C", "min"), x=("A", "max"), z=("C", "max")),
    ),
    (
        "frame-agg-named-one",
        "DataFrame.agg",
        ("kwargs",),
        lambda pd: _tail9_scores(pd).agg(x=("A", "max")),
    ),
    (
        "series-agg-named",
        "Series.agg",
        ("kwargs",),
        lambda pd: pd.Series([1, 2, 3]).agg(x="max", y="min"),
    ),
    (
        "factorize-numpy-objects",
        "pandas.factorize",
        ("values",),
        lambda pd: [
            x.tolist()
            for x in pd.factorize(__import__("numpy").array(["b", "b", "a", "c", "b"], dtype="O"))
        ],
    ),
    (
        "factorize-numpy-sorted",
        "pandas.factorize",
        ("sort",),
        lambda pd: [
            x.tolist()
            for x in pd.factorize(
                __import__("numpy").array(["b", "b", "a", "c", "b"], dtype="O"), sort=True
            )
        ],
    ),
    (
        "unique-numpy",
        "pandas.unique",
        ("values",),
        lambda pd: pd.unique(__import__("numpy").array([3, 1, 3])).tolist(),
    ),
    ("set-option-dict", "pandas.set_option", ("args",), _tail9_options),
    (
        "timestamp-month-first",
        "pandas.Timestamp",
        ("ts_input",),
        lambda pd: str(pd.Timestamp("1/2/2018")),
    ),
    (
        "timestamp-year-slashes",
        "pandas.Timestamp",
        ("ts_input",),
        lambda pd: str(pd.Timestamp("2018/01/02 10:30")),
    ),
    (
        "bdate-range-slashes",
        "pandas.bdate_range",
        ("start",),
        lambda pd: [str(x) for x in pd.bdate_range(start="1/1/2018", end="1/08/2018")],
    ),
)


def _tail10_shapes(pd):
    """Angles and degrees of three shapes."""
    return pd.DataFrame(
        {"angles": [0, 3, 4], "degrees": [360, 180, 360]},
        index=["circle", "triangle", "rectangle"],
    )


def _tail10_numpy():
    """numpy, read when a case needs it."""
    import numpy

    return numpy


def _tail10_relabelled(pd):
    """Three columns given two levels of labels after they are made."""
    frame = pd.DataFrame({"A": [1, 4], "B": [2, 5], "C": [3, 6]})
    frame.columns = [list("ABC"), list("DEF")]
    return _tail_levels(frame.columns)


def _tail10_gap_written(pd, kind):
    """A column of two values with NA written over the first."""
    values = {"int64": [1, 2], "float64": [1.5, 2.5]}[kind]
    column = pd.Series(values, dtype=kind)
    column.iloc[0] = pd.NA
    return column


def _tail10_frame_gap(pd):
    """A frame with NA written into one cell by position."""
    frame = pd.DataFrame({"a": [1.5, 2.5], "b": [3.5, 4.5]})
    frame.iloc[0, 1] = pd.NA
    return frame


TAIL10_CASES = (
    (
        "frame-mul-dict",
        "DataFrame.mul",
        ("other",),
        lambda pd: _tail10_shapes(pd).mul({"angles": 0, "degrees": 2}),
    ),
    (
        "frame-add-dict-order",
        "DataFrame.add",
        ("other",),
        lambda pd: _tail10_shapes(pd).add({"degrees": 1, "angles": 2}),
    ),
    (
        "frame-mul-dict-rows",
        "DataFrame.mul",
        ("axis",),
        lambda pd: _tail10_shapes(pd).mul(
            {"circle": 1, "triangle": 2, "rectangle": 3}, axis="index"
        ),
    ),
    (
        "frame-sub-dict",
        "DataFrame.sub",
        ("other",),
        lambda pd: _tail10_shapes(pd).sub({"angles": 1, "degrees": 10}),
    ),
    (
        "series-index-lists",
        "pandas.Series",
        ("index",),
        lambda pd: _tail_levels(
            pd.Series([1, 2, 3, 4], index=[["a", "a", "b", "b"], [1, 2, 1, 2]]).index
        ),
    ),
    (
        "series-index-lists-sum",
        "pandas.Series",
        ("index",),
        lambda pd: (
            pd.Series([1, 2, 3, 4], index=[["a", "a", "b", "b"], [1, 2, 1, 2]])
            .groupby(level=0)
            .sum()
        ),
    ),
    (
        "series-index-arrays",
        "pandas.Series",
        ("index",),
        lambda pd: _tail_levels(
            pd.Series(
                [5, 6], index=[_tail10_numpy().array(["x", "y"]), _tail10_numpy().array([1, 2])]
            ).index
        ),
    ),
    (
        "frame-index-lists",
        "pandas.DataFrame",
        ("index",),
        lambda pd: _tail_levels(pd.DataFrame({"g": [1, 2]}, index=[["x", "y"], ["p", "q"]]).index),
    ),
    (
        "frame-columns-lists",
        "pandas.DataFrame",
        ("columns",),
        lambda pd: _tail_levels(pd.DataFrame([[1, 2]], columns=[["a", "a"], ["x", "y"]]).columns),
    ),
    (
        "frame-columns-lists-values",
        "pandas.DataFrame",
        ("columns",),
        lambda pd: pd.DataFrame([[1, 2], [3, 4]], columns=[["a", "a"], ["x", "y"]])["a"][
            "y"
        ].tolist(),
    ),
    ("frame-columns-assigned-lists", "DataFrame.set_axis", ("labels",), _tail10_relabelled),
    (
        "date-range-compound-freq",
        "pandas.date_range",
        ("freq",),
        lambda pd: pd.date_range("2018-04-09", periods=4, freq="1D20min"),
    ),
    (
        "date-range-compound-hours",
        "pandas.date_range",
        ("freq",),
        lambda pd: pd.date_range("2020-01-01", periods=3, freq="2h30min"),
    ),
    (
        "datetimeindex-m8-dtype",
        "pandas.DatetimeIndex",
        ("dtype",),
        lambda pd: pd.DatetimeIndex(["2015-03-29 02:30:00", "2015-03-29 03:30:00"], dtype="M8[ns]"),
    ),
    (
        "series-ufunc-sqrt",
        "Series.transform",
        ("func",),
        lambda pd: _tail10_numpy().sqrt(
            pd.Series([1.0, 4.0, 9.0], index=["p", "q", "r"], name="v")
        ),
    ),
    (
        "series-ufunc-add",
        "Series.transform",
        ("func",),
        lambda pd: _tail10_numpy().add(pd.Series([1.5, 2.5], name="v"), 1),
    ),
    (
        "series-transform-ufuncs",
        "Series.transform",
        ("func",),
        lambda pd: pd.Series([1.0, 2.0, 3.0]).transform(
            [_tail10_numpy().sqrt, _tail10_numpy().exp]
        ),
    ),
    (
        "series-transform-ufunc-one",
        "Series.transform",
        ("func",),
        lambda pd: pd.Series([1.0, 4.0]).transform(_tail10_numpy().sqrt),
    ),
    ("series-iloc-na-int", "pandas.Series", ("data",), lambda pd: _tail10_gap_written(pd, "int64")),
    (
        "series-iloc-na-float",
        "pandas.Series",
        ("data",),
        lambda pd: _tail10_gap_written(pd, "float64"),
    ),
    ("frame-iloc-na", "pandas.DataFrame", ("data",), _tail10_frame_gap),
)


def _tail11_objects(pd, values):
    """A column of objects holding the values."""
    return pd.Series(values, dtype=object)


def _tail11_in_place(pd):
    """A frame given two columns by one eval in place."""
    frame = pd.DataFrame({"a": [1, 2], "b": [1.5, 2.5]})
    frame.eval("c = a + b\nd = c * 2", inplace=True)
    return frame


TAIL11_CASES = (
    (
        "objects-convert-whole",
        "Series.convert_dtypes",
        ("infer_objects",),
        lambda pd: _tail11_objects(pd, [1, 2, 3]).convert_dtypes(),
    ),
    (
        "objects-convert-floats",
        "Series.convert_dtypes",
        ("convert_floating",),
        lambda pd: _tail11_objects(pd, [1.5, 2.0, 3]).convert_dtypes(),
    ),
    (
        "objects-convert-whole-floats",
        "Series.convert_dtypes",
        ("convert_integer",),
        lambda pd: _tail11_objects(pd, [1.0, 2.0]).convert_dtypes(),
    ),
    (
        "objects-convert-no-whole",
        "Series.convert_dtypes",
        ("convert_integer",),
        lambda pd: _tail11_objects(pd, [1, 2]).convert_dtypes(convert_integer=False),
    ),
    (
        "objects-convert-text",
        "Series.convert_dtypes",
        ("convert_string",),
        lambda pd: _tail11_objects(pd, ["x", "y"]).convert_dtypes(),
    ),
    (
        "objects-convert-text-kept",
        "Series.convert_dtypes",
        ("convert_string",),
        lambda pd: _tail11_objects(pd, ["x", "y"]).convert_dtypes(convert_string=False),
    ),
    (
        "objects-convert-mixed",
        "Series.convert_dtypes",
        ("infer_objects",),
        lambda pd: _tail11_objects(pd, ["x", 1]).convert_dtypes(),
    ),
    (
        "frame-convert-object-column",
        "DataFrame.convert_dtypes",
        ("convert_string",),
        lambda pd: pd.DataFrame(
            {"a": pd.Series(["p", "q"], dtype=object), "b": pd.Series([4, 5], dtype=object)}
        ).convert_dtypes(),
    ),
    (
        "eval-two-lines",
        "DataFrame.eval",
        ("expr",),
        lambda pd: pd.DataFrame({"a": [1, 2], "b": [1.5, 2.5]}).eval("c = a + b\nd = c * 2"),
    ),
    (
        "eval-overwrite-line",
        "DataFrame.eval",
        ("expr",),
        lambda pd: pd.DataFrame({"a": [1, 2], "b": [1.5, 2.5]}).eval("c = a * 3\na = b - 1"),
    ),
    ("eval-lines-in-place", "DataFrame.eval", ("inplace",), lambda pd: _tail11_in_place(pd)),
)


def _tail12_moments(pd):
    """Whole numbers beside three days two apart."""
    return pd.DataFrame(
        {"a": [1, 2, 3], "t": pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-05"])}
    )


TAIL12_CASES = (
    (
        "complex-sizes",
        "Series.abs",
        (),
        lambda pd: pd.Series([3 + 4j, 1j, -2 + 0j], name="z").abs(),
    ),
    ("objects-abs-kept", "Series.abs", (), lambda pd: pd.Series([-1, 2.5, -3], dtype=object).abs()),
    (
        "frame-abs-complex-column",
        "DataFrame.abs",
        (),
        lambda pd: pd.DataFrame({"z": [3 + 4j, 1j], "n": [-1, 2]}).abs(),
    ),
    (
        "quantile-moments-beside",
        "DataFrame.quantile",
        ("numeric_only",),
        lambda pd: _tail12_moments(pd).quantile(0.5, numeric_only=False),
    ),
    (
        "quantile-moments-two",
        "DataFrame.quantile",
        ("q",),
        lambda pd: _tail12_moments(pd).quantile([0.25, 0.75], numeric_only=False),
    ),
)


def _tail13_pattern():
    """A compiled pattern that ignores case."""
    import re

    return re.compile("^f.", re.IGNORECASE)


def _tail13_moment(pd):
    """One afternoon reading, to be rounded to an hour and a half."""
    return pd.Timestamp("2020-03-14 15:32:52")


TAIL13_CASES = (
    (
        "ceil-hour-and-a-half",
        "Timestamp.ceil",
        ("freq",),
        lambda pd: _tail13_moment(pd).ceil("1h30min"),
    ),
    (
        "span-ceil-pieces",
        "Timedelta.ceil",
        ("freq",),
        lambda pd: pd.Timedelta("1h10min").ceil("1h30min"),
    ),
    (
        "labels-ceil-pieces",
        "DatetimeIndex.ceil",
        ("freq",),
        lambda pd: pd.DatetimeIndex(["2020-03-14 15:32", "2020-03-14 17:01"]).ceil("1h30min"),
    ),
    (
        "dt-floor-pieces",
        "Series.dt.floor",
        (),
        lambda pd: pd.Series(pd.to_datetime(["2020-03-14 15:32"])).dt.floor("1h30min"),
    ),
    (
        "dt-round-day-and-hours",
        "Series.dt.round",
        (),
        lambda pd: pd.Series(pd.to_datetime(["2020-03-14 15:32"])).dt.round("1D2h"),
    ),
    (
        "period-nan-text",
        "pandas.Period",
        ("value", "freq"),
        lambda pd: pd.Series([pd.Period("nan", freq="D") is pd.NaT]),
    ),
    (
        "short-moment-name-cast",
        "Series.astype",
        ("dtype",),
        lambda pd: pd.Series(["2015-03-29 02:30:00"]).astype("M8[ns]"),
    ),
    (
        "short-span-name-cast",
        "Series.astype",
        ("dtype",),
        lambda pd: pd.Series([1, 2]).astype("<m8[s]"),
    ),
    (
        "replace-compiled-pattern",
        "Series.str.replace",
        (),
        lambda pd: pd.Series(["foo", "fuz"]).str.replace(_tail13_pattern(), "X", regex=True),
    ),
    (
        "label-counts-repeats-gap",
        "Index.value_counts",
        ("dropna",),
        lambda pd: pd.Index([3, 1, 2, 3, 4, float("nan")]).value_counts(dropna=True),
    ),
    (
        "putmask-index-values",
        "Index.putmask",
        ("mask", "value"),
        lambda pd: pd.Index([1, 2, 3]).putmask([True, False, True], pd.Index([7, 8, 9])),
    ),
    (
        "union-two-widths",
        "Index.union",
        ("other",),
        lambda pd: pd.Index([1, 2], dtype="uint8").union(pd.Index([3, 4])),
    ),
    (
        "difference-keeps-width",
        "Index.difference",
        ("other",),
        lambda pd: pd.Index([1, 2], dtype="int8").difference(pd.Index([2, 4], dtype="uint16")),
    ),
    (
        "labels-nan-among-text",
        "pandas.Index",
        ("data",),
        lambda pd: pd.Index([float("nan"), "var1", float("nan")]),
    ),
)


for _id, _api, _covers, _build in (
    TAIL_CASES
    + TAIL2_CASES
    + TAIL3_CASES
    + TAIL4_CASES
    + TAIL5_CASES
    + TAIL6_CASES
    + TAIL7_CASES
    + TAIL8_CASES
    + TAIL9_CASES
    + TAIL10_CASES
    + TAIL11_CASES
    + TAIL12_CASES
    + TAIL13_CASES
):
    case(
        f"basics/{_id}",
        _api,
        level="L3" if _covers else "L2",
        covers=_covers,
        frames=("single",),
        expr=lambda pd, df, _build=_build: _build(pd),
        in_process=True,
        note="the parameters named, on a small self-built input. " + PARAMETER_IN_PROCESS,
        rules=Rules(tolerance=Tolerance.STATISTICAL, reason="moments and correlations"),
    )
case(
    "basics/multi-index-where-refused",
    "MultiIndex.where",
    level="L4",
    frames=("single",),
    expr=lambda pd, df: _tail_multi(pd).where([True, False, True, True]),
    raises=("NotImplementedError", ".where is not supported for MultiIndex operations"),
    in_process=True,
    note="where on a MultiIndex is refused as pandas refuses it. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/timestamp-to-numpy-copy",
    "Timestamp.to_numpy",
    level="L4",
    frames=("single",),
    expr=lambda pd, df: pd.Timestamp("2024-01-01").to_numpy(copy=True),
    raises=("ValueError", "dtype and copy arguments are ignored"),
    in_process=True,
    note="a copy of one instant is refused as pandas refuses it. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/timedelta-to-numpy-dtype",
    "Timedelta.to_numpy",
    level="L4",
    frames=("single",),
    expr=lambda pd, df: pd.Timedelta("1s").to_numpy(dtype="timedelta64[ms]"),
    raises=("ValueError", "dtype and copy arguments are ignored"),
    in_process=True,
    note="a dtype for one span is refused as pandas refuses it. " + PARAMETER_IN_PROCESS,
)
case(
    "basics/concat-levels-missing-key",
    "pandas.concat",
    level="L4",
    frames=("single",),
    expr=lambda pd, df: pd.concat(
        [pd.Series([1]), pd.Series([2])], keys=["p", "q"], levels=[["q"]]
    ),
    raises=("ValueError", "Values not found in passed level"),
    in_process=True,
    note="a key its given level lacks is refused. " + PARAMETER_IN_PROCESS,
)
