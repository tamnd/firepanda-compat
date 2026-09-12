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
    "hold. The frames with text and dates in them are back now that "
    "`engine/dtype-spelling` is registered, so this case asserts the divergence over the "
    "types that have one and the agreement over the types that do not. " + MEASURING,
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
case("basics/index", "DataFrame.index", frames=SHAPES, expr=lambda pd, df: df.index)
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
    "them are back now that `engine/integer-widening` is registered, and the entry has "
    "a frame list on it, so the eight integer widths are allowed to differ and the two "
    "float widths beside them are not. " + MEASURING,
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
    "than as the number itself, because the number is a divergence and is scored under "
    "`divergences/nbytes`. What is left after the divergence is taken out is still worth "
    "holding, since every rule here can break without the byte count changing at all. " + MEASURING
)

case(
    "basics/series-nbytes",
    "Series.nbytes",
    frames=SHAPES,
    expr=lambda pd, df: df[df.columns[0]].nbytes > 0,
    in_process=True,
    note="whether the column weighs anything, which both libraries agree about even "
    "though they disagree about how much. The empty frame is the half that makes this a "
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
    "answers on both sides, so the two members have to agree with each other even where "
    "neither agrees across the two libraries. " + INVARIANT,
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

case(
    "basics/sum-skipna-false",
    "Series.sum",
    level="L3",
    covers=("skipna",),
    frames=NUMERIC + FLOATS,
    expr=lambda pd, df: df["value"].sum(skipna=False),
)
case(
    "basics/sum-min-count",
    "Series.sum",
    level="L3",
    covers=("min_count",),
    frames=NUMERIC + FLOATS,
    expr=lambda pd, df: df["value"].sum(min_count=1),
    note="min_count is the parameter that makes an all null sum return null instead of zero",
)
case(
    "basics/mean-skipna-false",
    "Series.mean",
    level="L3",
    covers=("skipna",),
    frames=FLOATS,
    expr=lambda pd, df: df["value"].mean(skipna=False),
)
case(
    "basics/sum-tall",
    "Series.sum",
    frames=("tall",),
    expr=lambda pd, df: df["value"].sum(),
    rules=ACCUMULATED,
)
case(
    "basics/mean-tall",
    "Series.mean",
    frames=("tall",),
    expr=lambda pd, df: df["value"].mean(),
    rules=ACCUMULATED,
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
    "read is not written yet. The cases below are the same reductions without it",
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

case(
    "basics/idxmax",
    "Series.idxmax",
    frames=("float64_no_nulls", "int64_no_nulls", "tall"),
    expr=lambda pd, df: df["value"].idxmax(),
    note="the first maximum, and which one is first is the whole content of the case",
)
case(
    "basics/idxmin",
    "Series.idxmin",
    frames=("float64_no_nulls", "int64_no_nulls", "tall"),
    expr=lambda pd, df: df["value"].idxmin(),
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
    "the case that says whether a nan counts as missing, which it does",
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
)
case(
    "basics/round",
    "Series.round",
    level="L3",
    covers=("decimals",),
    frames=FLOATS,
    expr=lambda pd, df: df["value"].round(2),
    note="banker's rounding, and the tolerance is exact on purpose because a rounding "
    "case that tolerates a difference is testing nothing",
    rules=Rules(tolerance=Tolerance.EXACT),
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
)

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
        note="a comparison against a null is false rather than null, which is the "
        "numpy answer and not the SQL one",
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
)
case(
    "basics/rank",
    "Series.rank",
    frames=("keys_10", "float64_half_null"),
    expr=lambda pd, df: df["value"].rank(),
)
case(
    "basics/rank-method-min",
    "Series.rank",
    level="L3",
    covers=("method",),
    frames=("keys_10", "float64_half_null"),
    expr=lambda pd, df: df["value"].rank(method="min"),
)
case(
    "basics/rank-method-dense",
    "Series.rank",
    level="L3",
    covers=("method",),
    frames=("keys_10",),
    expr=lambda pd, df: df["key"].rank(method="dense"),
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
    "basics/assign",
    "DataFrame.assign",
    frames=("two", "tall"),
    expr=lambda pd, df: df.assign(extra=df.iloc[:, 0]),
)
case(
    "basics/insert-by-assignment",
    "DataFrame.__setitem__",
    frames=("two", "tall"),
    expr=lambda pd, df: df.assign(**{"new": 1}),
    note="the assign spelling, because a case that mutates its frame would change the "
    "input of the next case if frames were ever cached",
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
    expr=lambda pd, df: df.sort_values(df.columns[0]).reset_index(drop=True),
)
case(
    "basics/transpose",
    "DataFrame.transpose",
    frames=("single", "two"),
    expr=lambda pd, df: df[[df.columns[0], df.columns[1]]].transpose(),
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
    "hold nothing, and what can be compared about that is which rows they are. The "
    "values cannot be, because pandas widens the column and puts a nan in it where "
    "this keeps the column's type and puts a null in it, which is the divergence spec "
    "48 section 5 argues is the right one and which the harness is right to call a "
    "difference. The integer frame is here because that is where pandas' widening is "
    "visible. Python layer",
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
)
case(
    "basics/unique",
    "Series.unique",
    frames=("keys_10", "keys_awkward", "strings_null_heavy"),
    expr=lambda pd, df: df.iloc[:, 0].unique(),
    note="unique keeps first seen order, which is the part people get wrong",
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
    note="two halves of one column added together. Every value in the answer is null, "
    "because no label appears in both, and the answer is twice as long as either "
    "operand. A user who meant to add them elementwise gets no error at all",
)
case(
    "basics/alignment-frame-add",
    "DataFrame.add",
    frames=("float64_no_nulls",),
    expr=lambda pd, df: _top(df) + _bottom(df),
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
    "basics/alignment-align",
    "DataFrame.align",
    level="L3",
    covers=("other", "join"),
    frames=("float64_no_nulls",),
    expr=lambda pd, df: _top(df).align(_bottom(df), join="outer")[0],
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
