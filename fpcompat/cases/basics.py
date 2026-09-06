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

ACCUMULATED = Rules(
    tolerance=Tolerance.ACCUMULATION,
    reason="a sum over ten thousand doubles depends on the order they were added in",
)

# ---------------------------------------------------------------------------
# Shape and attributes
# ---------------------------------------------------------------------------

case(
    "basics/shape",
    "DataFrame.shape",
    frames=(*SHAPES, "wide"),
    expr=lambda pd, df: df.shape,
    note="the empty frame is here because a zero row shape is where an off by one shows",
)
case("basics/len", "DataFrame.__len__", frames=SHAPES, expr=lambda pd, df: len(df))
case("basics/size", "DataFrame.size", frames=(*SHAPES, "wide"), expr=lambda pd, df: df.size)
case("basics/ndim", "DataFrame.ndim", frames=SHAPES, expr=lambda pd, df: df.ndim)
case(
    "basics/columns",
    "DataFrame.columns",
    frames=(*SHAPES, "wide"),
    expr=lambda pd, df: df.columns,
)
case(
    "basics/dtypes",
    "DataFrame.dtypes",
    frames=(*SHAPES, "temporal_range", "categorical_ordered"),
    expr=lambda pd, df: df.dtypes.astype(str),
    note="compared as strings because a dtype object is not a value the comparison can hold",
)
case("basics/index", "DataFrame.index", frames=SHAPES, expr=lambda pd, df: df.index)
case("basics/empty", "DataFrame.empty", frames=SHAPES, expr=lambda pd, df: df.empty)
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

for name, symbol in (
    ("add", lambda s: s + 1),
    ("sub", lambda s: s - 1),
    ("mul", lambda s: s * 2),
    ("truediv", lambda s: s / 2),
    ("floordiv", lambda s: s // 2),
    ("mod", lambda s: s % 3),
    ("pow", lambda s: s**2),
):
    case(
        f"basics/{name}-scalar",
        f"Series.{name}",
        frames=WIDTHS,
        expr=(lambda op: lambda pd, df: op(df["value"]))(symbol),
        note="every width, because what an int8 does at 127 is not what an int64 does",
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
    frames=FLOATS + NUMERIC,
    expr=lambda pd, df: df["value"].clip(lower=-1, upper=1),
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
    expr=lambda pd, df: df.sort_values("key"),
    note="the awkward frame has nulls and an empty string in the key, so this is where "
    "the null position rule shows",
)
case(
    "basics/sort-values-descending",
    "DataFrame.sort_values",
    level="L3",
    covers=("by", "ascending"),
    frames=("keys_10", "keys_awkward"),
    expr=lambda pd, df: df.sort_values("key", ascending=False),
)
case(
    "basics/sort-values-na-first",
    "DataFrame.sort_values",
    level="L3",
    covers=("by", "na_position"),
    frames=("keys_awkward", "strings_null_heavy"),
    expr=lambda pd, df: df.sort_values(df.columns[0], na_position="first"),
)
case(
    "basics/sort-values-stable",
    "DataFrame.sort_values",
    level="L3",
    covers=("by", "kind"),
    frames=("keys_10", "keys_1000"),
    expr=lambda pd, df: df.sort_values("key", kind="stable"),
    note="ten distinct keys over sixty four rows means every group has ties, so an "
    "unstable sort would be visible here and nowhere else",
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
case(
    "basics/drop-column",
    "DataFrame.drop",
    level="L3",
    covers=("columns",),
    frames=("two", "tall", "wide"),
    expr=lambda pd, df: df.drop(columns=[df.columns[0]]),
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
    frames=FLOATS + NUMERIC,
    expr=lambda pd, df: df["value"].where(df["value"] > 0, -1),
)
case(
    "basics/mask",
    "Series.mask",
    level="L3",
    covers=("cond", "other"),
    frames=FLOATS + NUMERIC,
    expr=lambda pd, df: df["value"].mask(df["value"] > 0, -1),
)
case(
    "basics/replace",
    "Series.replace",
    level="L3",
    covers=("to_replace", "value"),
    frames=("keys_10",),
    expr=lambda pd, df: df["key"].replace(0, 99),
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
