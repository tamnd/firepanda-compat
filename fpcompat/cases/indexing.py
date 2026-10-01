"""Label and position indexing, and the operations that change the index.

Indexing is where the two mental models collide. `iloc` is position and `loc` is
label, and on a default RangeIndex they look identical until somebody sorts the frame,
at which point every case that only ever ran on an unsorted frame was testing nothing.
So most of these run on a frame that has been sorted first, which is the cheap way to
make the label and the position disagree.

The other thing this section exists for is the closed interval. `loc` includes its
right endpoint and `iloc` does not, and that difference is the single most common
source of an off by one in code written by somebody who came from numpy.
"""

from __future__ import annotations

from typing import Any

from fpcompat.cases import case, section
from fpcompat.compare import Rules

section("indexing")

SHAPES = ("single", "two", "tall")
KEYED = ("keys_10", "keys_1000", "keys_unique")

# Sorting by a value column shuffles the index, which is what makes a label and a
# position different things. Every case below that cares about the distinction starts
# with one of these.
STRICT = Rules(strict_index=True)


def _shuffled(df):
    """Sorts by the last column so that the index is no longer in order."""
    return df.sort_values(df.columns[-1])


case(
    "indexing/iloc-scalar",
    "DataFrame.iloc",
    frames=SHAPES,
    expr=lambda pd, df: df.iloc[0, 0],
)
case(
    "indexing/iloc-row",
    "DataFrame.iloc",
    in_process=True,
    frames=SHAPES,
    expr=lambda pd, df: df.iloc[0],
    note=(
        "a row across typed columns is an object column, which the Python surface "
        "holds and the driver cannot spell"
    ),
)
case(
    "indexing/iloc-slice",
    "DataFrame.iloc",
    frames=(*SHAPES, "wide"),
    expr=lambda pd, df: df.iloc[2:5],
    note="the right endpoint is excluded, which is the opposite of what loc does",
)
case(
    "indexing/iloc-negative",
    "DataFrame.iloc",
    frames=SHAPES,
    expr=lambda pd, df: df.iloc[-3:],
)
case(
    "indexing/iloc-step",
    "DataFrame.iloc",
    frames=("tall",),
    expr=lambda pd, df: df.iloc[::7],
)
case(
    "indexing/iloc-list",
    "DataFrame.iloc",
    frames=("two", "tall"),
    expr=lambda pd, df: df.iloc[[0, 0, 1]],
    note="a repeated position gives a repeated row, which the index has to show",
    rules=STRICT,
)
case(
    "indexing/iloc-column",
    "DataFrame.iloc",
    frames=(*SHAPES, "wide"),
    expr=lambda pd, df: df.iloc[:, 1],
)
case(
    "indexing/iloc-both",
    "DataFrame.iloc",
    frames=("wide", "tall"),
    expr=lambda pd, df: df.iloc[3:9, 1:4],
)
case(
    "indexing/loc-slice-closed",
    "DataFrame.loc",
    frames=(*SHAPES, "wide"),
    expr=lambda pd, df: df.loc[2:5],
    note="five rows and not four, because loc includes the label it stops at",
    rules=STRICT,
)
case(
    "indexing/loc-after-sort",
    "DataFrame.loc",
    frames=("tall", "keys_10"),
    expr=lambda pd, df: _shuffled(df).loc[3],
    in_process=True,
    note="sorted first, so the label three and the position three are different rows "
    "and a case that confused them would fail here. In process because firepanda #1103 "
    "reads one row in its Python layer. The tall frame mixes a flag with numbers, which "
    "pandas answers with an object column, so it is refused there and scored as a fail",
)
case(
    "indexing/loc-list-after-sort",
    "DataFrame.loc",
    frames=("tall", "keys_10"),
    expr=lambda pd, df: _shuffled(df).loc[[5, 1, 9]],
    rules=STRICT,
)
case(
    "indexing/loc-mask",
    "DataFrame.loc",
    frames=("tall",),
    expr=lambda pd, df: df.loc[df["value"] > 0],
    rules=STRICT,
)
case(
    "indexing/loc-mask-columns",
    "DataFrame.loc",
    frames=("tall",),
    expr=lambda pd, df: df.loc[df["flag"], ["value", "key"]],
    rules=STRICT,
)
case(
    "indexing/loc-column",
    "DataFrame.loc",
    frames=SHAPES,
    expr=lambda pd, df: df.loc[:, "b" if "b" in df else "value"],
)
case(
    "indexing/at",
    "DataFrame.at",
    frames=SHAPES,
    expr=lambda pd, df: df.at[0, "a" if "a" in df else "key"],
)
case(
    "indexing/iat",
    "DataFrame.iat",
    frames=SHAPES,
    expr=lambda pd, df: df.iat[0, 0],
)
case(
    "indexing/series-iloc",
    "Series.iloc",
    frames=("tall", "float64_half_null"),
    expr=lambda pd, df: df["value"].iloc[4:12],
)
case(
    "indexing/series-loc-after-sort",
    "Series.loc",
    frames=("tall",),
    expr=lambda pd, df: _shuffled(df)["value"].loc[7],
)
case(
    "indexing/series-getitem",
    "Series.__getitem__",
    frames=("tall",),
    expr=lambda pd, df: df["value"][3:8],
)
case(
    "indexing/take",
    "DataFrame.take",
    level="L3",
    covers=("indices",),
    frames=("tall", "keys_10"),
    expr=lambda pd, df: df.take([2, 0, 1]),
    rules=STRICT,
)
case(
    "indexing/take-negative",
    "DataFrame.take",
    level="L3",
    covers=("indices",),
    frames=("tall",),
    expr=lambda pd, df: df.take([-1, -2]),
    rules=STRICT,
)
case(
    "indexing/get",
    "DataFrame.get",
    level="L3",
    covers=("key", "default"),
    frames=SHAPES,
    expr=lambda pd, df: df.get("not_a_column", "missing"),
    in_process=True,
    note="get returns the default rather than raising, which is the only difference "
    "between it and square brackets and the only reason it exists. The answer here "
    "is the default, so a driver entry would have to hold a copy of the value the "
    "case asked for and hand it back, which is the driver scoring itself. See spec 36",
)
case(
    "indexing/squeeze",
    "DataFrame.squeeze",
    frames=("single",),
    expr=lambda pd, df: df[["a"]].squeeze(),
    in_process=True,
    note=(
        "squeeze reads the shape of the frame and decides between three shapes of "
        "answer, which is the whole method and is above the core. A driver entry "
        "would have to make that decision itself to know what to emit. See spec 36"
    ),
)

# ---------------------------------------------------------------------------
# Changing the index
# ---------------------------------------------------------------------------

case(
    "indexing/set-index",
    "DataFrame.set_index",
    level="L3",
    covers=("keys",),
    frames=(*KEYED, "keys_awkward"),
    expr=lambda pd, df: df.set_index("key"),
    rules=STRICT,
)
case(
    "indexing/set-index-drop-false",
    "DataFrame.set_index",
    level="L3",
    covers=("keys", "drop"),
    frames=("keys_10",),
    expr=lambda pd, df: df.set_index("key", drop=False),
    rules=STRICT,
)
case(
    "indexing/set-index-two",
    "DataFrame.set_index",
    level="L3",
    covers=("keys",),
    frames=("keys_two_column",),
    expr=lambda pd, df: df.set_index(["left", "right"]),
    rules=STRICT,
    note="a two level index, which the comparison flattens into two index columns. A "
    "MultiIndex is firepanda's Python layer and the driver has no entry for it, so the case "
    "runs in process",
    in_process=True,
)
case(
    "indexing/set-index-append",
    "DataFrame.set_index",
    level="L3",
    covers=("keys", "append"),
    frames=("keys_two_column",),
    expr=lambda pd, df: df.set_index("left").set_index("right", append=True),
    rules=STRICT,
    note="a MultiIndex is firepanda's Python layer and the driver has no entry for it, "
    "so the case runs in process",
    in_process=True,
)
LABELS_HANDED_IN = (
    "labels handed in rather than a column name are read by position and name the level "
    "after what was handed in. firepanda sets them in its Python layer, so the case runs in "
    "process"
)

case(
    "indexing/set-index-date-range",
    "DataFrame.set_index",
    level="L3",
    covers=("keys",),
    frames=("keys_two_column",),
    expr=lambda pd, df: repr(df.set_index(pd.date_range("2024-01-01", periods=len(df)))),
    in_process=True,
    note="a date range as the row labels, which keeps its step. " + LABELS_HANDED_IN,
)
case(
    "indexing/set-index-series",
    "DataFrame.set_index",
    level="L3",
    covers=("keys",),
    frames=("keys_two_column",),
    expr=lambda pd, df: repr(
        df.set_index(pd.Series(range(len(df)), index=list(range(len(df)))[::-1], name="s"))
    ),
    in_process=True,
    note="a column with labels of its own, read by position rather than aligned. "
    + LABELS_HANDED_IN,
)
case(
    "indexing/set-index-array-and-column",
    "DataFrame.set_index",
    level="L3",
    covers=("keys", "drop"),
    frames=("keys_two_column",),
    expr=lambda pd, df: repr(
        df.set_index(["left", pd.Index(range(len(df)), name="n")], drop=False)
    ),
    in_process=True,
    note="labels beside a column name make two levels. " + LABELS_HANDED_IN,
)
case(
    "indexing/set-index-length-refused",
    "DataFrame.set_index",
    level="L4",
    covers=("keys",),
    frames=("keys_two_column",),
    expr=lambda pd, df: df.set_index(pd.Index([1, 2])),
    raises=("ValueError", "Length mismatch: Expected 8 rows, received array of length 2"),
)
case(
    "indexing/reset-index-levels",
    "DataFrame.reset_index",
    level="L3",
    covers=("level",),
    frames=("keys_two_column",),
    expr=lambda pd, df: df.set_index(["left", "right"]).reset_index(level="right"),
    note="a MultiIndex is firepanda's Python layer and the driver has no entry for it, "
    "so the case runs in process",
    in_process=True,
)
case(
    "indexing/reset-index-two",
    "DataFrame.reset_index",
    frames=("keys_two_column",),
    expr=lambda pd, df: df.set_index(["left", "right"]).reset_index(),
    note="a MultiIndex is firepanda's Python layer and the driver has no entry for it, "
    "so the case runs in process",
    in_process=True,
)
case(
    "indexing/reset-index",
    "DataFrame.reset_index",
    frames=("keys_10",),
    expr=lambda pd, df: df.set_index("key").reset_index(),
)
case(
    "indexing/reset-index-drop",
    "DataFrame.reset_index",
    level="L3",
    covers=("drop",),
    frames=("keys_10",),
    expr=lambda pd, df: df.set_index("key").reset_index(drop=True),
)
case(
    "indexing/reindex",
    "DataFrame.reindex",
    level="L3",
    covers=("index",),
    frames=("keys_unique",),
    expr=lambda pd, df: df.set_index("key").reindex([0, 2, 999]),
    rules=STRICT,
    note="a label that is not there gives a row of nulls rather than an error, and it "
    "widens an integer column to float on the way, which is the surprising part",
)
case(
    "indexing/reindex-fill",
    "DataFrame.reindex",
    level="L3",
    covers=("index", "fill_value"),
    frames=("keys_unique",),
    expr=lambda pd, df: df.set_index("key").reindex([0, 2, 999], fill_value=0),
    rules=STRICT,
)
case(
    "indexing/reindex-columns",
    "DataFrame.reindex",
    level="L3",
    covers=("columns",),
    frames=("two",),
    expr=lambda pd, df: df.reindex(columns=["c", "a", "missing"]),
)
case(
    "indexing/reindex-ffill",
    "DataFrame.reindex",
    level="L3",
    covers=("index", "method", "limit"),
    frames=("keys_unique",),
    expr=lambda pd, df: (
        df.set_index("key")
        .sort_index()
        .reindex([-1, 0, 1, 2, 999, 1000, 1001], method="ffill", limit=1)
    ),
    rules=STRICT,
    in_process=True,
    note="a label that is not there reads from the one before it, at most once per label, "
    "which firepanda #1233 works out in the Python layer before the core reindexes",
)
case(
    "indexing/reindex-nearest",
    "DataFrame.reindex",
    level="L3",
    covers=("index", "method", "tolerance"),
    frames=("keys_unique",),
    expr=lambda pd, df: (
        df.set_index("key")
        .sort_index()
        .reindex([-5, 0, 3, 500, 5000], method="nearest", tolerance=2)
    ),
    rules=STRICT,
    in_process=True,
    note="the closer neighbour, the later one on a tie, and nothing further away than "
    "the tolerance, all worked out in firepanda's Python layer since #1233",
)
case(
    "indexing/xs",
    "DataFrame.xs",
    level="L3",
    covers=("key",),
    frames=("keys_10",),
    expr=lambda pd, df: df.set_index("key").xs(3),
    rules=STRICT,
    note="a MultiIndex is firepanda's Python layer and the driver has no entry for it, "
    "so the case runs in process",
    in_process=True,
)
case(
    "indexing/droplevel",
    "DataFrame.droplevel",
    level="L3",
    covers=("level",),
    frames=("keys_two_column",),
    expr=lambda pd, df: df.set_index(["left", "right"]).droplevel(0),
    rules=STRICT,
    note="a MultiIndex is firepanda's Python layer and the driver has no entry for it, "
    "so the case runs in process",
    in_process=True,
)
case(
    "indexing/swaplevel",
    "DataFrame.swaplevel",
    frames=("keys_two_column",),
    expr=lambda pd, df: df.set_index(["left", "right"]).swaplevel(),
    rules=STRICT,
    note="a MultiIndex is firepanda's Python layer and the driver has no entry for it, "
    "so the case runs in process",
    in_process=True,
)
case(
    "indexing/sort-index-multi",
    "DataFrame.sort_index",
    frames=("keys_two_column",),
    expr=lambda pd, df: df.set_index(["left", "right"]).sort_index(),
    rules=STRICT,
    note="a MultiIndex is firepanda's Python layer and the driver has no entry for it, "
    "so the case runs in process",
    in_process=True,
)
case(
    "indexing/sort-index-level",
    "DataFrame.sort_index",
    frames=("keys_two_column",),
    expr=lambda pd, df: df.set_index(["left", "right"]).sort_index(level="right"),
    rules=STRICT,
    note="a MultiIndex is firepanda's Python layer and the driver has no entry for it, "
    "so the case runs in process",
    in_process=True,
)
case(
    "indexing/reorder-levels",
    "DataFrame.reorder_levels",
    frames=("keys_two_column",),
    expr=lambda pd, df: df.set_index(["left", "right"]).reorder_levels(["right", "left"]),
    rules=STRICT,
    note="a MultiIndex is firepanda's Python layer and the driver has no entry for it, "
    "so the case runs in process",
    in_process=True,
)
case(
    "indexing/loc-prefix",
    "DataFrame.loc",
    frames=("keys_two_column",),
    expr=lambda pd, df: df.set_index(["left", "right"]).sort_index().loc["y"],
    rules=STRICT,
    note="a MultiIndex is firepanda's Python layer and the driver has no entry for it, "
    "so the case runs in process",
    in_process=True,
)
case(
    "indexing/series-xs-level",
    "Series.xs",
    frames=("keys_two_column",),
    expr=lambda pd, df: df.set_index(["left", "right"])["value"].xs(1, level="right"),
    rules=STRICT,
    note="a MultiIndex is firepanda's Python layer and the driver has no entry for it, "
    "so the case runs in process",
    in_process=True,
)

# ---------------------------------------------------------------------------
# Duplicates, filters and the top of the frame
# ---------------------------------------------------------------------------

case(
    "indexing/duplicated",
    "DataFrame.duplicated",
    frames=(*KEYED, "keys_awkward"),
    expr=lambda pd, df: df.duplicated(subset=[df.columns[0]]),
    level="L3",
    covers=("subset",),
)
case(
    "indexing/duplicated-keep-last",
    "DataFrame.duplicated",
    level="L3",
    covers=("subset", "keep"),
    frames=("keys_10", "keys_awkward"),
    expr=lambda pd, df: df.duplicated(subset=[df.columns[0]], keep="last"),
)
case(
    "indexing/duplicated-keep-false",
    "DataFrame.duplicated",
    level="L3",
    covers=("subset", "keep"),
    frames=("keys_10",),
    expr=lambda pd, df: df.duplicated(subset=["key"], keep=False),
)
case(
    "indexing/drop-duplicates",
    "DataFrame.drop_duplicates",
    level="L3",
    covers=("subset",),
    frames=(*KEYED, "keys_awkward"),
    expr=lambda pd, df: df.drop_duplicates(subset=[df.columns[0]]),
    rules=STRICT,
    note="which row survives is the whole content, so the index is compared strictly",
)
case(
    "indexing/nlargest",
    "DataFrame.nlargest",
    level="L3",
    covers=("n", "columns"),
    frames=("tall", "keys_1000"),
    expr=lambda pd, df: df.nlargest(5, "value"),
    rules=STRICT,
)
case(
    "indexing/nsmallest",
    "DataFrame.nsmallest",
    level="L3",
    covers=("n", "columns"),
    frames=("tall", "keys_1000"),
    expr=lambda pd, df: df.nsmallest(5, "value"),
    rules=STRICT,
)
case(
    "indexing/nlargest-keep-last",
    "DataFrame.nlargest",
    level="L3",
    covers=("n", "columns", "keep"),
    frames=("keys_10",),
    expr=lambda pd, df: df.nlargest(4, "key", keep="last"),
    rules=STRICT,
    note="ten keys over sixty four rows means the fourth largest is a tie, which is "
    "exactly when keep starts to matter",
)
QUERY = (
    "In process because the driver has no entry for it, and firepanda's query is its Python "
    "layer reading the expression with Python's own parser and working each node out over "
    "whole columns"
)
case(
    "indexing/query",
    "DataFrame.query",
    level="L3",
    covers=("expr",),
    frames=("tall", "keys_1000"),
    expr=lambda pd, df: df.query("key > 3"),
    rules=STRICT,
    in_process=True,
    note=QUERY,
)
case(
    "indexing/query-and",
    "DataFrame.query",
    level="L3",
    covers=("expr",),
    frames=("tall",),
    expr=lambda pd, df: df.query("key > 3 and value < 0"),
    rules=STRICT,
    in_process=True,
    note=QUERY,
)
case(
    "indexing/filter-like",
    "DataFrame.filter",
    level="L3",
    covers=("like",),
    frames=("wide",),
    expr=lambda pd, df: df.filter(like="01"),
    in_process=True,
    note=(
        "filter matches on the column names rather than on the columns, so the whole "
        "of it is above the core and the driver has nothing to call. See spec 36"
    ),
)
case(
    "indexing/filter-regex",
    "DataFrame.filter",
    level="L3",
    covers=("regex",),
    frames=("wide",),
    expr=lambda pd, df: df.filter(regex=r"^c00[0-4]$"),
    in_process=True,
    note=(
        "the regex rule is Python's re module applied to the column names, and Mojo "
        "has no regex engine for the driver to be wrong with. See spec 36"
    ),
)
case(
    "indexing/filter-items",
    "DataFrame.filter",
    level="L3",
    covers=("items",),
    frames=("two",),
    expr=lambda pd, df: df.filter(items=["c", "a"]),
    in_process=True,
    note=(
        "the same rule as the other two filter cases, and the one that keeps the "
        "frame's order rather than the caller's. See spec 36"
    ),
)
case(
    "indexing/select-dtypes",
    "DataFrame.select_dtypes",
    level="L3",
    covers=("include",),
    frames=("two", "tall", "temporal_range"),
    expr=lambda pd, df: df.select_dtypes(include="number"),
    in_process=True,
    note=(
        "the numpy type tree this matches against is a pandas compatibility rule and "
        "not a dataframe operation, so it lives in the Python layer and a driver "
        "branch would have to carry a second copy of it in Mojo. See spec 36"
    ),
)
case(
    "indexing/truncate",
    "DataFrame.truncate",
    level="L3",
    covers=("before", "after"),
    frames=("tall",),
    expr=lambda pd, df: df.truncate(before=10, after=20),
    rules=STRICT,
    in_process=True,
    note=(
        "truncate turns its two bounds into a slice through Index.slice_indexer, "
        "which is a Python layer method, so the core has no truncate to call. "
        "See spec 36"
    ),
)
case(
    "indexing/index-unique",
    "Index.unique",
    frames=("keys_10",),
    expr=lambda pd, df: df.set_index("key").index.unique(),
)
case(
    "indexing/index-is-unique",
    "Index.is_unique",
    frames=KEYED,
    expr=lambda pd, df: df.set_index("key").index.is_unique,
)
case(
    "indexing/index-monotonic",
    "Index.is_monotonic_increasing",
    frames=KEYED,
    expr=lambda pd, df: df.set_index("key").index.is_monotonic_increasing,
)
case(
    "indexing/index-get-loc",
    "Index.get_loc",
    level="L3",
    covers=("key",),
    frames=("keys_unique",),
    expr=lambda pd, df: df.set_index("key").index.get_loc(5),
)
case(
    "indexing/index-searchsorted",
    "Index.searchsorted",
    level="L3",
    covers=("value",),
    frames=("keys_unique",),
    expr=lambda pd, df: df.set_index("key").sort_index().index.searchsorted(5),
    in_process=True,
    note="an index read out of a frame is searched by the index's own kernel",
)
INDEX_THROUGH_A_COLUMN = (
    "In process because firepanda answers this for an index by reading the labels as a "
    "column in its Python layer, which the driver cannot reach"
)
case(
    "indexing/index-value-counts",
    "Index.value_counts",
    frames=("keys_10", "keys_awkward"),
    expr=lambda pd, df: df.set_index("key").index.value_counts(),
    in_process=True,
    note="counted as the column of labels would be, the index name on the answer. "
    + INDEX_THROUGH_A_COLUMN,
)
case(
    "indexing/index-argmax",
    "Index.argmax",
    frames=("keys_1000",),
    expr=lambda pd, df: df.set_index("value").index.argmax(),
    in_process=True,
    note="the first largest label's position. " + INDEX_THROUGH_A_COLUMN,
)
case(
    "indexing/index-where",
    "Index.where",
    level="L3",
    covers=("cond", "other"),
    frames=("keys_10",),
    expr=lambda pd, df: df.set_index("value").index.where((df["key"] > 4).tolist(), -1),
    in_process=True,
    note="the labels where the condition holds and minus one elsewhere. " + INDEX_THROUGH_A_COLUMN,
)
case(
    "indexing/index-isin",
    "Index.isin",
    level="L3",
    covers=("values",),
    frames=("keys_10",),
    expr=lambda pd, df: df.set_index("key").index.isin([1, 2, 3]),
)
# An index and a column hold the same thing in firepanda, so `Index.to_series` is the
# door between them and the eight names under it are that door plus a column method.
# Every one of them lives in firepanda's Python layer, which is what the flag on each
# of them is about: the core has no `to_series` to call, so a driver entry would have
# to compose the same series itself out of the labels and then go on to compose the
# answer, which is the driver writing the method it is scoring.
AS_A_COLUMN = (
    "the index members that read as a column are firepanda's Python layer on top of "
    "one door, and the core has no call for any of them, so a driver entry would have "
    "to build the column and then write the method itself. See spec 36"
)
case(
    "indexing/index-to-series",
    "Index.to_series",
    frames=("keys_10",),
    expr=lambda pd, df: df.set_index("key").index.to_series(),
    in_process=True,
    note=AS_A_COLUMN,
)
case(
    "indexing/index-to-series-given-both",
    "Index.to_series",
    level="L3",
    covers=("index", "name"),
    frames=("keys_10",),
    expr=lambda pd, df: df.set_index("key").index.to_series(
        index=list(range(len(df))), name="labels"
    ),
    in_process=True,
    note="the labels come back twice unless the caller says otherwise, which is what "
    "these two parameters are for. " + AS_A_COLUMN,
)
case(
    "indexing/index-isna",
    "Index.isna",
    frames=("int64_half_null",),
    expr=lambda pd, df: list(df.set_index("value").index.isna()),
    in_process=True,
    note=AS_A_COLUMN,
)
case(
    "indexing/index-isnull",
    "Index.isnull",
    frames=("int64_half_null",),
    expr=lambda pd, df: list(df.set_index("value").index.isnull()),
    in_process=True,
    note="the older spelling of isna, which pandas keeps and so does this. " + AS_A_COLUMN,
)
case(
    "indexing/index-notna",
    "Index.notna",
    frames=("int64_half_null",),
    expr=lambda pd, df: list(df.set_index("value").index.notna()),
    in_process=True,
    note=AS_A_COLUMN,
)
case(
    "indexing/index-notnull",
    "Index.notnull",
    frames=("int64_half_null",),
    expr=lambda pd, df: list(df.set_index("value").index.notnull()),
    in_process=True,
    note="the older spelling of notna. " + AS_A_COLUMN,
)
case(
    "indexing/index-dropna",
    "Index.dropna",
    level="L3",
    covers=("how",),
    frames=("float64_half_null",),
    expr=lambda pd, df: df.set_index("value").index.dropna(how="any"),
    in_process=True,
    note="how is any or all and on a flat index the two mean the same thing, since a "
    "label is one value. The frame here is the float one where the others are the "
    "integer one, because this is the only one of them that hands back labels and "
    "therefore a dtype, and a half null integer column is read as int64 by one engine "
    "and as float64 by the other before dropna is reached, so the integer frame would "
    "be scoring the read path under this name. " + AS_A_COLUMN,
)
case(
    "indexing/index-min",
    "Index.min",
    frames=KEYED,
    expr=lambda pd, df: df.set_index("key").index.min(),
    in_process=True,
    note=AS_A_COLUMN,
)
case(
    "indexing/index-max",
    "Index.max",
    frames=KEYED,
    expr=lambda pd, df: df.set_index("key").index.max(),
    in_process=True,
    note=AS_A_COLUMN,
)
case(
    "indexing/index-nunique",
    "Index.nunique",
    level="L3",
    covers=("dropna",),
    frames=("int64_half_null",),
    expr=lambda pd, df: df.set_index("value").index.nunique(dropna=False),
    in_process=True,
    note="a missing label counts as one more distinct label here, which is the "
    "parameter the column underneath still refuses. " + AS_A_COLUMN,
)
case(
    "indexing/assign-misaligned-column",
    "DataFrame.__setitem__",
    frames=("tall",),
    expr=lambda pd, df: (
        lambda copy: (copy.__setitem__("shifted", copy["value"].tail(len(copy) - 2)), copy)[1]
    )(df.copy()),
    in_process=True,
    note="assigning a shorter column back into the frame lines it up on the labels "
    "rather than on position, so the two rows it does not cover come back null instead "
    "of the column landing at the top. This is the one that surprises people who have "
    "used pandas for years. It was a divergence case until firepanda started aligning",
)


def _written(df: Any, key: Any, value: Any) -> Any:
    """A copy of the frame after `copy[key] = value`, where either may read the copy."""
    copy = df.copy()
    copy[key(copy) if callable(key) else key] = value(copy) if callable(value) else value
    return copy


def _numbers(df: Any) -> Any:
    return df[["key", "value"]]


case(
    "indexing/setitem-scalar",
    "DataFrame.__setitem__",
    frames=("tall",),
    expr=lambda pd, df: _written(df, "one", 1),
    in_process=True,
    note="one value for a new column is spread down every row",
)
case(
    "indexing/setitem-replace",
    "DataFrame.__setitem__",
    frames=("tall",),
    expr=lambda pd, df: _written(df, "value", lambda c: c["value"] * 2),
    in_process=True,
    note="writing over a column keeps its place among the others",
)
case(
    "indexing/setitem-several",
    "DataFrame.__setitem__",
    frames=("tall",),
    expr=lambda pd, df: _written(df, ["a", "b"], lambda c: c[["value", "key"]]),
    in_process=True,
    note="a frame written under a list of names is taken by position, not by its own names",
)
case(
    "indexing/setitem-marked-rows",
    "DataFrame.__setitem__",
    frames=("tall",),
    expr=lambda pd, df: _written(_numbers(df), lambda c: c["key"] > 50, 0),
    in_process=True,
    note="a column of flags as the key writes the value into every column of the rows it marks",
)
case(
    "indexing/setitem-marked-cells",
    "DataFrame.__setitem__",
    frames=("tall",),
    expr=lambda pd, df: _written(_numbers(df), lambda c: c > 50, -1),
    in_process=True,
    note="a frame of flags as the key writes the value into the cells it marks, as mask does",
)


def _put(target: Any, accessor: str | None, key: Any, value: Any) -> Any:
    """A copy after `copy.<accessor>[key] = value`, or `copy[key] = value` with no accessor."""
    copy = target.copy()
    place = copy if accessor is None else getattr(copy, accessor)
    place[key(copy) if callable(key) else key] = value(copy) if callable(value) else value
    return copy


case(
    "indexing/series-setitem-mask",
    "Series.__setitem__",
    frames=("tall",),
    expr=lambda pd, df: _put(df["value"], None, lambda s: s > 50, 0.0),
    in_process=True,
    note="one value written into the rows a mask marks, the type kept",
)
case(
    "indexing/series-setitem-gap-widens",
    "Series.__setitem__",
    frames=("tall",),
    expr=lambda pd, df: _put(df["key"], None, lambda s: s > 50, None).fillna(-1.0),
    in_process=True,
    note="a gap written into some rows of whole numbers makes the column float64. The gaps "
    "are filled before comparing, since firepanda spells a float gap as null and pandas as NaN",
)
case(
    "indexing/series-loc-enlarge",
    "Series.loc",
    frames=("tall",),
    expr=lambda pd, df: _put(df["key"].head(3), "loc", 99, 7),
    in_process=True,
    note="a label the series does not have puts a new row on the end",
)
case(
    "indexing/series-iloc-write",
    "Series.iloc",
    frames=("tall",),
    expr=lambda pd, df: _put(df["value"], "iloc", [4, 0, 2], [1.0, 2.0, 3.0]),
    in_process=True,
    note="a list of values goes into a list of positions in the order given",
)
case(
    "indexing/series-iat-write",
    "Series.iat",
    frames=("tall",),
    expr=lambda pd, df: _put(df["key"], "iat", 1, 5),
    in_process=True,
    note="one value written by position",
)
case(
    "indexing/loc-write-cells",
    "DataFrame.loc",
    frames=("tall",),
    expr=lambda pd, df: _put(df, "loc", lambda c: (c["key"] > 50, "value"), 0.0),
    in_process=True,
    note="the rows a mask marks, in one column",
)
case(
    "indexing/loc-write-new-column",
    "DataFrame.loc",
    frames=("tall",),
    expr=lambda pd, df: _put(df, "loc", lambda c: (c["key"] > 50, "big"), 1.0),
    in_process=True,
    note="a name the frame does not have makes a new column, a gap in the rows not marked",
)
case(
    "indexing/loc-write-new-row",
    "DataFrame.loc",
    frames=("tall",),
    expr=lambda pd, df: _put(_numbers(df).head(3), "loc", 99, [1, 2.5]),
    in_process=True,
    note="a row label the frame does not have puts a row on the end, one value a column",
)
case(
    "indexing/iloc-write-block",
    "DataFrame.iloc",
    frames=("tall",),
    expr=lambda pd, df: _put(_numbers(df), "iloc", ([0, 1], [0, 1]), [[7, 8.0], [9, 10.0]]),
    in_process=True,
    note="rows of values go one row of them a row",
)
case(
    "indexing/at-write",
    "DataFrame.at",
    frames=("tall",),
    expr=lambda pd, df: _put(df, "at", (2, "value"), 0.5),
    in_process=True,
    note="one cell written by its row label and column name",
)
case(
    "indexing/iat-write",
    "DataFrame.iat",
    frames=("tall",),
    expr=lambda pd, df: _put(df, "iat", (0, 0), 5),
    in_process=True,
    note="one cell written by its row and column positions",
)


# ---------------------------------------------------------------------------
# The members that read as a frame, on the column and on the index
# ---------------------------------------------------------------------------

AS_A_FRAME = (
    "the members that read as a frame are firepanda's Python layer on top of one door, "
    "and the core has no call for any of them on a column, so a driver entry would have "
    "to build the frame of one column and then write the method itself. See spec 36"
)

case(
    "indexing/series-take",
    "Series.take",
    frames=("keys_10",),
    expr=lambda pd, df: df["key"].take([0, 5, 2]),
    in_process=True,
    note=AS_A_FRAME,
)
case(
    "indexing/series-sort-index",
    "Series.sort_index",
    frames=("keys_unique",),
    expr=lambda pd, df: df.set_index("value")["key"].sort_index(),
    in_process=True,
    note="the labels are the value column of the unique frame, which arrives in no order "
    "and has no two labels the same, so the sort has real work and no tie for the two "
    "libraries to settle differently. Sorting a column by its own values rather than by "
    "its labels would read better here and is not written yet, on the column or on the "
    "frame. " + AS_A_FRAME,
)
case(
    "indexing/series-sort-index-descending",
    "Series.sort_index",
    level="L3",
    covers=("ascending",),
    frames=("keys_unique",),
    expr=lambda pd, df: df.set_index("value")["key"].sort_index(ascending=False),
    in_process=True,
    note=AS_A_FRAME,
)
case(
    "indexing/series-reset-index",
    "Series.reset_index",
    frames=("keys_10",),
    expr=lambda pd, df: df.set_index("key")["value"].reset_index(),
    in_process=True,
    note="the labels kept, which makes a frame of two columns rather than a column, "
    "because the labels have become values. " + AS_A_FRAME,
)
case(
    "indexing/series-reset-index-drop",
    "Series.reset_index",
    level="L3",
    covers=("drop",),
    frames=("keys_10",),
    expr=lambda pd, df: df.set_index("key")["value"].reset_index(drop=True),
    in_process=True,
    note="the same call with the labels dropped, which answers a column, and the two "
    "answers being different types is pandas rather than an invention here. " + AS_A_FRAME,
)
case(
    "indexing/series-truncate",
    "Series.truncate",
    frames=("keys_10",),
    expr=lambda pd, df: df["key"].truncate(),
    in_process=True,
    note="no bounds at all, which keeps every row and is the default this level is for. "
    + AS_A_FRAME,
)
case(
    "indexing/series-truncate-between",
    "Series.truncate",
    level="L3",
    covers=("before", "after"),
    frames=("keys_10",),
    expr=lambda pd, df: df["key"].truncate(10, 20),
    in_process=True,
    note="both ends are kept, which is the one thing about this method that surprises "
    "people who read it as a slice. " + AS_A_FRAME,
)
case(
    "indexing/index-to-frame",
    "Index.to_frame",
    frames=("keys_10",),
    expr=lambda pd, df: df.set_index("key").index.to_frame(),
    in_process=True,
    note="two doors end to end, the labels into a column and the column into a frame. "
    + AS_A_FRAME,
)
case(
    "indexing/index-to-frame-given-both",
    "Index.to_frame",
    level="L3",
    covers=("index", "name"),
    frames=("keys_10",),
    expr=lambda pd, df: df.set_index("key").index.to_frame(index=False, name="labels"),
    in_process=True,
    note="whether the labels stay on as labels as well is the choice this method exists "
    "to offer. " + AS_A_FRAME,
)
case(
    "indexing/index-duplicated",
    "Index.duplicated",
    frames=("keys_10",),
    expr=lambda pd, df: list(df.set_index("key").index.duplicated()),
    in_process=True,
    note=AS_A_FRAME,
)
case(
    "indexing/index-duplicated-keep",
    "Index.duplicated",
    level="L3",
    covers=("keep",),
    frames=("keys_10",),
    expr=lambda pd, df: list(df.set_index("key").index.duplicated(keep="last")),
    in_process=True,
    note=AS_A_FRAME,
)
case(
    "indexing/index-drop-duplicates",
    "Index.drop_duplicates",
    frames=("keys_10",),
    expr=lambda pd, df: df.set_index("key").index.drop_duplicates(),
    in_process=True,
    note=AS_A_FRAME,
)
case(
    "indexing/index-drop-duplicates-keep",
    "Index.drop_duplicates",
    level="L3",
    covers=("keep",),
    frames=("keys_10",),
    expr=lambda pd, df: df.set_index("key").index.drop_duplicates(keep="last"),
    in_process=True,
    note=AS_A_FRAME,
)

# ---------------------------------------------------------------------------
# The index put in order of its own labels
# ---------------------------------------------------------------------------

# Sorting is the first member of this family where the core has the operation and
# still cannot answer the case. `Index` in the core has no sort of its own: what
# firepanda does is turn the labels into a column, sort the column, and turn the
# answer back into an index of the class it started as. A driver entry would have
# to write those three steps, which is the method rather than a spelling of it, so
# these run in process for the reason the family above does. See spec 43.
IN_ORDER = (
    "an index in the core has no sort of its own and firepanda's is the column's sort "
    "with a door on each end, so a driver entry would have to write the method it is "
    "scoring. See spec 43"
)

case(
    "indexing/index-sort-values",
    "Index.sort_values",
    frames=("keys_unique",),
    expr=lambda pd, df: df.set_index("value").index.sort_values(),
    in_process=True,
    note="the value column of the unique frame, which arrives in no order and has no "
    "two labels the same, so the sort has real work in it and no tie for the two "
    "libraries to settle differently. " + IN_ORDER,
)
case(
    "indexing/index-sort-values-descending",
    "Index.sort_values",
    level="L3",
    covers=("ascending",),
    frames=("keys_unique",),
    expr=lambda pd, df: df.set_index("value").index.sort_values(ascending=False),
    in_process=True,
    note=IN_ORDER,
)
case(
    "indexing/index-sort-values-indexer",
    "Index.sort_values",
    level="L3",
    covers=("return_indexer",),
    frames=("keys_unique",),
    expr=lambda pd, df: list(df.set_index("value").index.sort_values(return_indexer=True)[1]),
    in_process=True,
    note="the permutation rather than the sorted index, which is the second of the two "
    "shapes this method answers in and the reason it is the one member of the family "
    "whose return type depends on an argument. The list is around it because pandas "
    "answers a numpy array here and firepanda answers a list, which is the shape the "
    "whole index family already differs in. " + IN_ORDER,
)
case(
    "indexing/index-argsort",
    "Index.argsort",
    frames=("keys_unique",),
    expr=lambda pd, df: list(df.set_index("value").index.argsort()),
    in_process=True,
    note=IN_ORDER,
)

A_COPY = (
    "the index is the one thing in the library a copy of can be told from its "
    "original, with is_, so firepanda builds a new index here rather than handing "
    "back another wrapper on the same one the way the frame and the column do. The "
    "renaming that does it is a Python layer call, so a driver entry would have to "
    "write the method it is scoring. See spec 44"
)

case(
    "indexing/index-copy",
    "Index.copy",
    frames=("keys_unique", "keys_awkward"),
    expr=lambda pd, df: df.set_index("value").index.copy(),
    in_process=True,
    note=A_COPY,
)
case(
    "indexing/index-copy-named",
    "Index.copy",
    level="L3",
    covers=("name",),
    frames=("keys_unique",),
    expr=lambda pd, df: df.set_index("value").index.copy(name="row").name,
    in_process=True,
    note="the name rather than the labels, because renaming on the way out is the "
    "only thing this parameter does and the labels would be the same either way. " + A_COPY,
)
case(
    "indexing/index-copy-deep",
    "Index.copy",
    level="L3",
    covers=("deep",),
    frames=("keys_unique",),
    expr=lambda pd, df: df.set_index("value").index.copy(deep=True),
    in_process=True,
    note="pandas documents deep as having no effect on an index, because an index is "
    "immutable once built and there is no later write for a deep copy to protect the "
    "original from. firepanda accepts it and does not read it for the same reason, so "
    "this asks both libraries for the answer they both say is the same one. " + A_COPY,
)
case(
    "indexing/index-copy-is-not-the-original",
    "Index.is_",
    frames=("keys_unique",),
    expr=lambda pd, df: (lambda index: [index.is_(index), index.copy().is_(index)])(
        df.set_index("value").index
    ),
    in_process=True,
    note="the one question in either library that can tell a copy from the object it "
    "was taken from, asked both ways round so that a method answering False to "
    "everything would not pass. It is here rather than beside the copy cases because "
    "the board scores the name being called and this one calls is_",
)


def _renamed_in_place(index, wanted):
    """Renames a level in place and answers what came back along with the name left behind.

    Returns:
        A two item list of the return value, which is None in both libraries
        because the whole point of inplace is that there is nothing to assign,
        and the name the index was left holding afterwards.
    """
    answered = index.rename(wanted, inplace=True)
    return [answered, index.name]


case(
    "indexing/index-rename-inplace",
    "Index.rename",
    level="L3",
    covers=("inplace",),
    frames=("keys_unique",),
    expr=lambda pd, df: _renamed_in_place(df.set_index("value").index.copy(), "row"),
    in_process=True,
    note="the one inplace parameter firepanda honours rather than refusing, which is "
    "why this case is here and not in the inplace divergence block with the other "
    "forty two. A level name is not data, renaming one does not move a single label, "
    "and pandas treats an index as mutable in that one respect for the same reason. "
    "The copy first is so that the frame the case was handed is not renamed underneath "
    "it. " + A_COPY,
)

A_LEVEL_NAME = (
    "the name an index level is under, which is metadata and not data. Changing it "
    "moves no labels, invalidates no lookup and reads no values, which is why it is "
    "here while mapping the labels themselves is not. See spec 45"
)

case(
    "indexing/rename-axis",
    "DataFrame.rename_axis",
    frames=("keys_unique",),
    expr=lambda pd, df: df.set_index("key").rename_axis("row"),
    rules=STRICT,
    note="the whole frame rather than just the name it was given, so that a rename_axis "
    "which quietly dropped a column or reordered the rows would not pass. " + A_LEVEL_NAME,
)
case(
    "indexing/rename-axis-cleared",
    "DataFrame.rename_axis",
    level="L3",
    covers=("mapper",),
    frames=("keys_unique",),
    expr=lambda pd, df: df.set_index("key").rename_axis(None),
    rules=STRICT,
    note="None is a name to clear and a missing argument is not, which is why the "
    "parameter defaults to a sentinel in both libraries rather than to None. " + A_LEVEL_NAME,
)
case(
    "indexing/rename-axis-keyword",
    "DataFrame.rename_axis",
    level="L3",
    covers=("index",),
    frames=("keys_unique",),
    expr=lambda pd, df: df.set_index("key").rename_axis(index=["row"]),
    rules=STRICT,
    in_process=True,
    note="the keyword spelling, and the sequence form of the name with it, because "
    "pandas takes both a name and a sequence of names here and the sequence is the one "
    "that generalises to a multi level index. Working out which of the two shapes was "
    "given is the Python layer's, so a driver entry would have to unwrap the list "
    "itself. " + A_LEVEL_NAME,
)
case(
    "indexing/series-rename-axis",
    "Series.rename_axis",
    frames=("keys_unique",),
    expr=lambda pd, df: df.set_index("key")["value"].rename_axis("row"),
    rules=STRICT,
    in_process=True,
    note="a column carries its frame's labels, so naming them is the same operation "
    "one level down. Reaching the column of a keyed frame is a Python layer walk here, "
    "so a driver entry would have to write it. " + A_LEVEL_NAME,
)
case(
    "indexing/index-names",
    "Index.names",
    frames=("keys_unique",),
    expr=lambda pd, df: list(df.set_index("key").index.names),
    in_process=True,
    note="the level names as a list, which for a flat index is one long. It is the "
    "property set_names is the setter for, and four of the cases in the inplace "
    "divergence block reported a missing attribute rather than the refusal they were "
    "registered for until it existed. " + A_LEVEL_NAME,
)
case(
    "indexing/index-set-names",
    "Index.set_names",
    frames=("keys_unique",),
    expr=lambda pd, df: df.set_index("key").index.set_names("row"),
    in_process=True,
    note="the returning form, which is the one a caller should use. The inplace form is "
    "in the divergence block, because it is one of the two places in the library where "
    "inplace is honoured rather than refused. " + A_LEVEL_NAME + ". " + A_COPY,
)
case(
    "indexing/index-set-names-sequence",
    "Index.set_names",
    level="L3",
    covers=("names",),
    frames=("keys_unique",),
    expr=lambda pd, df: list(df.set_index("key").index.set_names(["row"]).names),
    in_process=True,
    note="a one element sequence rather than a bare name, which pandas accepts and "
    "which is the shape that generalises to more than one level. The names read back "
    "rather than the index, because what this is checking is that the sequence was "
    "unwrapped and not stored whole. " + A_LEVEL_NAME,
)


def _multi_named_in_place(pd, df, call):
    """Renames the levels of a MultiIndex in place and answers what came back and the names.

    Returns:
        A two item list of the return value, None in both libraries, and the level
        names the index was left holding afterwards.
    """
    index = pd.MultiIndex.from_frame(df[["left", "right"]])
    answered = call(index)
    return [answered, list(index.names)]


case(
    "indexing/multi-index-rename-inplace",
    "MultiIndex.rename",
    level="L3",
    covers=("inplace",),
    frames=("keys_two_column",),
    expr=lambda pd, df: _multi_named_in_place(
        pd, df, lambda index: index.rename(["one", "two"], inplace=True)
    ),
    in_process=True,
    note="the MultiIndex half of the inplace naming pair, which sat in the inplace "
    "divergence block until firepanda had a MultiIndex to rename. It is checked the "
    "way indexing/index-rename-inplace is, by what came back as well as the names left "
    "behind, and it runs in process because the MultiIndex is a Python layer class",
)
case(
    "indexing/multi-index-set-names-inplace",
    "MultiIndex.set_names",
    level="L3",
    covers=("inplace",),
    frames=("keys_two_column",),
    expr=lambda pd, df: _multi_named_in_place(
        pd, df, lambda index: index.set_names(["one", "two"], inplace=True)
    ),
    in_process=True,
    note="the same door as indexing/multi-index-rename-inplace, spelled set_names, "
    "and it runs in process for the same reason",
)


def _set_names_in_place(index, wanted):
    """Sets a level name in place and answers what came back along with the names left.

    Returns:
        A two item list of the return value, which is None in both libraries
        because the whole point of inplace is that there is nothing to assign,
        and the level names the index was left holding afterwards.
    """
    answered = index.set_names(wanted, inplace=True)
    return [answered, list(index.names)]


case(
    "indexing/index-set-names-inplace",
    "Index.set_names",
    level="L3",
    covers=("inplace",),
    frames=("keys_unique",),
    expr=lambda pd, df: _set_names_in_place(df.set_index("key").index.copy(), "row"),
    in_process=True,
    note="the second of the four inplace parameters firepanda honours rather than "
    "refusing, which is why this is here and not in the inplace divergence block. It "
    "goes through the same door as indexing/index-rename-inplace and is checked the "
    "same way, by asking for what came back as well as the names left behind, because "
    "a method that returned the index rather than None would otherwise look right. "
    + A_LEVEL_NAME
    + ". "
    + A_COPY,
)

case(
    "indexing/series-sort-values-na-first",
    "Series.sort_values",
    level="L3",
    covers=("na_position",),
    frames=("keys_awkward",),
    expr=lambda pd, df: df["key"].sort_values(na_position="first"),
    note="the awkward key column, which holds real nulls and an empty string, so the "
    "front of this answer has both a missing value and the smallest value that is not "
    "one. The half null float frames are not used here because their missing rows are "
    "a mix of nulls and NaNs, and pandas cannot tell those apart while firepanda can, "
    "which is a divergence this case has no business measuring. Ten rows, so numpy "
    "falls back to insertion sort and settles the ties stably by accident, which is "
    "the same reason basics/sort-values-na-first gives",
)
case(
    "indexing/series-sort-values-ignore-index",
    "Series.sort_values",
    level="L3",
    covers=("ignore_index",),
    frames=("keys_awkward",),
    expr=lambda pd, df: df["key"].sort_values(ignore_index=True),
    in_process=True,
    note="numbering the rows again after the sort, which in firepanda is a reset_index "
    "on the answer and lives in the Python layer rather than in the core, so a driver "
    "entry would have to write it. See spec 43",
)

A_LABEL = (
    "the row half of pandas' drop, which finds the positions a set of labels sits at "
    "and builds every column again without them. firepanda reaches it by asking the "
    "index to drop the labels and then reindexing on what is left, which is two calls "
    "the core already had and no third one, so that composition is what the driver "
    "writes down as well. The column half is a schema edit and lives with the basics "
    "cases. See spec 46"
)

case(
    "indexing/drop-labels",
    "DataFrame.drop",
    frames=("keys_unique",),
    expr=lambda pd, df: df.set_index("key").drop([0, 2]),
    rules=STRICT,
    note="the plainest form of the method, which is a positional argument and every "
    "other parameter left alone, and on a frame it means the rows rather than the "
    "columns. The whole frame is compared rather than the index alone, so a drop that "
    "took the right labels out and reordered what was left would not pass. " + A_LABEL,
)
case(
    "indexing/drop-index",
    "DataFrame.drop",
    level="L3",
    covers=("index",),
    frames=("keys_unique",),
    expr=lambda pd, df: df.set_index("key").drop(index=[0, 2]),
    rules=STRICT,
    note="the keyword spelling of the same door, which is the one that says out loud "
    "which axis was meant and the one a reader of the call does not have to know the "
    "default to follow. " + A_LABEL,
)
case(
    "indexing/drop-missing-ignored",
    "DataFrame.drop",
    level="L3",
    covers=("errors",),
    frames=("keys_unique",),
    expr=lambda pd, df: df.set_index("key").drop([0, 999999], errors="ignore"),
    rules=STRICT,
    note="a label the index has and one it does not, with the word that says to skip "
    "what is not there. The index does this itself in the core rather than in the "
    "Python layer, which is the difference from the column half, where the same word "
    "is answered before the core is reached. " + A_LABEL,
)
case(
    "indexing/drop-missing-raised",
    "DataFrame.drop",
    level="L4",
    covers=("errors",),
    frames=("keys_unique",),
    expr=lambda pd, df: df.set_index("key").drop([999999]),
    raises=("KeyError", "not found in axis"),
    note="the default, which is that a label the index does not have stops the call. "
    "Both libraries name what was missing in the message, and firepanda puts the word "
    "index in front of the list where pandas does not, which is why the substring "
    "asserted here is the part they share. " + A_LABEL,
)
case(
    "indexing/series-drop-labels",
    "Series.drop",
    frames=("keys_unique",),
    expr=lambda pd, df: df.set_index("key")["value"].drop([0, 2]),
    rules=STRICT,
    note="the row half with one column under it, which is the whole of the method on a "
    "column, since a column has no second axis to take anything out of. " + A_LABEL,
)
case(
    "indexing/series-drop-index",
    "Series.drop",
    level="L3",
    covers=("index",),
    frames=("keys_unique",),
    expr=lambda pd, df: df.set_index("key")["value"].drop(index=[0, 2]),
    rules=STRICT,
    note="the keyword spelling on a column, where it reads oddly because index is the "
    "only axis there is, and pandas takes it for the same reason it takes columns here "
    "and does nothing with that one. " + A_LABEL,
)
case(
    "indexing/series-drop-labels-axis",
    "Series.drop",
    level="L3",
    covers=("labels", "axis"),
    frames=("keys_unique",),
    expr=lambda pd, df: df.set_index("key")["value"].drop([0, 2], axis=0),
    rules=STRICT,
    note="the axis named on an object that has one, which pandas accepts and checks "
    "rather than ignores, so naming the other one is an error on a column where it is "
    "a door on a frame. " + A_LABEL,
)
RANGE_IN_PROCESS = (
    "In process because the driver has no entry for it, and firepanda's RangeIndex is its "
    "Python layer writing the range out as an int64 index"
)
case(
    "indexing/range-index",
    "pandas.RangeIndex",
    level="L3",
    covers=("start", "stop", "step", "name"),
    frames=("single",),
    expr=lambda pd, df: pd.RangeIndex(start=10, stop=-5, step=-3, name="r"),
    in_process=True,
    note="a range that steps down past zero and stops before the stop. " + RANGE_IN_PROCESS,
)
case(
    "indexing/range-index-from-range",
    "pandas.RangeIndex",
    frames=("single",),
    expr=lambda pd, df: pd.RangeIndex.from_range(range(2, 20, 4), name="r"),
    in_process=True,
    note="a Python range taken whole. " + RANGE_IN_PROCESS,
)
case(
    "indexing/set-axis-range-index",
    "DataFrame.set_axis",
    frames=SHAPES,
    expr=lambda pd, df: df.set_axis(pd.RangeIndex(100, 100 + len(df))),
    in_process=True,
    note="a frame relabelled with a range that starts at a hundred. " + RANGE_IN_PROCESS,
)
INTERVAL_IN_PROCESS = (
    "In process because the driver has no entry for it, and firepanda's Interval is a Python scalar"
)
case(
    "indexing/interval-mid-length",
    "pandas.Interval",
    frames=("single",),
    expr=lambda pd, df: pd.Series(
        [pd.Interval(0, 3, closed="both").mid, pd.Interval(0.5, 4.0).length], name="v"
    ),
    in_process=True,
    note="the midpoint of whole number ends is a float, and the length is right less left. "
    + INTERVAL_IN_PROCESS,
)
case(
    "indexing/interval-contains-overlaps",
    "pandas.Interval",
    frames=("single",),
    expr=lambda pd, df: pd.Series(
        [
            0 in pd.Interval(0, 1),
            1 in pd.Interval(0, 1),
            pd.Interval(0, 1, closed="neither") in pd.Interval(0, 1),
            pd.Interval(0, 1).overlaps(pd.Interval(1, 2)),
            pd.Interval(0, 1, closed="both").overlaps(pd.Interval(1, 2, closed="both")),
        ],
        name="v",
    ),
    in_process=True,
    note="an open end leaves its point out, for a point, another interval and an overlap. "
    + INTERVAL_IN_PROCESS,
)
case(
    "indexing/index-repr",
    "Index.__repr__",
    frames=("keys_10", "tall"),
    expr=lambda pd, df: [
        repr(df.index),
        repr(df.head(3).index),
        repr(pd.Index([1.5, None, 3.25])),
        repr(pd.Index(["a", None, "tab\there"], name="k")),
        repr(pd.Index(list(range(150)))),
        repr(pd.Index([f"label_number_{i}" for i in range(40)])),
        repr(pd.TimedeltaIndex(["-1 days", "2 days", None])),
    ],
    in_process=True,
    note="an index prints its labels in brackets, wrapped and cut as pandas does, then its "
    "type, name and length, and a frame's untouched labels print as a RangeIndex",
)


def _flat_answer(out: Any) -> Any:
    """An index method's answer as plain lists and ints, so an array and a list compare."""
    if isinstance(out, tuple):
        return tuple(_flat_answer(x) for x in out)
    if out is None or isinstance(out, (str, float)):
        return out
    if hasattr(out, "tolist"):
        out = out.tolist()
        if isinstance(out, list):
            return [_flat_cell(x) for x in out]
        return out
    return int(out) if type(out).__name__.startswith("int") else out


def _flat_cell(x: Any) -> Any:
    if type(x).__name__.startswith("int"):
        return int(x)
    return "nan" if isinstance(x, float) and x != x else x


def _counted(counts: Any) -> Any:
    return (_flat_answer(counts.index), _flat_answer(counts), counts.name)


def _named(index: Any) -> Any:
    return (_flat_answer(index), index.name)


def _awkward_ints(pd: Any) -> Any:
    return pd.Index([3, 1, 2, 3, None, 1, 1])


def _key_pairs(pd: Any) -> Any:
    return pd.MultiIndex.from_tuples([("a", 1), ("b", 2), ("a", 2), ("b", 1)], names=["k", "n"])


def _days(pd: Any) -> Any:
    return pd.DatetimeIndex(["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-01"])


BUILT_INDEX = "In process because the index is built by the case and asked directly"
case(
    "indexing/index-value-counts-options",
    "Index.value_counts",
    level="L3",
    covers=("normalize", "sort", "ascending", "dropna", "bins"),
    frames=("single",),
    expr=lambda pd, df: [
        _counted(
            _awkward_ints(pd).value_counts(normalize=True, sort=True, ascending=True, dropna=False)
        ),
        _counted(_awkward_ints(pd).value_counts(sort=False, dropna=True)),
        pd.Index([1, 2, 3, 4, 5]).value_counts(bins=2).tolist(),
    ],
    in_process=True,
    note="shares, the gap counted or not, first seen order without sorting, and bins. "
    + BUILT_INDEX,
)
case(
    "indexing/multi-value-counts-options",
    "MultiIndex.value_counts",
    level="L3",
    covers=("normalize", "sort", "ascending", "dropna"),
    frames=("single",),
    expr=lambda pd, df: [
        _counted(
            _key_pairs(pd).value_counts(normalize=True, sort=True, ascending=True, dropna=False)
        ),
        _counted(_key_pairs(pd).value_counts(sort=False)),
    ],
    in_process=True,
    note="a pair of labels counted as a row, in label order when sorting is off. " + BUILT_INDEX,
)
case(
    "indexing/index-take-options",
    "Index.take",
    level="L3",
    covers=("indices", "axis", "allow_fill", "fill_value"),
    frames=("single",),
    expr=lambda pd, df: [
        _flat_answer(_awkward_ints(pd).take([0, 2], axis=0)),
        _flat_answer(_days(pd).take([0, 2], axis=0)),
        _flat_answer(pd.Index(["b", "a", "c"]).take([0, -1], allow_fill=True, fill_value=None)),
    ],
    in_process=True,
    note="positions taken, and minus one read as the last label when no fill is given. "
    + BUILT_INDEX,
)
case(
    "indexing/multi-take-options",
    "MultiIndex.take",
    level="L3",
    covers=("indices", "axis"),
    frames=("single",),
    expr=lambda pd, df: _key_pairs(pd).take([0, 2], axis=0).tolist(),
    in_process=True,
    note="the pairs at the given positions. " + BUILT_INDEX,
)
case(
    "indexing/index-sortlevel-options",
    "Index.sortlevel",
    level="L3",
    covers=("level", "ascending", "sort_remaining", "na_position"),
    frames=("single",),
    expr=lambda pd, df: _flat_answer(
        pd.Index(["b", "a", "c", "a"]).sortlevel(
            level=0, ascending=False, sort_remaining=True, na_position="last"
        )
    ),
    in_process=True,
    note="a flat index sorts by its labels and hands back the order it used. " + BUILT_INDEX,
)
case(
    "indexing/multi-sortlevel-options",
    "MultiIndex.sortlevel",
    level="L3",
    covers=("level", "ascending", "sort_remaining", "na_position"),
    frames=("single",),
    expr=lambda pd, df: _flat_answer(
        _key_pairs(pd).sortlevel(level=0, ascending=False, sort_remaining=True, na_position="last")
    ),
    in_process=True,
    note="pairs sorted by the first level falling, ties broken by the rest. " + BUILT_INDEX,
)
case(
    "indexing/index-sort-values-options",
    "Index.sort_values",
    level="L3",
    covers=("return_indexer", "ascending", "na_position", "key"),
    frames=("single",),
    expr=lambda pd, df: [
        _flat_answer(
            _awkward_ints(pd).sort_values(return_indexer=True, ascending=False, na_position="first")
        ),
        _flat_answer(_days(pd).sort_values(return_indexer=True)),
        _flat_answer(pd.Index(["b", "a", "c"]).sort_values(key=lambda v: v)),
    ],
    in_process=True,
    note="labels sorted with the gap placed first, the order handed back on request. "
    + BUILT_INDEX,
)
case(
    "indexing/index-to-numpy-options",
    "Index.to_numpy",
    level="L3",
    covers=("dtype", "copy"),
    frames=("single",),
    expr=lambda pd, df: [
        _flat_answer(pd.Index(["b", "a"]).to_numpy(dtype=object, copy=True)),
        _flat_answer(_days(pd).to_numpy(dtype=object, copy=True)),
    ],
    in_process=True,
    note="labels as plain objects, a moment as a Timestamp. " + BUILT_INDEX,
)
case(
    "indexing/multi-to-numpy-options",
    "MultiIndex.to_numpy",
    level="L3",
    covers=("dtype", "copy"),
    frames=("single",),
    expr=lambda pd, df: _flat_answer(_key_pairs(pd).to_numpy(dtype=object, copy=True)),
    in_process=True,
    note="each pair as a tuple. " + BUILT_INDEX,
)
case(
    "indexing/index-symmetric-difference-options",
    "Index.symmetric_difference",
    level="L3",
    covers=("other", "result_name", "sort"),
    frames=("single",),
    expr=lambda pd, df: [
        _named(
            pd.Index(["b", "a", "c", "a"]).symmetric_difference(
                pd.Index(["b", "z"]), result_name="r", sort=False
            )
        ),
        _days(pd).symmetric_difference(_days(pd)[1:3], sort=None).tolist(),
    ],
    in_process=True,
    note="labels in one side only, named as asked, sorted or in first seen order. " + BUILT_INDEX,
)
case(
    "indexing/multi-symmetric-difference-options",
    "MultiIndex.symmetric_difference",
    level="L3",
    covers=("other", "result_name", "sort"),
    frames=("single",),
    expr=lambda pd, df: [
        _key_pairs(pd).symmetric_difference(_key_pairs(pd)[:2], result_name=["x", "y"]).tolist(),
        _key_pairs(pd).symmetric_difference(_key_pairs(pd)[1:3], sort=False).tolist(),
    ],
    in_process=True,
    note="pairs in one side only. " + BUILT_INDEX,
)
case(
    "indexing/index-slice-locs-options",
    "Index.slice_locs",
    level="L3",
    covers=("start", "end", "step"),
    frames=("single",),
    expr=lambda pd, df: [
        _flat_answer(pd.Index([1, 2, 3, 4, 5]).slice_locs(start=2, end=4, step=1)),
        _flat_answer(pd.Index([5, 4, 3, 2, 1]).slice_locs(start=4, end=2, step=-1)),
    ],
    in_process=True,
    note="the positions bounding a label range, forward and backward. " + BUILT_INDEX,
)
case(
    "indexing/datetime-slice-locs-options",
    "DatetimeIndex.slice_locs",
    level="L3",
    covers=("start", "end"),
    frames=("single",),
    expr=lambda pd, df: [
        _flat_answer(_days(pd).sort_values().slice_locs(start="2024-01-01", end="2024-01-02")),
        _flat_answer(_days(pd).sort_values().slice_locs(pd.Timestamp("2024-01-02"), None)),
        _flat_answer(_days(pd).sort_values().slice_locs("2024-01", "2024-01")),
    ],
    in_process=True,
    note="a date string, a Timestamp and a month string bound the range. " + BUILT_INDEX,
)
case(
    "indexing/multi-slice-locs-options",
    "MultiIndex.slice_locs",
    level="L3",
    covers=("start", "end"),
    frames=("single",),
    expr=lambda pd, df: _flat_answer(
        _key_pairs(pd).sort_values().slice_locs(start=("a", 2), end=("b", 1))
    ),
    in_process=True,
    note="pairs bound the range in a sorted multi index. " + BUILT_INDEX,
)
case(
    "indexing/index-slice-indexer-options",
    "Index.slice_indexer",
    level="L3",
    covers=("start", "end", "step"),
    frames=("single",),
    expr=lambda pd, df: [
        repr(pd.Index([1, 2, 3, 4, 5]).slice_indexer(start=2, end=4, step=2)),
        repr(_days(pd).sort_values().slice_indexer(start="2024-01-02", end=None, step=1)),
    ],
    in_process=True,
    note="a slice of positions for a label range. " + BUILT_INDEX,
)
case(
    "indexing/index-reindex-options",
    "Index.reindex",
    level="L3",
    covers=("target", "method", "limit", "tolerance"),
    frames=("single",),
    expr=lambda pd, df: [
        _flat_answer(
            pd.Index([1, 2, 4]).reindex([1, 3, 4], method="nearest", limit=1, tolerance=1)
        ),
        _flat_answer(pd.Index([1, 2, 4]).reindex([0, 3, 5], method="ffill")),
    ],
    in_process=True,
    note="the new labels and where each was found, filling from a neighbour. " + BUILT_INDEX,
)
case(
    "indexing/index-get-indexer-options",
    "Index.get_indexer",
    level="L3",
    covers=("target", "method", "limit", "tolerance"),
    frames=("single",),
    expr=lambda pd, df: [
        _flat_answer(
            pd.Index([1, 2, 4]).get_indexer([0, 3, 5], method="bfill", limit=1, tolerance=2)
        ),
        _flat_answer(
            pd.DatetimeIndex(["2024-01-01", "2024-01-03"]).get_indexer(
                pd.DatetimeIndex(["2024-01-02"]), method="nearest", tolerance="2D"
            )
        ),
    ],
    in_process=True,
    note="the position of the next or nearest label within the tolerance. " + BUILT_INDEX,
)
case(
    "indexing/index-join-options",
    "Index.join",
    level="L3",
    covers=("other", "how", "level", "return_indexers", "sort"),
    frames=("single",),
    expr=lambda pd, df: [
        _flat_answer(
            pd.Index([1, 2, 3]).join(
                pd.Index([2, 3, 4]), how="outer", return_indexers=True, sort=True
            )
        ),
        _flat_answer(pd.Index([3, 1, 2]).join(pd.Index([2, 3, 4]), how="inner", sort=False)),
        _flat_answer(
            pd.Index([2, 1, 5], name="n").join(
                _key_pairs(pd), how="inner", level="n", return_indexers=True
            )
        ),
    ],
    in_process=True,
    note="labels joined with the positions each side came from, a multi index other joined "
    "on its level. " + BUILT_INDEX,
)
case(
    "indexing/multi-join-options",
    "MultiIndex.join",
    level="L3",
    covers=("other", "how", "level", "return_indexers", "sort"),
    frames=("single",),
    expr=lambda pd, df: (
        [
            _flat_answer(
                _key_pairs(pd).join(
                    pd.Index([2, 1, 5], name="n"), how=how, level="n", return_indexers=True
                )
            )
            for how in ("left", "right", "inner", "outer")
        ]
        + [
            _flat_answer(
                _key_pairs(pd).join(
                    pd.MultiIndex.from_tuples([("b", 2), ("c", 3), ("a", 1)], names=["k", "n"]),
                    how="outer",
                    return_indexers=True,
                    sort=True,
                )
            )
        ]
    ),
    in_process=True,
    note="a flat index joined on a named level for each way, and two multi indexes joined. "
    + BUILT_INDEX,
)
case(
    "indexing/datetime-searchsorted-options",
    "DatetimeIndex.searchsorted",
    level="L3",
    covers=("value", "side", "sorter"),
    frames=("single",),
    expr=lambda pd, df: [
        _flat_answer(
            _days(pd).sort_values().searchsorted(pd.Timestamp("2024-01-02"), side="right")
        ),
        _flat_answer(_days(pd).sort_values().searchsorted("2024-01-02")),
        _flat_answer(
            _days(pd)
            .sort_values()
            .searchsorted([pd.Timestamp("2024-01-02"), pd.Timestamp("2025-01-01")])
        ),
        _flat_answer(
            _days(pd).searchsorted(
                pd.Timestamp("2024-01-02"), side="left", sorter=_days(pd).argsort()
            )
        ),
    ],
    in_process=True,
    note="where a moment would go, on either side, through a sorter. " + BUILT_INDEX,
)
case(
    "indexing/datetime-std-options",
    "DatetimeIndex.std",
    level="L3",
    covers=("axis", "ddof", "skipna"),
    frames=("single",),
    expr=lambda pd, df: [
        str(_days(pd).std(skipna=True, ddof=0)),
        str(pd.DatetimeIndex(["2024-01-01", None, "2024-01-03"]).std(skipna=False)),
        str(_days(pd).std(axis=0, ddof=1)),
    ],
    in_process=True,
    note="the spread of moments as a Timedelta, NaT when a gap is not skipped. " + BUILT_INDEX,
)
case(
    "indexing/datetime-mean-options",
    "DatetimeIndex.mean",
    level="L3",
    covers=("skipna",),
    frames=("single",),
    expr=lambda pd, df: [
        str(pd.DatetimeIndex(["2024-01-01", None]).mean(skipna=False)),
        str(pd.DatetimeIndex([]).mean()),
    ],
    in_process=True,
    note="NaT for an unskipped gap and for no moments at all. " + BUILT_INDEX,
)
case(
    "indexing/datetime-between-time-options",
    "DatetimeIndex.indexer_between_time",
    level="L3",
    covers=("start_time", "end_time", "include_start", "include_end"),
    frames=("single",),
    expr=lambda pd, df: [
        _flat_answer(
            pd.date_range("2024-01-01", periods=6, freq="h").indexer_between_time(
                "01:00", "03:00", include_start=False, include_end=True
            )
        ),
        _flat_answer(
            pd.date_range("2024-01-01", periods=6, freq="h").indexer_between_time(
                start_time="04:00", end_time="01:00"
            )
        ),
    ],
    in_process=True,
    note="positions inside a time window, ends open or closed, and a window over midnight. "
    + BUILT_INDEX,
)


def _slice_parts(found):
    """A slice as its three plain bounds, so a numpy integer inside reads as a plain one."""
    return [None if x is None else int(x) for x in (found.start, found.stop, found.step)]


def _spelled_counts(counts):
    """Counts with their labels spelled out, so a NaT label compares equal to another."""
    return ([str(x) for x in counts.index], counts.tolist())


def _three_pairs(pd: Any) -> Any:
    return pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1)], names=["k", "n"])


def _gappy_days(pd: Any) -> Any:
    return pd.DatetimeIndex(["2024-01-03", "2024-01-01", "NaT", "2024-01-01"])


case(
    "indexing/multi-set-levels-options",
    "MultiIndex.set_levels",
    level="L3",
    covers=("levels", "level", "verify_integrity"),
    frames=("single",),
    expr=lambda pd, df: _named(
        _three_pairs(pd).set_levels(["x", "y"], level="k", verify_integrity=True)
    ),
    in_process=True,
    note="one level's values swapped by name, the codes kept. " + BUILT_INDEX,
)
case(
    "indexing/multi-set-levels-short",
    "MultiIndex.set_levels",
    level="L4",
    frames=("single",),
    expr=lambda pd, df: _three_pairs(pd).set_levels([1], level=0, verify_integrity=True),
    raises=("ValueError", "code max (1) >= length of level (1)"),
    in_process=True,
    note="a level too short for its codes is refused when integrity is checked. " + BUILT_INDEX,
)
case(
    "indexing/multi-set-codes-options",
    "MultiIndex.set_codes",
    level="L3",
    covers=("codes", "level", "verify_integrity"),
    frames=("single",),
    expr=lambda pd, df: _named(
        _three_pairs(pd).set_codes([1, 0, 0], level=1, verify_integrity=True)
    ),
    in_process=True,
    note="one level's codes swapped by position. " + BUILT_INDEX,
)
case(
    "indexing/multi-get-loc-level-options",
    "MultiIndex.get_loc_level",
    level="L3",
    covers=("key", "level", "drop_level"),
    frames=("single",),
    expr=lambda pd, df: [
        _slice_parts(_three_pairs(pd).get_loc_level("a", level="k", drop_level=False)[0]),
        _three_pairs(pd).get_loc_level("a", level="k", drop_level=False)[1].tolist(),
        _flat_answer(_three_pairs(pd).get_loc_level(1, level=1, drop_level=True)[0]),
        _three_pairs(pd).get_loc_level(1, level=1, drop_level=True)[1].tolist(),
    ],
    in_process=True,
    note="where a level holds a key, as a slice or a mask, and the rows with or without "
    "the level. " + BUILT_INDEX,
)
case(
    "indexing/multi-from-tuples-options",
    "MultiIndex.from_tuples",
    level="L3",
    covers=("tuples", "sortorder", "names"),
    frames=("single",),
    expr=lambda pd, df: _flat_answer(
        pd.MultiIndex.from_tuples([("a", 1), ("b", 2)], sortorder=0, names=["k", "n"]).names
    ),
    in_process=True,
    note="pairs built into an index with its level names. " + BUILT_INDEX,
)
case(
    "indexing/multi-from-product-options",
    "MultiIndex.from_product",
    level="L3",
    covers=("iterables", "sortorder", "names"),
    frames=("single",),
    expr=lambda pd, df: _named(
        pd.MultiIndex.from_product([["a", "b"], [1, 2]], sortorder=0, names=["k", "n"])
    ),
    in_process=True,
    note="every pair of the two lists, first level slowest. " + BUILT_INDEX,
)
case(
    "indexing/multi-slice-indexer-options",
    "MultiIndex.slice_indexer",
    level="L3",
    covers=("start", "end", "step"),
    frames=("single",),
    expr=lambda pd, df: (
        repr(_three_pairs(pd).slice_indexer(start=("a", 2), end=("b", 1), step=1))
        .replace("np.int64(", "")
        .replace("), ", ", ")
    ),
    in_process=True,
    note="a slice of positions between two pairs. " + BUILT_INDEX,
)
case(
    "indexing/multi-sort-values-options",
    "MultiIndex.sort_values",
    level="L3",
    covers=("return_indexer", "ascending", "na_position", "key"),
    frames=("single",),
    expr=lambda pd, df: [
        _flat_answer(
            _key_pairs(pd).sort_values(return_indexer=True, ascending=False, na_position="last")
        ),
        _flat_answer(_key_pairs(pd).sort_values(key=lambda x: x)),
    ],
    in_process=True,
    note="pairs sorted falling with the order handed back. " + BUILT_INDEX,
)
case(
    "indexing/multi-get-indexer-options",
    "MultiIndex.get_indexer",
    level="L3",
    covers=("target", "method", "limit"),
    frames=("single",),
    expr=lambda pd, df: [
        _flat_answer(_three_pairs(pd).get_indexer([("a", 2), ("z", 1)], method=None)),
        _flat_answer(
            _three_pairs(pd).get_indexer(
                [("a", 0), ("a", 1), ("a", 3), ("a", 4), ("b", 1), ("b", 2), ("b", 3)],
                method="pad",
                limit=1,
            )
        ),
        _flat_answer(
            _three_pairs(pd).get_indexer(
                [("a", 0), ("a", 1), ("a", 3), ("a", 4), ("b", 1), ("b", 2), ("b", 3)],
                method="bfill",
                limit=2,
            )
        ),
    ],
    in_process=True,
    note="a fill counts the inexact rows each pair fills and stops at the limit. " + BUILT_INDEX,
)
case(
    "indexing/multi-get-indexer-limit-order",
    "MultiIndex.get_indexer",
    level="L4",
    frames=("single",),
    expr=lambda pd, df: _three_pairs(pd).get_indexer([("b", 0), ("a", 3)], method="pad", limit=1),
    raises=("ValueError", "only well-defined if index and target are monotonic"),
    in_process=True,
    note="a limit needs the target in order. " + BUILT_INDEX,
)
case(
    "indexing/multi-reindex-options",
    "MultiIndex.reindex",
    level="L3",
    covers=("target", "method", "level", "limit"),
    frames=("single",),
    expr=lambda pd, df: [
        _flat_answer(_three_pairs(pd).reindex([("a", 1), ("c", 3)])),
        _flat_answer(_three_pairs(pd).reindex(["b", "a"], level="k")),
        _flat_answer(
            _three_pairs(pd).reindex([("a", 3), ("a", 4), ("b", 5)], method="ffill", limit=1)
        ),
    ],
    in_process=True,
    note="new pairs and where each was found, by a level or by filling. " + BUILT_INDEX,
)
case(
    "indexing/datetime-value-counts-options",
    "DatetimeIndex.value_counts",
    level="L3",
    covers=("normalize", "sort", "ascending", "dropna"),
    frames=("single",),
    expr=lambda pd, df: [
        _spelled_counts(
            _gappy_days(pd).value_counts(normalize=True, sort=True, ascending=True, dropna=False)
        ),
        _spelled_counts(_gappy_days(pd).value_counts(sort=False)),
    ],
    in_process=True,
    note="moments counted, the gap counted when asked. " + BUILT_INDEX,
)
case(
    "indexing/datetime-take-options",
    "DatetimeIndex.take",
    level="L3",
    covers=("indices", "axis", "allow_fill", "fill_value"),
    frames=("single",),
    expr=lambda pd, df: [
        _flat_answer(_days(pd).take([0, -1], axis=0, allow_fill=True, fill_value=None)),
        _flat_answer(_days(pd).take([2, 1])),
    ],
    in_process=True,
    note="moments at the given positions. " + BUILT_INDEX,
)
case(
    "indexing/datetime-sortlevel-options",
    "DatetimeIndex.sortlevel",
    level="L3",
    covers=("level", "ascending", "sort_remaining", "na_position"),
    frames=("single",),
    expr=lambda pd, df: _flat_answer(
        _days(pd).sortlevel(level=0, ascending=False, sort_remaining=True, na_position="first")
    ),
    in_process=True,
    note="moments sorted falling with the order handed back. " + BUILT_INDEX,
)
case(
    "indexing/datetime-get-indexer-options",
    "DatetimeIndex.get_indexer",
    level="L3",
    covers=("target", "method", "limit", "tolerance"),
    frames=("single",),
    expr=lambda pd, df: _flat_answer(
        pd.DatetimeIndex(["2024-01-01", "2024-01-05"]).get_indexer(
            pd.DatetimeIndex(["2024-01-02", "2024-01-09"]),
            method="ffill",
            limit=1,
            tolerance=pd.Timedelta("2D"),
        )
    ),
    in_process=True,
    note="the moment before each target, within the tolerance. " + BUILT_INDEX,
)
case(
    "indexing/datetime-reindex-options",
    "DatetimeIndex.reindex",
    level="L3",
    covers=("target", "method", "limit", "tolerance"),
    frames=("single",),
    expr=lambda pd, df: _flat_answer(
        pd.DatetimeIndex(["2024-01-01", "2024-01-05"]).reindex(
            pd.DatetimeIndex(["2024-01-04"]), method="nearest", limit=1, tolerance="2D"
        )
    ),
    in_process=True,
    note="the nearest moment within the tolerance. " + BUILT_INDEX,
)
case(
    "indexing/datetime-join-options",
    "DatetimeIndex.join",
    level="L3",
    covers=("other", "how", "return_indexers", "sort"),
    frames=("single",),
    expr=lambda pd, df: _flat_answer(
        pd.DatetimeIndex(["2024-01-03", "2024-01-01"]).join(
            pd.DatetimeIndex(["2024-01-01", "2024-01-09"]),
            how="outer",
            return_indexers=True,
            sort=True,
        )
    ),
    in_process=True,
    note="moments joined with the positions each side came from. " + BUILT_INDEX,
)
