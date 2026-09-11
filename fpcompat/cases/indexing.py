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
    frames=SHAPES,
    expr=lambda pd, df: df.iloc[0],
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
    note="sorted first, so the label three and the position three are different rows "
    "and a case that confused them would fail here",
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
    note="a two level index, which the comparison flattens into two index columns",
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
    "indexing/xs",
    "DataFrame.xs",
    level="L3",
    covers=("key",),
    frames=("keys_10",),
    expr=lambda pd, df: df.set_index("key").xs(3),
    rules=STRICT,
)
case(
    "indexing/droplevel",
    "DataFrame.droplevel",
    level="L3",
    covers=("level",),
    frames=("keys_two_column",),
    expr=lambda pd, df: df.set_index(["left", "right"]).droplevel(0),
    rules=STRICT,
)
case(
    "indexing/swaplevel",
    "DataFrame.swaplevel",
    frames=("keys_two_column",),
    expr=lambda pd, df: df.set_index(["left", "right"]).swaplevel(),
    rules=STRICT,
)
case(
    "indexing/sort-index-multi",
    "DataFrame.sort_index",
    frames=("keys_two_column",),
    expr=lambda pd, df: df.set_index(["left", "right"]).sort_index(),
    rules=STRICT,
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
case(
    "indexing/query",
    "DataFrame.query",
    level="L3",
    covers=("expr",),
    frames=("tall", "keys_1000"),
    expr=lambda pd, df: df.query("key > 3"),
    rules=STRICT,
)
case(
    "indexing/query-and",
    "DataFrame.query",
    level="L3",
    covers=("expr",),
    frames=("tall",),
    expr=lambda pd, df: df.query("key > 3 and value < 0"),
    rules=STRICT,
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
    note="assigning a shorter column back into the frame lines it up on the labels "
    "rather than on position, so the two rows it does not cover come back null instead "
    "of the column landing at the top. This is the one that surprises people who have "
    "used pandas for years. It was a divergence case until firepanda started aligning",
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
