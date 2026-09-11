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
