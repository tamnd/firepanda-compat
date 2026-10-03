"""Failure, which is part of the API and not the absence of it.

Level four. A library that computes the right answer and raises the wrong exception is
not a drop in replacement, because the code around it catches by type. `pandas.errors`
has forty six types in it and the difference between a `MergeError` and a `ValueError`
is the difference between a caller's error handler running and their process dying.

Two rules run through the whole section. The type has to match exactly, since a
subclass is not what the caller wrote in their except clause. The message is only
checked for a substring, and that substring is the piece a person would recognise,
which is a column name or a dtype or a value. Everything past it is pandas prose and
pinning it would turn a pandas point release into a hundred failures that are all the
same non bug.
"""

from __future__ import annotations

from typing import Any

from fpcompat.cases import case, section

section("errors")

# ---------------------------------------------------------------------------
# Missing things
# ---------------------------------------------------------------------------

case(
    "errors/missing-column",
    "DataFrame.__getitem__",
    level="L4",
    frames=("two", "tall"),
    expr=lambda pd, df: df["not_a_column"],
    raises=("KeyError", "not_a_column"),
    note="a KeyError and not a ValueError, and the message has the name in it, which "
    "is what makes it useful",
)
case(
    "errors/missing-column-list",
    "DataFrame.__getitem__",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df[["a", "not_a_column"]],
    raises=("KeyError", "not_a_column"),
)
case(
    "errors/missing-column-list-says-which",
    "DataFrame.__getitem__",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df[["a", "nope"]],
    raises=("KeyError", "['nope'] not in index"),
    note="pandas lists the names it could not find when some of the list is there",
)
case(
    "errors/missing-column-list-none-there",
    "DataFrame.__getitem__",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df[["nope", "nada"]],
    raises=("KeyError", "None of [Index(['nope', 'nada']"),
    note="when none of the list is there pandas repeats the whole key as an index",
)
case(
    "errors/iloc-column-off-the-end",
    "DataFrame.iloc",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.iloc[:, 999],
    raises=("IndexError", "single positional indexer is out-of-bounds"),
)
case(
    "errors/iloc-column-list-off-the-end",
    "DataFrame.iloc",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.iloc[:, [0, 999]],
    raises=("IndexError", "positional indexers are out-of-bounds"),
)
case(
    "errors/iloc-cell-column-off-the-end",
    "DataFrame.iloc",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.iloc[0, 999],
    raises=("IndexError", "index 999 is out of bounds for axis 0 with size"),
    note="one cell off the end is worded the way numpy words it, which pandas passes on",
)
case(
    "errors/take-off-the-end",
    "DataFrame.take",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.take([0, 10**6]),
    raises=("IndexError", "indices are out-of-bounds"),
)
case(
    "errors/grouped-column-not-found",
    "DataFrame.groupby",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.groupby(df.columns[0])["nope"],
    raises=("KeyError", "Column not found: nope"),
)
case(
    "errors/grouped-columns-not-found",
    "DataFrame.groupby",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.groupby(df.columns[0])[["nope", "nada"]],
    raises=("KeyError", "Columns not found: 'nada', 'nope'"),
    note="pandas sorts the missing names before it lists them",
)
case(
    "errors/ascending-not-a-flag",
    "Series.sort_values",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df[df.columns[0]].sort_values(ascending="yes"),
    raises=("ValueError", 'For argument "ascending" expected type bool, received type str.'),
)
case(
    "errors/sort-index-ascending-not-a-flag",
    "DataFrame.sort_index",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.sort_index(ascending="yes"),
    raises=("ValueError", 'For argument "ascending" expected type bool'),
)
case(
    "errors/diff-periods-not-whole",
    "Series.diff",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df[df.columns[0]].diff(1.5),
    raises=("ValueError", "periods must be an integer"),
)
case(
    "errors/duplicated-keep-wording",
    "Series.duplicated",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df[df.columns[0]].duplicated(keep="middle"),
    raises=("ValueError", 'keep must be either "first", "last" or False'),
)
case(
    "errors/replace-regex-not-a-flag",
    "Series.replace",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df[df.columns[0]].replace(1, 2, regex=5),
    raises=("ValueError", "'to_replace' must be 'None' if 'regex' is not a bool"),
)
case(
    "errors/transform-unknown-name",
    "Series.transform",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df[df.columns[0]].transform("nope"),
    raises=("ValueError", "Transform function failed"),
)
case(
    "errors/apply-unknown-name",
    "Series.apply",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df[df.columns[0]].apply("nope"),
    raises=("AttributeError", "'nope' is not a valid function for 'Series' object"),
)
case(
    "errors/concat-series-bad-axis",
    "pandas.concat",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: pd.concat([df[df.columns[0]], df[df.columns[0]]], axis=5),
    raises=("ValueError", "No axis named 5 for object type DataFrame"),
    note="pandas reads the axis as a frame's even when only columns are joined",
)
case(
    "errors/frame-columns-of-different-lengths",
    "pandas.DataFrame",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: pd.DataFrame({"a": [1, 2, 3], "b": [1]}),
    raises=("ValueError", "All arrays must be of the same length"),
)
case(
    "errors/fillna-list-value",
    "DataFrame.fillna",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.fillna([1]),
    raises=("TypeError", '"value" parameter must be a scalar or dict, but you passed a "list"'),
)
case(
    "errors/series-fillna-set-value",
    "Series.fillna",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df[df.columns[0]].fillna({1}),
    raises=("TypeError", '"value" parameter must be a scalar, dict or Series'),
)
case(
    "errors/index-fillna-list-value",
    "Index.fillna",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.index.fillna([1]),
    raises=("TypeError", "'value' must be a scalar, passed: list"),
)
case(
    "errors/std-ddof-not-a-number",
    "DataFrame.std",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.std(ddof="a"),
    raises=("ValueError", "could not convert string to float: 'a'"),
)
case(
    "errors/argmax-past-the-one-axis",
    "Series.argmax",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df[df.columns[0]].argmax(axis=1),
    raises=("ValueError", "`axis` must be fewer than the number of dimensions (1)"),
)
case(
    "errors/nlargest-text-column",
    "DataFrame.nlargest",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: pd.DataFrame({"c": ["a", "b"]}).nlargest(1, "c"),
    raises=("TypeError", "Column 'c' has dtype str, cannot use method 'nlargest' with this dtype"),
)
case(
    "errors/series-nsmallest-text",
    "Series.nsmallest",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: pd.Series(["a", "b"]).nsmallest(1),
    raises=("TypeError", "Cannot use method 'nsmallest' with dtype str"),
)
case(
    "errors/grouped-quantile-out-of-range",
    "GroupBy.quantile",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: pd.DataFrame({"k": [1, 1], "v": [1.0, 2.0]}).groupby("k").quantile(2),
    raises=("ValueError", "Each 'q' must be between 0 and 1. Got '2.0' instead"),
)
case(
    "errors/merge-validate-words-in-order",
    "DataFrame.merge",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.merge(df, on=list(df.columns), validate="x"),
    raises=("ValueError", '- "m:m"\n- "one_to_one"'),
)
case(
    "errors/round-to-unreadable-frequency",
    "Series.dt.round",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: pd.Series(pd.to_datetime(["2024-01-01"])).dt.round("xx"),
    raises=("ValueError", "Invalid frequency: xx. Failed to parse with error message"),
)
case(
    "errors/floor-to-retired-month-alias",
    "Series.dt.floor",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: pd.Series(pd.to_datetime(["2024-01-01"])).dt.floor("M"),
    raises=("ValueError", "'M' is no longer supported for offsets"),
)
case(
    "errors/timestamp-unknown-unit",
    "pandas.Timestamp",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: pd.Timestamp(1, unit="xx"),
    raises=("ValueError", "Unrecognized unit xx"),
)
case(
    "errors/timedelta-unknown-unit",
    "pandas.Timedelta",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: pd.Timedelta(1, unit="xx"),
    raises=("ValueError", "invalid unit abbreviation: xx"),
)
case(
    "errors/to-datetime-list-unknown-unit",
    "pandas.to_datetime",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: pd.to_datetime([1], unit="xx"),
    raises=("ValueError", "Unrecognized unit xx"),
)
case(
    "errors/dt-on-numbers",
    "Series.dt",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: pd.Series([1, 2]).dt,
    raises=("AttributeError", "Can only use .dt accessor with datetimelike values"),
)
case(
    "errors/str-on-whole-numbers",
    "Series.str",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: pd.Series([1, 2]).str,
    raises=("AttributeError", "with string values, not integer"),
)
case(
    "errors/instants-plus-whole-number",
    "Series.add",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: pd.Series(pd.to_datetime(["2024-01-01"])).add(1),
    raises=("TypeError", "Addition/subtraction of integers and integer-arrays with DatetimeArray"),
)
case(
    "errors/instants-times-two",
    "Series.mul",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: pd.Series(pd.to_datetime(["2024-01-01"])).mul(2),
    raises=("TypeError", "cannot perform __mul__ with this index type: DatetimeArray"),
)
case(
    "errors/timestamp-plus-whole-number",
    "pandas.Timestamp",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: pd.Timestamp("2024-01-01") + 1,
    raises=("TypeError", "integer-arrays with Timestamp is no longer supported"),
)
case(
    "errors/localize-a-zoned-column",
    "Series.dt.tz_localize",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: (
        pd.Series(pd.to_datetime(["2024-01-01"])).dt.tz_localize("UTC").dt.tz_localize("UTC")
    ),
    raises=("TypeError", "Already tz-aware, use tz_convert to convert."),
)
case(
    "errors/localize-to-unknown-zone",
    "Series.dt.tz_localize",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: pd.Series(pd.to_datetime(["2024-01-01"])).dt.tz_localize("Mars/Base"),
    raises=("ZoneInfoNotFoundError", "No time zone found with key Mars/Base"),
)
case(
    "errors/get-loc-missing-label",
    "Index.get_loc",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.columns.get_loc("zz"),
    raises=("KeyError", "zz"),
)
case(
    "errors/iloc-float-row",
    "DataFrame.iloc",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.iloc[1.0],
    raises=("TypeError", "Cannot index by location index with a non-integer key"),
)
case(
    "errors/iloc-text-row",
    "Series.iloc",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df["a"].iloc["a"],
    raises=("TypeError", "Cannot index by location index with a non-integer key"),
)
case(
    "errors/iloc-pair-with-float",
    "DataFrame.iloc",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.iloc[1.5, 0],
    raises=("ValueError", "Location based indexing can only have [integer, integer slice"),
)
case(
    "errors/iloc-pair-with-name",
    "DataFrame.iloc",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.iloc[0, "a"],
    raises=("ValueError", "Location based indexing can only have [integer, integer slice"),
)
case(
    "errors/iat-float",
    "DataFrame.iat",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.iat[1.5, 0],
    raises=("ValueError", "iAt based indexing can only have integer indexers"),
)
case(
    "errors/series-iat-text",
    "Series.iat",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df["a"].iat["a"],
    raises=("ValueError", "iAt based indexing can only have integer indexers"),
)
case(
    "errors/head-text-count",
    "DataFrame.head",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.head("2"),
    raises=(
        "TypeError",
        "cannot do positional indexing on RangeIndex with these indexers [2] of type str",
    ),
)
case(
    "errors/series-head-float-count",
    "Series.head",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df["a"].head(1.5),
    raises=(
        "TypeError",
        "cannot do positional indexing on RangeIndex with these indexers [1.5] of type float",
    ),
)
case(
    "errors/tail-text-count",
    "DataFrame.tail",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.tail("2"),
    raises=("TypeError", "bad operand type for unary -: 'str'"),
)
case(
    "errors/groupby-sum-numeric-only-text",
    "GroupBy.sum",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.groupby("a").sum(numeric_only="x"),
    raises=("ValueError", "numeric_only accepts only Boolean values"),
)
case(
    "errors/groupby-max-numeric-only-text",
    "GroupBy.max",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.groupby("a").max(numeric_only="x"),
    raises=("ValueError", "numeric_only accepts only Boolean values"),
)
case(
    "errors/text-diff",
    "Series.diff",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: pd.Series(["a", "b"]).diff(),
    raises=("TypeError", "operation 'sub' not supported for dtype 'str' with dtype 'str'"),
)
case(
    "errors/text-pct-change",
    "Series.pct_change",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: pd.Series(["a", "b"]).pct_change(),
    raises=("TypeError", "operation 'truediv' not supported for dtype 'str' with dtype 'str'"),
)
case(
    "errors/frame-text-diff",
    "DataFrame.diff",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df[["c"]].fillna("x").diff(),
    raises=("TypeError", "operation 'sub' not supported for dtype 'str' with dtype 'str'"),
)
case(
    "errors/frame-text-cumprod",
    "DataFrame.cumprod",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.fillna({"c": "x"}).cumprod(),
    raises=("TypeError", "operation 'cumprod' not supported for dtype 'str'"),
)
case(
    "errors/text-times-text",
    "Series.mul",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: pd.Series(["a", "b"]).mul(pd.Series(["a", "b"])),
    raises=("TypeError", "Can only string multiply by an integer."),
)
case(
    "errors/text-times-floats",
    "Series.mul",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: pd.Series(["a", "b"]).mul(pd.Series([1.5, 2.0])),
    raises=("TypeError", "Can only string multiply by an integer."),
)
case(
    "errors/text-clip-by-numbers",
    "Series.clip",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: pd.Series(["a", "b"]).clip(0, 1),
    raises=("TypeError", "Invalid comparison between dtype=str and int"),
)
case(
    "errors/merge-left-index-not-bool",
    "DataFrame.merge",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.merge(df, left_index=1, right_index=True),
    raises=("ValueError", "left_index parameter must be of type bool, not <class 'int'>"),
)
case(
    "errors/merge-right-index-not-bool",
    "pandas.merge",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: pd.merge(df, df, right_index="y"),
    raises=("ValueError", "right_index parameter must be of type bool, not <class 'str'>"),
)
case(
    "errors/merge-ordered-cross",
    "pandas.merge_ordered",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: pd.merge_ordered(df[["a", "b"]], df[["a"]], on="a", how="cross"),
    raises=("ValueError", "do not recognize join method cross"),
)
case(
    "errors/head-unknown-keyword",
    "DataFrame.head",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.head(zz=1),
    raises=("TypeError", "NDFrame.head() got an unexpected keyword argument 'zz'"),
)
case(
    "errors/fillna-method-keyword",
    "DataFrame.fillna",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.fillna(0, method="ffill"),
    raises=("TypeError", "NDFrame.fillna() got an unexpected keyword argument 'method'"),
)
case(
    "errors/pivot-without-columns",
    "DataFrame.pivot",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.pivot(index="a"),
    raises=("TypeError", "DataFrame.pivot() missing 1 required keyword-only argument: 'columns'"),
)
case(
    "errors/divide-without-other",
    "DataFrame.div",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.div(),
    raises=("TypeError", "DataFrame.truediv() missing 1 required positional argument"),
)
case(
    "errors/series-tolist-keyword",
    "Series.tolist",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df["a"].tolist(zz=1),
    raises=("TypeError", "IndexOpsMixin.tolist() got an unexpected keyword argument 'zz'"),
)
case(
    "errors/series-shift-keyword",
    "Series.shift",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df["a"].shift(zz=1),
    raises=("TypeError", "NDFrame.shift() got an unexpected keyword argument 'zz'"),
)
case(
    "errors/pivot-table-unknown-function",
    "DataFrame.pivot_table",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.pivot_table(index="a", columns="c", values="b", aggfunc="bogus"),
    raises=("AttributeError", "'bogus' is not a valid function for 'DataFrameGroupBy' object"),
)
case(
    "errors/pivot-table-margins-name-taken",
    "DataFrame.pivot_table",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.fillna({"c": "x"}).pivot_table(
        index="c", values="b", margins=True, margins_name="one"
    ),
    raises=("ValueError", 'Conflicting name "one" in margins'),
)
case(
    "errors/pivot-table-margins-name-number",
    "DataFrame.pivot_table",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.fillna({"c": "x"}).pivot_table(
        index="c", values="b", margins=True, margins_name=1
    ),
    raises=("ValueError", "margins_name argument must be a string"),
)
case(
    "errors/get-dummies-missing-column",
    "pandas.get_dummies",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: pd.get_dummies(df, columns=["zz", "c"]),
    raises=("KeyError", "['zz'] not in index"),
)
case(
    "errors/set-index-missing-column",
    "DataFrame.set_index",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.set_index("nope"),
    raises=("KeyError", "None of ['nope'] are in the columns"),
)
case(
    "errors/sort-by-missing-column",
    "DataFrame.sort_values",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.sort_values("nope"),
    raises=("KeyError", "nope"),
)
case(
    "errors/missing-label",
    "DataFrame.loc",
    level="L4",
    frames=("keys_unique",),
    expr=lambda pd, df: df.set_index("key").loc[99999],
    in_process=True,
    note="a label that is not in the index is a KeyError naming it. In process because "
    "firepanda #1103 reads one row across columns in its Python layer",
    raises=("KeyError", "99999"),
)
case(
    "errors/missing-label-list",
    "DataFrame.loc",
    level="L4",
    frames=("keys_unique",),
    expr=lambda pd, df: df.set_index("key").loc[[0, 99999]],
    raises=("KeyError", "not in index"),
    note="a list where one label is missing fails whole rather than returning what it "
    "found, which is a decision pandas made and then kept",
)
case(
    "errors/position-out-of-bounds",
    "DataFrame.iloc",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.iloc[9999],
    in_process=True,
    note="a position past the end is an IndexError whose message says out-of-bounds. "
    "In process because firepanda #1104 checks it in its Python layer",
    raises=("IndexError", "out-of-bounds"),
)
case(
    "errors/drop-missing",
    "DataFrame.drop",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.drop(columns=["not_a_column"]),
    raises=("KeyError", "not_a_column"),
)
case(
    "errors/set-index-missing",
    "DataFrame.set_index",
    level="L4",
    frames=("keys_10",),
    expr=lambda pd, df: df.set_index("not_a_column"),
    raises=("KeyError", "not_a_column"),
)
case(
    "errors/sort-missing",
    "DataFrame.sort_values",
    level="L4",
    frames=("keys_10",),
    expr=lambda pd, df: df.sort_values("not_a_column"),
    raises=("KeyError", "not_a_column"),
)
case(
    "errors/groupby-missing",
    "DataFrame.groupby",
    level="L4",
    frames=("keys_10",),
    expr=lambda pd, df: df.groupby("not_a_column").sum(),
    raises=("KeyError", "not_a_column"),
)
case(
    "errors/attribute-missing",
    "DataFrame.head",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.not_a_method(),
    raises=("AttributeError", "not_a_method"),
    note="the api field points at something real because the case is about a name that "
    "is not there, and the registry will not take a name that is not in pandas",
)

# ---------------------------------------------------------------------------
# The pandas.errors types, which are the ones that matter most
# ---------------------------------------------------------------------------

case(
    "errors/merge-error",
    "pandas.merge",
    level="L4",
    frames=("keys_10",),
    expr=lambda pd, df: pd.merge(df, df.rename(columns={"key": "k", "value": "v"})),
    raises=("MergeError", "No common columns"),
    note="a MergeError and not a ValueError, which is exactly the distinction this "
    "section exists for",
)
case(
    "errors/merge-validate-fails",
    "pandas.merge",
    level="L4",
    frames=("keys_10",),
    expr=lambda pd, df: pd.merge(df, df, on="key", validate="one_to_one"),
    raises=("MergeError", "unique"),
    note="validate is the parameter whose entire job is to raise, so a case that does "
    "not make it raise is not testing it",
)
case(
    "errors/merge-cross-with-key",
    "pandas.merge",
    level="L4",
    frames=("keys_10",),
    expr=lambda pd, df: pd.merge(df, df, on="key", how="cross"),
    raises=("MergeError", "Can not pass on, right_on, left_on"),
    note="a cross join has no key, and naming one is a MergeError rather than a key "
    "that is quietly ignored",
)
case(
    "errors/merge-suffix-collision",
    "pandas.merge",
    level="L4",
    frames=("keys_10",),
    expr=lambda pd, df: pd.merge(df, df, on="key", suffixes=(None, None)),
    raises=("ValueError", "columns overlap but no suffix specified"),
    note="a plain ValueError and not a MergeError, which is inconsistent with the two "
    "cases above it and is exactly the sort of thing a copy of the API has to copy "
    "rather than tidy up",
)
case(
    "errors/boolean-wrong-length",
    "DataFrame.loc",
    level="L4",
    frames=("two", "tall"),
    expr=lambda pd, df: df.loc[[True, False, True]],
    raises=("IndexError", "Boolean index has wrong length"),
    note="the message says both lengths, which is the only reason this error is ever quick to fix",
)
case(
    "errors/too-many-indexers",
    "DataFrame.loc",
    level="L4",
    frames=("two", "tall"),
    expr=lambda pd, df: df.loc[0, 0, 0],
    raises=("IndexingError", "Too many indexers"),
    note="an IndexingError, which is a pandas type and not a builtin, and which almost "
    "nobody knows exists until they catch one",
)
case(
    "errors/duplicate-label",
    "DataFrame.pivot",
    level="L4",
    frames=("keys_two_column",),
    expr=lambda pd, df: df.pivot(index="left", columns="right", values="value"),
    raises=("ValueError", "duplicate entries"),
    note="pivot refuses duplicates and pivot_table aggregates them, and the refusal is "
    "the entire difference between the two",
)
case(
    "errors/undefined-variable",
    "DataFrame.query",
    level="L4",
    frames=("tall",),
    expr=lambda pd, df: df.query("not_a_column > 1"),
    raises=("UndefinedVariableError", "not_a_column"),
    in_process=True,
    note="the query parser has its own error type, which is a subclass of NameError "
    "and still has to be exactly itself. In process because the driver has no entry for "
    "query, which is firepanda's Python layer",
)
case(
    "errors/out-of-bounds-datetime",
    "pandas.to_datetime",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: pd.to_datetime(["1500-01-01"]).as_unit("ns"),
    raises=("OutOfBoundsDatetime", "1500-01-01"),
    note="a nanosecond timestamp cannot reach the sixteenth century, and the type that "
    "says so is a pandas one",
)
case(
    "errors/invalid-index",
    "pandas.DataFrame",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.set_index("a").loc["a string label"],
    raises=("KeyError", "a string label"),
)

# ---------------------------------------------------------------------------
# Types that do not go together
# ---------------------------------------------------------------------------

case(
    "errors/astype-string-to-int",
    "Series.astype",
    level="L4",
    frames=("strings_ascii",),
    expr=lambda pd, df: df["value"].astype("int64"),
    raises=("ValueError", "invalid literal"),
)
case(
    "errors/astype-null-to-int",
    "Series.astype",
    level="L4",
    frames=("float64_half_null",),
    expr=lambda pd, df: df["value"].astype("int64"),
    raises=("IntCastingNaNError", "Cannot convert non-finite values"),
    note="a float column with nulls cannot become a plain integer one, which is the "
    "whole reason the nullable integer dtypes exist",
)
case(
    "errors/add-string-to-number",
    "Series.add",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df["a"] + df["c"],
    raises=("TypeError", "not supported for dtype"),
)
case(
    "errors/compare-unordered-categorical",
    "Series.lt",
    level="L4",
    frames=("categorical_unordered",),
    expr=lambda pd, df: df["value"] < df["value"].cat.categories[0],
    raises=("TypeError", "Unordered Categoricals"),
    note="the ordered frame does this happily, so the pair of cases is what pins down "
    "what ordered means",
)
case(
    "errors/fillna-unknown-category",
    "Series.fillna",
    level="L4",
    frames=("categorical_unordered", "categorical_ordered"),
    expr=lambda pd, df: df["value"].fillna("not a category"),
    raises=("TypeError", "Cannot setitem"),
)
case(
    "errors/fillna-frame-on-column",
    "Series.fillna",
    level="L4",
    frames=("float64_half_null",),
    expr=lambda pd, df: df["value"].fillna(df),
    raises=("TypeError", "must be a scalar, dict or Series"),
    note="a frame is the one shape a column refuses, because a frame lines up on both "
    "axes and a column has only one for it to line up against",
)
case(
    "errors/isin-scalar",
    "Series.isin",
    level="L4",
    frames=("float64_half_null",),
    expr=lambda pd, df: df["value"].isin(1),
    raises=("TypeError", "only list-like objects are allowed"),
    note="a set of one is still a set, and pandas refuses the bare value rather than "
    "wrapping it, because a caller who wrote a scalar there meant a comparison",
)
case(
    "errors/isin-string",
    "Series.isin",
    level="L4",
    frames=("strings_null_heavy",),
    expr=lambda pd, df: df["value"].isin("v1"),
    raises=("TypeError", "only list-like objects are allowed"),
    note="a string is list-like to Python and is not list-like to pandas, which is what "
    "stops isin on a column of words from quietly meaning a set of letters",
)
case(
    "errors/truth-value-frame",
    "DataFrame.__bool__",
    level="L4",
    frames=("single", "two"),
    expr=lambda pd, df: bool(df),
    raises=("ValueError", "truth value of a DataFrame is ambiguous"),
    in_process=True,
    note="a frame of one row is refused as flatly as a frame of a thousand, because the "
    "question is what the rows mean rather than how many there are, and Python's own "
    "answer would be the row count. See spec 56 section 1",
)
case(
    "errors/truth-value-column",
    "Series.__bool__",
    level="L4",
    frames=("single", "two"),
    expr=lambda pd, df: bool(df["a"]),
    raises=("ValueError", "truth value of a Series is ambiguous"),
    in_process=True,
    note="the sentence names four members to use instead and one of them does not exist "
    "here yet, which is copied whole anyway because a sentence people search for is not "
    "a sentence to edit. See spec 56 section 9",
)
case(
    "errors/add-existing-category",
    "cat.add_categories",
    level="L4",
    frames=("categorical_unordered",),
    expr=lambda pd, df: df["value"].cat.add_categories([df["value"].cat.categories[0]]),
    raises=("ValueError", "must not include old categories"),
)
case(
    "errors/remove-missing-category",
    "cat.remove_categories",
    level="L4",
    frames=("categorical_unordered",),
    expr=lambda pd, df: df["value"].cat.remove_categories(["not a category"]),
    raises=("ValueError", "not"),
)
case(
    "errors/duplicate-categories",
    "pandas.CategoricalDtype",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: pd.CategoricalDtype(["a", "a", "b"]),
    raises=("ValueError", "unique"),
)
case(
    "errors/tz-convert-naive",
    "dt.tz_convert",
    level="L4",
    frames=("temporal_range",),
    expr=lambda pd, df: df["second"].dt.tz_convert("UTC"),
    raises=("TypeError", "tz-naive"),
    note="converting a naive timestamp is the mistake everybody makes once, and the "
    "message telling them to localize instead is worth as much as the type",
)
case(
    "errors/tz-localize-twice",
    "dt.tz_localize",
    level="L4",
    frames=("temporal_dst_forward",),
    expr=lambda pd, df: df["zoned"].dt.tz_localize("UTC"),
    raises=("TypeError", "Already tz-aware"),
)
case(
    "errors/nonexistent-time",
    "dt.tz_localize",
    level="L4",
    frames=("temporal_dst_forward",),
    expr=lambda pd, df: pd.Series(pd.to_datetime(["2024-03-10 02:30:00"])).dt.tz_localize(
        "America/New_York"
    ),
    raises=("ValueError", "is a nonexistent time due to daylight savings time"),
    note="the timestamp is written out here rather than taken from the frame, because "
    "the naive column in the corpus holds the UTC reading of each instant and a UTC "
    "reading is never in the gap. The frame is still the one that documents the "
    "transition, which is why the case runs on it",
)
case(
    "errors/ambiguous-time",
    "dt.tz_localize",
    level="L4",
    frames=("temporal_dst_back",),
    expr=lambda pd, df: df["zoned"].dt.tz_localize(None).dt.tz_localize("America/New_York"),
    raises=("ValueError", "Cannot infer dst time from"),
    note="dropping the zone gives the local wall clock, and in the autumn that clock "
    "reads one in the morning twice, so putting the zone back cannot say which of the "
    "two a given row meant",
)
case(
    "errors/compare-different-lengths",
    "Series.eq",
    level="L4",
    frames=("tall",),
    expr=lambda pd, df: df["value"] == df["value"].head(3),
    raises=("ValueError", "identically-labeled"),
)
case(
    "errors/concat-nothing",
    "pandas.concat",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: pd.concat([]),
    raises=("ValueError", "No objects to concatenate"),
    note="In process because firepanda's concat lives in its Python layer, which the "
    "driver cannot reach",
    in_process=True,
)
case(
    "errors/quantile-out-of-range",
    "Series.quantile",
    level="L4",
    frames=("tall",),
    expr=lambda pd, df: df["value"].quantile(1.5),
    raises=("ValueError", "percentiles"),
)
case(
    "errors/bad-interpolation",
    "Series.quantile",
    level="L4",
    frames=("tall",),
    expr=lambda pd, df: df["value"].quantile(0.5, interpolation="not a method"),
    raises=("ValueError", "is not a valid method"),
)
case(
    "errors/bad-frequency",
    "dt.floor",
    level="L4",
    frames=("temporal_range",),
    expr=lambda pd, df: df["second"].dt.floor("not a frequency"),
    raises=("ValueError", "Invalid frequency"),
)
case(
    "errors/rolling-negative-window",
    "DataFrame.rolling",
    level="L4",
    frames=("tall",),
    expr=lambda pd, df: df["value"].rolling(-1).sum(),
    raises=("ValueError", "window must be"),
)
case(
    "errors/ewm-two-decays",
    "DataFrame.ewm",
    level="L4",
    frames=("tall",),
    expr=lambda pd, df: df["value"].ewm(span=5, alpha=0.3).mean(),
    raises=("ValueError", "comass, span, halflife, and alpha"),
    note="four ways of writing one decay and exactly one of them may be given, which "
    "is a validation rule and not a computation",
)
case(
    "errors/item-on-many",
    "Series.item",
    level="L4",
    frames=("tall",),
    expr=lambda pd, df: df["value"].item(),
    raises=("ValueError", "size 1"),
)
case(
    "errors/set-categories-not-unique",
    "cat.set_categories",
    level="L4",
    frames=("categorical_unordered",),
    expr=lambda pd, df: df["value"].cat.set_categories(["a", "a"]),
    raises=("ValueError", "unique"),
)
case(
    "errors/reindex-duplicate-axis",
    "DataFrame.reindex",
    level="L4",
    frames=("keys_10",),
    expr=lambda pd, df: df.set_index("key").reindex([0, 1]),
    raises=("ValueError", "cannot reindex"),
    note="reindexing off an index with duplicates in it is ambiguous rather than "
    "expensive, so it refuses instead of guessing",
)
case(
    "errors/cut-integer-bins-with-infinity",
    "pandas.cut",
    level="L4",
    frames=("float64_no_nulls",),
    expr=lambda pd, df: pd.cut(df["value"], 4),
    raises=("ValueError", "cannot specify integer `bins` when input data contains infinity"),
    note="the float frames carry both infinities at the top, and an integer bin count "
    "has to work out a range, which an infinity makes impossible. The passing half of "
    "this pair is in the reshape section on a frame with no infinity in it",
)
case(
    "errors/quantile-on-boolean",
    "DataFrame.quantile",
    level="L4",
    frames=("tall",),
    expr=lambda pd, df: df.quantile(0.5, numeric_only=True),
    in_process=True,
    raises=("TypeError", "numpy boolean subtract"),
    note="numeric_only keeps the boolean column, because a bool is a number as far as "
    "the selection is concerned, and then the interpolation cannot subtract two of "
    "them. That is a pandas bug in every reading except the one where it is the "
    "documented behaviour, and either way it is what a caller sees. In process "
    "because firepanda reads quantile in its Python layer",
)
case(
    "errors/melt-value-name-collision",
    "DataFrame.melt",
    level="L4",
    frames=("keys_two_column", "tall"),
    expr=lambda pd, df: df.melt(id_vars=[df.columns[0]], value_vars=[df.columns[1]]),
    raises=("ValueError", "cannot match an element in the DataFrame columns"),
    note="the default value_name is the string value, and a frame that already has a "
    "column called value cannot take it. The check looks at every column and not only "
    "at the ones being melted, which is why naming an untouched column value is enough "
    "to break the call",
)
case(
    "errors/where-condition-not-boolean",
    "Series.where",
    level="L4",
    frames=("float64_no_nulls",),
    expr=lambda pd, df: df["value"].where(df["row"]),
    raises=("TypeError", "Boolean array expected for the condition"),
    note="a column of ones and zeros is the obvious thing to write and is the thing "
    "pandas will not take, which makes this the refusal a caller is most likely to "
    "meet in this family",
)
case(
    "errors/where-other-needs-an-axis",
    "DataFrame.where",
    level="L4",
    frames=("float64_no_nulls",),
    expr=lambda pd, df: df.where(df["value"] > 0, df["row"]),
    raises=("ValueError", "Must specify axis=0 or 1"),
    note="a column offered to a frame is a value per row or a value per column and "
    "pandas will not guess between them, which is one of the few places it refuses "
    "rather than picking the reading that is usually meant",
)


def _setitem(target: Any, accessor: str | None, key: Any, value: Any) -> None:
    """Writes into a copy, for the cases that expect the write to raise."""
    copy = target.copy()
    (copy if accessor is None else getattr(copy, accessor))[key] = value


case(
    "errors/setitem-value-does-not-fit",
    "Series.__setitem__",
    level="L4",
    frames=("tall",),
    expr=lambda pd, df: _setitem(df["key"], None, 0, 1.5),
    in_process=True,
    raises=("TypeError", "Invalid value"),
    note="pandas 3 refuses a value the column cannot hold as it is, rather than widening",
)
case(
    "errors/iloc-cannot-enlarge",
    "DataFrame.iloc",
    level="L4",
    frames=("tall",),
    expr=lambda pd, df: _setitem(df, "iloc", (len(df), 0), 0),
    in_process=True,
    raises=("IndexError", "cannot enlarge"),
    note="a position past the end refuses under iloc, where loc with a new label adds a row",
)
