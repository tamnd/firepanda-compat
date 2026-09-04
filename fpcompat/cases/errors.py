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
    "errors/missing-label",
    "DataFrame.loc",
    level="L4",
    frames=("keys_unique",),
    expr=lambda pd, df: df.set_index("key").loc[99999],
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
    note="the query parser has its own error type, which is a subclass of NameError "
    "and still has to be exactly itself",
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
    raises=("TypeError", "numpy boolean subtract"),
    note="numeric_only keeps the boolean column, because a bool is a number as far as "
    "the selection is concerned, and then the interpolation cannot subtract two of "
    "them. That is a pandas bug in every reading except the one where it is the "
    "documented behaviour, and either way it is what a caller sees",
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
