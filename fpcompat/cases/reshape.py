"""Joins, concatenation and the long to wide operations.

A join is the operation where an implementation has the most freedom to be wrong in a
way that still looks plausible. The row count coming out of an inner join on keys that
repeat is a product and not a sum, the row order pandas produces is a consequence of
how it hashes rather than a promise, and a null key never matches another null key
even though a null value compares equal to itself in a groupby. Those three facts are
what most of this section is about.

Every join here joins a frame against a shape of itself, because the corpus has no
pair of frames designed to be joined and inventing one at case time would put the
interesting part of the fixture in the case file instead of in the corpus.
"""

from __future__ import annotations

from fpcompat.cases import case, section
from fpcompat.compare import Rules

section("reshape")

JOIN_ORDER = Rules(
    relaxations=frozenset({"row_order"}),
    reason="pandas does not promise the row order out of a merge, it is a consequence "
    "of the hash table, so the rows are compared as a multiset",
)


def _right(df, column="key"):
    """The right hand side of a self join, with the value column renamed."""
    return df.rename(columns={"value": "other"})


# ---------------------------------------------------------------------------
# Merges
# ---------------------------------------------------------------------------

for how in ("inner", "left", "right", "outer"):
    case(
        f"reshape/merge-{how}",
        "pandas.merge",
        level="L3",
        covers=("left", "right", "on", "how"),
        frames=("keys_10", "keys_1000", "keys_unique", "keys_awkward"),
        expr=(lambda kind: lambda pd, df: pd.merge(df, _right(df), on="key", how=kind))(how),
        rules=JOIN_ORDER,
        note="the ten key frame is the one where the row count is a product, since "
        "roughly six rows on each side of every key gives thirty six out",
    )

case(
    "reshape/merge-null-key",
    "pandas.merge",
    level="L3",
    covers=("left", "right", "on", "how"),
    frames=("keys_awkward",),
    expr=lambda pd, df: pd.merge(df, _right(df), on="key", how="outer"),
    rules=JOIN_ORDER,
    note="a null key does not match another null key, which is the SQL rule and the "
    "opposite of what groupby does with the same column",
)
case(
    "reshape/merge-two-keys",
    "pandas.merge",
    level="L3",
    covers=("left", "right", "on"),
    frames=("keys_two_column",),
    expr=lambda pd, df: pd.merge(df, _right(df), on=["left", "right"]),
    rules=JOIN_ORDER,
)
case(
    "reshape/merge-suffixes",
    "pandas.merge",
    level="L3",
    covers=("left", "right", "on", "suffixes"),
    frames=("keys_10",),
    expr=lambda pd, df: pd.merge(df, df, on="key", suffixes=("_a", "_b")),
    rules=JOIN_ORDER,
    note="an overlapping column name that is not a key gets a suffix, and what happens "
    "when the suffixed name collides with an existing one is the next case up",
)
case(
    "reshape/merge-indicator",
    "pandas.merge",
    level="L3",
    covers=("left", "right", "on", "how", "indicator"),
    frames=("keys_10", "keys_awkward"),
    expr=lambda pd, df: pd.merge(df, _right(df).head(10), on="key", how="outer", indicator=True),
    rules=JOIN_ORDER,
    note="the indicator column is a categorical with three categories in it whether or "
    "not all three occur, which is easy to get wrong",
)
case(
    "reshape/merge-left-on-right-on",
    "pandas.merge",
    level="L3",
    covers=("left", "right", "left_on", "right_on"),
    frames=("keys_two_column",),
    expr=lambda pd, df: pd.merge(df, df, left_on="left", right_on="left"),
    rules=JOIN_ORDER,
)
case(
    "reshape/merge-index",
    "pandas.merge",
    level="L3",
    covers=("left", "right", "left_index", "right_index"),
    frames=("keys_unique",),
    expr=lambda pd, df: pd.merge(
        df.set_index("key"), _right(df).set_index("key"), left_index=True, right_index=True
    ),
    rules=JOIN_ORDER,
)
case(
    "reshape/merge-validate",
    "pandas.merge",
    level="L3",
    covers=("left", "right", "on", "validate"),
    frames=("keys_unique",),
    expr=lambda pd, df: pd.merge(df, _right(df), on="key", validate="one_to_one"),
    rules=JOIN_ORDER,
)
case(
    "reshape/merge-method",
    "DataFrame.merge",
    level="L3",
    covers=("right", "on", "how"),
    frames=("keys_10",),
    expr=lambda pd, df: df.merge(_right(df), on="key", how="left"),
    rules=JOIN_ORDER,
    note="the method spelling of the same thing, which has to agree with the function",
)
case(
    "reshape/join",
    "DataFrame.join",
    level="L3",
    covers=("other", "how"),
    frames=("keys_unique",),
    expr=lambda pd, df: df.set_index("key").join(_right(df).set_index("key"), how="left"),
    rules=JOIN_ORDER,
)
case(
    "reshape/merge-asof",
    "pandas.merge_asof",
    level="L3",
    covers=("left", "right", "on"),
    frames=("keys_unique",),
    expr=lambda pd, df: pd.merge_asof(
        df.sort_values("key"), _right(df).sort_values("key"), on="key"
    ),
    note="the nearest earlier key rather than an equal one, which is a different "
    "algorithm and not a variation on the join",
)
case(
    "reshape/merge-ordered",
    "pandas.merge_ordered",
    level="L3",
    covers=("left", "right", "on"),
    frames=("keys_unique",),
    expr=lambda pd, df: pd.merge_ordered(df, _right(df), on="key"),
)

# ---------------------------------------------------------------------------
# Concatenation
# ---------------------------------------------------------------------------

case(
    "reshape/concat-rows",
    "pandas.concat",
    level="L3",
    covers=("objs",),
    frames=("two", "keys_10", "empty"),
    expr=lambda pd, df: pd.concat([df, df]),
    note="the index repeats rather than being renumbered, which is the default and the "
    "thing that surprises people",
    rules=Rules(strict_index=True),
)
case(
    "reshape/concat-ignore-index",
    "pandas.concat",
    level="L3",
    covers=("objs", "ignore_index"),
    frames=("two", "keys_10"),
    expr=lambda pd, df: pd.concat([df, df], ignore_index=True),
)
case(
    "reshape/concat-columns",
    "pandas.concat",
    level="L3",
    covers=("objs", "axis"),
    frames=("two",),
    expr=lambda pd, df: pd.concat([df["a"], df["b"]], axis=1),
)
case(
    "reshape/concat-mismatched",
    "pandas.concat",
    level="L3",
    covers=("objs",),
    frames=("two",),
    expr=lambda pd, df: pd.concat([df[["a", "b"]], df[["b", "c"]]]),
    note="the union of the columns with nulls in the gaps, and the column order of the "
    "result is not the order of either input",
    rules=Rules(strict_index=True),
)
case(
    "reshape/concat-join-inner",
    "pandas.concat",
    level="L3",
    covers=("objs", "join"),
    frames=("two",),
    expr=lambda pd, df: pd.concat([df[["a", "b"]], df[["b", "c"]]], join="inner"),
    rules=Rules(strict_index=True),
)
case(
    "reshape/concat-keys",
    "pandas.concat",
    level="L3",
    covers=("objs", "keys"),
    frames=("two",),
    expr=lambda pd, df: pd.concat([df, df], keys=["first", "second"]),
    rules=Rules(strict_index=True),
    note="a two level index built out of the keys, which is how anyone remembers which "
    "half a row came from",
)
case(
    "reshape/concat-empty",
    "pandas.concat",
    level="L3",
    covers=("objs",),
    frames=("empty",),
    expr=lambda pd, df: pd.concat([df, df]),
    note="concatenating two empty frames has to give an empty frame with the right "
    "column types, not an empty frame with no columns",
)

# ---------------------------------------------------------------------------
# Long and wide
# ---------------------------------------------------------------------------

case(
    "reshape/pivot",
    "DataFrame.pivot",
    level="L3",
    covers=("index", "columns", "values"),
    frames=("keys_two_column",),
    expr=lambda pd, df: df.drop_duplicates(subset=["left", "right"]).pivot(
        index="left", columns="right", values="value"
    ),
)
case(
    "reshape/pivot-table",
    "DataFrame.pivot_table",
    level="L3",
    covers=("index", "columns", "values", "aggfunc"),
    frames=("keys_two_column",),
    expr=lambda pd, df: df.pivot_table(
        index="left", columns="right", values="value", aggfunc="sum"
    ),
    note="the table form aggregates duplicates instead of raising, which is the only "
    "difference from pivot and the reason both exist",
)
case(
    "reshape/pivot-table-fill",
    "DataFrame.pivot_table",
    level="L3",
    covers=("index", "columns", "values", "aggfunc", "fill_value"),
    frames=("keys_two_column",),
    expr=lambda pd, df: df.pivot_table(
        index="left", columns="right", values="value", aggfunc="mean", fill_value=0
    ),
)
case(
    "reshape/melt",
    "DataFrame.melt",
    level="L3",
    covers=("id_vars", "value_vars", "value_name"),
    frames=("two", "keys_two_column"),
    expr=lambda pd, df: df.melt(
        id_vars=[df.columns[0]], value_vars=[df.columns[1]], value_name="amount"
    ),
    note="value_name is given rather than left at its default, because the default is "
    "the string value and half the corpus has a column called that. What happens then "
    "is in the errors section",
)
case(
    "reshape/melt-function",
    "pandas.melt",
    level="L3",
    covers=("frame", "id_vars"),
    frames=("two",),
    expr=lambda pd, df: pd.melt(df, id_vars=["a"]),
    note="melting mixed types gives an object value column, which is a real answer and not a bug",
)
case(
    "reshape/stack",
    "DataFrame.stack",
    frames=("keys_10", "wide"),
    expr=lambda pd, df: df.head(5).stack(),
)
case(
    "reshape/unstack",
    "DataFrame.unstack",
    frames=("keys_two_column",),
    expr=lambda pd, df: df.groupby(["left", "right"])["value"].sum().unstack(),
)
case(
    "reshape/unstack-fill",
    "DataFrame.unstack",
    level="L3",
    covers=("fill_value",),
    frames=("keys_two_column",),
    expr=lambda pd, df: df.groupby(["left", "right"])["value"].sum().unstack(fill_value=0),
    note="without a fill value the missing combinations become nulls and widen the "
    "integer column to float, which is the quiet part",
)
case(
    "reshape/transpose",
    "DataFrame.T",
    frames=("single", "keys_10"),
    expr=lambda pd, df: df.head(4).T,
)
case(
    "reshape/explode",
    "DataFrame.explode",
    level="L3",
    covers=("column",),
    frames=("nested_list",),
    expr=lambda pd, df: df.explode("value"),
    rules=Rules(strict_index=True),
    note="an empty list gives one row with a null in it rather than no rows, which is "
    "the case that separates explode from a flatten",
)
case(
    "reshape/get-dummies",
    "pandas.get_dummies",
    level="L3",
    covers=("data",),
    frames=("categorical_unordered", "keys_awkward"),
    expr=lambda pd, df: pd.get_dummies(df.iloc[:, -2 if "row" in df else 0]),
)
case(
    "reshape/crosstab",
    "pandas.crosstab",
    level="L3",
    covers=("index", "columns"),
    frames=("keys_two_column",),
    expr=lambda pd, df: pd.crosstab(df["left"], df["right"]),
)
case(
    "reshape/cut",
    "pandas.cut",
    level="L3",
    covers=("x", "bins"),
    frames=("tall", "keys_1000"),
    expr=lambda pd, df: pd.cut(df["value"], 4).value_counts().sort_index(),
    note="an integer bin count refuses a column containing an infinity, which is why "
    "the float frames are not here and why that refusal is its own case in the errors "
    "section",
)
case(
    "reshape/qcut",
    "pandas.qcut",
    level="L3",
    covers=("x", "q"),
    frames=("tall",),
    expr=lambda pd, df: pd.qcut(df["value"], 4).value_counts().sort_index(),
)
case(
    "reshape/factorize",
    "pandas.factorize",
    level="L3",
    covers=("values",),
    frames=("keys_10", "keys_awkward", "strings_null_heavy"),
    expr=lambda pd, df: pd.factorize(df.iloc[:, 0]),
    note="a null gets code minus one and does not appear in the uniques, which is the "
    "one rule in factorize that is not obvious",
)
case(
    "reshape/duplicated-frame",
    "DataFrame.drop_duplicates",
    frames=("keys_10", "keys_two_column"),
    expr=lambda pd, df: df.drop_duplicates(),
    rules=Rules(strict_index=True),
)
case(
    "reshape/combine-first",
    "DataFrame.combine_first",
    level="L3",
    covers=("other",),
    frames=("float64_half_null",),
    expr=lambda pd, df: df.combine_first(df.fillna(0.0)),
)
case(
    "reshape/align",
    "DataFrame.align",
    level="L3",
    covers=("other", "join"),
    frames=("keys_unique",),
    expr=lambda pd, df: df.head(10).align(df.tail(10), join="outer")[0],
    rules=Rules(strict_index=True),
)
case(
    "reshape/compare",
    "DataFrame.compare",
    level="L3",
    covers=("other",),
    frames=("float64_half_null",),
    expr=lambda pd, df: df.compare(df.fillna(0.0)),
    rules=Rules(strict_index=True),
)
