"""Grouping and aggregation.

The three key frames are the whole design of this section. Ten keys over sixty four
rows is the small group case where a hash table never resizes and every group has
ties. A thousand keys over sixty four rows is the case where almost every group has
one row, which is where a per group allocation shows up as a cost and where an
implementation that assumes groups are big falls over. Unique keys is the degenerate
case where grouping is a sort with extra steps.

The awkward frame is the one that carries a null key, an empty string key and a key
that is only different from another by case. Every one of those is a decision. A null
key is dropped by default and pandas has a parameter to keep it, and an implementation
that quietly kept it would pass every case that used a clean frame.
"""

from __future__ import annotations

from fpcompat.cases import case, section
from fpcompat.compare import Rules, Tolerance

section("groupby")

KEYED = ("keys_10", "keys_1000", "keys_unique", "keys_awkward")
SMALL = ("keys_10", "keys_awkward")

SPREAD = Rules(
    tolerance=Tolerance.STATISTICAL,
    reason="a variance within a group is a different summation order in every "
    "implementation and the groups here are small enough for that to show",
)

# ---------------------------------------------------------------------------
# The aggregations
# ---------------------------------------------------------------------------

for name in (
    "sum",
    "mean",
    "min",
    "max",
    "count",
    "size",
    "first",
    "last",
    "median",
    "prod",
    "nunique",
):
    case(
        f"groupby/{name}",
        f"GroupBy.{name}",
        frames=KEYED,
        expr=(lambda method: lambda pd, df: getattr(df.groupby("key"), method)())(name),
        note="the awkward frame has a null key, which is dropped, so the row count of "
        "the answer is smaller than the number of distinct keys in the column",
    )

for name in ("std", "var", "sem", "skew"):
    case(
        f"groupby/{name}",
        f"GroupBy.{name}",
        frames=("keys_10", "keys_awkward"),
        expr=(lambda method: lambda pd, df: getattr(df.groupby("key"), method)())(name),
        rules=SPREAD,
        note="a group of one gives a null and not a zero, which the thousand key frame "
        "would make into sixty four nulls and no information",
    )

case(
    "groupby/dropna-false",
    "DataFrame.groupby",
    level="L3",
    covers=("by", "dropna"),
    frames=("keys_awkward",),
    expr=lambda pd, df: df.groupby("key", dropna=False).sum(),
    note="the null key becomes a group of its own, and where it sorts is the second "
    "half of the question",
)
case(
    "groupby/sort-false",
    "DataFrame.groupby",
    level="L3",
    covers=("by", "sort"),
    frames=SMALL,
    expr=lambda pd, df: df.groupby("key", sort=False).sum(),
    rules=Rules(
        relaxations=frozenset({"grouped_order"}),
        reason="sort off means first seen order, which pandas documents as not "
        "guaranteed, so the groups are compared as a set",
    ),
)
case(
    "groupby/as-index-false",
    "DataFrame.groupby",
    level="L3",
    covers=("by", "as_index"),
    frames=SMALL,
    expr=lambda pd, df: df.groupby("key", as_index=False).sum(),
    note="the key comes back as a column rather than as the index, which is a "
    "different shape and not a different computation",
)
case(
    "groupby/two-keys",
    "DataFrame.groupby",
    level="L3",
    covers=("by",),
    frames=("keys_two_column",),
    expr=lambda pd, df: df.groupby(["left", "right"]).sum(),
)
case(
    "groupby/two-keys-size",
    "GroupBy.size",
    frames=("keys_two_column",),
    expr=lambda pd, df: df.groupby(["left", "right"]).size(),
)
case(
    "groupby/series",
    "Series.groupby",
    level="L3",
    covers=("by",),
    frames=KEYED,
    expr=lambda pd, df: df["value"].groupby(df["key"]).sum(),
)
case(
    "groupby/observed-categorical",
    "DataFrame.groupby",
    level="L3",
    covers=("by", "observed"),
    frames=("categorical_unordered",),
    expr=lambda pd, df: df.groupby("value", observed=False).size(),
    note="a category with no rows in it is still a group when observed is off, which "
    "is the only way to get a row for something that is not in the data",
)
case(
    "groupby/observed-true",
    "DataFrame.groupby",
    level="L3",
    covers=("by", "observed"),
    frames=("categorical_unordered", "categorical_ordered"),
    expr=lambda pd, df: df.groupby("value", observed=True).size(),
)
case(
    "groupby/on-tall",
    "GroupBy.mean",
    frames=("tall",),
    expr=lambda pd, df: df.groupby("key")["value"].mean(),
    rules=Rules(
        tolerance=Tolerance.ACCUMULATION,
        reason="ten thousand rows over a small number of keys means each group mean is "
        "a long summation",
    ),
)
case(
    "groupby/bool-key",
    "GroupBy.sum",
    frames=("tall",),
    expr=lambda pd, df: df.groupby("flag")["value"].sum(),
    rules=Rules(
        tolerance=Tolerance.ACCUMULATION,
        reason="two groups of five thousand doubles each",
    ),
)

# ---------------------------------------------------------------------------
# agg, which is the part with the most spellings
# ---------------------------------------------------------------------------

case(
    "groupby/agg-string",
    "GroupBy.agg",
    level="L3",
    covers=("func",),
    frames=SMALL,
    expr=lambda pd, df: df.groupby("key").agg("sum"),
)
case(
    "groupby/agg-list",
    "GroupBy.agg",
    level="L3",
    covers=("func",),
    frames=SMALL,
    expr=lambda pd, df: df.groupby("key")["value"].agg(["sum", "mean", "count"]),
    note="a list gives one column per function, named after the function",
)
case(
    "groupby/agg-dict",
    "GroupBy.agg",
    level="L3",
    covers=("func",),
    frames=("keys_two_column",),
    expr=lambda pd, df: df.groupby("left").agg({"right": "max", "value": "sum"}),
)
case(
    "groupby/agg-named",
    "GroupBy.agg",
    frames=SMALL,
    expr=lambda pd, df: df.groupby("key").agg(total=("value", "sum"), rows=("value", "count")),
    note="the named form, which is the only one that lets a column be aggregated twice "
    "under two different names",
)
case(
    "groupby/agg-lambda",
    "GroupBy.agg",
    level="L3",
    covers=("func",),
    frames=("keys_10",),
    expr=lambda pd, df: df.groupby("key")["value"].agg(lambda group: group.max() - group.min()),
    note="an arbitrary Python callable, which is the escape hatch and which any "
    "implementation with a fast path has to fall back out of",
)
case(
    "groupby/agg-multiple-columns",
    "GroupBy.agg",
    level="L3",
    covers=("func",),
    frames=("keys_two_column",),
    expr=lambda pd, df: df.groupby("left").agg(["min", "max"]),
    note="a two level column index, which is where the shape gets interesting",
)

# ---------------------------------------------------------------------------
# The transforms, which keep the row count
# ---------------------------------------------------------------------------

for name in ("cumsum", "cumcount", "cummax", "cummin", "cumprod", "ngroup", "rank"):
    case(
        f"groupby/{name}",
        f"GroupBy.{name}",
        frames=("keys_10", "keys_1000"),
        expr=(lambda method: lambda pd, df: getattr(df.groupby("key")["value"], method)())(name),
        note="a transform gives back the original row count in the original order, "
        "which is the property that separates it from an aggregation",
    )

case(
    "groupby/transform-sum",
    "GroupBy.transform",
    level="L3",
    covers=("func",),
    frames=("keys_10", "keys_1000"),
    expr=lambda pd, df: df.groupby("key")["value"].transform("sum"),
)
case(
    "groupby/transform-mean",
    "GroupBy.transform",
    level="L3",
    covers=("func",),
    frames=("keys_10", "tall"),
    expr=lambda pd, df: df.groupby("key")["value"].transform("mean"),
    rules=Rules(
        tolerance=Tolerance.ACCUMULATION,
        reason="the tall frame makes each group mean a five thousand element sum",
    ),
)
case(
    "groupby/transform-lambda",
    "GroupBy.transform",
    level="L3",
    covers=("func",),
    frames=("keys_10",),
    expr=lambda pd, df: df.groupby("key")["value"].transform(lambda group: group - group.mean()),
)
case(
    "groupby/shift",
    "GroupBy.shift",
    frames=("keys_10", "keys_1000"),
    expr=lambda pd, df: df.groupby("key")["value"].shift(),
    note="shifting within a group and not across the frame, so every group's first row "
    "is a null no matter where it sits in the frame",
)
case(
    "groupby/diff",
    "GroupBy.diff",
    frames=("keys_10",),
    expr=lambda pd, df: df.groupby("key")["value"].diff(),
)
case(
    "groupby/ffill",
    "GroupBy.ffill",
    frames=("keys_awkward",),
    expr=lambda pd, df: df.groupby("key")["value"].ffill(),
)
case(
    "groupby/pct-change",
    "GroupBy.pct_change",
    frames=("keys_10",),
    expr=lambda pd, df: df.groupby("key")["value"].pct_change(),
)

# ---------------------------------------------------------------------------
# Picking rows out of groups
# ---------------------------------------------------------------------------

case(
    "groupby/head",
    "GroupBy.head",
    level="L3",
    covers=("n",),
    frames=("keys_10", "keys_1000"),
    expr=lambda pd, df: df.groupby("key").head(2),
)
case(
    "groupby/tail",
    "GroupBy.tail",
    level="L3",
    covers=("n",),
    frames=("keys_10",),
    expr=lambda pd, df: df.groupby("key").tail(2),
)
case(
    "groupby/nth",
    "GroupBy.nth",
    frames=("keys_10",),
    expr=lambda pd, df: df.groupby("key").nth(0),
)
case(
    "groupby/nth-negative",
    "GroupBy.nth",
    frames=("keys_10",),
    expr=lambda pd, df: df.groupby("key").nth(-1),
)
case(
    "groupby/idxmax",
    "GroupBy.idxmax",
    frames=("keys_10", "keys_1000"),
    expr=lambda pd, df: df.groupby("key")["value"].idxmax(),
)
case(
    "groupby/idxmin",
    "GroupBy.idxmin",
    frames=("keys_10",),
    expr=lambda pd, df: df.groupby("key")["value"].idxmin(),
)
case(
    "groupby/quantile",
    "GroupBy.quantile",
    level="L3",
    covers=("q",),
    frames=("keys_10", "tall"),
    expr=lambda pd, df: df.groupby("key")["value"].quantile(0.5),
    rules=Rules(
        tolerance=Tolerance.STATISTICAL,
        reason="an interpolated quantile is an average of two neighbours",
    ),
)
case(
    "groupby/value-counts",
    "GroupBy.value_counts",
    frames=("keys_10",),
    expr=lambda pd, df: df.groupby("key")["value"].value_counts(),
    rules=Rules(
        relaxations=frozenset({"row_order"}),
        reason="sorted by count within each group, and the counts are almost all one",
    ),
)
case(
    "groupby/ngroups",
    "GroupBy.ngroups",
    frames=KEYED,
    expr=lambda pd, df: df.groupby("key").ngroups,
)
case(
    "groupby/describe",
    "GroupBy.describe",
    frames=("keys_10",),
    expr=lambda pd, df: df.groupby("key")["value"].describe(),
    rules=Rules(
        tolerance=Tolerance.STATISTICAL,
        reason="describe includes a standard deviation and three quantiles",
    ),
)
case(
    "groupby/apply-frame",
    "GroupBy.apply",
    level="L3",
    covers=("func",),
    frames=("keys_10",),
    expr=lambda pd, df: df.groupby("key")[["value"]].apply(lambda group: group.sum()),
    note="apply is the slow path that has to exist, and it is here so that an "
    "implementation cannot claim groupby coverage without it",
)
case(
    "groupby/filter",
    "GroupBy.filter",
    level="L3",
    covers=("func",),
    frames=("keys_10", "keys_1000"),
    expr=lambda pd, df: df.groupby("key").filter(lambda group: len(group) > 5),
)
case(
    "groupby/any",
    "GroupBy.any",
    frames=("tall",),
    expr=lambda pd, df: df.groupby("key")["flag"].any(),
)
case(
    "groupby/all",
    "GroupBy.all",
    frames=("tall",),
    expr=lambda pd, df: df.groupby("key")["flag"].all(),
)
case(
    "groupby/cov",
    "GroupBy.cov",
    frames=("keys_two_column",),
    expr=lambda pd, df: df.groupby("left").cov(),
    rules=SPREAD,
)
case(
    "groupby/corr",
    "GroupBy.corr",
    frames=("keys_two_column",),
    expr=lambda pd, df: df.groupby("left").corr(),
    rules=SPREAD,
)
case(
    "groupby/grouper",
    "pandas.Grouper",
    frames=("keys_10",),
    expr=lambda pd, df: df.groupby(pd.Grouper(key="key")).sum(),
)
