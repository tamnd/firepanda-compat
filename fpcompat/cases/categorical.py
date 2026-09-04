"""The categorical dtype and the `cat` accessor.

A categorical is two things at once and every bug in one comes from forgetting the
other. It is a set of categories, which has an order and which exists whether or not
any row uses it, and it is an array of codes into that set. Almost everything
interesting follows from the categories outliving the data: removing a category leaves
the rows that used it as nulls, grouping produces a row for a category nobody used,
and two categoricals with the same values but different category sets do not compare.

The ordered frame is the one that can be sorted and compared with less than. The
unordered one raises for both, and that is in the errors section.
"""

from __future__ import annotations

from fpcompat.cases import case, section
from fpcompat.compare import Rules

section("categorical")

BOTH = ("categorical_unordered", "categorical_ordered")
ORDERED = ("categorical_ordered",)

# ---------------------------------------------------------------------------
# What a categorical is made of
# ---------------------------------------------------------------------------

case(
    "categorical/categories",
    "cat.categories",
    frames=BOTH,
    expr=lambda pd, df: df["value"].cat.categories,
    note="the categories are in their own order, which is not the order they appear in "
    "the data and not necessarily sorted either",
)
case(
    "categorical/codes",
    "cat.codes",
    frames=BOTH,
    expr=lambda pd, df: df["value"].cat.codes,
    note="a null is code minus one and not a null code, which means the codes column "
    "has no nulls in it at all",
)
case(
    "categorical/ordered",
    "cat.ordered",
    frames=BOTH,
    expr=lambda pd, df: df["value"].cat.ordered,
)
case(
    "categorical/dtype",
    "Series.dtype",
    frames=BOTH,
    expr=lambda pd, df: str(df["value"].dtype),
)
case(
    "categorical/nunique",
    "Series.nunique",
    frames=BOTH,
    expr=lambda pd, df: df["value"].nunique(),
    note="the number of used categories and not the number of categories, which is the "
    "difference between this and the length of the categories index",
)
case(
    "categorical/count",
    "Series.count",
    frames=BOTH,
    expr=lambda pd, df: df["value"].count(),
)
case(
    "categorical/isna",
    "Series.isna",
    frames=BOTH,
    expr=lambda pd, df: df["value"].isna(),
)
case(
    "categorical/value-counts",
    "Series.value_counts",
    frames=BOTH,
    expr=lambda pd, df: df["value"].value_counts().sort_index(),
    note="every category gets a row whether or not anything used it, which is the "
    "single most useful property of the dtype and the easiest one to drop",
)
case(
    "categorical/value-counts-dropna-false",
    "Series.value_counts",
    level="L3",
    covers=("dropna",),
    frames=BOTH,
    expr=lambda pd, df: df["value"].value_counts(dropna=False).sort_index(),
)

# ---------------------------------------------------------------------------
# Changing the category set
# ---------------------------------------------------------------------------

case(
    "categorical/add-categories",
    "cat.add_categories",
    level="L3",
    covers=("new_categories",),
    frames=BOTH,
    expr=lambda pd, df: df["value"].cat.add_categories(["zzz"]).cat.categories,
    note="a category nothing uses, which is legal and which changes what a groupby "
    "produces without changing a single row",
)
case(
    "categorical/remove-categories",
    "cat.remove_categories",
    level="L3",
    covers=("removals",),
    frames=BOTH,
    expr=lambda pd, df: df["value"].cat.remove_categories([df["value"].cat.categories[0]]),
    note="the rows that used it become nulls rather than raising, which is the thing "
    "that surprises people",
)
case(
    "categorical/remove-unused",
    "cat.remove_unused_categories",
    frames=BOTH,
    expr=lambda pd, df: df["value"].cat.remove_unused_categories().cat.categories,
)
case(
    "categorical/rename-categories",
    "cat.rename_categories",
    level="L3",
    covers=("new_categories",),
    frames=BOTH,
    expr=lambda pd, df: df["value"].cat.rename_categories(
        {name: name.upper() for name in df["value"].cat.categories}
    ),
    note="the codes do not move, only the labels change, so this is free and a full "
    "rebuild would also be correct and much slower",
)
case(
    "categorical/reorder-categories",
    "cat.reorder_categories",
    level="L3",
    covers=("new_categories",),
    frames=ORDERED,
    expr=lambda pd, df: df["value"].cat.reorder_categories(
        list(reversed(df["value"].cat.categories))
    ),
    note="the same set in a different order, which changes every comparison and every "
    "sort while leaving the values alone",
)
case(
    "categorical/set-categories",
    "cat.set_categories",
    level="L3",
    covers=("new_categories",),
    frames=BOTH,
    expr=lambda pd, df: df["value"].cat.set_categories(["a", "b", "c"]),
    note="a set that does not contain what is in the data, so the rows that used the "
    "missing ones become nulls, which is remove and add in one call",
)
case(
    "categorical/as-ordered",
    "cat.as_ordered",
    frames=("categorical_unordered",),
    expr=lambda pd, df: df["value"].cat.as_ordered().cat.ordered,
)
case(
    "categorical/as-unordered",
    "cat.as_unordered",
    frames=ORDERED,
    expr=lambda pd, df: df["value"].cat.as_unordered().cat.ordered,
)

# ---------------------------------------------------------------------------
# Order, which only the ordered frame has
# ---------------------------------------------------------------------------

case(
    "categorical/sort-ordered",
    "Series.sort_values",
    frames=ORDERED,
    expr=lambda pd, df: df["value"].sort_values(),
    rules=Rules(strict_index=True),
    note="sorted by category order and not alphabetically, and the corpus sets those "
    "two to disagree on purpose",
)
case(
    "categorical/sort-descending",
    "Series.sort_values",
    level="L3",
    covers=("ascending",),
    frames=ORDERED,
    expr=lambda pd, df: df["value"].sort_values(ascending=False),
    rules=Rules(strict_index=True),
)
case(
    "categorical/min",
    "Series.min",
    frames=ORDERED,
    expr=lambda pd, df: df["value"].min(),
)
case(
    "categorical/max",
    "Series.max",
    frames=ORDERED,
    expr=lambda pd, df: df["value"].max(),
)
case(
    "categorical/compare-lt",
    "Series.lt",
    frames=ORDERED,
    expr=lambda pd, df: df["value"] < df["value"].cat.categories[-1],
)
case(
    "categorical/compare-eq",
    "Series.eq",
    frames=BOTH,
    expr=lambda pd, df: df["value"] == df["value"].cat.categories[0],
    note="equality works on an unordered categorical, which is why it is here and less than is not",
)
case(
    "categorical/sort-unordered",
    "Series.sort_values",
    frames=("categorical_unordered",),
    expr=lambda pd, df: df["value"].sort_values(),
    rules=Rules(strict_index=True),
    note="an unordered categorical still sorts, by the category order, which is a "
    "surprise given that comparing two of them raises",
)

# ---------------------------------------------------------------------------
# Categoricals meeting everything else
# ---------------------------------------------------------------------------

case(
    "categorical/astype-string",
    "Series.astype",
    level="L3",
    covers=("dtype",),
    frames=BOTH,
    expr=lambda pd, df: df["value"].astype("str"),
)
case(
    "categorical/astype-category",
    "Series.astype",
    level="L3",
    covers=("dtype",),
    frames=("strings_ascii", "keys_awkward", "strings_null_heavy"),
    expr=lambda pd, df: df.iloc[:, -1].astype("category").cat.categories,
    note="building a categorical from strings sorts the categories, and a null does "
    "not become a category",
)
case(
    "categorical/groupby",
    "GroupBy.size",
    frames=BOTH,
    expr=lambda pd, df: df.groupby("value", observed=False).size(),
)
case(
    "categorical/groupby-agg",
    "GroupBy.sum",
    frames=BOTH,
    expr=lambda pd, df: df.groupby("value", observed=True)["row"].sum(),
)
case(
    "categorical/isin",
    "Series.isin",
    level="L3",
    covers=("values",),
    frames=BOTH,
    expr=lambda pd, df: df["value"].isin([df["value"].cat.categories[0]]),
)
case(
    "categorical/fillna",
    "Series.fillna",
    level="L3",
    covers=("value",),
    frames=BOTH,
    expr=lambda pd, df: df["value"].fillna(df["value"].cat.categories[0]),
    note="filling with something that is a category works and filling with something "
    "that is not raises, and the raising half is in the errors section",
)
case(
    "categorical/dropna",
    "Series.dropna",
    frames=BOTH,
    expr=lambda pd, df: df["value"].dropna(),
    rules=Rules(strict_index=True),
)
case(
    "categorical/unique",
    "Series.unique",
    frames=BOTH,
    expr=lambda pd, df: df["value"].unique(),
    note="unique on a categorical gives a categorical back and it keeps the full "
    "category set, which is not what unique does anywhere else",
)
case(
    "categorical/concat-same",
    "pandas.concat",
    level="L3",
    covers=("objs",),
    frames=BOTH,
    expr=lambda pd, df: pd.concat([df, df])["value"].dtype.categories,
    note="two categoricals with the same categories concatenate to a categorical, and "
    "two with different ones fall back to strings, which is a silent dtype change",
)
case(
    "categorical/str-accessor",
    "str.upper",
    frames=BOTH,
    expr=lambda pd, df: df["value"].astype("str").str.upper(),
)
case(
    "categorical/from-codes",
    "pandas.Categorical",
    frames=BOTH,
    expr=lambda pd, df: pd.Categorical.from_codes(
        df["value"].cat.codes, categories=df["value"].cat.categories
    ),
)
case(
    "categorical/dtype-construct",
    "pandas.CategoricalDtype",
    level="L3",
    covers=("categories", "ordered"),
    frames=BOTH,
    expr=lambda pd, df: str(pd.CategoricalDtype(["b", "a"], ordered=True)),
)
