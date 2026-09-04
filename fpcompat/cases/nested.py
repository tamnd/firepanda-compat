"""Lists and structs, through the Arrow backed accessors.

The smallest section, because pandas itself barely supports these. A list column in a
numpy backed frame is a column of Python list objects and almost nothing works on it,
and the `list` and `struct` accessors only exist on an Arrow backed column. That is
exactly why the section matters for firepanda: a library whose memory is Arrow all the
way down should be good at this, and the only way to know whether it agrees with
pandas is to ask pandas the same questions through the accessors that do exist.

The frames are converted to Arrow backed dtypes inside the case expression rather than
in the corpus, because the corpus is Arrow already and how it lands in pandas is part
of what is being tested everywhere else.
"""

from __future__ import annotations

from fpcompat.cases import case, section
from fpcompat.compare import Rules

section("nested")

LISTS = ("nested_list",)
STRUCTS = ("nested_struct", "nested_deep")
STRICT = Rules(strict_index=True)


def _arrow(pd, series):
    """The same column with an Arrow backed dtype, which is where the accessors live.

    The type is inferred from the values rather than carried across from the corpus,
    because by the time a case sees the column it is already a column of Python
    objects and the Arrow type it came from is gone. That loss is itself worth
    pinning, which is what the dtype case below does.
    """
    import pyarrow as pa

    return series.astype(pd.ArrowDtype(pa.array(series).type))


# ---------------------------------------------------------------------------
# Lists
# ---------------------------------------------------------------------------

case(
    "nested/list-dtype",
    "Series.dtype",
    frames=LISTS,
    expr=lambda pd, df: str(_arrow(pd, df["value"]).dtype),
    note="the corpus stores this as a large list and pandas gives it back as objects, "
    "so the width of the offsets is gone by the time anything can look. That is a "
    "real loss and this case is what says so out loud",
)
case(
    "nested/list-len",
    "list.len",
    frames=LISTS,
    expr=lambda pd, df: _arrow(pd, df["value"]).list.len(),
    note="an empty list is length zero and a null list is a null, which are two "
    "different rows in the corpus on purpose",
)
case(
    "nested/list-getitem",
    "Series.list",
    level="L4",
    frames=LISTS,
    expr=lambda pd, df: _arrow(pd, df["value"]).list[0],
    raises=("ArrowInvalid", "out of bounds"),
    note="the corpus has an empty list in it and the accessor refuses the whole column "
    "rather than giving that one row a null, which is a real answer and a surprising "
    "one, so it is pinned as a failure rather than quietly left out",
)
case(
    "nested/list-getitem-nonempty",
    "Series.list",
    frames=LISTS,
    expr=lambda pd, df: _arrow(pd, df["value"].dropna()[df["value"].str.len() > 2]).list[0],
    note="the same call on rows that are all long enough, which is the half that works",
)
case(
    "nested/list-flatten",
    "list.flatten",
    frames=LISTS,
    expr=lambda pd, df: _arrow(pd, df["value"]).list.flatten(),
    note="flatten drops the empty and the null rows entirely, which is where it "
    "differs from explode and the reason both exist",
)
case(
    "nested/list-explode",
    "Series.explode",
    frames=LISTS,
    expr=lambda pd, df: df["value"].explode(),
    rules=STRICT,
    note="explode keeps a row for the empty list with a null in it, and the index says "
    "which original row each element came from",
)
case(
    "nested/list-explode-frame",
    "DataFrame.explode",
    level="L3",
    covers=("column", "ignore_index"),
    frames=LISTS,
    expr=lambda pd, df: df.explode("value", ignore_index=True),
)
case(
    "nested/list-isna",
    "Series.isna",
    frames=LISTS,
    expr=lambda pd, df: df["value"].isna(),
    note="a null list is missing and an empty list is not, which is the distinction "
    "everything else in this section rests on",
)
case(
    "nested/list-count",
    "Series.count",
    frames=LISTS,
    expr=lambda pd, df: df["value"].count(),
)

# ---------------------------------------------------------------------------
# Structs
# ---------------------------------------------------------------------------

case(
    "nested/struct-dtype",
    "Series.dtype",
    frames=STRUCTS,
    expr=lambda pd, df: str(_arrow(pd, df["value"]).dtype),
    note="field order is part of the type, so two structs with the same fields in a "
    "different order are different types and the comparison keeps them apart",
)
case(
    "nested/struct-dtypes",
    "struct.dtypes",
    frames=STRUCTS,
    expr=lambda pd, df: _arrow(pd, df["value"]).struct.dtypes.astype(str),
)
case(
    "nested/struct-field-name",
    "struct.field",
    level="L3",
    covers=("name_or_index",),
    frames=("nested_struct",),
    expr=lambda pd, df: _arrow(pd, df["value"]).struct.field("a"),
)
case(
    "nested/struct-field-second",
    "struct.field",
    level="L3",
    covers=("name_or_index",),
    frames=("nested_struct",),
    expr=lambda pd, df: _arrow(pd, df["value"]).struct.field("b"),
)
case(
    "nested/struct-field-index",
    "struct.field",
    level="L3",
    covers=("name_or_index",),
    frames=("nested_struct",),
    expr=lambda pd, df: _arrow(pd, df["value"]).struct.field(0),
    note="by position as well as by name, and the two have to agree",
)
case(
    "nested/struct-field-nested",
    "struct.field",
    level="L3",
    covers=("name_or_index",),
    frames=("nested_deep",),
    expr=lambda pd, df: _arrow(pd, df["value"]).struct.field("inner"),
    note="a struct inside a struct, which is where a flat implementation of the field "
    "lookup stops working",
)
case(
    "nested/struct-field-path",
    "struct.field",
    level="L3",
    covers=("name_or_index",),
    frames=("nested_deep",),
    expr=lambda pd, df: _arrow(pd, df["value"]).struct.field(["inner", "deep"]),
    note="a path rather than a name, which is the only way to reach the second level",
)
case(
    "nested/struct-explode",
    "struct.explode",
    frames=STRUCTS,
    expr=lambda pd, df: _arrow(pd, df["value"]).struct.explode(),
    note="one column per field, named after the field, which is the whole struct in "
    "one call and the thing anyone actually wants",
)
case(
    "nested/struct-isna",
    "Series.isna",
    frames=("nested_struct",),
    expr=lambda pd, df: df["value"].isna(),
    note="a null struct is different from a struct whose every field is null, and the "
    "corpus has both",
)
case(
    "nested/struct-field-of-null",
    "struct.field",
    level="L3",
    covers=("name_or_index",),
    frames=("nested_struct",),
    expr=lambda pd, df: _arrow(pd, df["value"]).struct.field("a").isna(),
    note="reading a field out of a null struct gives a null rather than raising",
)

# ---------------------------------------------------------------------------
# Nested columns meeting the rest of the library
# ---------------------------------------------------------------------------

case(
    "nested/list-head",
    "DataFrame.head",
    frames=LISTS + STRUCTS,
    expr=lambda pd, df: df.head(4),
    note="slicing a nested column has to keep the child data and the offsets in step, "
    "which is the operation an offset bug shows up in first",
)
case(
    "nested/list-take",
    "DataFrame.take",
    level="L3",
    covers=("indices",),
    frames=LISTS + STRUCTS,
    expr=lambda pd, df: df.take([3, 0, 3]),
    rules=STRICT,
    note="a repeated index, so the child data is read twice from one place",
)
case(
    "nested/list-concat",
    "pandas.concat",
    level="L3",
    covers=("objs",),
    frames=LISTS + STRUCTS,
    expr=lambda pd, df: pd.concat([df.head(3), df.tail(3)]),
    rules=STRICT,
)
case(
    "nested/list-sort-by-other",
    "DataFrame.sort_values",
    level="L3",
    covers=("by", "ascending"),
    frames=LISTS + STRUCTS,
    expr=lambda pd, df: df.sort_values("row", ascending=False),
    rules=STRICT,
)
case(
    "nested/struct-to-frame",
    "Series.to_frame",
    frames=STRUCTS,
    expr=lambda pd, df: df["value"].to_frame(),
)
