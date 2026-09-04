"""The cases that assert the registered divergences.

These are ordinary cases and that is the point. Each one runs on pandas as its own
oracle like every other case in the suite and has to pass there, because a case whose
pandas side is broken cannot be evidence about firepanda either. What makes them
different is that `fpcompat/divergences.toml` points at them, so when the subject is
firepanda the runner requires the outcome the registry declares instead of requiring
the pandas answer.

That is what turns a divergence from an excuse into an assertion. A case here that
starts agreeing with pandas fails the build and says the registry is out of date,
which is the opposite of how a known failure list normally behaves.

The `inplace` block is the largest and it is driven by a table rather than written out
one call at a time, because 42 pandas callables take that parameter and a hand written
list of 42 would be missing three of them within a month. The table is checked against
the committed inventory by a test, so adding a callable with an `inplace` parameter to
pandas shows up as a failing test with the name in it.
"""

from __future__ import annotations

from fpcompat.cases import case, section

section("divergences")

# ---------------------------------------------------------------------------
# Plotting and styling
# ---------------------------------------------------------------------------

# These return an accessor or a figure, neither of which is an answer this suite can
# compare, so the case asks for the type name instead. That is enough. The claim being
# made is that the name resolves at all, which is exactly what firepanda refuses to do,
# and comparing the string keeps the oracle side honest without dragging matplotlib
# into a pinned environment that has no use for it.

case(
    "divergences/plotting/frame-plot",
    "DataFrame.plot",
    level="L0",
    frames=("two",),
    expr=lambda pd, df: type(df.plot).__name__,
    note="pandas hands back a PlotAccessor. firepanda has no plotting and points at "
    "to_pandas() instead, which is a one line change for the user and several thousand "
    "for the library",
)
case(
    "divergences/plotting/series-plot",
    "Series.plot",
    level="L0",
    frames=("two",),
    expr=lambda pd, df: type(df["a"].plot).__name__,
)
case(
    "divergences/plotting/frame-hist",
    "DataFrame.hist",
    level="L0",
    frames=("two",),
    expr=lambda pd, df: type(df.hist).__name__,
)
case(
    "divergences/plotting/series-hist",
    "Series.hist",
    level="L0",
    frames=("two",),
    expr=lambda pd, df: type(df["a"].hist).__name__,
)
case(
    "divergences/plotting/frame-boxplot",
    "DataFrame.boxplot",
    level="L0",
    frames=("two",),
    expr=lambda pd, df: type(df.boxplot).__name__,
)
case(
    "divergences/plotting/frame-style",
    "DataFrame.style",
    level="L4",
    frames=("two",),
    expr=lambda pd, df: df.style,
    raises=("AttributeError", "requires jinja2"),
    note="pandas itself refuses this one in the pinned environment, because the styler "
    "needs jinja2 and jinja2 is not a dependency of a dataframe library. The message "
    "differs from firepanda's and the registry does not care which message it is, only "
    "that the operation refuses",
)

# ---------------------------------------------------------------------------
# Pickle
# ---------------------------------------------------------------------------


def _roundtrip(pd, value):
    """Writes a pickle to a temporary file and reads it back."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / "frame.pkl"
        value.to_pickle(target)
        return pd.read_pickle(target)


case(
    "divergences/pickle/frame-roundtrip",
    "DataFrame.to_pickle",
    frames=("two", "tall"),
    expr=lambda pd, df: _roundtrip(pd, df),
    note="a real round trip rather than a name check, because the thing being given up "
    "is the round trip. firepanda points at Parquet and Arrow IPC, both of which "
    "another program can read and neither of which executes code on load",
)
case(
    "divergences/pickle/series-roundtrip",
    "Series.to_pickle",
    frames=("two",),
    expr=lambda pd, df: _roundtrip(pd, df["b"]),
)
case(
    "divergences/pickle/read",
    "pandas.read_pickle",
    frames=("two",),
    expr=lambda pd, df: _roundtrip(pd, df).shape,
)

# ---------------------------------------------------------------------------
# dtype=object
# ---------------------------------------------------------------------------

case(
    "divergences/object-dtype/construct",
    "pandas.DataFrame",
    frames=("two",),
    expr=lambda pd, df: pd.DataFrame({"a": [b"one", b"two"]}).dtypes.astype(str).tolist(),
    note="the divergence a beginner hits first. pandas stores a column of arbitrary "
    "Python objects and reaches into the interpreter once per element to do anything "
    "with it, and firepanda raises at construction naming the column and the type",
)
case(
    "divergences/object-dtype/astype",
    "Series.astype",
    frames=("two", "tall"),
    expr=lambda pd, df: df.iloc[:, 0].astype(object).dtype.name,
)
case(
    "divergences/object-dtype/mixed-column",
    "pandas.Series",
    frames=("two",),
    expr=lambda pd, df: pd.Series([1, "two", 3.0]).dtype.name,
    note="one column holding an integer, a string and a float, which pandas accepts "
    "and which has no Arrow type at all",
)

# ---------------------------------------------------------------------------
# inplace
# ---------------------------------------------------------------------------

# Every pandas callable that takes an `inplace` parameter, one case each. The call is
# written out per name because there is no way to synthesize a valid call to `drop` and
# a valid call to `set_index` from a signature, but the set of names is not written out
# by hand: a test checks this table against the inventory, so a name that grows an
# `inplace` parameter in a future pandas shows up as a failing test rather than as a
# quiet hole.
#
# Each expression returns the object after the mutation rather than the return value of
# the call, because the return value of an inplace call is None and None is the same on
# every engine. What is being asserted is that the mutation happened.

NUMERIC = ("float64_half_null",)
PLAIN = ("two",)


def _series(expr):
    """The same call against the second column of the frame."""
    return lambda pd, df: expr(pd, df.iloc[:, 1].copy())


def _mutating(call):
    """Runs a call for its side effect and hands back the object it mutated."""

    def run(pd, df):
        target = df.copy()
        call(pd, target)
        return target

    return run


INPLACE = (
    ("DataFrame.bfill", "frame-bfill", NUMERIC, lambda pd, d: d.bfill(inplace=True)),
    ("DataFrame.clip", "frame-clip", NUMERIC, lambda pd, d: d.clip(0, 1, inplace=True)),
    ("DataFrame.drop", "frame-drop", PLAIN, lambda pd, d: d.drop(columns=["c"], inplace=True)),
    (
        "DataFrame.drop_duplicates",
        "frame-drop-duplicates",
        ("keys_10",),
        lambda pd, d: d.drop_duplicates(subset=["key"], inplace=True),
    ),
    ("DataFrame.dropna", "frame-dropna", NUMERIC, lambda pd, d: d.dropna(inplace=True)),
    ("DataFrame.eval", "frame-eval", PLAIN, lambda pd, d: d.eval("d = a + 1", inplace=True)),
    ("DataFrame.ffill", "frame-ffill", NUMERIC, lambda pd, d: d.ffill(inplace=True)),
    ("DataFrame.fillna", "frame-fillna", NUMERIC, lambda pd, d: d.fillna(0.0, inplace=True)),
    (
        "DataFrame.interpolate",
        "frame-interpolate",
        NUMERIC,
        lambda pd, d: d.interpolate(inplace=True),
    ),
    ("DataFrame.mask", "frame-mask", NUMERIC, lambda pd, d: d.mask(d > 0, inplace=True)),
    ("DataFrame.query", "frame-query", PLAIN, lambda pd, d: d.query("a > 1", inplace=True)),
    (
        "DataFrame.rename",
        "frame-rename",
        PLAIN,
        lambda pd, d: d.rename(columns={"a": "z"}, inplace=True),
    ),
    (
        "DataFrame.rename_axis",
        "frame-rename-axis",
        PLAIN,
        lambda pd, d: d.rename_axis("row", inplace=True),
    ),
    ("DataFrame.replace", "frame-replace", PLAIN, lambda pd, d: d.replace(1, 100, inplace=True)),
    (
        "DataFrame.reset_index",
        "frame-reset-index",
        PLAIN,
        lambda pd, d: d.reset_index(drop=True, inplace=True),
    ),
    (
        "DataFrame.set_index",
        "frame-set-index",
        PLAIN,
        lambda pd, d: d.set_index("a", inplace=True),
    ),
    (
        "DataFrame.sort_index",
        "frame-sort-index",
        PLAIN,
        lambda pd, d: d.sort_index(ascending=False, inplace=True),
    ),
    (
        "DataFrame.sort_values",
        "frame-sort-values",
        PLAIN,
        lambda pd, d: d.sort_values("b", inplace=True),
    ),
    ("DataFrame.where", "frame-where", NUMERIC, lambda pd, d: d.where(d > 0, inplace=True)),
    ("Series.bfill", "series-bfill", NUMERIC, lambda pd, d: d.bfill(inplace=True)),
    ("Series.clip", "series-clip", NUMERIC, lambda pd, d: d.clip(0, 1, inplace=True)),
    ("Series.drop", "series-drop", PLAIN, lambda pd, d: d.drop(0, inplace=True)),
    (
        "Series.drop_duplicates",
        "series-drop-duplicates",
        ("keys_10",),
        lambda pd, d: d.drop_duplicates(inplace=True),
    ),
    ("Series.dropna", "series-dropna", NUMERIC, lambda pd, d: d.dropna(inplace=True)),
    ("Series.ffill", "series-ffill", NUMERIC, lambda pd, d: d.ffill(inplace=True)),
    ("Series.fillna", "series-fillna", NUMERIC, lambda pd, d: d.fillna(0.0, inplace=True)),
    (
        "Series.interpolate",
        "series-interpolate",
        NUMERIC,
        lambda pd, d: d.interpolate(inplace=True),
    ),
    ("Series.mask", "series-mask", NUMERIC, lambda pd, d: d.mask(d > 0, inplace=True)),
    ("Series.rename", "series-rename", PLAIN, lambda pd, d: d.rename("z", inplace=True)),
    (
        "Series.rename_axis",
        "series-rename-axis",
        PLAIN,
        lambda pd, d: d.rename_axis("row", inplace=True),
    ),
    ("Series.replace", "series-replace", NUMERIC, lambda pd, d: d.replace(0.0, 1.0, inplace=True)),
    (
        "Series.reset_index",
        "series-reset-index",
        PLAIN,
        lambda pd, d: d.reset_index(drop=True, inplace=True),
    ),
    (
        "Series.sort_index",
        "series-sort-index",
        PLAIN,
        lambda pd, d: d.sort_index(ascending=False, inplace=True),
    ),
    (
        "Series.sort_values",
        "series-sort-values",
        NUMERIC,
        lambda pd, d: d.sort_values(inplace=True),
    ),
    ("Series.where", "series-where", NUMERIC, lambda pd, d: d.where(d > 0, inplace=True)),
)

for api, suffix, frames, call in INPLACE:
    body = _mutating(call) if api.startswith("DataFrame") else _series(_mutating(call))
    case(
        f"divergences/inplace/{suffix}",
        api,
        level="L3",
        covers=("inplace",),
        frames=frames,
        expr=(lambda made: lambda pd, df: made(pd, df))(body),
    )


def _index_names(pd, index, call):
    """Applies an inplace rename to an index and returns the names it ended up with."""
    copied = index.copy()
    call(copied)
    return list(copied.names)


INDEX_INPLACE = (
    ("Index.rename", "index-rename", lambda i: i.rename("row", inplace=True)),
    ("Index.set_names", "index-set-names", lambda i: i.set_names("row", inplace=True)),
    ("DatetimeIndex.rename", "datetime-index-rename", lambda i: i.rename("when", inplace=True)),
    (
        "DatetimeIndex.set_names",
        "datetime-index-set-names",
        lambda i: i.set_names("when", inplace=True),
    ),
)

for api, suffix, call in INDEX_INPLACE:
    if api.startswith("DatetimeIndex"):
        frames = ("temporal_resolutions",)

        def source(pd, df):
            return pd.DatetimeIndex(df["s"])

    else:
        frames = ("two",)

        def source(pd, df):
            return pd.Index(df["a"])

    case(
        f"divergences/inplace/{suffix}",
        api,
        level="L3",
        covers=("inplace",),
        frames=frames,
        expr=(lambda made, take: lambda pd, df: _index_names(pd, take(pd, df), made))(call, source),
    )

case(
    "divergences/inplace/multi-index-rename",
    "MultiIndex.rename",
    level="L3",
    covers=("inplace",),
    frames=("keys_two_column",),
    expr=lambda pd, df: _index_names(
        pd,
        pd.MultiIndex.from_frame(df[["left", "right"]]),
        lambda index: index.rename(["one", "two"], inplace=True),
    ),
)
case(
    "divergences/inplace/multi-index-set-names",
    "MultiIndex.set_names",
    level="L3",
    covers=("inplace",),
    frames=("keys_two_column",),
    expr=lambda pd, df: _index_names(
        pd,
        pd.MultiIndex.from_frame(df[["left", "right"]]),
        lambda index: index.set_names(["one", "two"], inplace=True),
    ),
)


def _module_eval(pd, df):
    """`pandas.eval` writing into its target, which is the module level inplace."""
    target = df.copy()
    pd.eval("d = a + 1", target=target, resolvers=[target], inplace=True)
    return target


case(
    "divergences/inplace/module-eval",
    "pandas.eval",
    level="L3",
    covers=("inplace",),
    frames=("two",),
    expr=_module_eval,
    note="the only inplace parameter that is not on a method, and the only one where "
    "the object being mutated is passed in rather than being self",
)

# ---------------------------------------------------------------------------
# Automatic index alignment
# ---------------------------------------------------------------------------

# The largest behavioural divergence in the project and the one most likely to change a
# user's answer without telling them. Every case here produces nulls on pandas, and the
# nulls are the whole point: the two operands do not line up, pandas silently takes the
# union of their labels, and the result is longer than either input and mostly empty.


def _halves(df):
    """The first and second halves of a frame, which share no index labels."""
    middle = len(df) // 2
    return df.iloc[:middle], df.iloc[middle:]


case(
    "divergences/alignment/series-add",
    "Series.add",
    frames=("tall", "float64_no_nulls"),
    expr=lambda pd, df: (lambda halves: halves[0].iloc[:, -1] + halves[1].iloc[:, -1])(_halves(df)),
    note="two halves of one column added together. Every value in the pandas answer is "
    "null, because no label appears in both, and the answer is twice as long as either "
    "operand. A user who meant to add them elementwise gets no error at all",
)
case(
    "divergences/alignment/frame-add",
    "DataFrame.add",
    frames=("float64_no_nulls",),
    expr=lambda pd, df: (lambda halves: halves[0] + halves[1])(_halves(df)),
)
case(
    "divergences/alignment/subtract-shifted",
    "Series.sub",
    frames=("tall",),
    expr=lambda pd, df: df["value"] - df["value"].shift(1),
    note="the one alignment that people rely on and that reads correctly, which is why "
    "removing it is a real cost rather than a free simplification",
)
case(
    "divergences/alignment/align",
    "DataFrame.align",
    level="L3",
    covers=("other", "join"),
    frames=("float64_no_nulls",),
    expr=lambda pd, df: (lambda halves: halves[0].align(halves[1], join="outer")[0])(_halves(df)),
)
case(
    "divergences/alignment/assign-misaligned-column",
    "DataFrame.__setitem__",
    frames=("tall",),
    expr=lambda pd, df: (
        lambda copy: (copy.__setitem__("shifted", copy["value"].iloc[::-1]), copy)[1]
    )(df.copy()),
    note="assigning a reversed column back into the frame puts every value back where "
    "it started, because the assignment aligns on the index rather than on position. "
    "This is the one that surprises people who have used pandas for years",
)

# ---------------------------------------------------------------------------
# No implicit index
# ---------------------------------------------------------------------------

case(
    "divergences/implicit-index/reindex",
    "DataFrame.reindex",
    level="L3",
    covers=("index",),
    frames=("tall",),
    expr=lambda pd, df: df.reindex([0, 2, 4, 9999]),
    note="reindexing against labels that were never declared, where the missing one "
    "comes back as a row of nulls rather than as an error",
)
case(
    "divergences/implicit-index/loc-default",
    "DataFrame.loc",
    frames=("tall",),
    expr=lambda pd, df: df.loc[3],
    note="the default index is positions pretending to be labels, so this reads as a "
    "position and is not one, and the difference only shows once the frame has been "
    "sorted or filtered",
)
case(
    "divergences/implicit-index/reset-index-keeps-old",
    "DataFrame.reset_index",
    level="L3",
    covers=("drop",),
    frames=("tall",),
    expr=lambda pd, df: df.sort_values("value").reset_index(drop=False).head(5),
    note="the old positions survive as a column, which is only meaningful because the "
    "index existed without anybody asking for it",
)

# ---------------------------------------------------------------------------
# Regex lookaround and backreferences
# ---------------------------------------------------------------------------

# RE2 semantics. Both constructs need backtracking, backtracking is what makes a regex
# engine take exponential time on an adversarial pattern, and a dataframe library
# running a user's pattern over a hundred million rows is exactly where that matters.

case(
    "divergences/regex/backreference-replace",
    "str.replace",
    level="L3",
    covers=("pat", "repl", "regex"),
    frames=("strings_pattern", "strings_ascii"),
    expr=lambda pd, df: df["value"].str.replace(r"(.)\1", "X", regex=True),
    note="a doubled character, which needs the engine to remember what the first group "
    "matched, which is the thing RE2 does not do",
)
case(
    "divergences/regex/backreference-contains",
    "str.contains",
    level="L3",
    covers=("pat", "regex"),
    frames=("strings_pattern", "strings_ascii"),
    expr=lambda pd, df: df["value"].str.contains(r"(.)\1", regex=True),
)
case(
    "divergences/regex/lookahead",
    "str.contains",
    level="L3",
    covers=("pat", "regex"),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.contains(r"a(?=b)", regex=True),
)
case(
    "divergences/regex/negative-lookahead",
    "str.contains",
    level="L3",
    covers=("pat", "regex"),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.contains(r"a(?!b)", regex=True),
)
case(
    "divergences/regex/lookbehind",
    "str.contains",
    level="L3",
    covers=("pat", "regex"),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.contains(r"(?<=a)b", regex=True),
    note="the construct a user is most likely to have written, and the message has to "
    "name it rather than say the pattern is invalid, because it is not invalid, it is "
    "unsupported and those are different words",
)
case(
    "divergences/regex/lookaround-extract",
    "str.extract",
    level="L3",
    covers=("pat", "expand"),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.extract(r"(?<=a)(b+)", expand=True),
)
