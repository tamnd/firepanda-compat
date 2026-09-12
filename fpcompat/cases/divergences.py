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
# Four callables are left here and they are the four whose method is not in firepanda
# at all: `eval` and `query` on a frame, and `interpolate` on a frame and on a column.
# The other thirty one moved to `fpcompat/cases/inplace.py` when firepanda started
# honouring the parameter, which document 51 over there is the argument for. Nothing
# below is about the parameter. Each of these refuses because the method is missing,
# and each will leave this file for the same reason the thirty one did.
#
# Each expression returns the object after the mutation rather than the return value of
# the call, because the return value of an inplace call is None for half of these and
# is the object itself for the other half, and neither is evidence that anything
# happened. What is being asserted is that the mutation happened.

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
    ("DataFrame.eval", "frame-eval", PLAIN, lambda pd, d: d.eval("d = a + 1", inplace=True)),
    (
        "DataFrame.interpolate",
        "frame-interpolate",
        NUMERIC,
        lambda pd, d: d.interpolate(inplace=True),
    ),
    ("DataFrame.query", "frame-query", PLAIN, lambda pd, d: d.query("a > 1", inplace=True)),
    (
        "Series.interpolate",
        "series-interpolate",
        NUMERIC,
        lambda pd, d: d.interpolate(inplace=True),
    ),
)

IN_PROCESS_NOTE = (
    "every case in this block takes a copy, runs the mutating call on it and hands "
    "back the object that was mutated, and copy is a Python layer member here rather "
    "than a core one, so a driver entry could only emit the frame it was handed and "
    "would be scoring itself. Until copy existed at all these cases were reported "
    "absent, which meant the registry was asserting the inplace divergence nowhere. "
    "See spec 44"
)

for api, suffix, frames, call in INPLACE:
    body = _mutating(call) if api.startswith("DataFrame") else _series(_mutating(call))
    case(
        f"divergences/inplace/{suffix}",
        api,
        level="L3",
        covers=("inplace",),
        frames=frames,
        in_process=True,
        note=IN_PROCESS_NOTE,
        # `body` and not a lambda around it. `_mutating` and `_series` already
        # return a fresh closure per iteration, so there is no late binding here
        # for a trampoline to fix, and the trampoline was not free: it put one more
        # frame under every case in this loop, which is enough to push an absent
        # method past the depth `_unimplemented` allows and have it scored as a
        # deliberate divergence instead of a gap.
        expr=body,
    )


def _index_names(pd, index, call):
    """Applies an inplace naming call to an index and returns the names it ended up with."""
    copied = index.copy()
    call(copied)
    return list(copied.names)


# There is no flat index block here any more, and there has not been one since the
# index slice. Renaming a level and setting its names were the first places in the
# library where firepanda honoured inplace rather than refusing it, and their cases
# live with the ordinary index cases as indexing/index-rename-inplace,
# indexing/index-set-names-inplace, temporal/index-rename-inplace and
# temporal/index-set-names-inplace. What is left below is the MultiIndex, which has
# no implementation to honour anything with.

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
    in_process=True,
    note=IN_PROCESS_NOTE,
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
    in_process=True,
    note=IN_PROCESS_NOTE,
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
    in_process=True,
    note="the only inplace parameter that is not on a method, and the only one where "
    "the object being mutated is passed in rather than being self. " + IN_PROCESS_NOTE,
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

# ---------------------------------------------------------------------------
# The two pass moments
# ---------------------------------------------------------------------------

# The only entry in this registry where firepanda is closer to the truth than pandas
# is, which is why it takes a column of its own rather than riding on a corpus frame.
# Both sides ignore the frame they are handed and build the same five values, because
# the divergence needs a column where the mean cannot be represented exactly and no
# corpus frame is shaped that way. Two to the fifty second is where the gap between
# neighbouring float64 values is exactly one, so every input here is exact and the
# rounded centre is the only thing either engine can get wrong.

SHIFTED = [2.0**52 + 1, 2.0**52 + 2, 2.0**52 + 4, 2.0**52 + 8, 2.0**52 + 16]

case(
    "divergences/moment-precision/skew",
    "Series.skew",
    level="L2",
    frames=("two",),
    expr=lambda pd, df: float(pd.Series(SHIFTED).skew()),
    note="pandas answers 1.4863469519931585 and the true value is 1.3253147098134046, "
    "so this is twelve percent rather than a rounding difference. firepanda answers "
    "the true value",
)
case(
    "divergences/moment-precision/var",
    "Series.var",
    level="L2",
    frames=("two",),
    expr=lambda pd, df: float(pd.Series(SHIFTED).var()),
    note="the same column and the same cause, smaller because the second moment is "
    "squared rather than cubed. pandas answers 37.25 and the true value is 37.2",
)

# ---------------------------------------------------------------------------
# The integer zero divisor
# ---------------------------------------------------------------------------

# These build their own two columns for the same reason the moment cases do, which is
# that the divergence needs a zero in the divisor on every frame it runs on and no
# corpus frame guarantees one. The numerator carries a positive, a negative and a zero,
# because the pandas answer is a different infinity for each of the first two and a NaN
# for the third, and an entry that only showed one of them would read as though the rule
# were about division by zero rather than about what the column's type becomes.
#
# The divisor is `[3, 0, 2]` and not three zeros. Two of its rows divide perfectly well,
# which is what makes the point: pandas turns the whole column into a float64 over one
# bad row, so `[7, -7, 0] % [3, 0, 2]` comes back `[1.0, nan, 0.0]` and the two good
# answers paid for the bad one.


def _zero_divisor(pd, op):
    """One integer column against a divisor column with a zero in the middle of it."""
    top = pd.Series([7, -7, 0], dtype="int64", name="value")
    bottom = pd.Series([3, 0, 2], dtype="int64", name="value")
    return op(top, bottom)


def _zero_constant(pd, op):
    """The same numerator against a literal zero, which is a different kernel path."""
    return op(pd.Series([7, -7, 0], dtype="int64", name="value"), 0)


case(
    "divergences/zero-divisor/floordiv-column",
    "Series.floordiv",
    frames=("two",),
    expr=lambda pd, df: _zero_divisor(pd, lambda a, b: a // b),
    note="pandas answers float64 [2.0, -inf, 0.0] and firepanda answers int64 [2, null, "
    "0], so the two engines disagree about the type as well as about the middle row",
)
case(
    "divergences/zero-divisor/mod-column",
    "Series.mod",
    frames=("two",),
    expr=lambda pd, df: _zero_divisor(pd, lambda a, b: a % b),
    note="the remainder loses the same way and is worse to read, because a NaN in a "
    "column of remainders looks like a missing input rather than like a zero divisor",
)
case(
    "divergences/zero-divisor/truediv-column",
    "Series.truediv",
    frames=("two",),
    expr=lambda pd, df: _zero_divisor(pd, lambda a, b: a / b),
    note="the control, and the only one of the three that is not registered. True "
    "division answers a float64 whatever the divisor is, so there is no type to lose "
    "and firepanda gives the same infinity pandas gives. This case has to pass, because "
    "without it the entry above reads as though the whole family were different",
)
case(
    "divergences/zero-divisor/floordiv-scalar",
    "Series.floordiv",
    frames=("two",),
    expr=lambda pd, df: _zero_constant(pd, lambda a, b: a // b),
    note="a constant divisor takes the const path in the kernel rather than the column "
    "path, and the two are separate loops, so a fix to one of them is not a fix to both",
)
case(
    "divergences/zero-divisor/mod-scalar",
    "Series.mod",
    frames=("two",),
    expr=lambda pd, df: _zero_constant(pd, lambda a, b: a % b),
    note="every row divides by zero here, so pandas answers three NaNs and firepanda "
    "answers three nulls, which is the clearest form of the difference",
)


# ---------------------------------------------------------------------------
# How a type is spelt
# ---------------------------------------------------------------------------

# Every type firepanda has is an Arrow type, and `dtype` gives Arrow's name for it.
# Every type pandas has is a numpy dtype or an extension dtype, and `dtype` gives that
# one's name. For most types the two names happen to be the same word, which is why
# this went unnoticed: `int64`, `float64`, `bool` and `category` all read identically on
# both sides and are compared by string all over this suite. Text and dates are where
# the two vocabularies do not share a word, and there is no third name that would be
# right for both.
#
# The frames are the ones the corpus already has rather than new ones, so these cases
# also say which corpus frames the ordinary dtype cases have to stay off.

DTYPE_SPELLING = (
    "compared as a string, because a dtype object is not a value the comparison can "
    "hold and because the string is what a user reads"
)

case(
    "divergences/dtype-spelling/text-column",
    "Series.dtype",
    frames=("single",),
    expr=lambda pd, df: str(df["c"].dtype),
    in_process=True,
    note="pandas 3 says `str` and firepanda says `string`, which is Arrow's name for "
    "the same thing. " + DTYPE_SPELLING,
)
case(
    "divergences/dtype-spelling/text-frame",
    "DataFrame.dtypes",
    frames=("single",),
    expr=lambda pd, df: list(df.dtypes.astype(str)),
    in_process=True,
    note="the same word through the plural member, which matters because `dtypes` is "
    "the shape people actually read and a list of three where one is wrong looks like "
    "a list of three that is right. " + DTYPE_SPELLING,
)
case(
    "divergences/dtype-spelling/date-column",
    "Series.dtype",
    frames=("temporal_range",),
    expr=lambda pd, df: str(df["date"].dtype),
    in_process=True,
    note="a date carrying a day and no clock is `date32[day]` in Arrow and firepanda keeps that "
    "name. pandas has no date dtype at all, so it reads the column into a column of "
    "Python date objects and calls it `object`, which is the type it refuses to have "
    "anywhere else. " + DTYPE_SPELLING,
)
case(
    "divergences/dtype-spelling/timestamp-column",
    "Series.dtype",
    frames=("temporal_range",),
    expr=lambda pd, df: str(df["second"].dtype),
    in_process=True,
    note="the control, and the reason the entry is about two types rather than about "
    "temporal columns in general. A timestamp is `datetime64[s]` in both libraries "
    "because pandas has a dtype for it, so this case has to pass. " + DTYPE_SPELLING,
)

# ---------------------------------------------------------------------------
# What a gap does to an integer
# ---------------------------------------------------------------------------

GAPPED_INTEGERS = (
    "int8_half_null",
    "int16_half_null",
    "int32_half_null",
    "int64_half_null",
    "uint8_half_null",
    "uint16_half_null",
    "uint32_half_null",
    "uint64_half_null",
)
"""Every integer width with a gap in it, because the answer is the same on all eight.

Eight frames rather than one, because a rule that is stated about int64 and happens to be
true of the other seven is a rule nobody has checked. This one is true of all eight and the
case says so by running on all eight.
"""

INTEGER_WIDENING = (
    "in process, because the divergence is in the read rather than in the member. The "
    "driver's own read widens the way pandas does, so an arm here would agree with "
    "pandas by doing the thing the case is about"
)

case(
    "divergences/integer-widening/column-type",
    "Series.dtype",
    frames=GAPPED_INTEGERS,
    expr=lambda pd, df: str(df["value"].dtype),
    in_process=True,
    note="an integer column with one missing row in it is still that integer here and "
    "is float64 in pandas, at every width and both signednesses. " + INTEGER_WIDENING,
)
case(
    "divergences/integer-widening/frame-type",
    "DataFrame.dtypes",
    frames=GAPPED_INTEGERS,
    expr=lambda pd, df: str(df.dtypes["value"]),
    in_process=True,
    note="the same widening read through the plural member, which is where a caller "
    "meets it, because `df.dtypes` is what gets printed and compared against. " + INTEGER_WIDENING,
)
case(
    "divergences/integer-widening/float-control",
    "Series.dtype",
    frames=("float32_half_null", "float64_half_null"),
    expr=lambda pd, df: str(df["value"].dtype),
    in_process=True,
    note="the first control, and it has to pass. A float already has a value for a "
    "missing row, so pandas has nothing to widen to and both libraries keep the width "
    "they were given. " + INTEGER_WIDENING,
)
case(
    "divergences/integer-widening/dense-control",
    "Series.dtype",
    frames=("int8_no_nulls", "int64_no_nulls", "uint64_no_nulls"),
    expr=lambda pd, df: str(df["value"].dtype),
    in_process=True,
    note="the second control, and it has to pass. The same integer widths with no gap "
    "in them, which is what makes this an entry about the gap rather than an entry "
    "about integers. " + INTEGER_WIDENING,
)
