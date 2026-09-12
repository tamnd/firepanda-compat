"""Putting an answer back into the object it was asked of.

These cases were the largest block in `fpcompat/cases/divergences.py` until firepanda
started honouring the parameter, and they have moved here unchanged apart from their
ids. What they assert has not changed at all. Each one takes a copy, runs a mutating
call on it and hands back the object that was mutated, so what is compared is the
mutation rather than the return value, which is None on every engine for half of these
and is the object itself for the other half.

The reason the block is driven by a table rather than written out one call at a time
is the same reason it was before. Thirty one callables take the parameter and a hand
written list of thirty one is missing three of them within a month. The table is
checked against the committed pandas inventory by a test in `tests/test_divergences.py`
that looks for every case declaring it covers the parameter, wherever that case lives,
so a callable pandas adds shows up as a failing test with the name in it.

Four callables are still in the divergence registry and they are the four whose method
is not in firepanda at all: `eval` and `query` on a frame, `interpolate` on a frame and
on a column. `MultiIndex.rename`, `MultiIndex.set_names` and `pandas.eval` are there
for the same reason. None of those is about the parameter.

The split in what pandas answers is undocumented and is asserted at the bottom of this
file. `drop`, `dropna`, `drop_duplicates`, `sort_values`, `sort_index`, `reset_index`,
`set_index`, `rename_axis` and `DataFrame.rename` answer None. `fillna`, `ffill`,
`bfill`, `where`, `mask`, `clip`, `replace` and `Series.rename` answer the object.
`Series.rename` sitting on the far side from `DataFrame.rename` is the clearest sign
that the second group is a wart rather than a decision, and it is asserted anyway,
because people write `df.fillna(0, inplace=True).sum()` and it works.
"""

from __future__ import annotations

from fpcompat.cases import case, section

section("inplace")

NUMERIC = ("float64_half_null",)
PLAIN = ("two",)

# The frame with no gaps in it, for the calls that would otherwise be comparing a null
# against a nan rather than comparing the parameter. While these cases sat in the
# divergence block their outcome was absorbed by the entry and the frame they ran on
# did not matter, so several of them ran on a gapped frame and nobody could tell. Now
# that they are scored the ordinary way the frame matters, and the rule is the one the
# basics cases already follow: a call that reads a missing value or leaves one behind
# runs where there are none, because the oracle loads a gap as a nan and this library
# keeps it a null, which is a difference about loading rather than about the call.
WHOLE = ("float64_no_nulls",)


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


IN_PROCESS_NOTE = (
    "every case in this block takes a copy, runs the mutating call on it and hands "
    "back the object that was mutated, and copy is a Python layer member here rather "
    "than a core one, so a driver entry could only emit the frame it was handed and "
    "would be scoring itself. See spec 51"
)

SETTLES = (
    ("DataFrame.bfill", "frame-bfill", NUMERIC, lambda pd, d: d.bfill(inplace=True)),
    ("DataFrame.clip", "frame-clip", WHOLE, lambda pd, d: d.clip(0, 1, inplace=True)),
    ("DataFrame.drop", "frame-drop", PLAIN, lambda pd, d: d.drop(columns=["c"], inplace=True)),
    (
        "DataFrame.drop_duplicates",
        "frame-drop-duplicates",
        ("keys_10",),
        lambda pd, d: d.drop_duplicates(subset=["key"], inplace=True),
    ),
    ("DataFrame.dropna", "frame-dropna", NUMERIC, lambda pd, d: d.dropna(inplace=True)),
    ("DataFrame.ffill", "frame-ffill", NUMERIC, lambda pd, d: d.ffill(inplace=True)),
    ("DataFrame.fillna", "frame-fillna", NUMERIC, lambda pd, d: d.fillna(0.0, inplace=True)),
    ("DataFrame.mask", "frame-mask", WHOLE, lambda pd, d: d.mask(d > 0, -1, inplace=True)),
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
    ("DataFrame.where", "frame-where", WHOLE, lambda pd, d: d.where(d > 0, -1, inplace=True)),
    ("Series.bfill", "series-bfill", NUMERIC, lambda pd, d: d.bfill(inplace=True)),
    ("Series.clip", "series-clip", WHOLE, lambda pd, d: d.clip(0, 1, inplace=True)),
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
    ("Series.mask", "series-mask", WHOLE, lambda pd, d: d.mask(d > 0, -1.0, inplace=True)),
    ("Series.rename", "series-rename", PLAIN, lambda pd, d: d.rename("z", inplace=True)),
    (
        "Series.rename_axis",
        "series-rename-axis",
        PLAIN,
        lambda pd, d: d.rename_axis("row", inplace=True),
    ),
    ("Series.replace", "series-replace", PLAIN, lambda pd, d: d.replace(1.5, 100.0, inplace=True)),
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
        WHOLE,
        lambda pd, d: d.sort_values(inplace=True),
    ),
    ("Series.where", "series-where", WHOLE, lambda pd, d: d.where(d > 0, -1.0, inplace=True)),
)

for api, suffix, frames, call in SETTLES:
    body = _mutating(call) if api.startswith("DataFrame") else _series(_mutating(call))
    case(
        f"inplace/{suffix}",
        api,
        level="L3",
        covers=("inplace",),
        frames=frames,
        in_process=True,
        note=IN_PROCESS_NOTE,
        # `body` and not a lambda around it. `_mutating` and `_series` already return
        # a fresh closure per iteration, so there is no late binding here for a
        # trampoline to fix, and the trampoline was not free: it put one more frame
        # under every case in this loop, which is enough to push an absent method past
        # the depth `_unimplemented` allows and have it scored as a deliberate
        # divergence instead of a gap.
        expr=body,
    )


# ---------------------------------------------------------------------------
# What the call answers, which is two different things
# ---------------------------------------------------------------------------


def _answer(call):
    """Whether the call handed back None, the object it was asked of, or something else."""

    def run(pd, df):
        target = df.copy()
        answered = call(target)
        if answered is None:
            return "none"
        return "self" if answered is target else "other"

    return run


case(
    "inplace/answers-none",
    "DataFrame.drop",
    level="L3",
    covers=("inplace",),
    frames=PLAIN,
    expr=_answer(lambda d: d.drop(columns=["c"], inplace=True)),
    in_process=True,
    note="the half of the split that answers nothing. " + IN_PROCESS_NOTE,
)
case(
    "inplace/answers-the-object",
    "DataFrame.fillna",
    level="L3",
    covers=("inplace",),
    frames=NUMERIC,
    expr=_answer(lambda d: d.fillna(0.0, inplace=True)),
    in_process=True,
    note="the half of the split that hands the object back, which is what makes "
    "df.fillna(0, inplace=True).sum() work. " + IN_PROCESS_NOTE,
)
case(
    "inplace/column-rename-answers-the-object",
    "Series.rename",
    level="L3",
    covers=("inplace",),
    frames=PLAIN,
    expr=_answer(lambda d: d["a"].rename("z", inplace=True)),
    in_process=True,
    note="Series.rename is on the far side of the split from DataFrame.rename, which "
    "is the clearest sign that the split is a wart rather than a decision. " + IN_PROCESS_NOTE,
)


# ---------------------------------------------------------------------------
# What pandas will accept as a flag
# ---------------------------------------------------------------------------


def _refusal(call):
    """The sentence a bad flag is refused with, so that both engines are compared on it."""

    def run(pd, df):
        try:
            call(df.copy())
        except (TypeError, ValueError) as error:
            # Named by the class a caller would catch rather than by the class the
            # engine happened to raise. firepanda raises InvalidArgumentError, which
            # is a ValueError with a name of its own, and a case that read the name
            # off the object would be asserting that the two libraries spell their
            # exception classes the same way, which is not what is being asked here.
            kind = "TypeError" if isinstance(error, TypeError) else "ValueError"
            return f"{kind}: {error}"
        return "accepted"

    return run


case(
    "inplace/a-one-is-not-a-flag",
    "DataFrame.drop",
    level="L3",
    covers=("inplace",),
    frames=PLAIN,
    expr=_refusal(lambda d: d.drop(columns=["c"], inplace=1)),
    in_process=True,
    note="pandas refuses a one even though 1 == True in Python, and the sentence names "
    "the type that arrived. " + IN_PROCESS_NOTE,
)
case(
    "inplace/a-word-is-not-a-flag",
    "DataFrame.fillna",
    level="L3",
    covers=("inplace",),
    frames=NUMERIC,
    expr=_refusal(lambda d: d.fillna(0.0, inplace="yes")),
    in_process=True,
    note=IN_PROCESS_NOTE,
)
case(
    "inplace/nothing-is-a-flag-and-means-false",
    "DataFrame.drop",
    level="L3",
    covers=("inplace",),
    frames=PLAIN,
    expr=lambda pd, df: list(df.copy().drop(columns=["c"], inplace=None).columns),
    in_process=True,
    note="None is allowed and reads as False, which nobody would guess. " + IN_PROCESS_NOTE,
)
case(
    "inplace/a-column-cannot-become-a-frame",
    "Series.reset_index",
    level="L3",
    covers=("inplace",),
    frames=PLAIN,
    expr=_refusal(lambda d: d["a"].reset_index(inplace=True)),
    in_process=True,
    note="the labels become a second column of values and a column has nowhere to put "
    "a frame, so this is the one refusal that is about the operation rather than about "
    "the library. " + IN_PROCESS_NOTE,
)
