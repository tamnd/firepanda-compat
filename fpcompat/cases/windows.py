"""Rolling, expanding and exponentially weighted windows.

The one thing worth knowing before reading these is that pandas does not compute a
rolling sum by summing each window. It carries a running total and adds the entering
element and subtracts the leaving one, which is fast and which means the answer
depends on the whole history rather than only on the window. That is why a rolling sum
over a column containing an infinity poisons every window after it, and why the float
frames are in here.

`min_periods` is the parameter that decides how many non null values a window needs
before it produces anything, and its default is not the same for every method, which
is the sort of thing a conformance suite exists to pin down.
"""

from __future__ import annotations

from fpcompat.cases import case, section
from fpcompat.compare import Rules, Tolerance

section("windows")

FRAMES = ("float64_no_nulls", "float64_half_null", "int64_no_nulls", "tall")
NULLY = ("float64_half_null", "float64_all_null")

RUNNING = Rules(
    tolerance=Tolerance.ACCUMULATION,
    reason="a rolling total is carried rather than recomputed, so the result depends "
    "on every value that has passed through the window",
)
SPREAD = Rules(
    tolerance=Tolerance.STATISTICAL,
    reason="a rolling variance is carried the same way and it squares the error",
)
SHAPE = Rules(
    tolerance=Tolerance.STATISTICAL,
    reason="a rolling third and fourth moment are carried the same way again and they "
    "raise the error to a cube and a fourth power, and pandas rebuilds them from raw "
    "sums of powers where firepanda carries the central moments themselves",
)

# ---------------------------------------------------------------------------
# Rolling
# ---------------------------------------------------------------------------

for name in ("sum", "mean", "min", "max", "count", "median"):
    case(
        f"windows/rolling-{name}",
        f"Rolling.{name}",
        frames=FRAMES,
        expr=(lambda method: lambda pd, df: getattr(df["value"].rolling(5), method)())(name),
        rules=RUNNING if name in ("sum", "mean") else Rules(),
        note="the first four rows are null because a five wide window is not full yet, "
        "and count is the one that is not",
    )

for name in ("std", "var", "sem"):
    case(
        f"windows/rolling-{name}",
        f"Rolling.{name}",
        frames=("float64_no_nulls", "tall"),
        expr=(lambda method: lambda pd, df: getattr(df["value"].rolling(8), method)())(name),
        rules=SPREAD,
    )

for name in ("skew", "kurt"):
    case(
        f"windows/rolling-{name}",
        f"Rolling.{name}",
        frames=("float64_no_nulls", "tall"),
        expr=(lambda method: lambda pd, df: getattr(df["value"].rolling(8), method)())(name),
        rules=SHAPE,
        note="a shape is a ratio of moments and it has no units, so the answer is a "
        "small number built out of large ones and the two libraries build it "
        "differently",
    )

case(
    "windows/rolling-min-periods",
    "DataFrame.rolling",
    level="L3",
    covers=("window", "min_periods"),
    frames=FRAMES,
    expr=lambda pd, df: df["value"].rolling(5, min_periods=1).sum(),
    rules=RUNNING,
    note="with one required period the leading nulls disappear and the first rows "
    "become partial sums, which is a completely different answer from the default",
)
case(
    "windows/rolling-min-periods-nulls",
    "DataFrame.rolling",
    level="L3",
    covers=("window", "min_periods"),
    frames=NULLY,
    expr=lambda pd, df: df["value"].rolling(4, min_periods=3).mean(),
    rules=RUNNING,
    note="the corpus nulls are every other row, so a four wide window never has more "
    "than two real values in it and this is almost all nulls, which is the point",
)
case(
    "windows/rolling-center",
    "DataFrame.rolling",
    level="L3",
    covers=("window", "center"),
    frames=FRAMES,
    expr=lambda pd, df: df["value"].rolling(5, center=True).sum(),
    rules=RUNNING,
    note="an even window centred is not symmetric and which side gets the extra "
    "element is not something anyone can guess",
)
case(
    "windows/rolling-center-even",
    "DataFrame.rolling",
    level="L3",
    covers=("window", "center"),
    frames=("float64_no_nulls",),
    expr=lambda pd, df: df["value"].rolling(4, center=True).sum(),
    rules=RUNNING,
)
case(
    "windows/rolling-closed",
    "DataFrame.rolling",
    level="L3",
    covers=("window", "closed"),
    frames=("float64_no_nulls",),
    expr=lambda pd, df: df["value"].rolling(5, closed="left").sum(),
    rules=RUNNING,
)
case(
    "windows/rolling-step",
    "DataFrame.rolling",
    level="L3",
    covers=("window", "step"),
    frames=("tall",),
    expr=lambda pd, df: df["value"].rolling(10, step=3).sum(),
    rules=RUNNING,
)
case(
    "windows/rolling-window-one",
    "DataFrame.rolling",
    level="L3",
    covers=("window",),
    frames=FRAMES,
    expr=lambda pd, df: df["value"].rolling(1).sum(),
    rules=Rules(tolerance=Tolerance.EXACT),
    note="a window of one is the identity and there is nothing to accumulate, so this "
    "one is compared exactly on purpose",
)
case(
    "windows/rolling-window-longer-than-frame",
    "DataFrame.rolling",
    level="L3",
    covers=("window",),
    frames=("single", "two"),
    expr=lambda pd, df: df["b"].rolling(100).sum(),
    note="a window wider than the frame gives all nulls rather than an error",
)
case(
    "windows/rolling-quantile",
    "Rolling.quantile",
    level="L3",
    covers=("q",),
    frames=("float64_no_nulls", "tall"),
    expr=lambda pd, df: df["value"].rolling(8).quantile(0.5),
    rules=Rules(tolerance=Tolerance.SINGLE, reason="an interpolated quantile"),
)
case(
    "windows/rolling-apply",
    "Rolling.apply",
    level="L3",
    covers=("func",),
    frames=("float64_no_nulls",),
    expr=lambda pd, df: df["value"].rolling(4).apply(lambda window: window.max() - window.min()),
    in_process=True,
    note="the escape hatch again, and the case that says an implementation cannot ship "
    "only the fast paths",
)
case(
    "windows/rolling-corr",
    "Rolling.corr",
    level="L3",
    covers=("other",),
    frames=("tall",),
    expr=lambda pd, df: df["value"].rolling(20).corr(df["key"].astype("float64")),
    in_process=True,
    note="only the rows where both columns hold a value count toward the window",
    rules=SPREAD,
)
case(
    "windows/rolling-cov",
    "Rolling.cov",
    level="L3",
    covers=("other",),
    frames=("tall",),
    expr=lambda pd, df: df["value"].rolling(20).cov(df["key"].astype("float64")),
    in_process=True,
    note="only the rows where both columns hold a value count toward the window",
    rules=SPREAD,
)
case(
    "windows/rolling-rank",
    "Rolling.rank",
    frames=("float64_no_nulls", "tall"),
    expr=lambda pd, df: df["value"].rolling(8).rank(),
)
case(
    "windows/rolling-frame",
    "DataFrame.rolling",
    level="L3",
    covers=("window",),
    frames=("tall",),
    expr=lambda pd, df: df[["value"]].rolling(6).mean(),
    rules=RUNNING,
    note="in process because the driver has no entry for this call, and the module answers it the "
    "way pandas does on every frame",
    in_process=True,
)
case(
    "windows/rolling-time",
    "DataFrame.rolling",
    level="L3",
    covers=("window", "on"),
    frames=("temporal_range",),
    expr=lambda pd, df: df.rolling("30s", on="second")["row"].sum(),
    note="a time based window is a different algorithm from a count based one, and it "
    "needs the index to be sorted and the offsets to be understood. firepanda's Python "
    "layer finds pandas' bounds for each row and reduces over them, so the case runs in "
    "process",
    in_process=True,
)

# ---------------------------------------------------------------------------
# Expanding
# ---------------------------------------------------------------------------

for name in ("sum", "mean", "min", "max", "count", "median"):
    case(
        f"windows/expanding-{name}",
        f"Expanding.{name}",
        frames=FRAMES,
        expr=(lambda method: lambda pd, df: getattr(df["value"].expanding(), method)())(name),
        rules=RUNNING if name in ("sum", "mean") else Rules(),
        note="an expanding window is a rolling one with no left edge, so the last row "
        "is the whole column reduction and it has to match the plain reduction",
    )

for name in ("std", "var", "sem"):
    case(
        f"windows/expanding-{name}",
        f"Expanding.{name}",
        frames=("float64_no_nulls", "tall"),
        expr=(lambda method: lambda pd, df: getattr(df["value"].expanding(), method)())(name),
        rules=SPREAD,
    )

for name in ("skew", "kurt"):
    case(
        f"windows/expanding-{name}",
        f"Expanding.{name}",
        frames=("float64_no_nulls", "tall"),
        expr=(lambda method: lambda pd, df: getattr(df["value"].expanding(), method)())(name),
        rules=SHAPE,
        note="an expanding window never drops a row, so this is the one place a shape "
        "is asked for over ten thousand values at once and the last row has to be the "
        "whole column shape",
    )

case(
    "windows/expanding-min-periods",
    "DataFrame.expanding",
    level="L3",
    covers=("min_periods",),
    frames=FRAMES,
    expr=lambda pd, df: df["value"].expanding(min_periods=5).sum(),
    rules=RUNNING,
)
case(
    "windows/expanding-quantile",
    "Expanding.quantile",
    level="L3",
    covers=("q",),
    frames=("tall",),
    expr=lambda pd, df: df["value"].expanding(10).quantile(0.75),
    rules=Rules(tolerance=Tolerance.SINGLE, reason="an interpolated quantile"),
)
case(
    "windows/expanding-rank",
    "Expanding.rank",
    frames=("float64_no_nulls", "tall"),
    expr=lambda pd, df: df["value"].expanding(10).rank(),
    note="the rank of a row that never leaves the window again, so the number it "
    "answers climbs as the window grows under it",
)
case(
    "windows/expanding-apply",
    "Expanding.apply",
    level="L3",
    covers=("func",),
    frames=("float64_no_nulls",),
    expr=lambda pd, df: df["value"].expanding(4).apply(lambda window: window.iloc[0]),
    in_process=True,
    note="the window is handed over as a column keeping its labels",
)

# ---------------------------------------------------------------------------
# Exponentially weighted, where every parameter is a different spelling of one number
# ---------------------------------------------------------------------------

for name, kwargs in (
    ("span", {"span": 5}),
    ("com", {"com": 2.0}),
    ("halflife", {"halflife": 3.0}),
    ("alpha", {"alpha": 0.3}),
):
    case(
        f"windows/ewm-{name}-mean",
        "DataFrame.ewm",
        level="L3",
        covers=(name,),
        frames=("float64_no_nulls", "tall"),
        expr=(lambda opts: lambda pd, df: df["value"].ewm(**opts).mean())(kwargs),
        rules=RUNNING,
        note="span, centre of mass, half life and alpha are four ways of writing one "
        "decay, and an implementation that converts between them differently is wrong "
        "in a way that shows in the last decimal of every row",
    )

case(
    "windows/ewm-adjust-false",
    "DataFrame.ewm",
    level="L3",
    covers=("alpha", "adjust"),
    frames=("float64_no_nulls", "tall"),
    expr=lambda pd, df: df["value"].ewm(alpha=0.3, adjust=False).mean(),
    rules=RUNNING,
    note="adjust off is the recursive form and adjust on is the finite sum form, and "
    "they only agree in the limit",
)
case(
    "windows/ewm-ignore-na",
    "DataFrame.ewm",
    level="L3",
    covers=("alpha", "ignore_na"),
    frames=NULLY,
    expr=lambda pd, df: df["value"].ewm(alpha=0.3, ignore_na=True).mean(),
    rules=RUNNING,
    note="whether a null takes up a slot in the decay or is skipped over, which is a "
    "different answer and not a rounding difference",
)
case(
    "windows/ewm-min-periods",
    "DataFrame.ewm",
    level="L3",
    covers=("span", "min_periods"),
    frames=("float64_half_null",),
    expr=lambda pd, df: df["value"].ewm(span=5, min_periods=3).mean(),
    rules=RUNNING,
)
case(
    "windows/ewm-std",
    "ExponentialMovingWindow.std",
    frames=("float64_no_nulls", "tall"),
    expr=lambda pd, df: df["value"].ewm(span=5).std(),
    rules=SPREAD,
)
case(
    "windows/ewm-var",
    "ExponentialMovingWindow.var",
    frames=("float64_no_nulls", "tall"),
    expr=lambda pd, df: df["value"].ewm(span=5).var(),
    rules=SPREAD,
)
case(
    "windows/ewm-sum",
    "ExponentialMovingWindow.sum",
    frames=("float64_no_nulls",),
    expr=lambda pd, df: df["value"].ewm(span=5).sum(),
    rules=RUNNING,
)
case(
    "windows/ewm-corr",
    "ExponentialMovingWindow.corr",
    level="L3",
    covers=("other",),
    frames=("tall",),
    expr=lambda pd, df: df["value"].ewm(span=10).corr(df["key"].astype("float64")),
    in_process=True,
    note="pandas' own recurrence, over the rows where both columns hold a value",
    rules=SPREAD,
)


def _uneven_times(pd, df):
    """Instants that drift further apart down the frame, so every step decays differently."""
    return pd.to_datetime([row * row for row in range(len(df))], unit="s")


case(
    "windows/ewm-times-mean",
    "DataFrame.ewm",
    level="L3",
    covers=("halflife", "times"),
    frames=("float64_no_nulls", "tall"),
    expr=lambda pd, df: df["value"].ewm(halflife="30s", times=_uneven_times(pd, df)).mean(),
    in_process=True,
    rules=RUNNING,
    note="with times the decay between two rows is the time between them in half lives, "
    "so rows far apart forget more than rows close together",
)
case(
    "windows/ewm-times-gaps",
    "DataFrame.ewm",
    level="L3",
    covers=("halflife", "times", "adjust"),
    frames=NULLY,
    expr=lambda pd, df: (
        df["value"].ewm(halflife="30s", times=_uneven_times(pd, df), adjust=False).mean()
    ),
    in_process=True,
    rules=RUNNING,
    note="pandas' kernel gives a value after a gap what the decayed old weight left over "
    "when adjust is off and the centre of mass is one, which it always is with times",
)
case(
    "windows/ewm-times-sum-refused",
    "DataFrame.ewm",
    level="L4",
    covers=("halflife", "times"),
    frames=("float64_no_nulls",),
    expr=lambda pd, df: df["value"].ewm(halflife="30s", times=_uneven_times(pd, df)).sum(),
    raises=("NotImplementedError", "sum is not implemented with times"),
)

case(
    "windows/rolling-text-refused",
    "DataFrame.rolling",
    level="L4",
    covers=("window",),
    frames=("keys_two_column",),
    expr=lambda pd, df: df.rolling(2).sum(),
    raises=("DataError", "Cannot aggregate non-numeric type"),
)
case(
    "windows/ewm-text-refused",
    "DataFrame.ewm",
    level="L4",
    covers=("span",),
    frames=("keys_two_column",),
    expr=lambda pd, df: df["left"].ewm(span=3).mean(),
    raises=("DataError", "No numeric types to aggregate"),
)
case(
    "windows/rolling-count-text",
    "DataFrame.rolling",
    level="L3",
    covers=("window",),
    frames=("keys_two_column",),
    expr=lambda pd, df: df.rolling(3).count(),
    in_process=True,
    note="a count reads the values present in each window whatever their kind, so a "
    "text column counts where every other reduction refuses it. The kernel reads numbers only, "
    "so firepanda's Python layer marks the values present, and the case runs in process",
)

# ---------------------------------------------------------------------------
# Windows inside groups, which is where the two features meet
# ---------------------------------------------------------------------------

case(
    "windows/groupby-rolling",
    "GroupBy.rolling",
    level="L3",
    covers=("window",),
    frames=("keys_10", "tall"),
    expr=lambda pd, df: df.groupby("key")["value"].rolling(3).sum(),
    rules=RUNNING,
    note="the window resets at every group boundary, which is the whole point and is "
    "also the thing that is easiest to implement by accident as one long window. "
    "The answer is labelled by a MultiIndex, which is firepanda's Python layer, so "
    "the case runs in process",
    in_process=True,
)
case(
    "windows/groupby-expanding",
    "GroupBy.expanding",
    frames=("keys_10",),
    expr=lambda pd, df: df.groupby("key")["value"].expanding().sum(),
    rules=RUNNING,
    note="the answer is labelled by a MultiIndex, which is firepanda's Python layer, "
    "so the case runs in process",
    in_process=True,
)
case(
    "windows/groupby-ewm",
    "GroupBy.ewm",
    level="L3",
    covers=("span",),
    frames=("keys_10",),
    expr=lambda pd, df: df.groupby("key")["value"].ewm(span=3).mean(),
    rules=RUNNING,
    note="the answer is labelled by a MultiIndex, which is firepanda's Python layer, "
    "so the case runs in process",
    in_process=True,
)
case(
    "windows/rolling-list-over-frame",
    "Rolling.aggregate",
    level="L3",
    covers=("func",),
    frames=("float64_no_nulls", "int64_no_nulls"),
    expr=lambda pd, df: df.rolling(3).agg(["sum", "mean"]),
    rules=RUNNING,
    note="a list over a frame answers under the column and then the reduction, two levels "
    "of column labels, which is firepanda's Python layer, so the case runs in process",
    in_process=True,
)
case(
    "windows/expanding-dict-of-lists",
    "Expanding.aggregate",
    level="L3",
    covers=("func",),
    frames=("float64_no_nulls",),
    expr=lambda pd, df: df.expanding().agg({df.columns[0]: ["sum", "max"]}),
    rules=RUNNING,
    note="a dict holding a list reduces that column under two levels of column labels",
    in_process=True,
)
case(
    "windows/ewm-list-over-frame",
    "ExponentialMovingWindow.aggregate",
    level="L3",
    covers=("func",),
    frames=("float64_no_nulls",),
    expr=lambda pd, df: df.ewm(com=1).agg(["mean", "std"]),
    rules=RUNNING,
    note="a list over a decayed window of a frame answers under two levels of column labels",
    in_process=True,
)
case(
    "windows/rolling-corr-every-pair",
    "Rolling.corr",
    level="L3",
    covers=("pairwise",),
    frames=("tall",),
    expr=lambda pd, df: df[["value", "key"]].astype("float64").rolling(20).corr(),
    in_process=True,
    note="with no other frame every pair of columns is answered, each row label once a "
    "column under a second level of row labels",
    rules=SPREAD,
)
case(
    "windows/ewm-cov-every-pair",
    "ExponentialMovingWindow.cov",
    level="L3",
    covers=("other", "pairwise"),
    frames=("tall",),
    expr=lambda pd, df: (
        df[["value", "key"]]
        .astype("float64")
        .ewm(span=10)
        .cov(df[["key"]].astype("float64"), pairwise=True)
    ),
    in_process=True,
    note="against another frame the second level of row labels is the other frame's columns",
    rules=SPREAD,
)


def _keyed(pd):
    return pd.DataFrame(
        {
            "g": ["a", "b", "a", "b", "a", "b"],
            "x": [1.0, 2.0, float("nan"), 4.0, 5.0, 7.0],
            "y": [1, 2, 3, 4, 5, 6],
            "t": pd.date_range("2024-01-01", periods=6, freq="D"),
        }
    )


def _wobbly(pd):
    return pd.Series([1.0, 3.0, 2.0, 8.0, 5.0, 4.0, 4.0])


def _mixed(pd):
    return pd.DataFrame({"a": [1.0, 2.0, 4.0, 1.0], "b": ["x", "y", "z", "w"]})


def _side_by_side(pd, answers):
    return pd.concat(list(answers.values()), axis=1, keys=list(answers))


SELF_BUILT = "In process because the case builds its own small frame"
case(
    "windows/grouped-ewm-options",
    "GroupBy.ewm",
    level="L3",
    covers=("com", "span", "halflife", "alpha", "min_periods", "adjust", "ignore_na", "times"),
    frames=("single",),
    expr=lambda pd, df: _side_by_side(
        pd,
        {
            "com": _keyed(pd).groupby("g")["x"].ewm(com=0.5, min_periods=1, adjust=True).mean(),
            "span": _keyed(pd).groupby("g")["x"].ewm(span=3).mean(),
            "halflife": _keyed(pd).groupby("g")["x"].ewm(halflife=2.0).mean(),
            "alpha": _keyed(pd)
            .groupby("g")["x"]
            .ewm(alpha=0.3, adjust=False, ignore_na=True)
            .mean(),
            "times": _keyed(pd).groupby("g")["x"].ewm(halflife="1D", times=_keyed(pd)["t"]).mean(),
        },
    ),
    in_process=True,
    note="each group weighted on its own under every decay setting. " + SELF_BUILT,
    rules=RUNNING,
)
case(
    "windows/grouped-rolling-options",
    "GroupBy.rolling",
    level="L3",
    covers=("window", "min_periods", "center", "win_type", "closed", "method"),
    frames=("single",),
    expr=lambda pd, df: _side_by_side(
        pd,
        {
            "closed": _keyed(pd).groupby("g")["x"].rolling(2, min_periods=1, closed="right").sum(),
            "center": _keyed(pd).groupby("g")["x"].rolling(3, min_periods=1, center=True).mean(),
            "win_type": _keyed(pd)
            .groupby("g")["x"]
            .rolling(2, win_type="triang", min_periods=1)
            .sum(),
            "method": _keyed(pd).groupby("g")["x"].rolling(2, method="single").max(),
            "attribute": _keyed(pd).groupby("g").rolling(2).y.mean(),
        },
    ),
    in_process=True,
    note="win_type is taken and left unused, as pandas never weights a window over groups, "
    "and a column is picked by attribute. " + SELF_BUILT,
    rules=RUNNING,
)
case(
    "windows/grouped-rolling-on",
    "GroupBy.rolling",
    level="L3",
    covers=("on", "closed"),
    frames=("single",),
    expr=lambda pd, df: _keyed(pd).groupby("g").rolling("2D", on="t", closed="both")["x"].sum(),
    in_process=True,
    note="a picked column of a window ordered by on is labelled by the group and the instants. "
    + SELF_BUILT,
    rules=RUNNING,
)
for _name, _rules in (("var", SPREAD), ("std", SPREAD), ("rank", None)):
    _extra = {} if _rules is None else {"rules": _rules}
    case(
        f"windows/rolling-{_name}-numeric-only",
        f"Rolling.{_name}",
        level="L3",
        covers=("numeric_only",),
        frames=("single",),
        expr=lambda pd, df, _name=_name: getattr(_mixed(pd).rolling(2), _name)(numeric_only=True),
        in_process=True,
        note="the text column left out of the answer. " + SELF_BUILT,
        **_extra,
    )
    case(
        f"windows/expanding-{_name}-numeric-only",
        f"Expanding.{_name}",
        level="L3",
        covers=("numeric_only",),
        frames=("single",),
        expr=lambda pd, df, _name=_name: getattr(_mixed(pd).expanding(), _name)(numeric_only=True),
        in_process=True,
        note="the text column left out of the answer. " + SELF_BUILT,
        **_extra,
    )
case(
    "windows/rolling-var-std-ddof",
    "Rolling.var",
    level="L3",
    covers=("ddof",),
    frames=("single",),
    expr=lambda pd, df: _side_by_side(
        pd,
        {
            "var0": _wobbly(pd).rolling(3).var(ddof=0),
            "var2": _wobbly(pd).rolling(3).var(ddof=2),
        },
    ),
    in_process=True,
    note="the divisor moved by ddof. " + SELF_BUILT,
    rules=SPREAD,
)
case(
    "windows/rolling-std-ddof",
    "Rolling.std",
    level="L3",
    covers=("ddof",),
    frames=("single",),
    expr=lambda pd, df: _side_by_side(
        pd,
        {
            "std0": _wobbly(pd).rolling(3).std(ddof=0),
            "std2": _wobbly(pd).rolling(3).std(ddof=2),
        },
    ),
    in_process=True,
    note="the divisor moved by ddof. " + SELF_BUILT,
    rules=SPREAD,
)
case(
    "windows/rolling-rank-options",
    "Rolling.rank",
    level="L3",
    covers=("method", "ascending", "pct"),
    frames=("single",),
    expr=lambda pd, df: _side_by_side(
        pd,
        {
            "min": _wobbly(pd).rolling(3).rank(method="min", ascending=False, pct=True),
            "max": _wobbly(pd).rolling(3).rank(method="max"),
            "average": _wobbly(pd).rolling(3).rank(method="average", pct=True),
        },
    ),
    in_process=True,
    note="the last row's rank in its window, ties by the method, falling or as a share. "
    + SELF_BUILT,
)
case(
    "windows/expanding-var-std-ddof",
    "Expanding.var",
    level="L3",
    covers=("ddof",),
    frames=("single",),
    expr=lambda pd, df: _side_by_side(
        pd,
        {
            "var0": _wobbly(pd).expanding().var(ddof=0),
            "var2": _wobbly(pd).expanding(min_periods=3).var(ddof=2),
        },
    ),
    in_process=True,
    note="the growing window's variance with the divisor moved. " + SELF_BUILT,
    rules=SPREAD,
)
case(
    "windows/expanding-std-ddof",
    "Expanding.std",
    level="L3",
    covers=("ddof",),
    frames=("single",),
    expr=lambda pd, df: _side_by_side(
        pd,
        {
            "std0": _wobbly(pd).expanding().std(ddof=0),
            "std1": _wobbly(pd).expanding(min_periods=2).std(ddof=1),
        },
    ),
    in_process=True,
    note="the growing window's spread with the divisor moved. " + SELF_BUILT,
    rules=SPREAD,
)
case(
    "windows/expanding-rank-options",
    "Expanding.rank",
    level="L3",
    covers=("method", "ascending", "pct"),
    frames=("single",),
    expr=lambda pd, df: _side_by_side(
        pd,
        {
            "average": _wobbly(pd).expanding().rank(method="average", ascending=True, pct=False),
            "max": _wobbly(pd).expanding().rank(method="max", ascending=False, pct=True),
            "min": _wobbly(pd).expanding().rank(method="min"),
        },
    ),
    in_process=True,
    note="each row ranked among every row so far. " + SELF_BUILT,
)
case(
    "windows/expanding-rank-dense",
    "Expanding.rank",
    level="L4",
    frames=("single",),
    expr=lambda pd, df: _wobbly(pd).expanding().rank(method="dense"),
    raises=("ValueError", "Method 'dense' is not supported"),
    in_process=True,
    note="a window rank has no dense method. " + SELF_BUILT,
)
