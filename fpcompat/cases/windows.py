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

for name in ("std", "var", "sem", "skew", "kurt"):
    case(
        f"windows/rolling-{name}",
        f"Rolling.{name}",
        frames=("float64_no_nulls", "tall"),
        expr=(lambda method: lambda pd, df: getattr(df["value"].rolling(8), method)())(name),
        rules=SPREAD,
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
    rules=SPREAD,
)
case(
    "windows/rolling-cov",
    "Rolling.cov",
    level="L3",
    covers=("other",),
    frames=("tall",),
    expr=lambda pd, df: df["value"].rolling(20).cov(df["key"].astype("float64")),
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
)
case(
    "windows/rolling-time",
    "DataFrame.rolling",
    level="L3",
    covers=("window", "on"),
    frames=("temporal_range",),
    expr=lambda pd, df: df.rolling("30s", on="second")["row"].sum(),
    note="a time based window is a different algorithm from a count based one, and it "
    "needs the index to be sorted and the offsets to be understood",
)

# ---------------------------------------------------------------------------
# Expanding
# ---------------------------------------------------------------------------

for name in ("sum", "mean", "min", "max", "count"):
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
    "windows/expanding-apply",
    "Expanding.apply",
    level="L3",
    covers=("func",),
    frames=("float64_no_nulls",),
    expr=lambda pd, df: df["value"].expanding(4).apply(lambda window: window.iloc[0]),
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
    rules=SPREAD,
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
    "also the thing that is easiest to implement by accident as one long window",
)
case(
    "windows/groupby-expanding",
    "GroupBy.expanding",
    frames=("keys_10",),
    expr=lambda pd, df: df.groupby("key")["value"].expanding().sum(),
    rules=RUNNING,
)
case(
    "windows/groupby-ewm",
    "GroupBy.ewm",
    level="L3",
    covers=("span",),
    frames=("keys_10",),
    expr=lambda pd, df: df.groupby("key")["value"].ewm(span=3).mean(),
    rules=RUNNING,
)
