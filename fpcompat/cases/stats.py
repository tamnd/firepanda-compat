"""Descriptive statistics.

The section where the tolerance classes earn their keep. A sum of ten thousand doubles
is not one number, it is a family of numbers that depend on the order and the width of
the accumulator, and pretending otherwise means either a suite that fails on a
different machine or a tolerance so loose it accepts a wrong answer. So everything
here that accumulates says so and says why, and everything that does not, like a
median of an odd length column, is compared exactly.

The quantile cases are the ones worth reading. Seven interpolation methods, and the
difference between them only shows when the quantile falls between two elements, which
is why they all run on a frame with an even number of rows.
"""

from __future__ import annotations

from fpcompat.cases import case, section
from fpcompat.compare import Rules, Tolerance

section("stats")

FLOATS = ("float64_no_nulls", "float64_half_null", "float64_all_null")
NUMERIC = ("int64_no_nulls", "int64_half_null")
BIG = ("tall",)

SPREAD = Rules(
    tolerance=Tolerance.STATISTICAL,
    reason="a variance is a sum of squares and every implementation associates it "
    "differently, so the last few bits are not a conformance question",
)
ACCUMULATED = Rules(
    tolerance=Tolerance.ACCUMULATION,
    reason="ten thousand doubles added in some order, and the order is not part of the API",
)

# ---------------------------------------------------------------------------
# describe, which is eight statistics in one call
# ---------------------------------------------------------------------------

case(
    "stats/describe-numeric",
    "Series.describe",
    frames=FLOATS + NUMERIC,
    expr=lambda pd, df: df["value"].describe(),
    rules=SPREAD,
    note="the all null column is the one to read, because describe on it still has a "
    "count of zero and seven nulls rather than raising",
)
case(
    "stats/describe-frame",
    "DataFrame.describe",
    frames=("two", "tall"),
    expr=lambda pd, df: df.describe(),
    rules=SPREAD,
)
case(
    "stats/describe-strings",
    "Series.describe",
    frames=("strings_ascii", "strings_null_heavy"),
    expr=lambda pd, df: df["value"].describe(),
    note="a different set of statistics for a string column, which is a shape change "
    "driven by the dtype and not by a parameter",
)
case(
    "stats/describe-percentiles",
    "Series.describe",
    level="L3",
    covers=("percentiles",),
    frames=("tall",),
    expr=lambda pd, df: df["value"].describe(percentiles=[0.1, 0.9]),
    rules=SPREAD,
    note="the median is always in the answer whether or not it was asked for",
)
case(
    "stats/describe-categorical",
    "Series.describe",
    frames=("categorical_ordered", "categorical_unordered"),
    expr=lambda pd, df: df["value"].describe(),
)

# ---------------------------------------------------------------------------
# Spread
# ---------------------------------------------------------------------------

for name in ("std", "var", "sem", "skew", "kurt"):
    case(
        f"stats/{name}",
        f"Series.{name}",
        frames=FLOATS + NUMERIC + BIG,
        expr=(lambda method: lambda pd, df: getattr(df["value"], method)())(name),
        rules=SPREAD,
        note="the default is the sample form with one degree of freedom taken out, "
        "which is not what a naive implementation writes",
    )

case(
    "stats/std-ddof-zero",
    "Series.std",
    level="L3",
    covers=("ddof",),
    frames=("float64_no_nulls", "tall"),
    expr=lambda pd, df: df["value"].std(ddof=0),
    rules=SPREAD,
    note="the population form, which is what most other libraries default to and pandas does not",
)
case(
    "stats/var-ddof-zero",
    "Series.var",
    level="L3",
    covers=("ddof",),
    frames=("float64_no_nulls", "tall"),
    expr=lambda pd, df: df["value"].var(ddof=0),
    rules=SPREAD,
)
case(
    "stats/std-single-row",
    "Series.std",
    frames=("single",),
    expr=lambda pd, df: df["b"].std(),
    note="one row and one degree of freedom removed leaves zero, and the answer is a "
    "null rather than a division by zero",
)

# ---------------------------------------------------------------------------
# Quantiles
# ---------------------------------------------------------------------------

case(
    "stats/median",
    "Series.median",
    frames=FLOATS + NUMERIC + BIG,
    expr=lambda pd, df: df["value"].median(),
    rules=Rules(
        tolerance=Tolerance.SINGLE,
        reason="an even row count makes the median an average of two neighbours",
    ),
)
for method in ("linear", "lower", "higher", "nearest", "midpoint"):
    case(
        f"stats/quantile-{method}",
        "Series.quantile",
        level="L3",
        covers=("q", "interpolation"),
        frames=("float64_no_nulls", "tall"),
        expr=(lambda kind: lambda pd, df: df["value"].quantile(0.25, interpolation=kind))(method),
        rules=Rules(
            tolerance=Tolerance.SINGLE,
            reason="only the interpolating methods average two numbers, and the exact "
            "ones cost nothing to include under the same rule",
        ),
        note="sixty four rows means the quarter quantile falls between two elements, "
        "which is the only situation where these five differ",
    )

case(
    "stats/quantile-list",
    "Series.quantile",
    level="L3",
    covers=("q",),
    frames=("tall",),
    expr=lambda pd, df: df["value"].quantile([0.0, 0.25, 0.5, 0.75, 1.0]),
    rules=Rules(
        tolerance=Tolerance.SINGLE,
        reason="three of the five interpolate",
    ),
    note="a list of quantiles gives a Series indexed by the quantile, which is a "
    "float index and the only one in the corpus",
)
case(
    "stats/quantile-frame",
    "DataFrame.quantile",
    level="L3",
    covers=("q",),
    frames=("two", "keys_two_column"),
    expr=lambda pd, df: df.quantile(0.5, numeric_only=True),
    rules=Rules(tolerance=Tolerance.SINGLE, reason="an interpolated median"),
)
case(
    "stats/quantile-nulls",
    "Series.quantile",
    level="L3",
    covers=("q",),
    frames=("float64_half_null", "float64_all_null"),
    expr=lambda pd, df: df["value"].quantile(0.5),
    rules=Rules(tolerance=Tolerance.SINGLE, reason="an interpolated median"),
    note="nulls are dropped before the quantile is taken rather than sorted to one "
    "end, and an all null column gives a null",
)
case(
    "stats/mode",
    "Series.mode",
    frames=("keys_10", "keys_awkward", "strings_null_heavy"),
    expr=lambda pd, df: df.iloc[:, 0].mode(),
    note="every tied value and not just one of them, sorted, which is why the return "
    "is a Series and not a scalar",
)

# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

case(
    "stats/corr-pearson",
    "Series.corr",
    level="L3",
    covers=("other", "method"),
    frames=("tall", "keys_two_column"),
    expr=lambda pd, df: df.iloc[:, -1].corr(df.iloc[:, -2], method="pearson"),
    rules=SPREAD,
)
# Spearman and Kendall are not here on purpose. pandas hands both of them to scipy and
# raises an ImportError when scipy is absent, so a case for either would be measuring
# scipy rather than pandas and would drag scipy into the pinned environment to do it.
# If firepanda ever implements them, the thing to compare against is scipy directly
# and that belongs in a different harness.

case(
    "stats/corr-frame",
    "DataFrame.corr",
    frames=("tall", "keys_two_column"),
    expr=lambda pd, df: df.corr(numeric_only=True),
    rules=SPREAD,
    level="L3",
    covers=("numeric_only",),
)
case(
    "stats/cov",
    "Series.cov",
    level="L3",
    covers=("other",),
    frames=("tall", "keys_two_column"),
    expr=lambda pd, df: df.iloc[:, -1].cov(df.iloc[:, -2]),
    rules=SPREAD,
)
case(
    "stats/cov-frame",
    "DataFrame.cov",
    frames=("tall",),
    expr=lambda pd, df: df.cov(numeric_only=True),
    rules=SPREAD,
    level="L3",
    covers=("numeric_only",),
)
case(
    "stats/autocorr",
    "Series.autocorr",
    level="L3",
    covers=("lag",),
    frames=("tall",),
    expr=lambda pd, df: df["value"].autocorr(lag=1),
    rules=SPREAD,
)
case(
    "stats/corr-with-nulls",
    "Series.corr",
    level="L3",
    covers=("other",),
    frames=("float64_half_null",),
    expr=lambda pd, df: df["value"].corr(df["row"].astype("float64")),
    rules=SPREAD,
    note="pairs where either side is null are dropped, so the count going into the "
    "correlation is not the row count",
)

# ---------------------------------------------------------------------------
# Running and cumulative
# ---------------------------------------------------------------------------

case(
    "stats/cumsum-tall",
    "Series.cumsum",
    frames=BIG,
    expr=lambda pd, df: df["value"].cumsum(),
    rules=ACCUMULATED,
    note="every element of the answer is a partial sum, so the error grows along the "
    "column and the last element is the worst case",
)
case(
    "stats/cumsum-float-edges",
    "Series.cumsum",
    frames=("float64_no_nulls",),
    expr=lambda pd, df: df["value"].cumsum(),
    rules=Rules(tolerance=Tolerance.EXACT),
    note="a nan at offset zero means the whole running total is nan, and that is not a "
    "tolerance question, it is exact",
)
case(
    "stats/prod",
    "Series.prod",
    frames=("int64_no_nulls", "float64_no_nulls"),
    expr=lambda pd, df: df["value"].prod(),
    note="an integer product over sixty four values overflows, and what pandas does "
    "with that is the case",
)
case(
    "stats/nlargest-series",
    "Series.nlargest",
    level="L3",
    covers=("n",),
    frames=("tall", "keys_1000"),
    expr=lambda pd, df: df["value"].nlargest(10),
    rules=Rules(strict_index=True),
)
case(
    "stats/nsmallest-series",
    "Series.nsmallest",
    level="L3",
    covers=("n",),
    frames=("tall",),
    expr=lambda pd, df: df["value"].nsmallest(10),
    rules=Rules(strict_index=True),
)
case(
    "stats/rank-pct",
    "Series.rank",
    level="L3",
    covers=("pct",),
    frames=("tall", "keys_10"),
    expr=lambda pd, df: df["value"].rank(pct=True),
    rules=Rules(tolerance=Tolerance.SINGLE, reason="a rank divided by a count"),
)
case(
    "stats/rank-na-option",
    "Series.rank",
    level="L3",
    covers=("na_option",),
    frames=("float64_half_null",),
    expr=lambda pd, df: df["value"].rank(na_option="bottom"),
    note="where the nulls rank is a choice with three answers and the default is to "
    "leave them out of the ranking entirely",
)
case(
    "stats/interpolate",
    "Series.interpolate",
    frames=("float64_half_null",),
    expr=lambda pd, df: df["value"].interpolate(),
    rules=Rules(tolerance=Tolerance.SINGLE, reason="a linear interpolation is an average"),
    note="the nulls in the corpus are every other row rather than contiguous, which is "
    "the arrangement that makes interpolation and forward fill differ everywhere",
)
case(
    "stats/count-frame",
    "DataFrame.count",
    frames=("two", "float64_half_null", "strings_null_heavy"),
    expr=lambda pd, df: df.count(),
)
case(
    "stats/monotonic",
    "Series.is_monotonic_increasing",
    frames=("keys_unique", "keys_10", "tall"),
    expr=lambda pd, df: df.iloc[:, 0].is_monotonic_increasing,
)
case(
    "stats/hasnans",
    "Series.hasnans",
    frames=(*FLOATS, "strings_null_heavy"),
    expr=lambda pd, df: df["value"].hasnans,
)
case(
    "stats/argmax",
    "Series.argmax",
    frames=("tall", "keys_10"),
    expr=lambda pd, df: df["value"].argmax(),
    note="a position and not a label, which is the difference from idxmax and the only "
    "reason both exist",
)
case(
    "stats/argsort",
    "Series.argsort",
    frames=("keys_10", "float64_no_nulls"),
    expr=lambda pd, df: df["value"].argsort(),
)
case(
    "stats/searchsorted",
    "Series.searchsorted",
    level="L3",
    covers=("value",),
    frames=("keys_unique",),
    expr=lambda pd, df: df["key"].sort_values().searchsorted([0, 5, 1000]),
)
