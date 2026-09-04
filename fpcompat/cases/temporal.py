"""Timestamps, time zones, durations and the `dt` accessor.

Four of the corpus frames exist only for this section and each one is a specific trap.

The resolutions frame carries the same instants at second, millisecond, microsecond
and nanosecond precision, because pandas 2 made the unit part of the dtype and an
implementation that assumes nanoseconds everywhere passes every case that only ever
uses nanoseconds.

The two New York frames straddle a daylight saving transition. The forward one
contains a wall clock time that does not exist and the back one contains a wall clock
time that happens twice, and localizing either of those is a decision with three
possible answers.

The Lord Howe frame is there because its offset changes by half an hour rather than a
whole one, which breaks any code that stores an offset in whole hours, and there is a
lot of that code.
"""

from __future__ import annotations

from fpcompat.cases import case, section
from fpcompat.compare import Rules

section("temporal")

RESOLUTIONS = ("temporal_resolutions",)
ZONED = ("temporal_dst_forward", "temporal_dst_back", "temporal_dst_lord_howe")
RANGE = ("temporal_range",)
UNITS = ("s", "ms", "us", "ns")

STRICT = Rules(strict_index=True)

# ---------------------------------------------------------------------------
# The parts of a timestamp
# ---------------------------------------------------------------------------

for name in (
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "microsecond",
    "nanosecond",
    "dayofweek",
    "dayofyear",
    "quarter",
    "days_in_month",
    "is_leap_year",
    "is_month_start",
    "is_month_end",
    "is_quarter_start",
    "is_quarter_end",
    "is_year_start",
    "is_year_end",
):
    case(
        f"temporal/{name.replace('_', '-')}",
        f"dt.{name}",
        frames=RESOLUTIONS + RANGE,
        expr=(lambda field: lambda pd, df: getattr(df["us" if "us" in df else "second"].dt, field))(
            name
        ),
        note="the range frame crosses a month end, a quarter end and a leap day, which "
        "is what makes these more than a division",
    )

for unit in UNITS:
    case(
        f"temporal/nanosecond-{unit}",
        "dt.nanosecond",
        frames=RESOLUTIONS,
        expr=(lambda column: lambda pd, df: df[column].dt.nanosecond)(unit),
        note="the same instant at four precisions, and only the nanosecond column can "
        "have anything but zero here",
    )
    case(
        f"temporal/unit-{unit}",
        "dt.unit",
        frames=RESOLUTIONS,
        expr=(lambda column: lambda pd, df: df[column].dt.unit)(unit),
        note="the unit is part of the dtype since pandas 2, so an implementation that "
        "normalizes everything to nanoseconds fails here and nowhere else",
    )
    case(
        f"temporal/dtype-{unit}",
        "Series.dtype",
        frames=RESOLUTIONS,
        expr=(lambda column: lambda pd, df: str(df[column].dtype))(column := unit),
    )

case(
    "temporal/day-name",
    "dt.day_name",
    frames=RANGE,
    expr=lambda pd, df: df["second"].dt.day_name(),
)
case(
    "temporal/month-name",
    "dt.month_name",
    frames=RANGE,
    expr=lambda pd, df: df["second"].dt.month_name(),
)
case(
    "temporal/isocalendar",
    "dt.isocalendar",
    frames=RANGE,
    expr=lambda pd, df: df["second"].dt.isocalendar(),
    note="the ISO week year is not the calendar year at the turn of January, which is "
    "the only reason this is not three subtractions",
)
case(
    "temporal/date",
    "dt.date",
    frames=RANGE + RESOLUTIONS,
    expr=lambda pd, df: df["second" if "second" in df else "us"].dt.date,
)
case(
    "temporal/time",
    "dt.time",
    frames=RANGE,
    expr=lambda pd, df: df["second"].dt.time,
)
case(
    "temporal/normalize",
    "dt.normalize",
    frames=RANGE + RESOLUTIONS,
    expr=lambda pd, df: df["second" if "second" in df else "us"].dt.normalize(),
)
case(
    "temporal/strftime",
    "dt.strftime",
    level="L3",
    covers=("date_format",),
    frames=RANGE + RESOLUTIONS,
    expr=lambda pd, df: df["second" if "second" in df else "us"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
)

# ---------------------------------------------------------------------------
# Rounding, where the unit and the tie breaking both matter
# ---------------------------------------------------------------------------

for name in ("floor", "ceil", "round"):
    case(
        f"temporal/{name}",
        f"dt.{name}",
        level="L3",
        covers=("freq",),
        frames=RANGE + RESOLUTIONS,
        expr=(
            lambda method: (
                lambda pd, df: getattr(df["second" if "second" in df else "us"].dt, method)("h")
            )
        )(name),
        note="round breaks a tie to the even hour, which is the same rule as the "
        "numeric round and is not what anyone expects from a clock",
    )

case(
    "temporal/round-minute",
    "dt.round",
    level="L3",
    covers=("freq",),
    frames=RANGE,
    expr=lambda pd, df: df["second"].dt.round("min"),
)
case(
    "temporal/as-unit",
    "dt.as_unit",
    level="L3",
    covers=("unit",),
    frames=RESOLUTIONS,
    expr=lambda pd, df: df["ns"].dt.as_unit("s"),
    note="going down in precision truncates rather than rounding, and going back up "
    "does not recover what was lost",
)
case(
    "temporal/as-unit-up",
    "dt.as_unit",
    level="L3",
    covers=("unit",),
    frames=RESOLUTIONS,
    expr=lambda pd, df: df["s"].dt.as_unit("ns"),
)

# ---------------------------------------------------------------------------
# Time zones
# ---------------------------------------------------------------------------

case(
    "temporal/tz",
    "dt.tz",
    frames=ZONED,
    expr=lambda pd, df: str(df["zoned"].dt.tz),
)
case(
    "temporal/tz-convert-utc",
    "dt.tz_convert",
    level="L3",
    covers=("tz",),
    frames=ZONED,
    expr=lambda pd, df: df["zoned"].dt.tz_convert("UTC"),
    note="the instant does not move, only the wall clock reading does, which is the "
    "one sentence that separates convert from localize",
)
case(
    "temporal/tz-convert-half-hour",
    "dt.tz_convert",
    level="L3",
    covers=("tz",),
    frames=ZONED,
    expr=lambda pd, df: df["zoned"].dt.tz_convert("Asia/Kolkata"),
    note="a zone whose offset is not a whole number of hours, which is where an "
    "implementation storing offsets in hours falls over",
)
case(
    "temporal/tz-convert-hour",
    "dt.hour",
    frames=ZONED,
    expr=lambda pd, df: df["zoned"].dt.tz_convert("UTC").dt.hour,
    note="the wall clock hour after converting, which is the number a person actually "
    "reads and the one a wrong offset changes",
)
case(
    "temporal/tz-localize",
    "dt.tz_localize",
    level="L3",
    covers=("tz",),
    frames=RANGE,
    expr=lambda pd, df: df["second"].dt.tz_localize("UTC"),
    note="the range frame is a plain hourly sequence with no transition in it, so this "
    "is the case that works, and the two that do not are below",
)
case(
    "temporal/tz-localize-nonexistent-shift",
    "dt.tz_localize",
    level="L3",
    covers=("tz", "nonexistent"),
    frames=("temporal_dst_forward",),
    expr=lambda pd, df: df["naive"].dt.tz_localize("America/New_York", nonexistent="shift_forward"),
    note="a wall clock time that never happened, shifted forward past the gap, which "
    "is one of four possible answers and the other three are the next cases",
)
case(
    "temporal/tz-localize-nonexistent-nat",
    "dt.tz_localize",
    level="L3",
    covers=("tz", "nonexistent"),
    frames=("temporal_dst_forward",),
    expr=lambda pd, df: df["naive"].dt.tz_localize("America/New_York", nonexistent="NaT"),
)
case(
    "temporal/tz-localize-ambiguous-true",
    "dt.tz_localize",
    level="L3",
    covers=("tz", "ambiguous"),
    frames=("temporal_dst_back",),
    expr=lambda pd, df: df["naive"].dt.tz_localize("America/New_York", ambiguous=True),
    note="a wall clock time that happens twice, resolved to the first one, which is "
    "what true means and is not obvious from the name",
)
case(
    "temporal/tz-localize-ambiguous-false",
    "dt.tz_localize",
    level="L3",
    covers=("tz", "ambiguous"),
    frames=("temporal_dst_back",),
    expr=lambda pd, df: df["naive"].dt.tz_localize("America/New_York", ambiguous=False),
)
case(
    "temporal/tz-localize-ambiguous-nat",
    "dt.tz_localize",
    level="L3",
    covers=("tz", "ambiguous"),
    frames=("temporal_dst_back",),
    expr=lambda pd, df: df["naive"].dt.tz_localize("America/New_York", ambiguous="NaT"),
)
case(
    "temporal/tz-localize-lord-howe",
    "dt.tz_localize",
    level="L3",
    covers=("tz", "ambiguous", "nonexistent"),
    frames=("temporal_dst_lord_howe",),
    expr=lambda pd, df: df["naive"].dt.tz_localize(
        "Australia/Lord_Howe", ambiguous="NaT", nonexistent="NaT"
    ),
    note="a half hour transition, so the gap and the repeat are thirty minutes wide "
    "rather than sixty and an implementation with an hour hardcoded gets it wrong",
)
case(
    "temporal/tz-localize-none",
    "dt.tz_localize",
    level="L3",
    covers=("tz",),
    frames=ZONED,
    expr=lambda pd, df: df["zoned"].dt.tz_localize(None),
    note="dropping the zone keeps the wall clock reading and changes the instant, "
    "which is the opposite of what convert does",
)
case(
    "temporal/dst-difference",
    "Series.sub",
    frames=ZONED,
    expr=lambda pd, df: df["zoned"].diff(),
    note="the difference across a transition is not what the wall clocks suggest, "
    "which is the whole reason these frames exist",
)

# ---------------------------------------------------------------------------
# Durations
# ---------------------------------------------------------------------------

case(
    "temporal/duration-dtype",
    "Series.dtype",
    frames=("temporal_durations",),
    expr=lambda pd, df: str(df["value"].dtype),
)
case(
    "temporal/total-seconds",
    "Series.dt",
    frames=("temporal_durations",),
    expr=lambda pd, df: df["value"].dt.total_seconds(),
    note="a float, so a duration longer than a couple of hundred years loses "
    "precision, and the corpus has one",
)
case(
    "temporal/duration-days",
    "Series.dt",
    frames=("temporal_durations",),
    expr=lambda pd, df: df["value"].dt.days,
)
case(
    "temporal/duration-sum",
    "Series.sum",
    frames=("temporal_durations",),
    expr=lambda pd, df: df["value"].sum(),
)
case(
    "temporal/duration-mean",
    "Series.mean",
    frames=("temporal_durations",),
    expr=lambda pd, df: df["value"].mean(),
)
case(
    "temporal/duration-abs",
    "Series.abs",
    frames=("temporal_durations",),
    expr=lambda pd, df: df["value"].abs(),
    note="the corpus has negative durations in it, which are legal and which a lot of "
    "code does not expect",
)
case(
    "temporal/timestamp-minus-timestamp",
    "Series.sub",
    frames=RANGE,
    expr=lambda pd, df: df["second"] - df["second"].iloc[0],
)
case(
    "temporal/timestamp-plus-duration",
    "Series.add",
    frames=RANGE,
    expr=lambda pd, df: df["second"] + pd.Timedelta(hours=1),
)
case(
    "temporal/timedelta-construct",
    "pandas.Timedelta",
    level="L3",
    covers=("value", "unit"),
    frames=RANGE,
    expr=lambda pd, df: df["second"] + pd.Timedelta(90, unit="s"),
)
case(
    "temporal/to-timedelta",
    "pandas.to_timedelta",
    level="L3",
    covers=("arg", "unit"),
    frames=("int64_no_nulls",),
    expr=lambda pd, df: pd.to_timedelta(df["value"], unit="s"),
)

# ---------------------------------------------------------------------------
# Ranges, parsing and resampling
# ---------------------------------------------------------------------------

case(
    "temporal/to-datetime-strings",
    "pandas.to_datetime",
    level="L3",
    covers=("arg",),
    frames=RANGE,
    expr=lambda pd, df: pd.to_datetime(df["second"].dt.strftime("%Y-%m-%d %H:%M:%S")),
    note="a round trip through text, which is where a resolution quietly becomes nanoseconds again",
)
case(
    "temporal/to-datetime-format",
    "pandas.to_datetime",
    level="L3",
    covers=("arg", "format"),
    frames=RANGE,
    expr=lambda pd, df: pd.to_datetime(df["second"].dt.strftime("%d/%m/%Y"), format="%d/%m/%Y"),
)
case(
    "temporal/date-range",
    "pandas.date_range",
    level="L3",
    covers=("start", "periods", "freq"),
    frames=RANGE,
    expr=lambda pd, df: pd.date_range(start=df["second"].iloc[0], periods=10, freq="D"),
)
case(
    "temporal/date-range-tz",
    "pandas.date_range",
    level="L3",
    covers=("start", "periods", "freq", "tz"),
    frames=RANGE,
    expr=lambda pd, df: pd.date_range(
        start="2024-03-09", periods=6, freq="12h", tz="America/New_York"
    ),
    note="a range that steps over the spring transition, so the wall clock readings "
    "are not evenly spaced even though the instants are",
)
case(
    "temporal/resample-sum",
    "DataFrame.resample",
    level="L3",
    covers=("rule",),
    frames=RANGE,
    expr=lambda pd, df: df.set_index("second")["row"].resample("D").sum(),
)
case(
    "temporal/resample-mean",
    "Resampler.mean",
    frames=RANGE,
    expr=lambda pd, df: df.set_index("second")["row"].resample("6h").mean(),
)
case(
    "temporal/resample-count",
    "Resampler.count",
    frames=RANGE,
    expr=lambda pd, df: df.set_index("second")["row"].resample("D").count(),
    note="an empty bucket produces a row with a zero in it rather than no row, which "
    "is the difference between resampling and grouping by a truncated timestamp",
)
case(
    "temporal/resample-ohlc",
    "Resampler.ohlc",
    frames=RANGE,
    expr=lambda pd, df: df.set_index("second")["row"].resample("D").ohlc(),
)
case(
    "temporal/asfreq",
    "DataFrame.asfreq",
    level="L3",
    covers=("freq",),
    frames=RANGE,
    expr=lambda pd, df: df.set_index("second").asfreq("2h"),
    rules=STRICT,
)
case(
    "temporal/groupby-day",
    "GroupBy.sum",
    frames=RANGE,
    expr=lambda pd, df: df.groupby(df["second"].dt.date)["row"].sum(),
)
case(
    "temporal/sort-timestamps",
    "Series.sort_values",
    frames=RANGE + RESOLUTIONS,
    expr=lambda pd, df: df["second" if "second" in df else "us"].sort_values(),
    rules=STRICT,
)
case(
    "temporal/max",
    "Series.max",
    frames=RANGE + RESOLUTIONS + ZONED,
    expr=lambda pd, df: df["second" if "second" in df else ("us" if "us" in df else "zoned")].max(),
)
case(
    "temporal/date-column",
    "Series.dtype",
    frames=RANGE,
    expr=lambda pd, df: str(df["date"].dtype),
    note="a date32 column comes back as objects holding Python dates, which is a "
    "pandas fact rather than a good idea and it has to be copied anyway",
)
