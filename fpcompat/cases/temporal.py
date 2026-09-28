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
    "temporal/fillna-instant",
    "Series.fillna",
    covers=("value",),
    frames=RESOLUTIONS,
    expr=lambda pd, df: df["us"].where(df["row"] % 3 != 0).fillna(pd.Timestamp("2000-01-01")),
    in_process=True,
    note="an instant with no zone fills a column of instants with no zone and keeps its "
    "unit. In process because firepanda reads the fill value in its Python layer, which "
    "the driver cannot reach",
)
case(
    "temporal/fillna-instant-text",
    "Series.fillna",
    covers=("value",),
    frames=RESOLUTIONS,
    expr=lambda pd, df: df["s"].where(df["row"] % 2 == 0).fillna("2000-01-01 12:00"),
    in_process=True,
    note="text naming an instant is read as one, as pandas reads it. In process because "
    "firepanda reads the fill value in its Python layer, which the driver cannot reach",
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
    note="going down in precision floors rather than truncating towards zero, so half "
    "a second before the epoch is minus one second and not zero, and going back up "
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
    "temporal/to-datetime-guessed-month-first",
    "pandas.to_datetime",
    level="L3",
    covers=("arg",),
    frames=RANGE,
    expr=lambda pd, df: pd.to_datetime(df["second"].dt.strftime("%m/%d/%Y %H:%M")),
    in_process=True,
    note="no format, so the format is guessed from the first value and every row is held to it;"
    " the guesser is in Python, past the core the driver reaches",
)
case(
    "temporal/to-datetime-guessed-month-name",
    "pandas.to_datetime",
    level="L3",
    covers=("arg",),
    frames=RANGE,
    expr=lambda pd, df: pd.to_datetime(df["second"].dt.strftime("%B %d, %Y %I:%M %p")),
    in_process=True,
    note="a month name and a twelve hour clock, guessed with no format given;"
    " the guesser is in Python, past the core the driver reaches",
)
case(
    "temporal/to-datetime-guessed-short-month",
    "pandas.to_datetime",
    level="L3",
    covers=("arg",),
    frames=RANGE,
    expr=lambda pd, df: pd.to_datetime(df["second"].dt.strftime("%b %d %Y")),
    in_process=True,
    note="a short month name; the guesser is in Python, past the core the driver reaches",
)
case(
    "temporal/to-datetime-format-day-of-year",
    "pandas.to_datetime",
    level="L3",
    covers=("arg", "format"),
    frames=RANGE,
    expr=lambda pd, df: pd.to_datetime(df["second"].dt.strftime("%Y-%j"), format="%Y-%j"),
    in_process=True,
    note="the day of the year, which the core's format reader does not know and the Python layer"
    " reads the way pandas does",
)
case(
    "temporal/to-datetime-dayfirst",
    "pandas.to_datetime",
    level="L3",
    covers=("arg", "dayfirst"),
    frames=RANGE,
    expr=lambda pd, df: pd.to_datetime(df["second"].dt.strftime("%d/%m/%Y %H:%M"), dayfirst=True),
    in_process=True,
    note="the format is guessed day first; the guesser is in Python, past the core",
)
case(
    "temporal/to-datetime-mixed-yearfirst",
    "pandas.to_datetime",
    level="L3",
    covers=("arg", "format", "yearfirst"),
    frames=RANGE,
    expr=lambda pd, df: pd.to_datetime(
        df["second"].dt.strftime("%y/%m/%d"), format="mixed", yearfirst=True
    ),
    in_process=True,
    note="each row read on its own, two digit year first; the reader is in Python, past the core",
)
case(
    "temporal/to-datetime-iso8601-slashes",
    "pandas.to_datetime",
    level="L3",
    covers=("arg", "format"),
    frames=RANGE,
    expr=lambda pd, df: pd.to_datetime(
        df["second"].dt.strftime("%Y/%m/%d %H:%M:%S"), format="ISO8601"
    ),
    in_process=True,
    note="pandas' ISO 8601 reader takes slashes; the reader is in Python, past the core",
)
case(
    "temporal/date-range",
    "pandas.date_range",
    level="L3",
    covers=("start", "periods", "freq"),
    frames=RANGE,
    expr=lambda pd, df: pd.date_range(start=df["second"].iloc[0], periods=10, freq="D"),
    in_process=True,
    note="a range of days from a start held in seconds, which keeps the answer in seconds",
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
    in_process=True,
    note="a range that steps over the spring transition, so the wall clock readings "
    "are not evenly spaced even though the instants are",
)
CALENDAR_IN_PROCESS = (
    "In process because the driver has no entry for it, and firepanda's calendar steps are "
    "its Python layer walking the landing dates on the wall clock"
)
case(
    "temporal/date-range-month-end",
    "pandas.date_range",
    level="L3",
    covers=("start", "end", "freq"),
    frames=RANGE,
    expr=lambda pd, df: pd.date_range(start="2020-01-15 12:00", end="2020-06-30", freq="ME"),
    in_process=True,
    note="the start rolls forward to the month end and keeps its time of day, so the last "
    "month end at noon is past the end and left out. " + CALENDAR_IN_PROCESS,
)
case(
    "temporal/date-range-business",
    "pandas.date_range",
    level="L3",
    covers=("end", "periods", "freq"),
    frames=RANGE,
    expr=lambda pd, df: pd.date_range(end="2020-03-15 06:00", periods=6, freq="B"),
    in_process=True,
    note="an end on a Sunday rolls back to the Friday and the points count back from it. "
    + CALENDAR_IN_PROCESS,
)
case(
    "temporal/date-range-week-anchor",
    "pandas.date_range",
    level="L3",
    covers=("start", "periods", "freq", "name"),
    frames=RANGE,
    expr=lambda pd, df: pd.date_range(start="2020-01-04", periods=5, freq="2W-WED", name="w"),
    in_process=True,
    note="every other Wednesday from the first one on or after the start. " + CALENDAR_IN_PROCESS,
)
case(
    "temporal/date-range-quarter-anchor",
    "pandas.date_range",
    level="L3",
    covers=("start", "periods", "freq"),
    frames=RANGE,
    expr=lambda pd, df: pd.date_range(start="2020-02-04", periods=5, freq="BQE-JAN"),
    in_process=True,
    note="the last business day of quarters that end in January, April, July and October. "
    + CALENDAR_IN_PROCESS,
)
case(
    "temporal/date-range-year-start-tz",
    "pandas.date_range",
    level="L3",
    covers=("start", "periods", "freq", "tz"),
    frames=RANGE,
    expr=lambda pd, df: pd.date_range(
        start="2020-02-04", periods=3, freq="YS", tz="America/New_York"
    ),
    in_process=True,
    note="year starts counted on the wall clock of the zone. " + CALENDAR_IN_PROCESS,
)
case(
    "temporal/date-range-month-backwards",
    "pandas.date_range",
    level="L3",
    covers=("start", "end", "freq", "inclusive"),
    frames=RANGE,
    expr=lambda pd, df: pd.date_range(
        start="2020-06-30", end="2020-01-31", freq="-1ME", inclusive="left"
    ),
    in_process=True,
    note="a backwards step walks down from the start and drops the end. " + CALENDAR_IN_PROCESS,
)
case(
    "temporal/bdate-range",
    "pandas.bdate_range",
    level="L3",
    covers=("start", "end"),
    frames=RANGE,
    expr=lambda pd, df: pd.bdate_range(start="2020-03-13 10:00", end="2020-03-31"),
    in_process=True,
    note="business days with the start moved to midnight. " + CALENDAR_IN_PROCESS,
)
case(
    "temporal/bdate-range-custom",
    "pandas.bdate_range",
    level="L3",
    covers=("start", "periods", "freq", "weekmask", "holidays"),
    frames=RANGE,
    expr=lambda pd, df: pd.bdate_range(
        start="2020-03-13", periods=6, freq="C", weekmask="Mon Wed Fri", holidays=["2020-03-18"]
    ),
    in_process=True,
    note="a custom business day that works three days a week and skips a holiday. "
    + CALENDAR_IN_PROCESS,
)
OFFSETS_IN_PROCESS = (
    "In process because the driver has no entry for it, and firepanda's offsets are its Python "
    "layer moving each moment on the wall clock"
)


def _moments(pd, texts, tz=None):
    return pd.Series(pd.to_datetime(texts), name="t").dt.tz_localize(tz)


case(
    "temporal/offset-month-end",
    "pandas.offsets.MonthEnd",
    frames=RANGE,
    expr=lambda pd, df: (
        _moments(pd, ["2024-01-31 10:00", "2024-02-15 00:00", "2024-12-31 23:59"])
        + pd.offsets.MonthEnd(2)
    ),
    in_process=True,
    note="a moment on a month end moves two month ends on, one inside a month counts the end "
    "of its own month as the first, and the time of day is kept. " + OFFSETS_IN_PROCESS,
)
case(
    "temporal/offset-business-day",
    "pandas.offsets.BusinessDay",
    frames=RANGE,
    expr=lambda pd, df: (
        _moments(pd, ["2024-03-08 09:30", "2024-03-09 00:00", "2024-03-10 18:00"])
        - pd.offsets.BDay(3)
    ),
    in_process=True,
    note="three business days back, from a Friday, a Saturday and a Sunday. " + OFFSETS_IN_PROCESS,
)
case(
    "temporal/offset-custom-business-day",
    "pandas.offsets.CustomBusinessDay",
    frames=RANGE,
    expr=lambda pd, df: (
        _moments(pd, ["2024-12-23 00:00", "2024-12-24 12:00", "2024-12-31 00:00"])
        + pd.offsets.CustomBusinessDay(weekmask="Mon Tue Wed Thu", holidays=["2024-12-25"])
    ),
    in_process=True,
    note="a four day week that skips a holiday. " + OFFSETS_IN_PROCESS,
)
case(
    "temporal/offset-date-offset",
    "pandas.DateOffset",
    frames=RANGE,
    expr=lambda pd, df: (
        _moments(pd, ["2024-01-31 00:00", "2024-02-29 06:00", "2023-03-31 00:00"])
        + pd.DateOffset(years=1, months=1, days=2)
    ),
    in_process=True,
    note="years and months first, clipped to the end of a short month, then the days. "
    + OFFSETS_IN_PROCESS,
)
case(
    "temporal/offset-day-over-dst",
    "pandas.offsets.Day",
    frames=RANGE,
    expr=lambda pd, df: (
        _moments(pd, ["2024-03-09 12:00", "2024-11-02 12:00"], "America/New_York")
        + pd.offsets.Day(1)
    ),
    in_process=True,
    note="a day keeps the wall clock over a transition in pandas 3, where an hour count would "
    "not. " + OFFSETS_IN_PROCESS,
)
case(
    "temporal/offset-business-hour",
    "pandas.offsets.BusinessHour",
    frames=RANGE,
    expr=lambda pd, df: (
        _moments(pd, ["2024-03-08 16:30", "2024-03-09 10:00", "2024-03-11 08:00"])
        + pd.offsets.BusinessHour(3)
    ),
    in_process=True,
    note="three working hours from 9 to 5, carried over the weekend. " + OFFSETS_IN_PROCESS,
)
case(
    "temporal/date-range-offset",
    "pandas.date_range",
    level="L3",
    covers=("start", "periods", "freq"),
    frames=RANGE,
    expr=lambda pd, df: pd.date_range(
        start="2024-01-10 00:00", periods=5, freq=pd.offsets.BQuarterEnd(startingMonth=2)
    ),
    in_process=True,
    note="an offset as the frequency, business quarter ends anchored on February. "
    + OFFSETS_IN_PROCESS,
)
SPANS_IN_PROCESS = (
    "In process because the driver has no entry for it, and firepanda's span index and "
    "span range are its Python layer counting along the whole numbers of a unit"
)
case(
    "temporal/timedelta-range",
    "pandas.timedelta_range",
    level="L3",
    covers=("start", "periods", "freq", "name"),
    frames=RANGE,
    expr=lambda pd, df: pd.timedelta_range(start="1s", periods=6, freq="12h", name="span"),
    in_process=True,
    note="text ends read at microsecond resolution, stepped by half days. " + SPANS_IN_PROCESS,
)
case(
    "temporal/timedelta-range-closed",
    "pandas.timedelta_range",
    level="L3",
    covers=("start", "end", "freq", "closed"),
    frames=RANGE,
    expr=lambda pd, df: pd.timedelta_range(start="-1D", end="1D", freq="6h", closed="left"),
    in_process=True,
    note="a range across zero that drops the end it does not name. " + SPANS_IN_PROCESS,
)
case(
    "temporal/timedelta-range-even",
    "pandas.timedelta_range",
    level="L3",
    covers=("start", "end", "periods", "unit"),
    frames=RANGE,
    expr=lambda pd, df: pd.timedelta_range(start="1s", end="2s", periods=5, unit="ms"),
    in_process=True,
    note="no step, so the points are spaced evenly between the ends, in milliseconds. "
    + SPANS_IN_PROCESS,
)
case(
    "temporal/timedelta-range-compound-step",
    "pandas.timedelta_range",
    level="L3",
    covers=("start", "periods", "freq"),
    frames=RANGE,
    expr=lambda pd, df: pd.timedelta_range(start=1, periods=4, freq="2D3h"),
    in_process=True,
    note="a whole number start is nanoseconds, and a step of two units is read as a span. "
    + SPANS_IN_PROCESS,
)
case(
    "temporal/timedelta-index",
    "pandas.TimedeltaIndex",
    level="L3",
    covers=("data", "name"),
    frames=RANGE,
    expr=lambda pd, df: pd.TimedeltaIndex(["1s", "-2D3h", None, "1.5ms"], name="span"),
    in_process=True,
    note="text read as spans with a gap kept missing. " + SPANS_IN_PROCESS,
)
case(
    "temporal/timedelta-index-seconds",
    "pandas.TimedeltaIndex",
    frames=RANGE,
    expr=lambda pd, df: pd.TimedeltaIndex(["1s", "-2D3h", None, "1.5ms"], name="span").seconds,
    in_process=True,
    note="the seconds past the whole days, floored, so a negative span counts up from its "
    "day, and float64 because of the gap. " + SPANS_IN_PROCESS,
)
case(
    "temporal/timedelta-index-total-seconds",
    "pandas.TimedeltaIndex",
    frames=RANGE,
    expr=lambda pd, df: pd.TimedeltaIndex(["1s", "-2D3h", "1.5ms"]).total_seconds(),
    in_process=True,
    note="every label as a float number of seconds. " + SPANS_IN_PROCESS,
)
case(
    "temporal/dt-components",
    "Series.dt",
    frames=RANGE,
    expr=lambda pd, df: pd.to_timedelta(pd.Series(["1s", "-2D3h1ns", "3us"])).dt.components,
    in_process=True,
    note="every span cut into days down to nanoseconds, one int64 column each. "
    + SPANS_IN_PROCESS,
)
case(
    "temporal/dt-microseconds-gap",
    "Series.dt",
    frames=RANGE,
    expr=lambda pd, df: pd.to_timedelta(pd.Series(["1.5ms", None, "-1us"])).dt.microseconds,
    in_process=True,
    note="the microseconds past the whole seconds, float64 because of the gap. "
    + SPANS_IN_PROCESS,
)
RESAMPLE = (
    "In process because the driver has no entry for it, and firepanda's resample is its "
    "Python layer's group by on the bin number of every timestamp, reindexed onto every "
    "bin from the first to the last"
)
case(
    "temporal/resample-sum",
    "DataFrame.resample",
    level="L3",
    covers=("rule",),
    frames=RANGE,
    expr=lambda pd, df: df.set_index("second")["row"].resample("D").sum(),
    in_process=True,
    note="six centuries of days, most of them empty, and an empty day sums to zero. " + RESAMPLE,
)
case(
    "temporal/resample-mean",
    "Resampler.mean",
    frames=RANGE,
    expr=lambda pd, df: df.set_index("second")["row"].resample("6h").mean(),
    in_process=True,
    note="an empty bucket is NaN, which widens the int column to float64. " + RESAMPLE,
)
case(
    "temporal/resample-count",
    "Resampler.count",
    frames=RANGE,
    expr=lambda pd, df: df.set_index("second")["row"].resample("D").count(),
    in_process=True,
    note="an empty bucket produces a row with a zero in it rather than no row, which "
    "is the difference between resampling and grouping by a truncated timestamp. " + RESAMPLE,
)
case(
    "temporal/resample-ohlc",
    "Resampler.ohlc",
    frames=RANGE,
    expr=lambda pd, df: df.set_index("second")["row"].resample("D").ohlc(),
    in_process=True,
    note="four columns, each NaN in an empty bucket and so float64. " + RESAMPLE,
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
    note="In process because the driver has no entry for it, and firepanda reads a key that "
    "is not a column name in its Python layer",
    in_process=True,
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


def _instants(pd, df):
    """The second resolution column as an index of instants."""
    return pd.DatetimeIndex(df["s"])


# The same ten questions the indexing section asks of `Index`, asked again of
# `DatetimeIndex`, because the board scores a name rather than a method and
# `DatetimeIndex.isna` is a different name from `Index.isna` even though one class
# inherits the other. The flag on each is what the indexing section's note says: these
# are firepanda's Python layer on top of one door between an index and a column, the
# core has no call for any of them, and a driver entry would have to write the method
# it is scoring. See spec 36.
AS_A_COLUMN = (
    "the index members that read as a column are firepanda's Python layer on top of "
    "one door, and the core has no call for any of them, so a driver entry would have "
    "to build the column and then write the method itself. See spec 36"
)
case(
    "temporal/index-to-series",
    "DatetimeIndex.to_series",
    frames=RESOLUTIONS,
    expr=lambda pd, df: _instants(pd, df).to_series(),
    in_process=True,
    note=AS_A_COLUMN,
)
case(
    "temporal/index-to-series-given-both",
    "DatetimeIndex.to_series",
    level="L3",
    covers=("index", "name"),
    frames=RESOLUTIONS,
    expr=lambda pd, df: _instants(pd, df).to_series(index=list(range(len(df))), name="when"),
    in_process=True,
    note="the labels come back twice unless the caller says otherwise. " + AS_A_COLUMN,
)
case(
    "temporal/index-isna",
    "DatetimeIndex.isna",
    frames=RESOLUTIONS,
    expr=lambda pd, df: list(_instants(pd, df).isna()),
    in_process=True,
    note=AS_A_COLUMN,
)
case(
    "temporal/index-isnull",
    "DatetimeIndex.isnull",
    frames=RESOLUTIONS,
    expr=lambda pd, df: list(_instants(pd, df).isnull()),
    in_process=True,
    note="the older spelling of isna. " + AS_A_COLUMN,
)
case(
    "temporal/index-notna",
    "DatetimeIndex.notna",
    frames=RESOLUTIONS,
    expr=lambda pd, df: list(_instants(pd, df).notna()),
    in_process=True,
    note=AS_A_COLUMN,
)
case(
    "temporal/index-notnull",
    "DatetimeIndex.notnull",
    frames=RESOLUTIONS,
    expr=lambda pd, df: list(_instants(pd, df).notnull()),
    in_process=True,
    note="the older spelling of notna. " + AS_A_COLUMN,
)
case(
    "temporal/index-dropna",
    "DatetimeIndex.dropna",
    level="L3",
    covers=("how",),
    frames=RESOLUTIONS,
    expr=lambda pd, df: _instants(pd, df).dropna(how="any"),
    in_process=True,
    note="what comes back is still an index of instants, which is the one member of "
    "this family that says so by wrapping the answer in the class it was given rather "
    "than in a plain index. " + AS_A_COLUMN,
)
case(
    "temporal/index-nunique",
    "DatetimeIndex.nunique",
    level="L3",
    covers=("dropna",),
    frames=RESOLUTIONS,
    expr=lambda pd, df: _instants(pd, df).nunique(dropna=False),
    in_process=True,
    note=AS_A_COLUMN,
)


# ---------------------------------------------------------------------------
# The index of instants read as a frame
# ---------------------------------------------------------------------------

AS_A_FRAME = (
    "the members that read as a frame are firepanda's Python layer on top of two doors "
    "end to end, the labels into a column and the column into a frame, and the core has "
    "no call for any of them, so a driver entry would have to build both and then write "
    "the method itself. See spec 36"
)

case(
    "temporal/index-to-frame",
    "DatetimeIndex.to_frame",
    frames=RESOLUTIONS,
    expr=lambda pd, df: _instants(pd, df).to_frame(),
    in_process=True,
    note=AS_A_FRAME,
)
case(
    "temporal/index-to-frame-given-both",
    "DatetimeIndex.to_frame",
    level="L3",
    covers=("index", "name"),
    frames=RESOLUTIONS,
    expr=lambda pd, df: _instants(pd, df).to_frame(index=False, name="labels"),
    in_process=True,
    note=AS_A_FRAME,
)
case(
    "temporal/index-duplicated",
    "DatetimeIndex.duplicated",
    frames=RESOLUTIONS,
    expr=lambda pd, df: list(_instants(pd, df).duplicated()),
    in_process=True,
    note=AS_A_FRAME,
)
case(
    "temporal/index-drop-duplicates",
    "DatetimeIndex.drop_duplicates",
    level="L3",
    covers=("keep",),
    frames=RESOLUTIONS,
    expr=lambda pd, df: _instants(pd, df).drop_duplicates(keep="last"),
    in_process=True,
    note="what comes back is still an index of instants, the same way dropna says so. "
    + AS_A_FRAME,
)

# ---------------------------------------------------------------------------
# The index of instants put in order
# ---------------------------------------------------------------------------

IN_ORDER = (
    "an index in the core has no sort of its own and firepanda's is the column's sort "
    "with a door on each end, so a driver entry would have to write the method it is "
    "scoring. See spec 43"
)

case(
    "temporal/index-sort-values",
    "DatetimeIndex.sort_values",
    frames=RESOLUTIONS,
    expr=lambda pd, df: _instants(pd, df).sort_values(),
    in_process=True,
    note=IN_ORDER,
)
case(
    "temporal/index-sort-values-descending",
    "DatetimeIndex.sort_values",
    level="L3",
    covers=("ascending",),
    frames=RESOLUTIONS,
    expr=lambda pd, df: _instants(pd, df).sort_values(ascending=False),
    in_process=True,
    note="what comes back is still an index of instants, the same way drop_duplicates "
    "says so. " + IN_ORDER,
)
case(
    "temporal/index-argsort",
    "DatetimeIndex.argsort",
    frames=RESOLUTIONS,
    expr=lambda pd, df: list(_instants(pd, df).argsort()),
    in_process=True,
    note=IN_ORDER,
)

A_COPY = (
    "the renaming firepanda copies an index with is a Python layer call, so a driver "
    "entry would have to write the method it is scoring. See spec 44"
)

case(
    "temporal/index-copy",
    "DatetimeIndex.copy",
    frames=RESOLUTIONS,
    expr=lambda pd, df: _instants(pd, df).copy(),
    in_process=True,
    note="what comes back is still an index of instants, which is worth a case of its "
    "own because Index.copy used to answer a plain index here and the calendar members "
    "stopped resolving on it. " + A_COPY,
)
case(
    "temporal/index-copy-named",
    "DatetimeIndex.copy",
    level="L3",
    covers=("name",),
    frames=RESOLUTIONS,
    expr=lambda pd, df: _instants(pd, df).copy(name="when").name,
    in_process=True,
    note=A_COPY,
)
case(
    "temporal/index-copy-deep",
    "DatetimeIndex.copy",
    level="L3",
    covers=("deep",),
    frames=RESOLUTIONS,
    expr=lambda pd, df: _instants(pd, df).copy(deep=True),
    in_process=True,
    note=A_COPY,
)


def _renamed_in_place(index, wanted):
    """Renames a level in place and answers what came back along with the name left behind.

    Returns:
        A two item list of the return value, which is None in both libraries,
        and the name the index was left holding afterwards.
    """
    answered = index.rename(wanted, inplace=True)
    return [answered, index.name]


case(
    "temporal/index-rename-inplace",
    "DatetimeIndex.rename",
    level="L3",
    covers=("inplace",),
    frames=RESOLUTIONS,
    expr=lambda pd, df: _renamed_in_place(_instants(pd, df).copy(), "when"),
    in_process=True,
    note="the same call the flat index takes in indexing/index-rename-inplace, asked "
    "of an index of instants so that the calendar is carried through a rename rather "
    "than quietly turning back into a plain index. It is not in the inplace divergence "
    "block because renaming a level is the one inplace parameter firepanda honours. " + A_COPY,
)

case(
    "temporal/index-names",
    "DatetimeIndex.names",
    frames=RESOLUTIONS,
    expr=lambda pd, df: list(_instants(pd, df).rename("when").names),
    in_process=True,
    note="the same property indexing/index-names reads, asked of an index of instants, "
    "because the board scores a name and DatetimeIndex.names is a different name from "
    "Index.names even though one class inherits the other. The rename first is so that "
    "there is a name to read rather than the empty level both libraries start with. " + AS_A_COLUMN,
)
case(
    "temporal/index-set-names",
    "DatetimeIndex.set_names",
    frames=RESOLUTIONS,
    expr=lambda pd, df: _instants(pd, df).set_names("when"),
    in_process=True,
    note="the returning form, and what it has to answer is still an index of instants, "
    "which is the same thing temporal/index-copy watches for. The inplace form is in "
    "the divergence block. " + AS_A_COLUMN,
)


def _set_names_in_place(index, wanted):
    """Sets a level name in place and answers what came back along with the names left.

    Returns:
        A two item list of the return value, which is None in both libraries,
        and the level names the index was left holding afterwards.
    """
    answered = index.set_names(wanted, inplace=True)
    return [answered, list(index.names)]


case(
    "temporal/index-set-names-inplace",
    "DatetimeIndex.set_names",
    level="L3",
    covers=("inplace",),
    frames=RESOLUTIONS,
    expr=lambda pd, df: _set_names_in_place(_instants(pd, df).copy(), "when"),
    in_process=True,
    note="the same call the flat index takes in indexing/index-set-names-inplace, asked "
    "of an index of instants. It is not in the inplace divergence block because naming "
    "a level is one of the four inplace parameters firepanda honours rather than "
    "refusing. " + A_COPY,
)

# ---------------------------------------------------------------------------
# Moments, spans and dates handed out and taken in
# ---------------------------------------------------------------------------

HANDED = (
    "a temporal column is stored as counts, and what it hands out has to be a Timestamp "
    "with its unit and zone, a Timedelta or a date, which is where the count used to leak"
)


def _handed(value):
    """A value handed out, with the type, unit and zone that came with it.

    Returns:
        The value's repr, which names the type and carries the unit and zone.
    """
    return repr(value)


for unit in UNITS:
    case(
        f"temporal/tolist-{unit}",
        "Series.tolist",
        frames=RESOLUTIONS,
        expr=(lambda column: lambda pd, df: [_handed(v) for v in df[column].tolist()])(unit),
        in_process=True,
        note=HANDED,
    )
case(
    "temporal/tolist-zoned",
    "Series.tolist",
    frames=ZONED,
    expr=lambda pd, df: [_handed(v) for v in df["zoned"].tolist()],
    in_process=True,
    note=HANDED + ". The zone has to come out on every value, across the transition",
)
case(
    "temporal/tolist-spans",
    "Series.tolist",
    frames=("temporal_durations",),
    expr=lambda pd, df: [_handed(v) for v in df["value"].tolist()],
    in_process=True,
    note=HANDED,
)
case(
    "temporal/iterate",
    "Series.__iter__",
    frames=RESOLUTIONS,
    expr=lambda pd, df: [_handed(v) for v in df["ms"]],
    in_process=True,
    note=HANDED,
)
case(
    "temporal/items",
    "Series.items",
    frames=RESOLUTIONS,
    expr=lambda pd, df: [(k, _handed(v)) for k, v in df["s"].head(4).items()],
    in_process=True,
    note=HANDED,
)
case(
    "temporal/iloc-cell",
    "Series.iloc",
    frames=RESOLUTIONS + ZONED,
    expr=lambda pd, df: _handed(df["ns" if "ns" in df else "zoned"].iloc[3]),
    in_process=True,
    note=HANDED,
)
case(
    "temporal/iat-cell",
    "DataFrame.iat",
    frames=("temporal_durations",),
    expr=lambda pd, df: _handed(df.iat[5, 1]),
    in_process=True,
    note=HANDED,
)
for name in ("min", "max", "median"):
    case(
        f"temporal/reduce-{name}",
        f"Series.{name}",
        frames=RESOLUTIONS + ZONED + ("temporal_durations",),
        expr=(
            lambda how: (
                lambda pd, df: _handed(
                    getattr(
                        df["us" if "us" in df else ("zoned" if "zoned" in df else "value")], how
                    )()
                )
            )
        )(name),
        in_process=True,
        note=HANDED + ". A reduction that answers a value of the column's own kind hands it "
        "out the same way",
    )
case(
    "temporal/itertuples",
    "DataFrame.itertuples",
    frames=("temporal_durations",),
    expr=lambda pd, df: [tuple(_handed(v) for v in row) for row in df.itertuples()],
    in_process=True,
    note=HANDED,
)
case(
    "temporal/minus-first",
    "Series.sub",
    frames=RESOLUTIONS,
    expr=lambda pd, df: df["ms"] - df["ms"].min(),
    in_process=True,
    note="a Timestamp on the other side of an operator is lined up as a column, so the "
    "span from the first moment is a column of spans at the column's unit",
)
case(
    "temporal/after-a-moment",
    "Series.gt",
    frames=RESOLUTIONS,
    expr=lambda pd, df: df["s"] > df["s"].iloc[5],
    in_process=True,
    note="a comparison with one moment taken out of the column",
)
case(
    "temporal/built-from-datetimes",
    "pandas.Series",
    frames=RANGE,
    expr=lambda pd, df: pd.Series(
        [pd.Timestamp("2024-01-01 10:00"), pd.Timestamp("2024-06-01 00:00:00.5")], name="when"
    ),
    in_process=True,
    note="a list of moments is a column of moments at the finest unit any of them is "
    "quoted at, which pandas infers rather than defaulting to nanoseconds",
)
case(
    "temporal/built-from-spans",
    "pandas.Series",
    frames=RANGE,
    expr=lambda pd, df: pd.Series([pd.Timedelta(seconds=90), pd.Timedelta(days=2)]),
    in_process=True,
    note="a list of spans is a column of spans",
)
case(
    "temporal/built-zoned",
    "pandas.DataFrame",
    frames=RANGE,
    expr=lambda pd, df: pd.DataFrame(
        {"when": [pd.Timestamp("2024-01-01", tz="Asia/Tokyo")] * 2, "n": [1, 2]}
    ),
    in_process=True,
    note="a zone every value shares is kept on the column",
)
case(
    "temporal/zoned-repr",
    "Series.__repr__",
    frames=ZONED,
    expr=lambda pd, df: [
        repr(df["zoned"].head(3)),
        repr(pd.DataFrame({"when": df["zoned"].head(3)})),
    ],
    in_process=True,
    note="a zoned column prints each instant with its offset, not the count behind it",
)
case(
    "temporal/zoned-index-repr",
    "Index.__repr__",
    frames=ZONED,
    expr=lambda pd, df: [
        repr(df.set_index("zoned").index[:3]),
        repr(pd.DatetimeIndex(["2024-01-02", None, "2024-01-04"])),
        repr(pd.DatetimeIndex(["2024-01-02 10:00:00.123", "2024-01-03"], name="when")),
    ],
    in_process=True,
    note="a date index prints each instant, with its offset when it has a zone, and a day "
    "at midnight as a date",
)
case(
    "temporal/parts-with-gap",
    "Series.dt",
    frames=RANGE,
    expr=lambda pd, df: [
        [str(part.dtype), part.tolist()]
        for part in (
            getattr(df["second"].head(4).mask([False, True, False, False]).dt, name)
            for name in ("year", "month", "hour", "dayofweek", "quarter")
        )
    ],
    in_process=True,
    note="a whole number read from a column with a gap is a float with NaN in the gap. It "
    "is read out with `tolist`, since a float column here keeps its gap as a null where "
    "pandas keeps a NaN, and that is `divergences/missing-spelling` rather than this",
)
case(
    "temporal/flags-with-gap",
    "Series.dt",
    frames=RANGE,
    expr=lambda pd, df: pd.DataFrame(
        {
            name: getattr(df["second"].head(4).mask([False, True, False, False]).dt, name)
            for name in ("is_month_start", "is_month_end", "is_leap_year")
        }
    ),
    in_process=True,
    note="a yes or no field reads a gap as no, since pandas answers a numpy bool array",
)
case(
    "temporal/concat-units",
    "pandas.concat",
    frames=RANGE,
    expr=lambda pd, df: pd.concat(
        [df["second"].head(2).dt.as_unit("s"), df["second"].tail(2).dt.as_unit("us")],
        ignore_index=True,
    ),
    in_process=True,
    note="instants in two units stack at the finer of them",
)
case(
    "temporal/astype-unit",
    "Series.astype",
    frames=RANGE,
    expr=lambda pd, df: df["second"].head(3).astype("datetime64[ms]"),
    in_process=True,
    note="astype to instants in another unit changes the unit, which is as_unit",
)
case(
    "temporal/to-datetime-mixed",
    "pandas.to_datetime",
    frames=RANGE,
    expr=lambda pd, df: pd.to_datetime([df["second"].iat[0], "2020-05-05", None]),
    in_process=True,
    note="a moment among text is taken as it is and the text is read around it",
)
case(
    "temporal/object-values",
    "DataFrame.to_numpy",
    frames=RANGE,
    expr=lambda pd, df: [
        repr(value)
        for row in pd.DataFrame(
            {"a": df["second"].head(3).mask([False, True, False]), "s": ["x", "y", "z"]}
        )
        .to_numpy()
        .tolist()
        for value in row
    ],
    in_process=True,
    note="an instant in an object array is a Timestamp and a gap is NaT",
)
def _typed_values(result):
    """A column or an index as its type and its values written out, so a gap reads as NaT."""
    return [str(result.dtype), [str(value) for value in result.tolist()]]


def _index_names(labels):
    """The text and the times read from labels, as plain lists."""
    return [
        labels.day_name().tolist(),
        labels.month_name().tolist(),
        labels.strftime("%Y-%m").tolist(),
        [str(value) for value in labels.time],
    ]


case(
    "temporal/to-datetime-count",
    "pandas.to_datetime",
    frames=RANGE,
    expr=lambda pd, df: pd.to_datetime([df["second"].iat[0], 5, float("nan"), "2020-05-05"]),
    in_process=True,
    note="a number among moments is nanoseconds since 1970, and the column is held in nanoseconds",
)
case(
    "temporal/to-timedelta-mixed",
    "pandas.to_timedelta",
    frames=RANGE,
    expr=lambda pd, df: _typed_values(
        pd.to_timedelta([pd.Timedelta("1s").as_unit("s"), "2h", None])
    ),
    in_process=True,
    note="text beside spans is read at microseconds, the finer of that and the spans' unit",
)
case(
    "temporal/index-names-with-gap",
    "DatetimeIndex.day_name",
    frames=RANGE,
    expr=lambda pd, df: _index_names(
        pd.DatetimeIndex(df["second"].head(3).mask([False, True, False]))
    ),
    in_process=True,
    note="text read from labels with a gap keeps the gap as NaN, and the time of a gap is NaT",
)
