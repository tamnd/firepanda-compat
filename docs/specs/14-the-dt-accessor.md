# The dt accessor

A datetime column exists in firepanda now and nothing can ask it anything. The reader opens a timestamp at any unit with or without a zone, a date and a duration, the writer writes all three back, and the `temporal` section of this suite has no failures left in it. It also has no passes. All 122 of its runs report `unimplemented`, because a column that can be read and not asked is a column with no methods on it.

This document says what those methods are, how many runs each group of them is worth, and where the line falls between this work and M7. It follows `13-the-type-gap.md`, which scoped the column types and closed four of its five items. This is the fifth.

## The measurement

From the run that produced `results/firepanda.json` on 2026-09-07, against pandas 3.0.3 and firepanda at `0.6.51`. The `temporal` section is 122 runs over 80 distinct cases, and every one of them is `unimplemented`.

The split that matters is not by name, it is by which milestone owns it.

| group | cases | runs |
| --- | --- | --- |
| the `dt` accessor and what holds it up | 68 | 108 |
| resampling, `asfreq`, DST arithmetic and the ambiguous time policy | 12 | 14 |

108 of 122 runs are M6 work and 14 are M7. That is the whole temporal section, and the 14 are `resample-sum`, `resample-mean`, `resample-count`, `resample-ohlc`, `asfreq`, `dst-difference`, the three `tz-localize-ambiguous-*` cases, the two `tz-localize-nonexistent-*` cases and `tz-localize-lord-howe`. Everything else is a field read off a timestamp, a rounding, a name, a format string, a zone conversion or a constructor.

## The shape of it

`dt` is 42 names in pandas 3.0.3 and 13 callables. The other 29 are properties, and 19 parameters hang off the 13.

The 29 properties: `date`, `day`, `day_of_week`, `day_of_year`, `dayofweek`, `dayofyear`, `days_in_month`, `daysinmonth`, `freq`, `hour`, `is_leap_year`, `is_month_end`, `is_month_start`, `is_quarter_end`, `is_quarter_start`, `is_year_end`, `is_year_start`, `microsecond`, `minute`, `month`, `nanosecond`, `quarter`, `second`, `time`, `timetz`, `tz`, `unit`, `weekday`, `year`.

The 13 callables: `as_unit`, `ceil`, `day_name`, `floor`, `isocalendar`, `month_name`, `normalize`, `round`, `strftime`, `to_period`, `to_pydatetime`, `tz_convert`, `tz_localize`.

Four of those 42 are pairs. `day_of_week`, `dayofweek` and `weekday` are the same number under three names, and `days_in_month` and `daysinmonth` are the same number under two. A user's code has whichever one they typed, so all six exist and three of them are aliases at the binding layer and nothing deeper.

The scoreboard reads this badly and `07-scoreboard.md` already says why. The L3 rate over `dt` has 13 in its denominator because a property is not a callable, so an accessor with every property working and no method working scores zero, and one with `strftime` alone scores 7.7 percent. The runs are the honest number here and there are 108 of them.

## What holds them up

A timestamp column is an int64 count from the Unix epoch at one of four units, which is what Arrow stores and what pandas stores. Every property in the first group is a civil calendar computed from that count, and the calendar is the work. `dt.year` on a nanosecond column is not a field read, it is a division down to days followed by a conversion from a day number to a year, a month and a day.

That conversion should be Howard Hinnant's `civil_from_days`, which is branch free integer arithmetic over a four hundred year cycle with no table and no loop, and it should run as a kernel over the whole column rather than per row through a scalar path. `dt.year`, `dt.month`, `dt.day`, `dt.quarter`, `dt.dayofweek`, `dt.dayofyear`, `dt.days_in_month`, `dt.is_leap_year` and the six `is_*_start` and `is_*_end` properties are all the same conversion with a different field taken out of it, so the conversion is written once and the fifteen names are cheap after it.

The clock half is easier and is not free. `dt.hour`, `dt.minute`, `dt.second`, `dt.microsecond` and `dt.nanosecond` are a remainder against the day, and the remainder depends on the unit, so a second column has no nanosecond field to answer with and answers zero rather than raising. The corpus tests exactly that with `nanosecond-s`, `nanosecond-ms`, `nanosecond-us` and `nanosecond-ns` side by side, which is why `temporal_resolutions` exists and why the reader was written not to normalize.

Negative counts are where implementations go wrong and the corpus has them. A timestamp before 1970 is a negative int64, and the last second of 1969 lands in a different day from the first second of 1970 only if the division down to days floors rather than truncating toward zero. Mojo's `//` floors, which is the right default and is not the end of it, because the fast paths a kernel reaches for do not: a hardware divide truncates, and an arithmetic shift floors only for a power of two divisor, which 86400 is not. So this is correct today and stays correct only if a test fails when it stops flooring, and that is the first test to write.

## The four groups

**The field extractors.** The 15 calendar properties and the 5 clock properties, plus `dt.date`, `dt.time` and `dt.normalize`, which are the same arithmetic returning a column rather than a number. This is the largest group by runs and the smallest by decisions, and it is where the work starts because everything else can be written against it.

**The rounding trio.** `dt.floor`, `dt.ceil` and `dt.round`, each taking a frequency string. Rounding at a fixed frequency is arithmetic on the count and needs no calendar at all, so `floor('h')` on a nanosecond column is one division and one multiplication. The frequency string parser is shared with M7 and should be written here and used there, and only the fixed frequencies belong here: an hour, a minute, a second and their multiples. Anything month based is an offset and offsets are M7. `dt.round` uses banker's rounding on a half, which is the one behaviour in the group that a reimplementation gets wrong by default.

**The names and the formats.** `dt.day_name`, `dt.month_name`, `dt.strftime` and `dt.isocalendar`. The first two take a locale argument that pandas honours and this suite only exercises in English, so the honest scope is English names and a refusal by name for any other locale rather than silently ignoring the argument. `strftime` is a format string interpreter and is the one item in this document that is a parser rather than arithmetic. `isocalendar` returns a frame of three columns and is the only member of `dt` that does.

**The zone pair.** `dt.tz`, `dt.tz_localize` and `dt.tz_convert`. A conversion between two fixed offset zones is arithmetic on the count. A conversion that crosses a DST boundary is a lookup in the zone database, and the corpus separates the two deliberately: `tz-convert-utc`, `tz-convert-hour` and `tz-convert-half-hour` are in scope and the three ambiguous cases, the two nonexistent cases and Lord Howe are M7. `tz_localize(None)` is in scope and is one of the four, since dropping a zone is not a conversion.

## The three names outside the accessor

`pandas.to_datetime`, which the corpus asks for twice: `to-datetime-strings` parses ISO 8601 and `to-datetime-format` takes an explicit `format`. Parsing is the CSV date parser this library already has, reached from a different door.

`pandas.date_range` in its plain form, which is a start, an end or a period count, and a fixed frequency. The offset frequencies are M7 with the rest of `offsets`, and `date-range` and `date-range-tz` are the two cases here.

`Series.dtype` reporting a datetime dtype. The corpus has eight cases on this, `dtype-s` through `dtype-ns` and `unit-s` through `unit-ns`, and they are the cheapest eight runs in the section: the type already carries the unit and the zone, and this is the string it prints. pandas spells it `datetime64[ns]` and `datetime64[ns, Europe/Paris]`, which is what `LogicalType.write_to` already produces.

## The duration column comes with it

Ten of the 108 runs are the duration column rather than the timestamp: `duration-dtype`, `duration-days`, `duration-sum`, `duration-mean`, `duration-abs`, `total-seconds`, `timedelta-construct`, `to-timedelta`, `timestamp-minus-timestamp` and `timestamp-plus-duration`. They are in this workstream and not in one of their own, because subtracting two timestamps produces a duration and a user reaching for `dt` on a timestamp column reaches for `dt.total_seconds` on the result of it in the next line. `dt` over a duration column is a different set of names from the 42 above, since `Series.dt` dispatches on the column type, and the four the corpus asks for are `days`, `total_seconds`, `abs` and the reductions. The arithmetic is the easy half and the unit rule is the whole of it: a nanosecond timestamp minus a second timestamp is a nanosecond duration, and a reduction over a duration column is a duration and not an integer.

## The driver is the other half

Every one of these 108 runs needs two things and only one of them is in the library. The `nested` section is the demonstration: list and struct columns read in firepanda #282 and all 36 of its runs moved from `fail` to `unimplemented` reporting `Absent: the firepanda driver has no entry for nested/list-len`, which is a message about `drivers/firepanda/main.mojo` and not about the library.

So the unit of work here is a pair, and a pull request in the library that does not have a driver entry behind it moves nothing on the board. The driver entry should land in the same session as the library change, which is what `03-harness.md` means by the driver being a first class part of the suite rather than a script that grew.

## Exit criteria

- [ ] All 29 `dt` properties and all 13 `dt` callables exist, with the three `weekday` spellings and the two `daysinmonth` spellings all present
- [ ] Every field is correct for a timestamp before 1970, tested against a count that floors rather than truncates
- [ ] Every field is correct at all four units, and a field the unit cannot express answers zero rather than raising
- [ ] `pandas.to_datetime`, `pandas.date_range` in its fixed frequency form, and `Series.dtype` on a datetime column
- [ ] The driver has an entry for each of the 68 cases, landed with the library change rather than after it
- [ ] The `temporal` section reports passes rather than `unimplemented` for the 68 in scope cases, measured rather than predicted, with the run posted, and any that do not pass are a stated reason in the divergence registry or an issue rather than a method that is not there
- [ ] The 14 M7 runs still report `unimplemented` and not a wrong answer

## What stays out

`offsets`, `Resampler`, `resample`, `asfreq`, `merge_asof`, time based rolling and the ambiguous and nonexistent local time policy. All of it is M7 and all of it assumes the column and the accessor this document describes.

`DatetimeIndex` and its 91 callables go with the index work in firepanda #154 and #155, because a datetime index is an index before it is a datetime, and the accessor on a column is not the same surface as the methods on an index.

`Period` and `Interval` are their own column types and neither is in the corpus today. `dt.to_period` is in the 42 and it is the one member of the accessor that cannot be finished here, so it raises by name and points at the issue that adds the type.
