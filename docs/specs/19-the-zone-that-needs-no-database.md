# The zone that needs no database

## A name is a rule or a number

The temporal section of the board had thirty seven absent cases and the largest single group in it was the time zone one. That group looked like one piece of work sitting behind one dependency, which is the IANA time zone database. `America/New_York` is four hours ahead of UTC in July and five in January, was on a different pair of dates before 2007, and will be on another pair whenever somebody legislates, and none of that is derivable. It is a table, the table is a file, and a library that builds with a Mojo toolchain and nothing else does not have the file yet.

So the group looked blocked. It was not, and finding that out took one measurement rather than one argument.

The measurement is that a zone name is one of two different kinds of thing wearing the same syntax. `America/New_York` is a rule. It names a jurisdiction and a history of decisions that jurisdiction made, and the only way to turn it into a number is to look the number up for the instant you are asking about. `UTC` is not a rule. Neither is `+05:30`. Those are numbers that have been spelled as names. They state the whole answer in themselves, every instant in such a column is read against the same offset for as long as the column exists, and no database anywhere can tell you a single thing about them the name does not already say.

That split is the entire slice. Everything on the number side works today and everything on the rule side is refused with a sentence saying which database it needs, and because the split is a property of the name rather than a list of zones, nothing has to be kept up to date when the tzdb next changes.

## The operation that moves nothing

pandas has two operations here that sound like the same operation, and telling them apart was the second measurement.

A zoned column is UTC instants plus a name. The integers are the moments; the name says which clock a person reads them against. `tz_convert` changes which clock, and therefore changes nothing at all about the moments. This is checkable rather than arguable, so it was checked:

```python
converted = s.dt.tz_convert("Asia/Kolkata")
(converted.astype("int64") == s.astype("int64")).all()  # True
```

Every integer is where it was. Only the dtype moved, from `datetime64[us, America/New_York]` to `datetime64[us, Asia/Kolkata]`. Which means `tz_convert` never asks what the offset is, which means it needs no database, which means it works for every zone there is including the rule ones. The largest single operation in the blocked group turns out not to be in the blocked group.

`tz_localize` is the other one, and it is the opposite one. It takes a column of readings that are on no clock, says which clock they were read off, and therefore has to move every instant, because nine in the morning is a different moment in each place somebody might have said it. That one needs the offset and that one is where the rule zones stop.

There is a third, `tz_localize(None)`, which takes the clock off and keeps the reading. It is the exact reverse of what converting does even though both are spelled as a zone going away, and it needs the offset for the same reason localising does. Measured, a column holding 06:00 UTC labelled `America/New_York` becomes a naive column holding 01:00.

pandas refuses the two wrong-direction calls in so many words, and both messages were measured rather than guessed:

- `naive.dt.tz_convert(...)` gives `TypeError: Cannot convert tz-naive timestamps, use tz_localize to localize`
- `zoned.dt.tz_localize(...)` gives `TypeError: Already tz-aware, use tz_convert to convert.`

Those two sentences are the API telling its users the same thing this section is telling its implementers, which is a good sign that the distinction is real and not an artefact of how firepanda happens to store a column.

## The spellings that are numbers

`TimeZone.fixed_offset` reads a name and answers either the offset in seconds or nothing. The forms it takes are `UTC` in any case, the Arrow form `+HH:MM` with the colon optional and the minutes optional after that, and the `UTC+HH:MM` form, which matters because it is the form pandas prints back when it built the zone from an offset itself. Measured: `pd.to_datetime(...).dt.tz_localize('+05:30')` has dtype `datetime64[s, UTC+05:30]` and `str(tz)` of `UTC+05:30`, while `'utc'` normalises to `UTC`.

`Etc/GMT+5` is deliberately not one of the forms. It looks like a number, it parses like a number, and its sign runs backwards from every other spelling in that list, because the POSIX convention it comes from counts west as positive. It is an IANA name and it goes to the database with the rest of them. Reading it as a number would put a column five hours out in the direction where nothing crashes and no test fails until somebody notices their morning rows are in the afternoon. There is a test asserting it is refused, which is a strange thing to assert until you know why.

The parser also refuses `+5:30` and `+05:99`, because a spelling that is nearly right is the one worth being strict about.

## One turn, at the top

Every calendar kernel underneath reads its integers as a wall clock reading. That is correct for a naive column and wrong for a zoned one by exactly the offset. There were two ways to fix that: thread a shift argument down through the five kernels and have each of them remember, or turn the column into its local readings once at the entry point and let the kernels stay as they are.

The second one, for three reasons. A naive column pays nothing, because the turn is skipped entirely rather than being a shift of zero through five loops. A zoned column pays a single extra pass. And no kernel below has to hold the fact that its integers might be UTC, which is the kind of fact that gets forgotten by the sixth kernel somebody adds.

`dt.date`, `dt.day_name`, `dt.month_name`, `dt.strftime` and the calendar fields take the turn and stop there, because a date and a name are not on a clock any more. `dt.normalize` and the rounding trio make the round trip, going down to local readings, doing the work and coming back up to instants on the same clock, because local midnight is a moment and has to be labelled with the clock it was a midnight on. That round trip is exact when the offset never moves, which is another way of saying it would not be exact for a rule zone, which is the argument for refusing those rather than approximating them.

The three old scattered refusals are now one `_refuse_zone`, and no public entry point reaches it any more. It stays anyway. The bug it catches is not a crash, it is an hour that is quietly seven off, and that is worth a backstop for whoever writes the next entry point.

## The thirteen runs

`temporal/tz` reads the name back, and pandas answers the IANA string, so `str(s.dt.tz)` is `America/New_York`. `temporal/tz-convert-utc` and `temporal/tz-convert-half-hour` convert the three DST frames and assert the instants did not move. `temporal/tz-convert-hour` goes the whole way round, converting to UTC and reading the hour off, which is int32 in pandas and is the check that the two halves compose. `temporal/tz-localize` puts UTC on the naive range frame.

Four cases over the three zoned frames plus one over the range frame is thirteen runs, and they take the board from 481 passing to 494 with the failure count unchanged at the baseline. The temporal section goes from 85 passing and 37 absent to 98 and 24.

## What stays out

The seven `tz_localize` cases that pass `nonexistent` or `ambiguous` stay out, and they are the ones that most need the database, because the questions they ask cannot arise without it. A reading in the hour a clock skips forward denotes no instant, a reading in the hour it repeats denotes two, and a zone whose offset never changes has neither hour. So the hard cases and the blocked cases are the same cases, which is tidy but does mean the remaining work is all of one kind.

`tz-localize-none` over the DST frames stays out for the same reason, since those frames are on rule zones by construction.

`temporal/dst-difference` stays out for a different reason entirely. It is `zoned.diff()`, which needs `Series.shift`, which does not exist anywhere in the library yet. Measured, the answer is `timedelta64[us]` with NaT in the first row and ten minutes in every other row including the one that crosses the transition, because the stored instants are evenly spaced and the transition is a fact about the labels rather than about the moments. That is a nice result and it is waiting on a shift kernel rather than on a tzdb.

The next piece of work in this area is a TZif reader over `/usr/share/zoneinfo`. That is a file read at runtime rather than a build dependency, so it does not cost the library its one toolchain, and it would unblock everything listed above at once.

## How it was checked

Every pandas fact above came out of pandas 3.0.3 in the compat environment, not out of documentation. The offsets, the two error messages, the dtype spellings and the int32 of the converted hour were all printed before anything was written.

On the library side: eight new tests in `tests/test_temporal_zones.mojo`, the full suite at 61 files passing, 200,000 fuzz cases at seed 20260909 with five fuzzers passing, 191 files formatted, and a conformance run showing 494 passing with failures at the baseline of 94 and no temporal failures at all. The ratchet says nothing went backwards.
