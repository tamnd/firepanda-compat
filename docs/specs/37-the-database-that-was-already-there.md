# The database that was already there

## Where the zone work stopped

[19](19-the-zone-that-needs-no-database.md) split the time zone work in two. A zone name is either a number, like `UTC` or `+05:30`, or a rule, like `America/New_York`, and everything on the number side has worked since then. Everything on the rule side was refused with a sentence saying it needed a time zone database, and the refusal covered more than it sounds like. It covered every calendar field read off a zoned column, `strftime`, `day_name`, `month_name`, `date`, `normalize`, `floor`, `ceil`, `round`, `tz_localize` and `tz_localize(None)`, on any zone that has ever had daylight saving time, which is most of the zones anybody has a column in.

Two cases on the board were waiting on it by name, `errors/nonexistent-time` and `errors/ambiguous-time`, and firepanda issue #349 carried the plan on the record, which was a reader for the TZif files under `/usr/share/zoneinfo`. This document is what that reader turned out to need, and the three places where it was less obvious than it looked.

## Why read the files and not bundle the data

The IANA database changes several times a year, when some government moves its clocks with a few weeks' notice. A library that bundles the data is wrong about those places from the day of the change until somebody releases it again and somebody else upgrades. The files under `/usr/share/zoneinfo` are updated by the operating system, which is the thing that already has to be right about the local clock, and they are what Python's `zoneinfo` reads first, which is what pandas answers from. Reading the same files is the only way to get the same answer on the same machine, and it costs no dependency.

The format is RFC 8536. A file is a header, a table of transitions with 32 bit times, the same table again with 64 bit times in version 2 and later, and a footer that is a POSIX TZ string such as `EST5EDT,M3.2.0,M11.1.0`. The second table is the one read when there is one, and leap second records are skipped, since pandas counts no leap seconds.

## The table runs out

The first thing that was less obvious is that the table is not the whole answer. On the machines this was tested on, `America/New_York` has transitions up to 2037, and a distribution that builds the database in the slim form RFC 8536 allows stops the table at the last change of rule, 2007 for New York, and leaves the rest to the footer. Either way, every instant after the last transition is answered by the footer rule, so the footer cannot be treated as optional, and a timestamp in seconds can be tens of thousands of years out.

The reader writes the footer rule out year by year for four hundred years after the table, and folds any later instant back into that window by whole four hundred year cycles. Four hundred is not a tuning parameter. The Gregorian calendar repeats exactly every 146097 days, which is a whole number of weeks, so "the second Sunday in March" falls on the same day of the cycle every time round and the fold is exact. The table stays a few kilobytes and a lookup is one bisection. The test that holds this is a reading in July 2400, which is past both the file's table and the written out part.

## Before the table starts

The second one is the other end. Before the first transition there is no transition to say which offset applies. RFC 8536 says the first time type, and Python's `zoneinfo` says the first standard time type, falling back to the first type of any kind, and those differ for some zones. pandas answers from `zoneinfo`, so the reader takes the `zoneinfo` choice. For New York that is local mean time, four hours fifty six minutes and two seconds behind Greenwich, and a reading in 1800 is the test that holds it.

## A reading that is no instant, or two

The third is `tz_localize`, which goes the hard way round. Reading a zoned column is one lookup per row: the stored integer is UTC, the table is indexed by UTC, and the answer is unique. Localising starts from a wall clock reading, and the table is not indexed by those. On the morning a zone puts its clocks forward an hour of readings never happens, and on the evening it puts them back an hour happens twice.

The test used is the one pandas uses. The offsets in force a day either side of the reading are the only two candidates, since no zone has changed its clock twice in two days, and each candidate is kept when the zone really was on that offset at the instant it gives. An ordinary reading keeps one candidate, a skipped reading keeps none, and a repeated reading keeps two that differ. pandas with its defaults raises a `ValueError` for the last two, and the sentences were measured on pandas 3.0.3 rather than remembered:

```
2024-03-10 02:30:00 is a nonexistent time due to daylight savings time. Try using the 'nonexistent' argument.
Cannot infer dst time from 2024-11-03 01:30:00, try using the 'ambiguous' argument
```

Both name a row, and pandas names the first one. The kernel runs in parallel morsels, so the first morsel to hit a bad row is not necessarily the one holding the first bad row. The fix costs nothing on the path that succeeds: when any morsel fails, the rows are walked again in order and the first bad one raises. That second walk only happens on the way to an error. A test with two hundred thousand good rows ahead of the bad ones holds this, because at that size the rows span several morsels.

The same test serves `floor`, `ceil`, `round` and `normalize` on a zoned column, since each of those reads the local clock, rounds the reading, and has to put the result back on the clock, and a rounded reading can land in a gap or a fold like any other. pandas raises the same two errors there, so that falls out for free.

## A name the database does not hold

`tz_convert` never needed the database, because converting keeps the instant and changes only the name, and [19](19-the-zone-that-needs-no-database.md) made the point that this meant it accepted any name at all. pandas does not. It raises `ZoneInfoNotFoundError`, a `KeyError`, saying `No time zone found with key Nowhere/Land`. firepanda now checks a rule name against the database at the point it is attached, and raises with the same sentence, as a `ValueError`. That is the one difference here, and it is deliberate, because the error classes firepanda raises are fixed by its own table and a zone name is an argument value rather than a key. A name that would climb out of the database directory, such as one with `..` in it, is refused before any file is opened.

## What is still refused

`ambiguous` and `nonexistent` away from `"raise"` are still refused by name. Every one of them is now a few lines in the same kernel, since `"NaT"` is a null where the error is raised, `"shift_forward"` is the first instant after the gap, and a boolean array is a choice between the two candidates the kernel already has. They are the next slice rather than this one, so that this one could be checked against the board on its own. A zoned literal in a SQL comparison against a rule zone is also still refused, since it goes through a different entry point, and so is a `Timestamp` scalar on a rule zone, since the scalars are pure Python and that is a question about using Python's own `zoneinfo`.
