# The format string and the ISO calendar

`dt.day_name`, `dt.month_name`, `dt.isocalendar` and `dt.strftime` are the four members of the accessor that answer text or answer a frame, and they are the third slice of the group that `14-the-dt-accessor.md` scoped. Three of them look like lookups and the fourth looks like a printf, and in both cases the appearance is misleading. This document is what a running pandas 3.0.3 answered when it was asked directly, on this machine, one directive at a time.

## The ISO calendar is not the calendar

`dt.isocalendar()` answers a frame of three columns named `year`, `week` and `day`, each of them a nullable unsigned 32 bit integer, and the year in it is frequently not the year in the timestamp. The 1st of January 2021 is a Friday, it belongs to the week that started on the Monday of the previous December, and so its ISO year is 2020 and its ISO week is 53. The 31st of December 2019 goes the other way and lands in ISO week 1 of 2020. Anybody who implements this by taking the calendar year and dividing the day of the year by seven gets a column that is right for about eighty percent of rows and wrong every winter, which is the worst possible failure shape because it survives a casual test.

The rule that produces the right answer without a table is the Thursday rule. Every ISO week contains exactly one Thursday, the ISO year of a row is the calendar year of that Thursday, and the ISO week is that Thursday's day of the year divided by seven and rounded up. So the whole thing is one shift of the day number by three minus the ISO weekday, one call to the civil date conversion that was already there for the calendar fields, and one division. There is no week count table, no 52 or 53 rule and no special case for a leap year, which matters because the 52 or 53 rule is where a hand written implementation goes wrong.

The ISO day is the weekday numbered Monday as one through Sunday as seven, which is the one place in pandas where the week starts on Monday and is numbered from one. `dt.dayofweek` on the same row is Monday as zero, and `dt.strftime('%w')` on the same row is Sunday as zero. Three numberings of the same fact, all of them live at once, and none of them is a typo.

## Three week numbers that disagree on purpose

`%U`, `%W` and `%V` are three different week numbers and a format string may contain all three. `%U` counts weeks starting on Sunday and calls the days before the first Sunday week zero. `%W` does the same starting on Monday. `%V` is the ISO week and is never zero. On the 1st of January 2021 those are 00, 00 and 53.

The first two are one expression each once the day of the year and the weekday are in hand. `%U` is the day of the year plus six minus the Sunday first weekday, all divided by seven. `%W` is the same with the Monday zero weekday. Both were checked against every day of a run of years rather than reasoned about, because the off by one in either direction is invisible in the middle of a month.

## The padding flag, which is a measurement and not a principle

A directive may carry a flag between the percent and the letter. `-` means no padding, `_` means space padding and `0` means zero padding. The pandas documentation does not say what happens when the flag is put in front of a directive that has no width to move, and the answer is not uniform, so it was measured across the whole directive set.

The flag changes the answer for exactly thirteen directives, which are `m`, `d`, `e`, `j`, `V`, `U`, `W`, `H`, `k`, `I`, `l`, `M` and `S`. For everything else pandas ignores it. `%-A` is `%A`, `%-Y` is `%Y`, `%-Z` is `%Z`, and `%-%` is `%%`. That includes four directives that do carry digits and still ignore it, which are `Y`, `y`, `C`, `G` and `g`, and the two single digit weekday numbers `u` and `w`. So the rule is not "numeric directives take the flag". The rule is a list of thirteen, and the list is what firepanda holds.

The one exception is `%f`. With a flag in front of it pandas emits the literal letter f and loses the microseconds entirely, so `%-f` on a row with 123456 microseconds prints `f`. That is a bug in something below pandas rather than a decision, and copying it would mean writing a column of the letter f. firepanda refuses the format instead, at parse time, with a message that says what pandas does and why it is not being reproduced.

## The four digit year

`%Y` and `%G` are always zero padded to four digits, on both `time.strftime` and `datetime.strftime` on this machine, so the year 353 prints as `0353`. That is libc doing it rather than pandas, it is not what the C standard requires, and it is not what a from scratch implementation produces. It was found by a cross check that came back with sixty six differing lines all of which were early years, and it is the single most likely difference for anybody reimplementing this to ship without noticing, because no test corpus that starts at 1970 can see it.

The default padding for the rest is worth writing down too. `%e`, `%k` and `%l` are space padded rather than zero padded. `%j` is three digits. `%f` is always six. Everything else in the thirteen is two digits of zero padding.

## What is refused rather than reproduced

Eight directives depend on the machine, the locale or the C library and are refused at parse time. They are `%c`, `%x`, `%X`, `%s`, `%P`, `%Q`, `%E` and `%v`. Each of them either reads the environment or is a vendor extension, which means the same format string answers differently on two machines running the same pandas. A conformance suite cannot hold an answer to those and a user cannot rely on one, so the honest thing is to say no once, before any row is read, rather than to write a column that happens to match on the machine that made it.

The `locale` keyword on `dt.day_name` and `dt.month_name` is refused the same way. Anything other than the default raises, rather than being accepted and quietly answered in English, because an accepted argument that is ignored is worse than a rejected one.

The parse happens once per call and not once per row. A format string with a bad directive in it fails before the first row is touched, which is both faster and the only way to make the refusals above mean anything.

## A pandas limit that firepanda does not share

`Timestamp.strftime` raises `ValueError: year must be in 1..9999` for a row outside that range, even though the column holds the row perfectly well and every other accessor member reads it. A nanosecond column cannot reach outside that range but a second column reaches the year 292 million, and `dt.strftime` on such a column raises in pandas for the offending rows. firepanda renders them. This is a real divergence and it is on the list to write up in `06-divergences.md` rather than to paper over, because the alternative is refusing to print a date that the library was happy to store.

## The text dtypes

`day_name`, `month_name` and `strftime` all answer the pandas `str` dtype, backed by Arrow `large_string`, and a null input row comes back as NaN rather than as an empty string. `isocalendar` answers null in all three of its columns for a null row rather than zero. Those are the comparison rules the suite checks against, and the difference between NaN and the empty string is the sort of thing that passes a length check and fails a real one.

## What this is worth

Four cases and five runs, which are `temporal/day-name`, `temporal/month-name` and `temporal/isocalendar` over the range frame, and `temporal/strftime` over the range frame and the resolutions frame. The temporal section moves from 63 passing runs to 68 and the board moves from 459 to 464.

The parse is worth more than the five runs, because `pandas.to_datetime` takes the same directive set in the other direction and is the next slice of the same group. The thirteen directive list, the refusals and the four digit year are all facts about the same grammar, and reading a format string is written once.

## How it was checked

One thousand two hundred and fifty instants, drawn from a fixed seed and spread over the range each resolution can hold, at all four resolutions plus a date column, through a format string carrying about a hundred directives and flag combinations, plus the day name, the month name and all three ISO columns of every row. That is generated once by pandas and once by firepanda and diffed, and after the four digit year was fixed the diff was empty.

Twenty five tests hold the parts a reader can check by eye, including twelve hand picked rows whose pandas answers are written into the test file as literals rather than computed. The fuzzer runs the vector kernel against a one row at a time twin one case in eight, and the twin computes the ISO fields from the ordinal date formula with an explicit 52 or 53 week count rather than from the Thursday trick, so agreement between the two is evidence about the answer rather than about the shortcut. Two hundred thousand cases at a fixed seed found nothing.

## What stays out

`dt.time` and `dt.timetz`, which need a time of day column type that does not exist yet. The zone aware members. `pandas.to_datetime` and `pandas.date_range`, which are the parser going the other way. Anything on a duration column.
