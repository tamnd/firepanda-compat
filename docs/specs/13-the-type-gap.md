# The type gap

Of the 334 runs failing against firepanda today, 228 never ran an operation. The frame could not be read. Three whole sections of this suite score exactly zero, and they score zero for one reason rather than for many: firepanda has six column types and pandas has eleven.

This document says what the missing five are, corrects the reading of the scoreboard that hid them, and says which of them belong in M6 and which belong in M7.

## The measurement

Per section, from the run that produced `results/firepanda.json` on 2026-09-06, against pandas 3.0.3 and firepanda at `46ddd6a`.

| section | pass | fail | divergent | unimplemented |
| --- | --- | --- | --- | --- |
| basics | 232 | 92 | 1 | 168 |
| categorical | 0 | 61 | 0 | 3 |
| groupby | 111 | 5 | 0 | 57 |
| nested | 0 | 36 | 0 | 0 |
| stats | 52 | 4 | 0 | 51 |
| temporal | 0 | 121 | 0 | 1 |

Three of those rows are a zero. All 121 temporal failures, all 61 categorical failures and 35 of the 36 nested failures are the same sentence in different words, and the sentence is that `read_arrow` refused the file. The thirty sixth is an error type case that also raised at the read and was scored on the exception type instead.

| what the reader refused | frames | runs |
| --- | --- | --- |
| Arrow type 10, a timestamp | `temporal_range`, `temporal_resolutions`, `temporal_dst_back`, `temporal_dst_forward`, `temporal_dst_lord_howe` | 118 |
| a dictionary encoded column | `categorical_ordered`, `categorical_unordered` | 68 |
| a nested type | `nested_list`, `nested_struct`, `nested_deep` | 36 |
| Arrow type 18, a duration | `temporal_durations` | 6 |

That is 228 runs across the whole board. 217 of them are inside the three zero sections and 11 are spread over `groupby`, `basics`, `reshape`, `stats`, `indexing` and `windows`, where a case happens to name one of those frames for a reason that has nothing to do with the section it lives in.

So two thirds of everything this suite calls a failure is not a wrong answer. It is a file that could not be opened.

## What it is

`firepanda/dtype/logical.mojo` has a `TypeKind` with six members: `NULL`, `BOOL`, `INT`, `FLOAT_KIND`, `STRING`, `BINARY`. `LogicalType` builds fifteen types out of them, from `INT8` to `BINARY`, and that is the whole type system. There is no timestamp, no duration, no dictionary and no nested type anywhere in the library, not in the reader, not in the kernels and not on the Python side.

`_format_for` in `firepanda/io/arrow_ipc.mojo` is where that shows up to a user. It reads a schema field, maps it to a format string for the C Data Interface importer, and raises when the Arrow type has no format string to map to, which is every temporal type, every decimal, every interval, anything dictionary encoded and anything with children.

The refusals are correct and the messages are good. `column 'second' has Arrow type 10, which firepanda cannot read yet` names the column and the type and says the honest thing. Nothing here is a bug in the reader.

## The reading of the scoreboard that hid this

The suite has published a headline number since the harness landed and the number has always been dominated by `unimplemented`, which is 3340 of 4079 runs today. Two of the sections in that count are the surface sweeps, `resolution` at 1413 and `signature` at 1034, and they are supposed to be large: they are an L0 case for every pandas name and an L1 case for every pandas callable, so they measure breadth and they were designed to start red.

Reading past those two, the eye lands on the failure count and the failure count is a single number. 334 looks like 334 different problems. It is about a dozen, and the largest of them is one line in a type table.

This is the third time this project has found its own instrument reporting something other than what it looked like it was reporting, and it is worth putting the three next to each other because the shape repeats. The groupby section reported 58 failures that were one missing index. The scalar arithmetic family reported 70 unimplemented that were one missing driver entry. Now three sections report 217 failures that are one missing type kind. Each time the number was accurate and the summary of it was not, and each time the fix was to group the failures by their message rather than to count them.

The rule that comes out of it: a failure count is not a work list until it has been bucketed. `pixi run report` should print the buckets, and until it does, anybody quoting the failure count should have grouped it by message first.

## Why the workstream list did not catch it

M6 is fourteen issues and two of them are exactly this problem for two of the four types. #159 is categoricals, and #160 is nested data. Both are written as method level work, "8 callables plus the dtype and the `observed` paths" and "`.list`, `.struct`, `explode`, `json_normalize`", and neither says that the first thing standing in front of those methods is a column type that does not exist and a reader that refuses the file. That is not a mistake in the issues, it is what a scope written before there was a measurement looks like, and the measurement now exists.

The temporal case is different and it is a real hole. There is no temporal workstream in M6 at all. The fourteen are the index, strings, categoricals, nested, windows, reshaping, groupby, statistics, the cheap surface, errors and `api.types`, and a datetime column is in none of them. M7 is titled Time series and its scope is `resample`, `merge_asof`, time based rolling, timezones, `date_range`, the forty offsets, `at_time` and `between_time`, Period and Interval. Every item on that list assumes a timestamp column already exists to operate on.

So a datetime column fell between two milestones. M6 did not claim it because M6 is organized by pandas namespace and the column type is not a namespace. M7 did not claim it because M7 is the time series algorithms and it took the column for granted. It is the single largest cause on the board and it was scoped by nobody.

## The size of what is missing

From the committed inventory, `surface/pandas-3.0.3.json`, pandas 3.0.3 is 1125 public callables and 1413 names.

| namespace | names | callables |
| --- | --- | --- |
| `dt` | 42 | 13 |
| `Timestamp` | 75 | 42 |
| `Timedelta` | 22 | 10 |
| `DatetimeIndex` | 144 | 91 |
| `offsets` | 47 | 46 |
| `Resampler` | 34 | 27 |
| `cat` | 11 | 8 |
| `list` | 2 | 2 |
| `struct` | 3 | 2 |

The temporal six are 364 names and 229 callables, which is 20 percent of the pandas callable surface sitting behind a column type nothing can construct. The categorical and nested three are 16 names and 12 callables, which is small as a surface and is behind exactly the same wall.

The `dt` count is the one to read carefully. 42 names and only 13 callables, because 29 of them are properties: `dt.year`, `dt.month`, `dt.hour`, `dt.is_leap_year`, `dt.days_in_month` and the rest. A property is not a callable and does not appear in the callable denominator, so the L3 rate over `dt` would be measured on 13 names while the work is on 42, and this suite's `temporal` section already has cases for all of them.

## Scope

Four column types, in this order, and the order is by what unblocks the most per unit of work rather than by what is most interesting.

**One, the timestamp.** A `TIMESTAMP` type kind with a unit and an optional time zone, stored as int64, which is what Arrow stores and what pandas stores. The reader maps Arrow type 10 to it, honouring the unit rather than normalizing to nanoseconds, because `temporal_resolutions` exists in the corpus specifically to hold a second, a millisecond, a microsecond and a nanosecond column side by side, and a reader that normalized would pass that frame by destroying the thing it tests. This alone unblocks 118 runs and it is the entire `temporal` section minus the durations.

**Two, the duration.** A `DURATION` kind, also int64 with a unit, mapped from Arrow type 18. Six runs, and it is here rather than later because subtracting two timestamps produces one and `Series.sub` on a timestamp column is already a case in the suite.

**Three, the dictionary.** A dictionary encoded column, which is a values array and an index array and an ordered flag. 68 runs and the whole of #159, and it is third rather than first because the categorical section is smaller than the temporal one and because a dictionary column touches every kernel that dispatches on type, where a timestamp is an int64 wearing a hat.

**Four, the nested types.** List and struct, mapped from a field with children. 36 runs and the whole of #160.

None of this is M7. M7 is what you do with a timestamp column once you have one: resampling it, joining as of it, rolling over it, and the forty offsets. This is the column.

The split against M6 is the `dt` accessor, `pandas.to_datetime`, `pandas.date_range` in its plain form, and `Series.dtype` reporting a datetime dtype, all of which the `temporal` section already asks for and none of which needs a single M7 algorithm. `DatetimeIndex` with its 91 callables belongs with the index work in #154 and #155 and not here, and `offsets` and `Resampler` stay in M7 entirely.

## What this does not close

The other third. 92 failures are in `basics` and only 2 of them are a read, so 90 are real comparisons of real answers, and 87 of those 90 are one thing: the missing value policy in firepanda #170 and the dtype backend question in firepanda #171.

That family breaks down as 26 runs answering a signed integer where pandas answers a double, 24 answering an unsigned integer where pandas answers a double, 23 answering a null where pandas answers a NaN, 6 scalar reductions with the same integer against double difference, and 8 dtype cases reporting the name of the type rather than a value. It is one decision with four faces, and firepanda #171 states it exactly: pandas on the numpy backend has no nullable integer, so an int64 column with a null in it arrives as float64 and every reduction over it is a float reduction.

That decision is not settled and this document does not settle it. What the numbers say is that it is the second largest cause on the board and the largest one that is about semantics rather than about a type that has not been written, and that it should be decided in a document of its own before it is decided inside a kernel.

The rest of the board once the reads are taken out is 19 runs: 3 in `basics` that are the index, 11 in `errors`, 2 in `groupby`, 2 in `stats` and 1 in `nested`. Every one of those is scoped somewhere already and none of them is a family.

## What a green board would take

Adding it up rather than estimating it. The four column types close 228 runs across nine sections, since eleven of them are cases in other sections that happen to name one of those frames. The missing value decision closes 87. Everything else on the board today is 19 runs across five sections. That is the whole of the failure column, and two changes are 94 percent of it.

The unimplemented column is a different question and a much larger one, and nothing here touches it. It is 3340 runs, 2447 of which are the two surface sweeps, and it is a schedule rather than a bug. This document is only about the 334.
