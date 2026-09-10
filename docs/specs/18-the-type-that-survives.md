# The type that survives

Sorting a column of timestamps in firepanda used to give back a column of numbers. The rows were in the right order, every value was correct, nothing raised, and the answer was wrong in the only way that a conformance suite can see, which is that pandas says `datetime64[s]` and firepanda said `int64`. This document is about the class of bug that is, where else it was, what the reductions over a temporal column actually answer in a running pandas 3.0.3, and the two places where firepanda now refuses instead of guessing.

## The kernels are written against the bits

A timestamp column is an int64 count of units since the epoch with a label on it saying what the units are. That is the Arrow layout and it is the right layout, and it means that a sort, a gather, a filter, a forward fill and a concatenation do not need to know anything about time at all. They move int64 values around. Every one of them was already correct on a temporal column before this slice, in the sense that every number that came out was the number that should have come out.

What each of them did was rebuild the result from the physical dtype and drop the label. There is no warning available for that, because to the kernel the label was never there. The bug is invisible from inside the kernel and obvious from outside it, which is why it survived: a unit test written against a sort checks the order, and the order was fine.

So the fix is small and it is in seven places. `take_any`, `filter_any`, `concat_refs_any`, `concat_two_any`, `coalesce_any`, `fill_forward_any` and `fill_backward_any` now put the incoming column's logical type back on the outgoing one. Sorting goes through `take_any` and needed nothing of its own. None of the arithmetic changed and none of the loops changed.

## The reduction table nobody would guess

Reductions are the other half and they are not a relabelling, because a reduction is allowed to answer a different type than it read. Which reductions exist on a temporal column, and what each one answers, was measured against a running pandas rather than reasoned about, and three rows of the result are not what reasoning would give.

| | timestamp, whole column | timestamp, inside a group by | duration, either |
|---|---|---|---|
| sum | raises | raises | duration |
| mean | timestamp | timestamp | duration |
| min, max, median, quantile | timestamp | timestamp | duration |
| first, last | not on a Series | timestamp | duration |
| std | duration | duration | duration |
| sem | raises | duration | same split |
| var, skew | raises | raises | raises |
| count, size, nunique | int64 | int64 | int64 |

The first surprise is that a standard deviation of instants is a length of time. It reads perfectly once you say it out loud, since how far apart a set of points in time are is a span and not a point, and nobody writes it that way when they are writing the dispatch table from memory.

The second is that a variance is refused while its own square root is given. The variance of a set of instants is in units of time multiplied by itself, and there is no dtype anywhere in pandas that holds such a thing, so it raises. The standard deviation is back in units of time and lands in a duration. Skewness is refused for the same reason as the variance and so are correlation and covariance.

The third is not a rule at all. A standard error over a whole column raises, and the same standard error over the same column inside a group by answers a duration. That is an inconsistency in pandas rather than a design, and firepanda copies both halves of it on purpose, because the user comparing the two libraries is comparing whichever of the two spellings they wrote and does not care which of them pandas is right about.

A date column gets the same treatment with one extra refusal. Its maximum is a date, since the answer is a value the column held. Its mean and its standard deviation raise, because both would have to invent a unit and pandas has no date dtype to measure a span of days against.

## One table, two callers

All of that lives in a single function, `temporal_agg_type`, which takes the column type, the aggregation kind and one flag saying whether this is a whole column or a group. It is the only thing in the library that knows the table above, and both the grouped path and the whole column path call it. The flag exists solely to hold the standard error inconsistency, and it is worth one boolean to have the inconsistency named in one place rather than reproduced by accident in two.

Putting the table in `group.mojo` rather than in `reduce.mojo` is a small thing that is worth saying, because `group.mojo` is where `AggKind` is defined and putting it the other way round would have meant an import cycle. The consequence that matters is that the grouped and the whole column answers cannot drift apart, since there is only one table and the only way to disagree with it is to stop reading it.

The relabelling of a reduction's result decides from the raw result's physical dtype rather than from a second table of what each core returns. A mean comes back as a float64 and gets truncated toward zero into the answer's own integer type, and a minimum comes back already in the right integer type and is retagged without touching a value. That is one branch instead of a per kind list, and it stays correct if a core ever changes what it produces, which a second table would not.

This also fixed a regression that the previous slice introduced and that nothing caught at the time. Making the whole column path handle temporal types had left it raising for every reduction not on its fast route, which quietly took away the median, the quantile and the distinct count on a temporal column. Those three fall through to the grouped path over a single group, the way they did before, and the table is what makes falling through safe, since the refusals happen before the fall through rather than after it.

## Two resolutions are not one column

Concatenation and coalescing are the two kernels here that take two columns instead of one, and they are the two that can now refuse.

A second column and a millisecond column have the same physical dtype. Stacking one onto the other with the label of the first is a silent factor of a thousand applied to half the rows, and that is the worst answer a library is capable of: right in shape, right in dtype, right in every row count, wrong by three orders of magnitude in the values. Coalescing has the same hole, where a row filled from the second column comes back off by the ratio between the units.

pandas reconciles the two at the finer unit, which is what the arithmetic in the previous slice already does. firepanda refuses here and says which two types it was given. That is a smaller answer than pandas gives and it is deliberate, because the reconciliation is real work that belongs with the concatenation rules rather than being bolted on, and a missing feature costs less than an answer that is wrong in a way nobody reading the output can detect. The refusal is written down here so that the direction is on the record rather than being rediscovered.

## The harness lost a half each way

One of the seven new runs failed for a reason that was not in firepanda at all, and finding it took two attempts because the first fix broke three other frames.

The comparison layer turns a returned scalar into a canonical type string. For a `pd.Timestamp` it was calling `pa.scalar` and letting pyarrow infer, and pyarrow answers `timestamp[us]` for every `pd.Timestamp` regardless of what the scalar's own `unit` attribute says. So a correct `datetime64[s]` maximum was reported as a mismatch against an expectation the harness had also got wrong, in the other direction.

The obvious fix is to hand pyarrow the numpy value instead, since that carries the unit. It does, and it drops the time zone, so the three daylight saving frames started failing instead. Both halves are attributes on the scalar itself, so the type is now built from `value.unit` and `value.tz` rather than inferred from anything. Same for a `pd.Timedelta` and its unit.

This is worth a section because it is the second harness bug this month that presented as a library failure, and the shape is the same both times. An inference call in the comparison layer is a piece of code with an opinion, and when its opinion is wrong the suite blames the library. The rule this suggests is that anywhere the harness can read a fact directly off the value it should, and inferring it from a reconstructed copy is a shortcut that costs more than it saves.

## What this is worth

Seven runs, all seven passing. `temporal/sort-timestamps` over the range frame and the resolutions frame, and `temporal/max` over those two plus all three daylight saving frames. The temporal section moves from 78 passing runs to 85 and from 44 absent to 37, with no failures in it. The board moves from 474 passing to 481, with the 94 failures unchanged.

The three daylight saving frames are worth a note. They pass now without any of the zone work being done, because the maximum of a zoned column is a value the column already held and the zone rides on the type rather than on the operation. That is the general shape of this slice: the work that makes a type survive is worth runs on frames that the operations behind those runs were never the problem for.

What the seven runs undercount is the sort and the fill and the gather, which are not seven runs each but are underneath a large share of everything the frame layer does. A group by on a temporal key, a merge on one, a rolling window over one and a reindex all go through the kernels this slice fixed, and every one of them would have handed back a number.

## How it was checked

Eleven tests, in a file whose subject is that a type survives rather than that a kernel is correct, since the kernels were already correct. The five value sample they reduce is unsorted and straddles the epoch, because a reduction that lost the sign or the order would still pass on five ascending positive numbers. Every expected value was read off pandas, including the standard deviation of seventeen seconds that is a truncation of 17.176 and not a rounding of it.

Three of the eleven are refusals rather than answers, which is unusual for a test file and is the point of this one. A test that asserts a variance raises is a test that the table was read, and a table that is only exercised by its non raising rows is a table with half its content untested.

The full suite is sixty files and passed. Two hundred thousand fuzzer cases at a fixed seed found nothing, which was expected, since no loop changed and the fuzzer compares values rather than dtype labels. It was run because a change inside `group.mojo` touches the dispatch that every aggregation in the library goes through, and the cost of being sure is ninety seconds.

## What stays out

The reconciliation of two units in a concatenation or a coalesce, which is the refusal above turned into an answer. `temporal/groupby-day`, which needs a computed group key and the indexed group by shape rather than anything about types. `temporal/date-column`, where pandas reports a date32 column as holding Python objects and firepanda does not, which is a divergence to write up rather than a driver entry to add. The zone aware members, `dt.time`, `pandas.to_datetime` and `pandas.date_range`, all of which were out of the previous slice too and are the next real group.
