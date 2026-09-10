# The elapsed time

A duration column is what you get when you subtract one datetime column from another, and until that subtraction worked there was nothing in firepanda that could produce one. So the ten cases in this slice arrived together rather than one at a time, and the work behind them is mostly not in the ten operations that the cases name. It is in the rule that decides what type an expression between two temporal columns has, which is a different rule from the one the rest of the library uses and does not fit the machinery that was there. This document is what a running pandas 3.0.3 answered when it was asked directly, on this machine, and what firepanda now does about it.

## Promotion is the wrong shape for time

Every other binary operation in firepanda works the same way. Take the two operand types, find the one type both of them can be converted to without losing anything, convert both, run one loop. An int32 column plus an int64 column is two int64 columns and an addition. That model is simple, it is what numpy does, and it is wrong here in two separate ways.

The first way is that reconciling two resolutions is not a conversion. A second column and a millisecond column meet in milliseconds, and getting there means multiplying every value in the second column by a thousand. That is arithmetic on the values, not a reinterpretation of the bits, and it can overflow where a widening cast never can.

The second way is worse. A timestamp plus a duration has no common operand type at all. The answer is a timestamp, neither operand can be converted into the other, and there is no third type both of them could become. An instant is not a length of time and a length of time is not an instant, and any machinery that insists on making them the same type before it can add them has already lost the distinction that makes the answer meaningful.

So the temporal arithmetic is a parallel pair of functions rather than an extension of the existing one. `temporal_binary_type` decides the answer type from the pair of operand types and the operator, and a second function does the work by rescaling each side to the working unit and then calling the same integer loops everything else calls. The general promotion was extended in exactly one place, which is that two temporals of the same kind and the same zone now reconcile at the finer of their two units. That one change is what makes comparing a second column against a millisecond column start working, which it did not before and which is not an operation this slice set out to fix.

## The finer unit wins, in all three shapes

The rule is one rule and it applies to every pairing that has an answer. Second minus second is a second. Second minus millisecond is a millisecond. A second timestamp plus a microsecond duration is a microsecond timestamp. A second duration plus a millisecond duration is a millisecond duration. There is no case where the coarser unit wins and no case where the answer is at a resolution neither operand had.

What has no answer is worth listing, because each refusal is pandas' refusal and not a gap. Two instants cannot be added: `s + s` raises `TypeError: cannot add DatetimeArray and DatetimeArray`, and the reason is that the sum of two points in time is not a point in time and is not a length of one either. A duration minus a timestamp has no answer, which is the one asymmetry in the set, because subtraction is the operation whose operands are not interchangeable. A timestamp and a plain integer have no common type, which is what stops `s + 1` from silently meaning one of whatever the column happens to be stored in. Each of those raises here with a message that says why rather than saying the dtype is unsupported.

## A constant carries a resolution, and the resolution is in the spelling

This is the rule most likely to surprise somebody reading their own code. `pd.Timedelta(90, unit='s')` is a second resolution constant. `pd.Timedelta(hours=1)` is a microsecond one. So `df['second'] + pd.Timedelta(90, unit='s')` is a `datetime64[s]` column and `df['second'] + pd.Timedelta(hours=1)` is a `datetime64[us]` column, on the same input, from the same accessor, with the same kind of amount on the right. `Timedelta(1, unit='D')` is a second and `Timedelta(days=1)` is a microsecond, which is the same pair of spellings for the same length of time landing on two different resolutions.

Two of the ten cases in this slice are those two spellings and nothing else, which is why they look nearly identical and are not redundant. firepanda models it with `Value.duration(count, unit)`, so a constant has a resolution the way a column does, and the answer type falls out of the same finer unit rule rather than out of a special case.

The constant path does the rescale with one multiplication on the constant rather than by building a column. That is not an optimisation worth much on its own and it is worth stating because it is the one place where the column and the constant paths could have drifted, and they do not.

## The mean goes through a float, on purpose

pandas computes the mean of a duration column in float64 and then truncates toward zero. Not a round and not a floor. The mean of one and two seconds is one second, the mean of minus two and minus one seconds is minus one second, and the mean of minus one and zero seconds is zero. Above 2 to the 53 counts in the column's own unit the float has run out of mantissa, and there the mean of two identical durations is not that duration.

firepanda reproduces this rather than fixing it. Doing the division in int64 would be more accurate, and it would disagree with pandas on columns that people who care about the last microsecond of a century long span would notice, which is a set with nobody in it. The argument is the one already written at the top of firepanda's own binary arithmetic file, which is that being wrong the way people already expect costs less than being right in a way that makes an answer depend on how large the values happened to be. It is also simpler, which matters less but is not nothing.

The sum has its own rule and it is not the same one. The sum of an empty duration column and the sum of a column that is entirely null are both a zero length span rather than a missing one, which is what pandas gives and follows from a sum having an identity. The mean of either is `NaT`. Those two disagreeing about the empty column looks like an inconsistency and is not one.

The sum of a datetime column raises in pandas, because adding two instants raises and a total is additions in a row. The mean of one does not raise, because the average of a set of instants is an instant pandas will name for you. Both of those are now true in firepanda for the same reasons.

## The two readers

`dt.days` rounds downward and not toward zero, so a span of minus one microsecond is minus one day. That is the rule that keeps the days and the remainder adding back up to the original span, it is what pandas gives, and it is the single most likely thing for a from scratch implementation to get wrong, because a truncating division agrees with it on every positive row.

`dt.total_seconds` answers float64 whatever the column's resolution is, including on a column of whole seconds. That means it is lossy on a long duration: a float64 runs out of mantissa at 2 to the 53, a count of microseconds passes that at about 285 years, and the corpus has a row longer than that. The precision loss is pandas' and firepanda copies it, because the number a user compares against came out of pandas.

`pandas.to_timedelta` on an integer column with a unit given is a relabelling and not a conversion. The integers are already the counts and the unit says what they are counts of, so nothing is computed and no null becomes a zero length span on the way through. A column that already carries a resolution is answered unchanged and the unit argument is ignored, which is what pandas does with one.

## Where firepanda and pandas still differ

`dt.days` on a column with missing rows is int64 in firepanda and float64 in pandas. That is not a decision about durations. pandas has no missing value for a numpy int64 and so promotes the whole column to float to make room for one, and firepanda puts the missing behind a validity bit and leaves the dtype alone. It is the same difference that shows up on every integer valued accessor member and it belongs to the general missing value question rather than to this slice.

The duration mean losing precision above 2 to the 53 is documented here and is deliberately not registered as a divergence, because it is not one. firepanda reproduces the loss exactly. It is written down so that somebody who finds it later knows it was measured rather than missed.

## What this is worth

Ten cases and ten runs, all ten passing. They are `temporal/duration-dtype`, `temporal/total-seconds`, `temporal/duration-days`, `temporal/duration-sum`, `temporal/duration-mean` and `temporal/duration-abs` over the durations frame, `temporal/timestamp-minus-timestamp`, `temporal/timestamp-plus-duration` and `temporal/timedelta-construct` over the range frame, and `temporal/to-timedelta` over an integer frame. The temporal section moves from 68 passing runs to 78 and from 54 absent to 44. The board moves from 464 passing to 474, with the 94 failures unchanged.

The reductions are worth more than the two runs that name them. Making `reduce_any` carry the logical type through means the maximum of a timestamp column is now a timestamp rather than the int64 it is stored in. `temporal/max` has five absent runs and this unblocks two of them, over the range frame and the resolutions frame, since the other three are over the daylight saving frames and need the zone work rather than this. Neither of the two is in this slice, because adding them is a driver entry and belongs with whatever group brings the other three. Sorting and grouping need the same logical type treatment and now have a model to copy.

## How it was checked

Eighteen tests, every expected value read off pandas rather than worked out, including both signs of the mean truncation and both signs of the day rounding, which are the two places where an implementation can be wrong in a way that a positive only test corpus cannot see.

The fuzzer runs the two new column readers against one row at a time twins one case in eight, with the twin written to use an explicitly floored division rather than the language operator, so agreement between the two is evidence about the answer rather than about the operator. The duration columns it draws are not the ones the instant cases use: the magnitude has a random number of bits in it and the sign is drawn separately, so a run sees spans of a few units beside spans of a few centuries and the negative side of both roundings gets the same coverage as the positive one. Four hundred thousand cases at a fixed seed found nothing. The three reductions are fuzzed against their scalar twins too, for the type as much as for the arithmetic.

The arithmetic between two temporal columns is deliberately not fuzzed. It is a rescale followed by the same add and subtract loops that the fuzzer already runs over every dtype, so a case for the pair would mean drawing operands small enough that the rescale cannot overflow and then testing nothing that is not already tested twice.

## What stays out

`dt.time` and `dt.timetz`, which need a time of day column type that does not exist yet. The zone aware members. `pandas.to_datetime` and `pandas.date_range`. `dt.seconds`, `dt.microseconds` and `dt.nanoseconds`, which are the remainder fields that go with `dt.days` and which no case asks for yet. `dt.components`, which answers a frame. Duration arithmetic inside a group by, and sorting a duration column, both of which need the same logical type work done to two more kernels.
