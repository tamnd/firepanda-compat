# The row before

## Twenty five runs behind one call

`Series.shift` was worth doing next for a reason that is easy to state and was not true a week ago. Twenty five runs on the board go through it: six for `basics/shift`, six for `basics/shift-negative`, one for `basics/shift-fill`, six for `basics/diff`, two for `basics/pct-change`, one for `basics/alignment-subtract-shifted`, and three for `temporal/dst-difference`. That is a good ratio for one function. But before document 20 landed it was a bad one, because every single shift of an integer column would have answered an integer column with a hole in it where pandas answers a float column with a NaN, and twenty five runs would have gone from absent to failing rather than from absent to passing. Implementing shift first would have made the board worse and the library no more correct. The order of these two slices was not a preference.

## A shift is a copy with a gap at one end

Row `i` of the answer is row `i - periods` of the input. Every row that reaches off the end of the input is missing. Nothing is read, compared or computed on the way, which makes shift the one operation in the kernel package with no opinion about values at all.

The obvious implementation is a `take` with an index array: build `[-1, -1, 0, 1, 2, ...]`, hand it to the gather that already exists, done in four lines. It would be correct. It would also allocate eight bytes a row for indices that nobody looks at twice, and it would turn a memory copy into a gather, which on a tall column is the difference between reading the input once in order and reading it once out of order. Shift is the operation people put inside a loop over a hundred lags. Doing it as a gather is paying a gather a hundred times for a memcpy.

So it is written as what it is. A block for the gap, a slice for the overlap, and a concat to put the two together. All three already existed, all three are already parallel, and the slice is a byte move. There are three shapes and they are the whole function: a shift of nothing is a copy, a shift further than the column is tall is a column of nothing but gap, and anything else is a gap and an overlap in one order or the other. The middle one is written out rather than left to fall out of a slice of negative length, because a slice of negative length is the sort of thing that is correct on one implementation and silently wrong on the next.

## The type is part of the answer

Measured against pandas 3.0.3, on `pd.Series([1, 2, 3, 4, 5], dtype="int64")`.

| Call | dtype out | values |
|---|---|---|
| `shift()` | `float64` | `[nan, 1.0, 2.0, 3.0, 4.0]` |
| `shift(2)` | `float64` | `[nan, nan, 1.0, 2.0, 3.0]` |
| `shift(-2)` | `float64` | `[3.0, 4.0, 5.0, nan, nan]` |
| `shift(0)` | `int64` | `[1, 2, 3, 4, 5]` |
| `shift(1, fill_value=0)` | `int64` | `[0, 1, 2, 3, 4]` |
| `shift(10)` | `float64` | `[nan, nan, nan, nan, nan]` |
| `diff()` | `float64` | `[nan, 1.0, 1.0, 1.0, 1.0]` |
| `diff(2)` | `float64` | `[nan, nan, 2.0, 2.0, 2.0]` |
| `diff(-1)` | `float64` | `[-1.0, -1.0, -1.0, -1.0, nan]` |
| `pct_change()` | `float64` | `[nan, 1.0, 0.5, 0.33333333333333326, 0.25]` |

Three of those rows are one rule looked at from three sides. `shift()` opens a gap in a column that has no way to say absent, so the column widens. `shift(0)` opens no gap and stays an integer column. `shift(1, fill_value=0)` opens a gap and immediately fills it, so nothing is missing and it stays an integer column too. The type is not a property of the operation, it is a property of whether anything went missing, which is exactly the rule document 20 wrote down. That is why the three are separate cases on the board rather than one case with three parameters: a library that hardcoded float64 for every shift would pass two of the three and be wrong about the API in a way no value check would find.

A float32 column stays float32 through a shift, because a NaN already fits in it. A timestamp column stays a timestamp and gets a NaT, a string column stays a string and gets a null. Only the integers move, which is the whole of document 20 restated for one function.

## The invariant moved up a layer

Document 20 said the widening happens on the read path and called it the only place the turn from Arrow to pandas happens. That sentence stopped being true here, and rather than leave it in the docstring to rot it has been replaced with the thing that was actually meant.

The rule is not about the read path. It is about a layer. In the pandas facing part of the library, a numeric column never carries a null in its bitmap. That invariant is established at the door by `DataFrame.widen_for_missing` and it is restored by anything above the kernels that opens a new gap, which today is `Series.shift` and the differences built on it. No kernel calls it and no kernel ever will: `shift_any` gives the Arrow answer, an int64 column with cleared bits, and `Series.shift` decides whether that is an answer a pandas program is allowed to see. The precedent was already in the codebase, in `Series.null_count`, which counts NaN because it sits on the pandas side of the same line.

Stated the other way: the number of places the turn happens is not the thing to keep at one. The number of layers it happens in is.

## Divide, then subtract

`pct_change` looks like `(v - v_prev) / v_prev` and pandas does not compute it that way. pandas computes `v / v_prev - 1`.

On ordinary numbers those are the same number and it does not matter. The corpus has a float column of edge values in it precisely so that it does matter somewhere, and it does: row 3 of `float64_no_nulls` is minus zero and row 2 is minus infinity. Dividing first gives `-0.0 / -inf`, which is `0.0`, and then `-1.0`. Subtracting first gives `(-0.0 - -inf) / -inf`, which is `inf / -inf`, which is NaN. pandas answers minus one.

This is the entire value of running a driver against a corpus that contains infinities and signed zeros rather than against a column of one to ten. The first implementation used the algebraically equivalent form, passed every unit test written for it, passed `basics/pct-change` on the tall frame, and failed on exactly one row of one frame. That row is now a unit test in `tests/test_shift.mojo` with the reason attached, because the next person to simplify that function will simplify it back.

## The difference across a transition

`temporal/dst-difference` is three runs and it needed no code. A zoned column counts instants rather than wall clocks, the shift carries the column's logical type through the concat, and the subtraction of two timestamps is already a duration. So the gap across a daylight saving transition comes out as the hour that was really there rather than the hour the clock face suggests, and nothing in the driver entry says so. It falls out of the type surviving three operations that had no reason to know about zones.

That is the shape a compatibility result should have when the underlying model is right. When it needs a special case, the special case is usually the bug.

## The block that forgot what it was

One real defect turned up, in a function this slice only borrowed. `all_null(type, rows)` builds a column of a given type with every row missing. It took a `LogicalType`, matched it against the physical layouts, allocated an array of the matching physical dtype and returned that, dropping the logical type on the floor. A block of missing timestamps came back as a block of missing integers of the same width.

Nothing had noticed, because the two other callers used it in positions where the type was about to be overwritten anyway. Shift noticed immediately and loudly, because the concat that stacks the gap onto the overlap refuses to stack a `datetime64[s]` onto an `int64` and says why: they are counts of different things and stacking them would put one of the two out by the ratio between their units. A guard written for one reason caught a bug of a different kind, which is the argument for guards that check meaning rather than width.

The fix is one line and it is in `all_null` rather than in shift, because the other caller that passes a real logical type, the frame reindex path in `frame.mojo`, had the same bug waiting in it.

## The board after

580 passing runs before this slice, 605 after, which is the twenty five that were sized for it and no fewer. Failing runs unchanged at eight, and the ratchet's callable count is unchanged at five. The remaining eight are the same eight document 20 left: three wrong rows out of a sort, two missing index levels on a two key groupby, two skew values differing in the twelfth significant figure, and one cast to string that loses a null.

The next slice is the running accumulations, `cumsum` and its three relatives, which is twelve runs plus the two tall stats cases. They need a real kernel rather than a rearrangement of existing ones, because a running total skips a null in the total and emits it in place, measured as `pd.Series([1, None, 3]).cumsum()` giving `[1.0, nan, 4.0]`, and that is not what any loop anybody writes by accident does.
