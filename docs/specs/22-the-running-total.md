# The running total

## Fourteen runs and a kernel that had to be written

`cumsum`, `cumprod`, `cummax` and `cummin` are fourteen runs on the board: twelve for the four `basics` cases over `int64_half_null`, `float64_half_null` and `float64_no_nulls`, plus `stats/cumsum-tall` over ten thousand doubles and `stats/cumsum-float-edges` over the frame with the infinities in it.

The slice before this one, the shift, needed no new kernel at all. It was a block, a slice and a concat, all three of which already existed, arranged into a shape nobody had asked for yet. This one is the opposite. There is no way to build a running total out of the operations already in the package, and the thing that has to be written is a scan, which is the one shape in a column library that does not decompose into independent rows. Row `i` of a running total is a function of every row before it. That is the definition, and it is also the reason a scan is the operation people assume cannot be vectorised.

## The ladder

The assumption is wrong, and it is worth being precise about why, because the reason it is wrong here is not the reason it was wrong for the fills.

A forward fill also looks sequential: row `i` is the last present value at or before `i`. But a fill has no arithmetic in it. It carries a value, and the only thing a vector unit can do for it is skip over long runs quickly. That is what `nulls.mojo` does and it is the right answer for that operation.

A running total has arithmetic, and arithmetic can be regrouped. The Hillis and Steele ladder is the standard trick and it fits in a register. Fold the register against itself shifted by one lane, and every lane now holds two rows folded together. Fold against a shift of two, and every lane holds four. After log2 of the register width steps, lane `i` holds rows zero through `i`, which is the prefix of the block. Fold in the carry from the block before, store, and take the last lane as the next carry. The lanes that a shift brings in from below the register are filled with the operator's identity, so the low lanes fold against something that leaves them alone.

The dependency chain is then one step per block rather than one step per row. On a sixty four bit dtype with four lanes that is four times fewer serial steps and each step is a vector instruction rather than a scalar one.

## Where the ladder stops being free

The ladder computes row three as rows two and three folded, then folded with rows zero and one folded. The loop computes row three as rows zero through two folded, then row three. Those are the same answer when the operator is associative and they are a different answer when it is not.

Integer addition and multiplication wrap, and wrapping is associative, so the ladder is exact on every integer width and on bool. A running maximum or minimum is associative on every dtype including float, because it selects one of its inputs rather than computing a new number, and a NaN never reaches it since an arriving NaN counts as missing.

Floating point addition is not associative. A float running total computed with the ladder is within an ulp or two of the one pandas computes, at nearly every row rather than only at the last one, and it is not the one pandas computes. So the float running total and the float running product are summed one row at a time, on purpose, and `_reassociates` is the single line that decides it.

This is a real cost and it is worth naming rather than burying. It gives up the vector speedup on the most common column type in the corpus. The alternative is a library that is fast and answers a slightly different column than the one the user asked for, and there is no tolerance argument that rescues that, because a scan is not a reduction: it produces every partial sum, so a reassociated version disagrees at almost every row of the output rather than at one number a reader might reasonably round. `stats/cumsum-tall` is the case that would have been quietly relaxed to hide this, and it was left alone. `stats/cumsum-float-edges` compares exactly and passes.

The integers keep the ladder, which is where the speedup was most defensible anyway, since an integer scan has nothing to trade.

## What a missing row does

Measured. `pd.Series([1.0, nan, 3.0, nan, 5.0]).cumsum()` gives `[1.0, nan, 4.0, nan, 9.0]`.

Two things are true in that answer at once. The gap is emitted in place, so the output is missing exactly where the input was. And the gap is skipped in the total rather than restarting it or poisoning it, which is the four after the first gap: one plus three, not three, and not NaN.

Writing that as a branch per row would undo the whole of the section above, so it is not written that way. A missing lane is replaced by the operator's identity before the scan runs. An identity folded into a total leaves the total alone, which is what identity means, so the scan computes the right answer with no knowledge that anything was missing, and the output rows are blanked afterwards from the same bitmap the lanes were selected with. Zero for a sum, one for a product, the lowest value of the dtype for a running maximum, the highest for a running minimum.

## Two kinds of NaN

`pd.Series([nan, inf, -inf, -0.0]).cumsum()` gives `[nan, inf, nan, nan]`.

That looks like a bug in pandas and it is not, and getting it right is the difference between a compatible library and a library that is more sensible than pandas. The NaN in row zero arrived in the column. It is missing, so it is skipped and emitted in place. The NaN in row two was produced by adding positive infinity to negative infinity. It is a value, so it is carried, and every row after it is that value plus something and is therefore a NaN as well, even though every one of those rows is present.

So a NaN means missing on the way in and means a number on the way out, in the same function, and nothing looks at the running total again once it has been folded. An implementation that treated its own NaN as missing would answer the rest of the column instead, and it would look more reasonable while being wrong. `stats/cumsum-float-edges` compares exactly for this reason.

## The types

Measured against pandas 3.0.3, on a three element column of each dtype.

| Column | `cumsum` | `cumprod` | `cummax` | `cummin` |
|---|---|---|---|---|
| `int8` | `int64` | `int64` | `int8` | `int8` |
| `int16` | `int64` | `int64` | `int16` | `int16` |
| `int32` | `int64` | `int64` | `int32` | `int32` |
| `int64` | `int64` | `int64` | `int64` | `int64` |
| `uint8` | `uint64` | `uint64` | `uint8` | `uint8` |
| `uint32` | `uint64` | `uint64` | `uint32` | `uint32` |
| `uint64` | `uint64` | `uint64` | `uint64` | `uint64` |
| `float32` | `float32` | `float32` | `float32` | `float32` |
| `float64` | `float64` | `float64` | `float64` | `float64` |
| `bool` | `int64` | `int64` | `bool` | `bool` |
| `timedelta64[s]` | `timedelta64[s]` | raises | `timedelta64[s]` | `timedelta64[s]` |
| `datetime64[s]` | raises | raises | `datetime64[s]` | `datetime64[s]` |

Three rules and one exception. A running extreme answers the column's own type, always, because a maximum is one of the values that went in. A running sum or product widens the way a whole column sum widens, so every signed width answers int64, every unsigned one answers uint64, and bool answers int64, which makes `cumsum` on a bool column a running count and `cummin` on the same column a running and.

The exception is float. A float32 running total is float32, where `Series.sum` over the same column accumulates in float64. That is not an inconsistency waiting to be tidied up. A reduction produces one number and can afford a wider accumulator for free. A scan produces a column, and widening it would double the size of the answer to buy accuracy nobody asked for.

The two temporal rows fall out of the same argument `unary.mojo` makes about negation. A duration has a running sum, because two elapsed times add up to an elapsed time, and has no running product, because the product of two elapsed times is an area and there is no type for it. An instant has running extremes, because the latest instant so far is an instant, and has no running sum, because there is no point in time that is the sum of two points in time. firepanda raises in both places with those sentences in the message.

## The invariant, again, from the other side

Document 21 replaced the read path rule with a layer rule: in the pandas facing part of the library a numeric column never carries a null in its bitmap, the door establishes it, and anything above the kernels that opens a new gap restores it.

The scans are the case that shows the rule is about gaps and not about kernels in general. `Series.shift` calls `widen_for_missing` on its way out because it makes room that has nothing in it. `Series.cumsum` does not call it and must not, because a running total is missing in exactly the rows its input was missing, no more and no fewer. If the input came through the door it was already widened and the answer is a float column with NaNs in it; if it did not, the answer is the Arrow one. Either way nothing new went absent, so there is nothing to restore.

That is one line of code that is not there, and the argument for its absence is the reason document 21 was rewritten rather than patched.

## The twin, and what fuzzing it is for

`cumulative_scalar` in `scalar.mojo` is the loop with no ladder, no blocks and no carry between them, and the fuzz harness runs the two against each other at int8, int32, int64, uint16, float32 and float64, twice on the float dtypes so that both spellings of missing get exercised. Two hundred thousand cases found nothing, which is the outcome to expect and not the outcome to assume: the twin exists because the block seam is the kind of thing that is right for every length the author thought of.

It is also the reason the float decision above is testable rather than merely stated. Where the kernel uses the ladder the twin proves the ladder agrees with the loop, and where the kernel does not use the ladder both sides are the same loop and the twin proves nothing at all, which is exactly right: what pandas gives is what the conformance corpus is for, and `stats/cumsum-tall` compares ten thousand partial sums bit for bit.

## The board after

605 passing runs before this slice, 619 after, which is the fourteen that were sized for it. Failing runs unchanged at eight, and the ratchet's callable count unchanged at five. The remaining eight are the same eight document 21 left: three wrong rows out of a sort, two missing index levels on a two key groupby, two skew values differing in the twelfth significant figure, and one cast to string that loses a null.

The four sort rows are next. They are the oldest failures on the board, they are a wrong answer rather than an absence, and a wrong answer is worse than a missing one because a program that calls `sort_values` gets rows back and has no way to tell.
