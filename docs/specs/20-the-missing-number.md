# The missing number

## Eighty seven of ninety four

The board had ninety four failing runs. Reading them one at a time would have produced ninety four issues, so they were bucketed by message first, and the buckets were these.

| Bucket | Runs |
|---|---|
| `dtype int8` and the other seven integer widths, `expected double` | 56 |
| `N nulls, expected 0. A null is not a NaN` | 23 |
| `basics/series-dtype` saying `'int8', expected 'float64'` | 8 |
| everything else | 7 |

The first three buckets are the same sentence said three ways, and the sentence is that firepanda gave back an integer column with a validity bitmap where pandas gave back a float column with a NaN in it. Eighty seven of the ninety four failures were one decision that had never been written down, and the seven that were left are three wrong rows out of a sort, two missing index levels on a two key groupby, and two skew values that differ in the twelfth significant figure. The board was not eighty seven bugs deep. It was one policy question deep, with a short tail behind it.

That is worth stating plainly because it changes what the remaining work looks like. A hundred failures spread over a hundred causes is a year. A hundred failures over one cause is an afternoon and a decision, and the decision is the hard part.

## Two answers to one question

The question is where a number goes when there is no number.

Arrow answers it with a second buffer. A column has its values and it has a validity bitmap, one bit per row, and a cleared bit means the row is absent. The value buffer still holds an integer at that row and nobody is meant to look at it. The type of the column is unaffected by whether anything is missing, so an int64 column with a gap in it is an int64 column, and it can hold every int64 there is including the ones a float64 cannot represent exactly.

numpy answers it by spending a value. A numpy float64 array has NaN sitting inside the type already, so a float column needs nothing extra to record a gap. A numpy int64 array has no spare value at all, every bit pattern being a number somebody might mean, so there is nowhere to put the information. pandas on the numpy backend resolves that by not keeping the column an integer column. It widens to float64 on the way in and writes NaN in the gap.

The widening was measured against pandas 3.0.3 and pyarrow 25.0.0, one Arrow type at a time, and the rule is narrower than it first sounds.

| Arrow type in | pandas dtype out | what the gap became |
|---|---|---|
| `int8` through `int64`, `uint8` through `uint64` | `float64` | `nan` |
| `float` | `float32` | `nan` |
| `double` | `float64` | `nan` |
| `bool` | `object` | `None` |
| `string`, `large_string` | `str` | `nan` |
| `date32` | `object` | `None` |
| `timestamp[s]` | `datetime64[s]` | `NaT` |
| `duration[s]` | `timedelta64[s]` | `NaT` |

Every integer width goes to float64 and none of them goes to float32, even though a float32 holds every int8 and every int16 exactly. A float32 column stays float32, because a NaN already fits in it and there is nothing to make room for. Everything that is not a number has its own way of being absent already and keeps it, so this is a rule about numbers and only about numbers.

pandas has a second backend where none of this happens. `Int64` with a capital I is a nullable integer, it is a values array plus a mask, and it is Arrow's answer wearing pandas clothes. It is not the default, and the conformance oracle loads its frames with `to_pandas()`, which produces the numpy backend, so the numpy backend is what the board is scored against. That is a decision and it is recorded here rather than left implicit in a line of harness code.

## The turn happens once, on the way in

There were four places the two models could have been reconciled and only one of them is cheap.

It could have gone in the kernels, so that adding two to an integer column with a gap in it produced a float column. That is pandas semantics pushed into the inner loops, it makes every arithmetic kernel ask a question about nullability before it can pick a type, and it would make the native Arrow API wrong for every consumer who is not a pandas program. Rejected.

It could have gone in the comparison rules, as a relaxation that lets an int64 answer match a float64 one. That would have turned eighty seven failures green by agreeing not to look, which is exactly the green washing `07-scoreboard.md` exists to forbid. Rejected, and it is worth saying that this was the tempting one, because it is a six line change.

It could have gone in the driver only, so that the compat program widens and the shipped library does not. That scores the board honestly for a frame the library will never hand anybody, which is a different kind of dishonest. Rejected.

It goes in the read path, which is where pandas itself does it. `pandas.read_csv` widens. `to_pandas` widens. Neither `DataFrame.__add__` nor `Series.sum` widens anything, because by the time they run there is nothing left to widen. So firepanda widens in the same place: `firepanda.read_csv`, which is a pandas name with a pandas meaning, and the compat driver, which is reading the same corpus file the oracle read and must start from the same frame the oracle started from. Everything downstream then agrees with pandas without being told anything about pandas.

The implementation is one function, `widen_for_missing`, and it is in `firepanda/kernel/nulls.mojo` next to the other things that know what missing means. It asks the column two questions, whether the type is numeric and whether the null count is above zero, and if either answer is no it hands the column back untouched. Otherwise it casts to float64 if it is not already floating, writes a NaN into every cleared row, and drops the bitmap.

The decision is made once for the whole column rather than once per chunk. A column is a `ChunkedArray` and the pieces are an implementation detail of how the file was read, so asking each piece whether it holds a missing row would widen the pieces that do and leave the ones that do not, and a column whose first chunk is int64 and whose second is float64 is not a column at all.

The bitmap is dropped rather than kept alongside the NaN. Keeping both would mean a row could be absent in two different ways and the two could disagree, and the first bug from that shape would be a `null_count` of one on a column with two NaN in it.

## What it costs, and what it does not cost

One pass over the numeric columns that have a gap, at read time, and nothing anywhere else.

A column with nothing missing is returned as it was, which is the common case and the one that decides whether the rule is affordable. A file of ten columns where two have gaps pays for two casts and eight pointer copies. No kernel gained a branch, no loop gained a check, and no type gained a field. This is the entire reason the read path was the right place: a rule that fires once on the way in costs one pass, and the same rule expressed inside the kernels costs a branch per operation forever.

## The door that keeps Arrow's answer

`firepanda.from_arrow` does not widen and will not.

The Arrow C Data Interface is a protocol with a promise attached, which is that what came in comes out. A pyarrow table handed to firepanda and taken back out is meant to be the same table, and an int64 column that arrives with three nulls and leaves as a float64 column with three NaN has been quietly altered by a library that was asked to hold it. That is a bug against every consumer, and pandas is not the consumer on the other end of that door most of the time. It is how firepanda talks to Polars and to DuckDB, neither of which has ever heard of NaN meaning absent, and both of which would be handed corrupted data by a widening they never asked for.

So the two doors say different things on purpose. `read_csv` is a pandas name and gives the pandas reading. `from_arrow` is an Arrow name and gives the Arrow one. A caller who wants the exact bytes has a way to get them and a caller who typed a pandas program gets what their program expects, and neither of them has to know the other exists.

## The rule that was already there

The reason the widening is safe on its own, with no downstream change of any kind, is that firepanda had already decided the harder half of this question and built it.

`firepanda/kernel/nulls.mojo` treats a NaN in a float column as missing. `present_bitmap`, `missing_count_any`, `is_null` and the fill functions all read the values on a float dtype rather than only the bitmap, and `firepanda/kernel/agg.mojo` steps over a NaN in its vectorised sum, min and max. So after the widening, `Series.null_count` still reports one for a column whose bitmap is now entirely set, `dropna` still drops the row, and `sum` still skips it. The row is still missing to everything that asks.

That is what turns a dtype change into a compatibility change. If NaN had not already counted as missing, widening would have converted eighty seven dtype failures into eighty seven counting failures and the board would not have moved. Because it had, arithmetic propagates NaN exactly as pandas does, for the same reason pandas does, with no rule about pandas anywhere in the multiply.

## The board after

| | before | after |
|---|---|---|
| passing runs | 494 | 580 |
| failing runs | 94 | 8 |
| ratchet failures | 16 | 5 |

The eight that are left are the seven from the tail plus one, and they are four separate small things rather than one large one. Three runs of `basics/sort-values` and `basics/sort-values-descending` come back with the wrong rows, which is a real bug in the sort and not a dtype question. Two runs of `groupby/two-keys` return no index levels where pandas returns two, which is the MultiIndex work. Two runs of `stats/skew` differ in the twelfth significant figure, which is the moment accumulation already written up as a divergence. One run of `basics/astype-string` has no null where pandas has one, which is a cast that lost the gap rather than kept it.

None of those is a policy. They are four bugs, and now they are visible, which they were not underneath eighty seven copies of the same message.
