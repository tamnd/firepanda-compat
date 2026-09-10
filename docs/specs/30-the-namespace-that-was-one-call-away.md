# 30. The namespace that was one call away

Document 28 was about an accessor that could not be opened because a constructor did not exist. This one is about a namespace that could not be opened because a method did not exist, and the difference between the two is that this time the whole implementation was already in the library. The kernels were written, the reductions were written, the frame level entry points were written, and 120 rows of the board reported the same sentence, which was that `DataFrame` has no attribute called `groupby`.

## The number

Passing runs went from 955 to 987 out of 4081. Failures stayed at nine and divergences at seven. Unimplemented went from 3110 to 3078.

Thirty two runs, and they are all reflection. Fifteen names resolve where none did before and the same fifteen match pandas' signature, which is thirty, plus `resolution/dataframe.groupby` and `signature/dataframe.groupby`, which is two. Not one behaviour case changed hands, and that is worth saying plainly rather than hiding in a table.

The groupby section reads 173 runs of which 111 pass, and 111 was very nearly the number before this slice as well. The behaviour cases in that section have been answered for a long time by the out of process driver, which calls the Mojo API directly and has never needed a Python `groupby` to exist. What was missing was the half of the board that asks about the API rather than about the arithmetic, and that half can only be answered by reflection on an importable module. Document 25 made this argument when the engine learned to hold both forms at once. This slice is the first time the argument has been paid back at this size.

## Why this and not the other thing

The choice was made by counting rather than by preference. `pandas.date_range` was the obvious next piece of temporal work and it is worth three to five runs, because most of its cases need a real zone database that firepanda does not have yet. The GroupBy family had 199 rows reporting unimplemented, of which 120 reported one sentence, and that sentence named one missing attribute.

A hundred and twenty rows behind one attribute is not a schedule, it is a door. The board is useful exactly because it can tell those apart, and a slice that opens a door is worth more than a slice that adds a feature, right up until the doors are gone.

## What was actually missing

Nothing in the core. `DataFrame` had `group_by`, `group_agg` and `group_count`, `AggKind` had seventeen reductions, and `firepanda/py/reduce.mojo` was already reading twelve pandas method names off the boundary for the whole column reductions. The grouping kernel, the hashing, the null key handling and the sort were all there and all tested.

What the slice added is one Mojo binding, `PyDataFrame.group_agg`, which reads a list of key names and a reduction name and hands back a frame. Three more names had to be readable, `size`, `first` and `last`, and those three are grouped shapes rather than whole column ones, so they went into a second entry point that is the first one plus three rather than a second table. A change to how `std` reads its delta degrees of freedom is still written once.

The rest is Python. Two classes, thirty methods, and a hundred lines of refusals.

## The thirty methods nobody should type

Fifteen reductions on a frame's grouping and the same fifteen on a column's grouping is thirty methods that differ from each other in a word and a return type. They are generated, from a table of fifteen names against ten parameter shapes, and the generator writes both classes from the same table.

That is not tidiness. A hand written set of thirty would drift, and the way it would drift is the subject of the next section.

## What pandas actually does, which is not what anybody remembers

Every signature here was measured against a running pandas 3.0.3 rather than copied out of the documentation, and the measurement found four things.

`min` and `max` take an engine and `first` and `last` do not. All four take `numeric_only`, `min_count` and `skipna`, in that order, with the same defaults, and it is entirely reasonable to look at those four and see one shape. They are two shapes. The first version of the table had them as one, the board reported `signature/groupby.min` and `signature/groupby.max` as failures on the first conformance run of the slice, and they were split. That is the parity check earning its place: the mistake was made, it survived a reading, and it was caught by the thing that compares against a live pandas instead of against a memory.

`size` is the one reduction with two shapes. With the key in the index it is a Series, because the answer is one number per group and there is nothing for a second column to hold, and pandas gives that Series no name at all. With `as_index=False` it is a two column frame whose count column is called `size`. Both shapes are the answer as much as the numbers are, so both are here.

A `SeriesGroupBy` does not always answer a series. `df.groupby("k", as_index=False)["v"].mean()` is a frame in pandas, with the key back as a column, because `as_index` outranks the narrowing. That makes the honest return type of every method on that class a union, which is what firepanda declares, and it is a place where a stub that says `Series` would be simpler and wrong.

Asking for no groups raises two different exceptions. `df.groupby()` is a `TypeError` saying you have to supply one of `by` and `level`, and `df.groupby([])` is a `ValueError` saying no group keys passed. Same intent from the caller, two classes, and a program that catches one of them does not catch the other.

## What is absent, and absent on purpose

pandas can hand back `groups`, `indices` and `get_group`, which are the grouping itself made visible. firepanda computes the grouping inside the reduction and throws it away, and keeping it would mean every group by object holds one index per group whether or not anybody ever asks for one. That is the cost pandas pays and it is the wrong default for a library whose entire claim is the other one.

So those three names are absent rather than slow, and the board counts them as unimplemented, which is the correct reading. If they arrive later they should arrive as a computation somebody asked for, not as a side effect of calling `groupby`.

`agg`, `apply` and `transform` are also absent, and those are a schedule rather than a decision. They are the doors that take a function or a mapping, and each of them is a piece of work.

## The type parameter that was not decoration

The two classes share all their state and all their argument refusals, so the shared half is one mixin. Written the obvious way, that mixin's helpers return whatever the subclass decided, which a type checker reads as `Any`, and thirty generated methods then each declare a return type that nothing checks. mypy said so, thirty six times, the moment the classes were generated.

The fix is one type parameter on the mixin, bound to `DataFrame` by the frame's class and to the union by the column's, with `size` given its own door on the frame's class because it is the one that does not answer what the other fourteen answer. That is four lines of typing and it turns thirty unchecked declarations into thirty checked ones. A wrapper this thin exists to be exactly the shape it says it is, so this is the one kind of mistake it cannot be allowed to make.

## What this argues for next

The two remaining failures in the groupby section are `groupby/two-keys` and `groupby/two-keys-size`, and both fail on the index rather than on the grouping. pandas puts two keys into a MultiIndex and firepanda has no index at all, so the numbers are right and the shape is not. They are the same failure the rest of the board keeps reporting and they belong to the index work in document 12, not to grouping.

`prod` is on the board and is not one of the seventeen kinds the core has. It is one kernel and it would move the section.

`Series.groupby(other_series)` is a case the board runs and firepanda cannot answer, because the key arrives as a column that is not in the frame and there is nowhere to put it. That is the same missing piece as grouping by a function or by a mapping, which is why all three are refused with the same sentence.

The `agg` family is the largest remaining block: `agg("sum")`, `agg(["sum", "mean"])`, `agg({"column": "function"})`, the named aggregation form, and a lambda. The first three are the fifteen reductions arranged differently and are worth doing together. The last one needs a way to call Python back from a kernel, which is a different question and probably a different document.
