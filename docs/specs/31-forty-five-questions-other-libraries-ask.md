# 31. Forty five questions other libraries ask

`pandas.api.types` is forty five names and none of them compute anything. Every one is either a question about a dtype or a question about a Python object. `is_numeric_dtype` is what scikit-learn calls before it fits, `is_list_like` is what half of pandas' own argument handling calls on the way in, and `is_scalar` is what decides whether a value is a cell or a column. The code asking is almost never the code that made the frame, which is the whole point of the namespace: `pandas.api` is the part of pandas that is public by contract rather than by convention, and it exists so that a library which has never heard of your frame can ask what is in it.

That makes it a strange thing for a compatibility project to be behind on, and firepanda was behind on all forty five. The conformance board reported the whole namespace with one message, `AttributeError: the api namespace could not be built`, ninety rows deep, and every one of those rows was a name some other library would have called.

## What the board said before

The board had three thousand and seventy eight unimplemented records. Bucketing them by their detail message rather than reading them one at a time turned that into a much shorter list, and twelve of the buckets were the same shape: a namespace that could not be opened, with every name behind it reported separately. `api.types` was the one worth taking first, and the reason was arithmetic rather than preference. Forty five names, ninety runs, and not one of them needs anything from the Mojo side. The `str` namespace is a hundred and fourteen runs and is blocked on the cast machinery. `DatetimeIndex` is two hundred and thirty five and needs an index type that does not exist yet. `api.types` was the only door on the board that could be opened all the way in one change.

## Why it is written by hand

Every other part of the pandas surface in firepanda is generated from the table in `tools/bindings.py`, because it crosses the language boundary and the two halves would otherwise drift apart with nothing to notice. None of this crosses anything. It is forty five pure Python functions over Python objects and dtype names, and the Mojo side has no opinion about any of them. Generating them would mean inventing a table that describes forty five different function bodies, which is not a table, it is the source with extra steps. The table earns its place when the bodies are the same shape, and here they are not.

## The dtype is a string, and that was decided already

`Series.dtype` in firepanda hands back `'int64'` rather than `numpy.dtype('int64')`. That is document 13's call and it predates this work. Everything here follows it. A dtype is normalised to its pandas spelling as a string and the predicates read the string, which is what lets the whole namespace work in an install with no numpy in it. That matters more here than anywhere else in firepanda, because this is the one corner of pandas where numpy is normally unavoidable.

It also means the four dtype classes hand back their `name` rather than wrapping a numpy dtype, and that `pandas_dtype` hands back a string for every dtype that carries no parameters. A program that compares the result against a string keeps working. A program that compares it against `numpy.dtype` was going to notice anyway, because `Series.dtype` told it first.

## numpy, when it is there but not required

Some of these have to answer correctly about a `numpy.float32`, because a caller who has numpy will hand one over and expect `is_float` to say yes. None of them import numpy.

The trick is small and it is worth writing down, because it generalises. A value the caller is holding cannot be a numpy scalar unless numpy is already imported, since there is no other way for that value to exist. So the numpy scalar types are reached through `sys.modules.get("numpy")` rather than through an import. In an install with numpy the lookup is a dict hit and the answer is right. In an install without numpy the lookup returns None, the predicate falls through to the builtin answer, and numpy never gets pulled into a dependency free library. Where that is not enough, the abstract base classes in `numbers` do the work, because numpy registers its scalar types with them and has done for years.

## Six answers that measurement changed

The plan was to write the forty five bodies from the documented behaviour and then check them against a running pandas. Six of them came out wrong, and they are worth listing individually because none of them is a typo. Each one is a place where the name says one thing and pandas does another.

`is_int64_dtype("Int64")` is True. The nullable spelling counts, even though `is_int64_dtype` is documented as the question of whether the dtype is `np.int64` and `Int64` is not that. The first draft said False.

`pandas_dtype(bytes)` is `dtype('S')`, and therefore `is_string_dtype(bytes)` is True while `is_object_dtype(bytes)` is False. The bare `bytes` class is a numpy string dtype rather than an object dtype, which nobody remembers. The first draft had bytes as an object column and got both predicates wrong.

`is_string_dtype("object")` is True. This one is documented and still surprises, because it means `is_string_dtype` is not the question of whether a column holds strings, it is the question of whether it might.

`IntervalDtype.name` is `"interval"` while `str(IntervalDtype("int64", "left"))` is `"interval[int64, left]"`. The name and the printed form disagree, which reads like an oversight and is load bearing, because `df.select_dtypes("interval")` matches on the name and would otherwise match nothing. The first draft had `name` returning the full spelling and the test that should have caught it asserted the same wrong thing, which is a reminder that a test written from the same misunderstanding as the code is not a check.

`pandas_dtype(list)` raises `TypeError`, even though `is_object_dtype(list)` is True. Being handed a dtype and being asked whether something is one turn out to be different questions with different strictness. That is defensible: answering `object` to `pandas_dtype(list)` would let a typo through as a dtype.

`pandas_dtype(None)` is `float64`, and `is_float_dtype(None)` is False. This is numpy's convention leaking through one function and being screened out by the other. It is an inconsistency inside pandas and it is copied here on purpose, both halves, because a program that reads either half breaks if only the other is right.

There is a seventh that is not in the list because it is not a difference of fact, it is a bug. `is_dtype_equal` does not go through `pandas_dtype`. `is_dtype_equal(list, "object")` is True and `is_dtype_equal(None, None)` is False, which is exactly the opposite of what two calls to `pandas_dtype` would produce for both pairs. The obvious implementation, the one that reads best, is wrong in two directions at once and the differential test is the only thing that says so.

## The one deliberate difference

`is_re_compilable("[")` raises `re.PatternError` in pandas rather than answering False. pandas catches `TypeError` around the compile and not `re.error`, so a string that is not a valid pattern comes back out as an exception. The name is a question and firepanda answers it, which means firepanda returns False where pandas raises.

That decision belongs in the divergence registry and it is not there, for a mechanical reason worth recording. The registry verifies every entry by checking that its case patterns point at real, runnable cases, which is what stops it becoming the usual list of known failures that nothing ever reads. The board has no way to run a case against a pure Python predicate yet: the L0 and L1 levels are reflection over an imported firepanda and the L2 and L3 levels go out of process to a Mojo driver, and a question like this one lives in neither. So the registry would reject the entry rather than record it. The decision lives in the specification and in `test_is_re_compilable_answers_rather_than_raising` instead, and the gap in the board is worth its own issue.

## Deprecation is part of the contract

Six of the forty five warn in pandas 3.0: `is_sparse`, `is_categorical_dtype`, `is_datetime64tz_dtype`, `is_period_dtype`, `is_interval_dtype` and `is_int64_dtype`. They warn here too, with the same message and at the same call, because a program running under `-W error::DeprecationWarning` has to break in the same place in both libraries. A compatibility layer that quietly stops warning has changed the behaviour of every strict test suite that imports it.

The warning class is `DeprecationWarning` rather than `Pandas4Warning`, which is the class pandas actually uses. That name belongs to pandas' release schedule and firepanda has no version four to point at. `Pandas4Warning` is a subclass of `DeprecationWarning`, so a filter written against the base class catches both and a filter written against the pandas name catches only pandas. That is the smaller of the two possible mistakes.

## The staging bug underneath it

`firepanda.api` is the first subpackage the library has ever had. Everything before it was a flat directory of modules, and two places had quietly assumed that: the test suite's staging helper and the conformance driver's build script, both of which copied the package by listing its top level files. Both of them therefore produced a firepanda with no `api` in it, and the board would have gone on reporting the namespace as unbuildable on the strength of a copy loop rather than of anything in the library.

This is the same class of bug as document 26, where the binding was behind the library and the board was measuring the binding. A staging step that does not produce the layout a wheel has is a measurement of the staging step. Both copies are recursive now, and the comment at each one says why rather than what.

## What it moved

Ninety runs, from unimplemented to passing, with no change to any kernel. Forty five resolution records and forty five signature records, all green. The board goes from nine hundred and eighty seven passing runs to one thousand and seventy eight out of four thousand and eighty one, failures unchanged at nine and divergences unchanged at seven.

None of that is L2 or L3, and it should not be read as though it were. These are reflection levels: the name resolves and its parameters are in the right order. What it buys is not correctness on the board, it is that `pd.api.types.is_numeric_dtype(df["a"])` now returns rather than raising, in code that nobody in this project wrote. One hundred and seventy five tests behind it compare every answer against a running pandas, and that is where the correctness is recorded, because the board has nowhere to put it.
