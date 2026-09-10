# 32. The scalar a column hands back

Ask a datetime column for its maximum and something has to come back. Index into it and something has to come back. Read one cell out of a grouped result and something has to come back. Until this work there was nothing in firepanda for any of those to hand over, which meant the whole temporal half of the library could produce columns and could not produce a value. `firepanda.Timestamp` and `firepanda.Timedelta` are that value.

This document is the compatibility side of firepanda #367. The library side is in the changelog entry that landed with it. What belongs here is what the conformance board could and could not see, what the differential sweep measured that no amount of reading would have produced, and why the board's failure count went up.

## Why a subclass and not a wrapper

Both classes subclass `datetime.datetime` and `datetime.timedelta`, which is what the pandas ones do and for the same reason. A scalar that is not a `datetime` breaks every piece of code that is holding a `datetime`, and almost all of that code belongs to somebody else. A library that hands back its own opaque moment type has not built a pandas compatible scalar, it has built a reason to call `.to_pydatetime()` everywhere.

The nanoseconds ride alongside as an extra field rather than inside the inherited value, because `datetime` stores microseconds and there is no room. That is the three digits Arrow has and the standard library does not, and it is the entire reason pandas needed its own class in the first place.

## The sweep, not the documentation

Every answer in both classes was compared against a running pandas 3.0.3 through a differential harness that made the same call both ways and compared either the value or the class of the failure. The sweep opened at ninety two differences for `Timedelta` and one hundred and four for `Timestamp`. It closed at one and three, and all four of those are deliberate and listed below.

The point of writing that number down is that none of the ninety two and none of the one hundred and four came from a misread of the documentation. They came from behaviour the documentation does not describe, and a compatibility layer written from the documentation would have shipped every one of them.

Six are worth naming because each one produces a wrong answer that nobody would question.

A zone handed to a naive text value was converting where it should have been localizing, so `Timestamp("2026-09-05 13:45:06", tz="Europe/Paris")` came back as a different instant. Two hours of error, no exception, no warning.

The int64 range was being checked in nanoseconds where pandas checks it in the unit the value is quoted at. The year 1000 fits comfortably in microseconds and was being refused.

Arithmetic kept the left operand's unit rather than the finer of the two, so adding a nanosecond span to a microsecond moment silently dropped the nanoseconds.

`hash` disagreed with pandas. Equal things have to hash the same, or a dictionary keyed on moments grows two entries for one key and the second one is never found.

The printed form of a negative span put the sign on the whole reading where pandas puts it on the days alone, so `str()` did not read back as itself.

`to_pytimedelta` floors in the obvious implementation and pandas rounds to nearest with the tie going upwards. That is measured. It is none of the three rules somebody would guess.

Most of the exception classes were wrong as well, which matters because a caller catching `ValueError` around a parse is the normal way to handle bad input. An out of bounds moment is a `ValueError`, since `OutOfBoundsDatetime` subclasses it. An unknown zone is a `KeyError` raised by `zoneinfo`. Misusing `tz_convert` is a `TypeError`. `as_unit` splits two ways, giving `NotImplementedError` for a real unit a scalar cannot be quoted at and `TypeError` for a word that is not a unit at all.

## Two parsers for one vocabulary

`round`, `floor` and `ceil` take a frequency string, and the library already parses frequency strings in the Mojo kernel for `Series.dt.floor` and its two neighbours. A scalar cannot reach that parser without building a one row column to carry a single value through it, so there is now a second parser in the Python file.

Two parsers for one vocabulary drift apart, and the thing that manages the drift is not a comment. One test asks both of them the same question over every alias, every count and all three directions, and compares. A second checks that they refuse the same things.

That second test is what turned up the one fix in this work that has nothing to do with scalars. A frequency the kernel cannot round to came back as a type complaint from a column and a value complaint from a scalar, and pandas raises the value one. The cause was that the `dt` accessor wrapped its whole call in a handler that tags anything coming out of it as a dtype error, which is right for a column of the wrong type and wrong for an argument of the wrong value. The accessor checks the frequency before that handler opens now, asking the kernel rather than keeping a third copy of the vocabulary, and the tagging helper leaves an already tagged error alone, which is what its own docstring had been claiming all along.

## Four gaps left open on purpose

There is no `NaT`, so `Timestamp(None)` and `Timedelta(None)` are refused. A missing temporal value inside a column is an Arrow null and works today. A missing one standing on its own needs an object that compares unequal to everything including itself, and that is a separate piece of work rather than a line in this one.

A moment past the year 9999 is out of range. pandas will build one, but only by way of its `Timestamp` claiming to be a `datetime` without really being one. This one really is a `datetime`, and the standard library stops at 9999.

`to_julian_date` gives a plain float where pandas gives a numpy one. That is the same call document 13 made when `Series.dtype` became a string, applied again.

`replace` works on a moment before 1678 where pandas refuses, because the pandas `replace` checks the nanosecond range rather than the range of the unit the value is quoted at. That is an inconsistency inside pandas rather than a rule worth copying.

All four are asserted in tests, so a future change that closes one of them fails a test that says out loud it was deliberate.

## The failure count went up, and that is the board working

The board moves from one thousand and seventy eight passing runs to one thousand one hundred and ninety five out of four thousand and eighty one. Unimplemented falls by one hundred and twenty, to two thousand eight hundred and sixty seven. Divergences are unchanged at seven. Failures go from nine to twelve.

That last number is the interesting one and it is not a regression. Before this work, `firepanda.Timestamp` did not exist, so every reflection case in the `Timestamp` and `Timedelta` namespaces reported one missing attribute and scored as unimplemented. A namespace that is not there cannot be wrong about anything. Now the namespace is there, one hundred and seventeen of its runs pass, and three of them are measurably wrong, which is a thing the board could not have told anybody yesterday.

This is worth stating as a general property rather than as an excuse for three rows. Unimplemented is not a better outcome than fail. It is a less informative one. A project reading only the passing count would see this change as progress and a project reading only the failure count would see it as damage, and both of those readings are wrong in the same way: the denominator that matters is what the board is able to ask, and this work raised it by one hundred and twenty questions.

The three are small and all three are in the reflection level, where the board compares parameter names and kinds.

`Timestamp.fromisoformat` takes a parameter firepanda calls `date_string` and pandas calls `object`. It is positional only in both, so no caller can pass it by name and nothing observable depends on it, and the board compares names anyway because a name that is wrong for no reason is usually a name that was never checked.

`Timestamp.strftime` is inherited from the standard library in firepanda, and the standard library one is a C level callable whose signature `inspect.signature` cannot read. pandas defines its own taking `format`. So firepanda reports that the signature could not be read where pandas reports one parameter, which is the failure the board printed.

`pandas.Timedelta` collects its keyword fields into a parameter firepanda calls `fields` and pandas calls `kwargs`. Same shape of miss as the first one.

None of the three needs anything from the Mojo side. All three are closed in document 33, which is the follow up this measurement caused, along with a much larger door the same three rows led to.

## What is still shut behind this door

The scalars are what a column hands back, and there is a second half to that sentence which is not done: the columns that would hand them back are still missing in places. `temporal/groupby-day`, `temporal/date-column` and the four `temporal/resample` cases have no driver entry, and the resample family needs an index the library does not have yet.

The thirty five reflection runs still unimplemented in these two namespaces are, at the time of writing, almost all in the `Timedelta` namespace, and the reason turned out not to be that the names are missing. That is document 33 and it is the reason this one has a sequel. The rest are `NaT`, the `Period` neighbours and the numpy valued members, and they are a smaller and more specific list than the one this work started from. That is the shape progress takes on a board this size: a door opens, most of what is behind it is now measured, and what remains is named.
