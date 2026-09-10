# 34. Sixty ways to spell a dozen types

`astype` looks like the easiest method on a dataframe. It converts a column from one type to another, the conversion is a loop, and firepanda had the loop written and tested in Mojo before any of this started. It was still missing from the Python surface, and it stayed missing for months, and the reason is worth writing down because it is not the reason anybody would guess.

The conversion is about a tenth of the method. The rest is working out what type the caller asked for.

## The table

pandas resolves a dtype through numpy, and numpy has spent thirty years accepting names. There is no single list anywhere; there is a resolution function that tries several things in order and the set of names that get through is whatever falls out. So the only way to know the vocabulary is to ask a running pandas, one name at a time, which is what we did.

`int64` is also `int`, `int_`, `intp`, `long`, `longlong`, `l`, `q`, `p` and `i8`. That is ten spellings of one type, and a program written by somebody who learned numpy first uses whichever of those they learned. There are about sixty spellings in total across the dozen types firepanda has.

Four of them are surprises, in the sense that we would have written the table down wrong from memory:

`i` is int32 and not int64. The single letter codes come from the array interface and there `i` means signed integer of the default width, which is four bytes. Somebody who reads `i` as short for `int` gets a column half the width they expected and no warning about it.

`u` is not a name at all, even though `i`, `f`, `b` and `d` are. There is no reason for this. `u1`, `u2`, `u4` and `u8` all work, and the bare `u` raises.

`long` is int64 and `longdouble` is float64, on this machine. Both are platform dependent and both are what pandas would report on the same machine, since pandas and firepanda are asking the same C compiler the same question. A table written by hand would have to decide what to say here and would probably say the wrong thing on somebody else's machine.

`unicode`, `str_`, `U` and `O` are all the object dtype rather than text. This is the one that costs a caller real time. Somebody who writes `astype("U")` meaning text gets an object column, which in pandas holds Python string objects one per row and is the slowest thing in the library. In firepanda it is refused, and the refusal says that the type that holds text is `str`, because a message saying only that there is no object column is accurate and useless.

## The test walks the table

The obvious way to test sixty spellings is to pick five and trust the rest. We did the other thing. The sweep imports the resolution table out of the module under test, walks every row of it, and asks a live pandas what each name means. A row added to the table without being measured fails there.

This is the same argument document 29 made about the parser and document 33 made about the two unit tables. A compatibility layer is a set of claims about another library's behaviour, and a claim that is not measured is a belief. The cost of measuring is that the test needs pandas installed, which it already did, and about a second of runtime.

## Four types firepanda has and refuses anyway

The interesting rows are not the ones that are missing. They are the four names that resolve to a type firepanda has and are refused regardless.

`datetime64[ns]`, `timedelta64[ns]` and `date32[day]` are refused because the cast underneath falls through to the physical layout. A timestamp column is an int64 count of microseconds and a date column is an int32 day number, and asking the kernel to cast to them hands the counts and the day numbers back as an integer column with the right dtype label on it. `binary` is refused because the cast to it produces a column of text.

All four of those are kernel bugs. Fixing them is not hard and it is not this slice. The question this document is actually about is what to do in the meantime, and there were two answers.

One: leave the names out of the table. Then `astype("datetime64[ns]")` fails on the lookup with `data type 'datetime64[ns]' not understood`, which is a lie, because firepanda does understand it.

Two: put the names in a refusal table with the reason. Then the same call raises `NotImplementedError` and says the cast would hand back the integers underneath.

We took the second. It costs a table and a paragraph of comment and it means a caller who hits it knows whether they have found a missing feature or made a mistake, which is the difference between filing an issue and rereading the documentation for an hour.

The general rule this is an instance of: a wrong answer that looks right is worse than an error, and an error that says the wrong thing is worse than an error that says nothing. There is a third position between "silently wrong" and "not implemented" and it is "implemented and deliberately refused, here is why", and most compatibility work lives there.

## Three ways a cast fails, and firepanda gets all three wrong

Measuring the successes turned up three failures, and they were only visible because we asked what happens when the conversion cannot be done rather than what happens when it can.

A column of text with a value that is not a number. pandas raises `ValueError: invalid literal for int() with base 10: 'x'`, which is the message the built in `int` raises, because that is literally what pandas is calling. firepanda raises a `TypeError` saying `cast: row 1 holds x, which is not a int64`. firepanda's message is better, since it names the row, and firepanda's class is wrong, since a program written against pandas catches `ValueError`.

A float column with a null in it, converted to a plain integer. pandas raises `IntCastingNaNError`, which is a `ValueError`, saying that non finite values cannot become integers and suggesting you replace or remove them. firepanda hands back an integer column with a null in it. That is a legal Arrow column and it is not a legal pandas one, and it is the worst of the three, because nothing fails and the caller gets a column pandas could not have made.

A float column with an infinity in it. pandas raises the same error. firepanda hands back the largest int64 there is.

Those are one issue and one slice and they are not in this one. What matters for this document is how they were found. They were found by adding a conformance case for the failure and not only for the success, and every one of the three had been sitting in the board as an absent driver entry for as long as the board has existed.

## What it moved

`Series.astype` and `DataFrame.astype` resolve and pass their signature checks, and four value level cases pass in `basics`. `dtype=` in both constructors stops being a refusal, which unblocks the constructor forms of a good deal else.

The callable does not count as complete on the board yet, because completeness is measured over every case that names it and three of the cases that name `Series.astype` are the failure cases above and two more are categorical. That is the board working correctly. A method that converts nine types out of nine and reports the wrong error on all three failures is not a finished method, and a scoreboard that said otherwise would be measuring the wrong thing.
