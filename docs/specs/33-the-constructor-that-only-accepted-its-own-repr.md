# 33. The constructor that only accepted its own repr

Document 32 ended with thirty five reflection runs still unimplemented across the two scalar namespaces and a note that most of them were in `Timedelta`. That reads like a list of missing names. It was not. Every one of those names was present and finished, and the board could not ask about any of them, because the board builds the `Timedelta` namespace by evaluating `Timedelta("1D")` and firepanda refused that string.

`1D` is the spelling in the pandas documentation, in every tutorial, and in most of the code anybody has written against pandas. firepanda accepted `1 days 02:03:00`, which is the shape a span prints as, and nothing else. A constructor that accepts its own repr and refuses the documentation is a library you can only call with output you already have.

## How it was found, which is the part worth copying

Not by reading the board. The board said the namespace could not be built and gave the parse error, and that message had been sitting there through two measurement rounds without anybody following it, because a namespace reported as unbuildable reads like a namespace that does not exist yet.

It was found by writing the test that document 32 said should exist. The three parameter name failures at the end of 32 were each one line to fix, and fixing three lines by hand is the kind of change that comes back, so the fix came with a test that compares every readable signature on both classes against a running pandas. That test builds its sample objects the way the board does, which meant it started with `Timedelta("1D")`, which meant it failed on the first line of its own setup.

The general point is not about scalars. A conformance board and a test suite in the library repository are the same measurement taken in two places, and the reason to have both is that they fail differently. The board reports a namespace it could not open and moves on, because it has three thousand other rows to get through. A test in the library stops. Document 31 made the same point from the other end, where a staging loop that could not copy a subpackage would have kept the board reporting a library problem forever.

## The vocabulary is not the vocabulary

pandas has two unit tables and they are not the same table, which is a thing nobody would design and everybody has to copy.

`Timedelta(1, unit="W")` is a week. `Timedelta("1 week")` raises. `Timedelta("1hr")` is an hour. `Timedelta(1, unit="hr")` raises. The string form is matched without regard to case, so `Timedelta("1Days")` is a day, and the `unit=` form is not. There are thirty three spellings in the string table against twenty nine in the other, and the overlap is most but not all of both.

Eight spellings still work and warn: `w`, `d`, `H`, `S`, `MIN`, `MS`, `US` and `NS`. They warn in firepanda too, for the reason document 31 gave the first time: a program running under `-W error::DeprecationWarning` has to break in the same place in both libraries, and a compatibility layer that quietly stops warning has changed the behaviour of every strict test suite that imports it. The class is `DeprecationWarning` rather than pandas' `Pandas4Warning`, which is a subclass of it.

Four spellings that used to work now raise, which is `T`, `t`, `L` and `l`, the old offset aliases for minutes and milliseconds. A program carrying one of them finds out rather than gets a number. `M`, `Y` and `y` raise as well and always have, because a month and a year have no fixed length and picking an average would answer a question nobody asked.

## Three rules and a wart decide the unit

The nanosecond count is the easy half. The unit the result is quoted at is the half that is not documented anywhere, and it took a sweep to find. Three rules apply and any of them is enough.

A count written in nanoseconds is quoted in nanoseconds. That is the spelling deciding and not the size, so `Timedelta("0ns")` is nanoseconds even though it is zero and `Timedelta("1000ns")` is nanoseconds even though a thousand nanoseconds is exactly one microsecond.

A count that spelled more than six fractional digits is quoted in nanoseconds, again on the digits rather than the value, so `Timedelta("1.000000000s")` is nanoseconds and `Timedelta("1.5s")` is microseconds even though the two are the same kind of thing written differently.

A total that is not a whole number of microseconds is quoted in nanoseconds, which is the only one of the three that reads off the answer rather than off the text, and it is what makes `Timedelta("1.5us")` nanoseconds.

Then there is a fourth thing which is a wart and not a rule. `0ns` is quoted in nanoseconds and `0NS` in microseconds. Every other unit in the table is matched without regard to case, and this one is not, because `NS` is a deprecated spelling that gets rewritten on the way in and the rewrite is what the resolution ends up being read off. `0nano` and `0NANO` are both nanoseconds, so it is not even a general property of upper case. It is copied, because a program that reads `.unit` gets the same answer from both libraries or it gets a surprise, and this project's job is not to have opinions about which of those two a caller deserves.

The fractions themselves have their own pair of rules, which are also not consistent with each other. The whole part and the fractional part are scaled separately, and the fraction is first rounded to as many decimals as its unit has nanoseconds in it, and then truncated. So `1.0009us` is a thousand and one nanoseconds, because its fraction rounds up to a whole nanosecond before anything is truncated, while `1.5ns` is one nanosecond, because a nanosecond has no decimals left to round to and the truncation is the only thing that happens. A reader expects one rule. There are two, and the second one contradicts the first.

## Eight hundred and eighty five spellings

The sweep that produced all of the above ran every combination of forty one unit spellings against fifteen numbers, plus a hand written list of the compound, signed, spaced and overlapping forms, and compared value and unit and the presence of a warning against a running pandas 3.0.3. It went from four differences to zero, and the four were the `NS` wart.

Forty five of those spellings are kept as a parametrised test in the library repository, chosen so that there is one per thing that can go wrong rather than one per spelling. A sweep of eight hundred is evidence while it is running and noise once it is committed, because a suite that takes an extra thirty seconds on every commit to re-derive a rule that has not changed is a suite people start skipping.

## The three parameter names

`Timestamp.fromisoformat` took `date_string` where pandas takes `object`. Positional only in both, so no caller can be holding the name, and the board compares names anyway because a name that is wrong for no reason is usually a name nobody checked.

`Timedelta` collected its keyword fields into `fields` where pandas calls them `kwargs`. `fields` is the better name and `kwargs` is the compatible one.

`Timestamp.strftime` was inherited from the standard library, and the standard library one is a C level callable whose signature `inspect.signature` refuses to read. pandas spells its own out. It is spelled out here now, delegating to the inherited one and doing nothing else, so that a program reading either library with `inspect` sees the same parameter.

None of the three changes what any call does. All three are what reflection sees, and reflection is what a great deal of tooling runs on.

## What it moved

The board goes from one thousand one hundred and ninety five passing runs to one thousand two hundred and thirty, out of four thousand and eighty one. Unimplemented falls from two thousand eight hundred and sixty seven to two thousand eight hundred and thirty five. Failures go from twelve back to nine, which is where they were before document 32's work opened the namespace, and the ratchet is green again without its floor being touched. Divergences are unchanged at seven.

Thirty two of the thirty five runs are the `Timedelta` namespace, which was never missing and is now visible. Three are the parameter names. The interesting number is not the thirty five, it is that all of it was already written and none of it could be measured, which is the fourth time this project has found that shape and the second time in two consecutive documents.

## What is left in these two namespaces

Nothing. `Timestamp` is a hundred and sixteen runs and all of them pass, `Timedelta` is thirty two and all of them pass, and neither namespace has an unimplemented or a failing row left in it. That is the first pair of namespaces on this board to be finished at the reflection levels, and it is worth saying plainly because it is a small and specific claim: the names are all there and their parameters are all right. It says nothing about whether the answers are correct, which is what L2 and L3 are for and which these two classes have a hundred and seventy nine tests of in the library repository instead, because the board has no level that runs a scalar.

The temporal work that is left has moved out of these two namespaces and into the `pandas` one, where seven names are still absent: `NaT`, `to_timedelta`, `TimedeltaIndex`, `timedelta_range`, `Period`, `period_range` and `offsets`. `NaT` is the deliberate gap document 32 named and it stays open until there is a singleton equal to nothing including itself, which is a piece of work rather than a line. `to_timedelta` is the smallest of the other six and is the obvious next one. The four after it are index and offset types, which are their own documents.

Outside the reflection levels, `temporal/groupby-day` and `temporal/date-column` still have no driver entry and the four `temporal/resample` cases need an index the library does not have.
