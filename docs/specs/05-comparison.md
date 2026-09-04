# Comparison

When two answers count as the same answer. This is the part of a differential suite that decides whether it produces bugs or noise, and it is the part that is usually left to whatever `assert_frame_equal` does by default.

## The default is strict

Two answers are equal when they have the same shape, the same column names in the same order, the same dtype per column, the same null mask, and the same values under the per dtype rule below. Everything that relaxes this is opt in per case and is recorded in the case declaration, so the report can say how many cases passed strictly and how many passed under a relaxation.

That last part is the interesting one. A suite where relaxations are invisible drifts, because each one is reasonable on the day it is added. A suite that prints `1841 pass, of which 63 under a declared relaxation` does not drift, because the second number is embarrassing enough to be looked at.

## Per dtype

**Integers and booleans.** Exact. No tolerance, ever. An integer answer that is off by one is a bug and there is no reading of that sentence where it is not.

**Floats.** There is a fourth class, `EXACT`, for the cases about values that have to round trip unchanged rather than be computed at all, and it is a class rather than a tolerance of zero because picking it is a statement about the case. Otherwise a relative tolerance of 1e-12 by default, which is tighter than the 1e-5 `assert_frame_equal` uses and looser than exact. Sums, means and anything else that accumulates get 1e-9 because a tree reduction and a linear one differ in the last bits, and that is expected, documented in the parent folder's document 09 section 5, and not a conformance failure. Variance, standard deviation, skew, kurtosis and correlation get 1e-7, because they subtract quantities of similar magnitude and a two pass and a Welford implementation genuinely differ more than the inputs suggest.

Every one of those three tolerances is attached to the case rather than to a global, and a case picks a class rather than a number. A case that needs more room than its class gives it picks the next class up and says why, and the report counts those. There is no field anywhere in a case declaration where a float can be written, because that is exactly how a suite ends up with one case at 1e-3 that nobody remembers agreeing to. An earlier draft of this section said a case could set its own tolerance with a reason, which contradicts the paragraph above it, and the implementation follows the paragraph above it.

NaN equals NaN. Positive and negative infinity compare exactly and against each other are not equal. Negative zero equals positive zero for value comparison and is compared bitwise in the cases specifically about sign preservation, of which there are four: `abs`, `min`, `max` and `sum` over a column of zeroes of both signs.

**Strings.** Byte exact, after neither side normalizes. A library that silently normalizes unicode has changed the user's data, and the `strings_unicode` frame carries a combining sequence and its precomposed equivalent adjacent to each other for exactly this check.

**Categoricals.** Categories compared as an ordered list when the dtype is ordered and as a set when it is not, codes compared after mapping through the categories rather than directly, because two libraries can represent the same categorical with different code assignments and both be right. The `ordered` flag itself is compared exactly. Whether unused categories survive an operation is compared exactly, because that is a real semantic pandas users depend on and it is what `observed=` is about.

**Temporal.** Compared at the resolution of the answer, with the resolution itself compared exactly. A microsecond answer where pandas gives nanoseconds is a failure and not a rounding question. Timezone comparison is on the zone name and the instant, not on the offset, so a frame in `America/New_York` and the same instants in `UTC` are not equal, which matches pandas.

**Nested.** Lists compared element wise with the same rules recursively, a null list distinguished from an empty list at every level. Structs compared field by field with field order significant, because Arrow struct field order is part of the type.

## The index

Normalized to columns before comparison, per document 03, and then compared like any other column, with three exceptions written down here because each one is a place where a naive comparison produces hundreds of false failures.

**Order of grouped output.** pandas sorts group keys by default and firepanda's group by does not have to, so a case on `groupby(...).agg(...)` compares after sorting both answers by the key columns, unless the case is specifically about `sort=`, in which case it does not. This relaxation is declared per case and counted.

**A default RangeIndex is not data.** When both sides carry a plain 0 to n-1 index it is dropped before comparison. When either side carries anything else it is compared. This is the one relaxation applied globally rather than per case, and it is the reason the frame `df.reset_index()` produces is comparable to the frame firepanda produces without one.

**Row order.** Compared exactly everywhere except the grouped case above and the four cases where pandas itself documents the order as undefined. Not "sorted before comparing to be safe". An engine that returns the right rows in the wrong order has a bug that a user will hit, and hiding it here means finding it in somebody's program instead.

## Errors

An error case declares the exception type and a substring the message must contain. The type must match exactly, including which of the 46 `pandas.errors` types it is, and the substring is the thing a user would search for, which is a column name, a dtype name or a value. Message text beyond that substring is not compared, because pandas rewords its messages between releases and pinning them would make a pandas upgrade a hundred failing cases.

Warnings are compared the same way: the type, and a substring. A case that expects no warning asserts that none was raised, which catches the opposite failure of a library that warns where pandas does not and breaks somebody's `-W error` build.

## The oracle self test

Every rule above is checked by running the whole registry with pandas on both sides. That run must be a perfect score, and when it is not, the harness is wrong and no result from it is publishable until it is fixed.

This is worth more than it sounds. The first version of any normalizer has a bug in it, usually in the index handling, and without the self test that bug is published as ten failures attributed to the library being tested. Every one of those is an hour of somebody's day spent debugging the wrong repository.

The self test also protects the relaxations. A relaxation that is wrong in the direction of being too loose does not show up in the self test, so the self test is run a second way, with each relaxation individually disabled, and the cases that then fail are exactly the cases that need that relaxation. A relaxation that is declared and not needed is deleted, automatically, by a task that reports it.
