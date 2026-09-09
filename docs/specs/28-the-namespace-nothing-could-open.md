# 28. The namespace nothing could open

Document 27 ended by saying that thirty nine callables had proven behaviour and no Python door, that groupby was sixteen of them and the `.dt` accessor was twelve, and that the accessor already had a document written for it. So the accessor was built. This document is about what the board did with it, which was almost nothing, and about why that is the correct reading rather than a disappointment.

## The number

Thirty seven names now exist on `s.dt`. Twenty four are the calendar and clock parts that take nothing and answer a column, `tz` and `unit` answer a word, ten are methods, and `isocalendar` answers a frame of three columns. Every one of them was in the core already. All 82 tests in the library's own `test_dt.py` pass, and every value the accessor computes was compared against a live pandas and agreed.

Passing runs went from 901 to 903. Out of 4081.

Two runs for thirty seven names. Document 27 moved fifty runs for fourteen names. Something is different, and it is not the work.

## Where the fifty five went

The board asks about the `.dt` family in fifty five places: forty two `resolution/dt.*` cases and thirteen `signature/dt.*` cases. Every one of them reports the same sentence.

    AttributeError: the dt namespace could not be built: module 'firepanda' has no attribute 'to_datetime'

The harness builds the namespace it is about to probe, and it builds this one by writing `pd.Series(pd.to_datetime(["2026-01-01"])).dt` against whichever library is under test. firepanda has no `to_datetime`. So the namespace could not be constructed, so every name in it reports as unimplemented, and it reports that whether the accessor behind it has thirty seven names on it or none.

This is the same shape as document 27 one level up. That document said a name that computes the right answer and cannot be called scores nothing, and that the cumulative ladder is right to say so. This one says a namespace that is complete and cannot be constructed scores nothing either, and the thing blocking it is not in the namespace. `Series.shift` was blocked by its own missing door. The `.dt` accessor is blocked by a door belonging to something else entirely, which is the constructor for the column type it lives on.

## What they would score, measured

This is worth measuring rather than assuming, because the interesting question is whether the accessor is actually finished or whether the missing constructor is hiding a second problem behind the first.

Running the board's own two probes, unchanged, against a timestamp column built through Arrow instead of through `to_datetime`:

| | cases | pass | unimplemented | differ |
|---|---|---|---|---|
| `resolution/dt.*` | 42 | 37 | 5 | 0 |
| `signature/dt.*` | 13 | 11 | 2 | 0 |

Forty eight of the fifty five would pass today. Nothing differs. The seven that would not pass are the five names that are deliberately absent because they need a column type that does not exist yet, which are `time`, `timetz`, `to_period`, `to_pydatetime` and `freq`, plus the two of those that also have a signature case. The accessor is finished to the limit of what the type system underneath it allows, and the board cannot see any of it.

So `pandas.to_datetime` is worth forty eight runs on its own, before it computes a single answer, and that is a lower bound because it also unblocks the temporal behaviour cases that document 14 costed and the `temporal` section is still reporting one attained level across 236 callables.

## The property that became two failures

The two runs that did move are worth the rest of this document, because they were failures first.

The accessor's first version spelled `Series.dt` as a Python property. That is the obvious spelling, it works for every program that writes `s.dt.year`, and all 82 library tests passed against it, because a test writes `s.dt.year`. The board then reported two new failures, where before it had reported two unimplemented.

    L0 | resolution/series.dt | Series.dt | 1 differences: 'value', expected 'callable'
    L1 | signature/series.dt | Series.dt | 1 differences: scalar type string, expected list<string>

A property read off the class answers itself. `pd.Series.dt` is the accessor class, and `firepanda.Series.dt` was a `property` object, which is not callable and has no signature to read. So a program that looks the accessor's members up on the class rather than on an instance would have found an object with none of them on it. The board is such a program. It is not an unusual one: documentation generators, type checkers and anything doing introspection all read the class.

The fix is a descriptor that answers the accessor class when it is read off the class and builds an accessor when it is read off a column, which is what pandas does, down to naming the parameter `data` because the L1 case compares parameter names. After it, both cases pass and the failure count goes back to the nine it was before the slice.

Two things follow. The first is that this was a real defect, found by the board and not by 82 tests written specifically for the feature, and it was found because the board probes the shape of the surface and not only the answers coming out of it. That is the L0 and L1 rungs earning their place, and it is the clearest example so far of why a conformance suite that only compares computed values would be a weaker instrument.

The second is that a failure appearing where an unimplemented used to be is a signal rather than a regression to be papered over. The name went from absent to present and wrong, which is strictly more information than absent, and the board said so within seconds. The scoring rule treats a fail as worse than an unimplemented, and that is right for a score, but a fail is better than an unimplemented for a reader, because it names the defect.

## What the board should do about this

The namespace builder is a single expression per namespace and it conflates two different answers. "This library has no `dt` accessor" and "this library cannot construct the column the accessor lives on" are not the same finding, and today they produce the same fifty five unimplemented rows carrying the same sentence.

Nothing in the board is wrong here, but it could be more useful, in two ways that do not weaken it.

The first is that the reason should be reported separately rather than only in a detail string. Fifty five cases blocked by one missing constructor is a fact the report should be able to surface without a reader grepping detail text, and document 13's proposal for message bucketing is the mechanism.

The second is that the builder should be allowed more than one way to build the same namespace, and should record which one it used. A timestamp column is a timestamp column whether it arrived through `to_datetime` or through Arrow, and a library that can produce one but spells it differently is not the same as a library that cannot produce one. The counter argument is real and should be written down rather than dismissed: if the board accepts any route to the object, it stops measuring whether pandas code runs unchanged, which is the whole claim. The answer is that the case still fails at the constructor, because `pandas.to_datetime` is itself a callable on the board with its own unimplemented rows, and it should keep failing there. What changes is that the accessor's own forty eight rows stop being charged for it a second time. Each missing thing should be counted once, in the place it is missing.

## What this argues for next

`pandas.to_datetime` and `pandas.date_range`. They are the remaining two groups of the constructor work, they are worth forty eight runs on the accessor alone before anything they do themselves, and until one of them lands, the entire temporal section of the board is measuring a constructor rather than the two hundred and thirty six callables it contains.

After that, groupby, which document 27 sized at sixteen callables and which needs a third bound type rather than a third accessor.

The general rule this adds to document 26 and document 27 is short. Document 26 said to read what is behind an unimplemented count before treating it as a work estimate. Document 27 said that once the behaviour cases exist, the board will tell you which part of that count is a door problem without anybody reading anything. This one says the door is not always on the thing you are looking at, and that when a whole namespace reports the same sentence, the sentence is the work item and the fifty five rows are one item rather than fifty five.
