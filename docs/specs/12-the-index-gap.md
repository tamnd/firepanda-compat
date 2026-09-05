# The index gap

Of the 98 cases still failing across basics, stats and groupby, 76 fail because firepanda has no index. That is one cause, not several, and it is now by a wide margin the largest thing standing between this suite and a green board. This document says what the gap actually is, corrects a wrong reading of it, and scopes the smallest index that closes it.

## What it is not

The first reading of these numbers was that the groupby section had never run its comparison, that 59 grouped results were failing before anything looked at a value, and that whether firepanda's grouped arithmetic is correct was therefore unknown. That is wrong, and the way it is wrong is worth keeping, because it is the failure mode this whole repository is built to avoid: reading a failure count instead of reading the cases.

The suite already solved this. Every grouped reduction in `fpcompat/cases/groupby.py` has a twin declared with `as_index=False`, which is the pandas spelling for returning the key as a column rather than as the index, and the note on one of them says so in as many words: "the key comes back as a column rather than as the index, which is a different shape and not a different computation". There are 17 such case ids covering sum, mean, min, max, count, first, last, median, nunique, sem, skew, std, var, two keys, `dropna=False` and `sort=False`, across four key frames. All 50 runs pass.

So the grouped arithmetic is measured, and it is right. The 59 indexed failures are the shape and nothing else, and they are failing on purpose. The driver's own docstring is explicit about this: "five lines here could manufacture an index column and turn a failure into a pass. That would be a lie, and worse than a lie it would be an invisible one". Any plan that ends with the driver declaring the key column to be an index level is that lie, and it would buy nothing, because the values it would let through are already compared by the flat twins.

The same goes for the 14 in basics. `tail`, `boolean-mask`, `dropna` and `sort_values` fail because pandas keeps the original row labels through all four, so `tail(5)` of a ten row frame comes back labelled 5 through 9 and `sort_values` comes back permuted. There is no header field that makes those equal, because they are not equal. A user who reads `result.index` gets different values from the two libraries and a user who calls `.loc` on the result gets different rows.

There is nothing to fix in the harness here. The harness is right and it is telling the truth.

## What it is

firepanda has no index, at all, anywhere. `DataFrame` is a schema, a list of chunked columns and a row count. `Series` is a name and an array. Neither has a field for row labels and no operation carries any. The frame module's docstring lists this as the first deliberate divergence and points at `docs/specs/04-python-dx.md`.

That was the right call for M1 and it is the wrong call for a milestone whose stated goal is parity. Registering it as a divergence would be the other way of avoiding the work, and it would freeze 76 cases as permanently red while the scoreboard called them expected. A divergence registry entry is a decision that something should differ. Nobody has decided that firepanda should not have an index. What was decided is that it did not have one yet.

So the gap closes by building one.

## The smallest index that closes it

Two stages, because the two halves of the 76 need different things and the first half is a much smaller change than the second.

**Stage one, row labels.** An `Index` holding an array of labels and an optional name, and an `index` field on `DataFrame` and on `Series` that defaults to the range zero to n minus one. Every operation that chooses rows already gathers or slices the columns, and the labels go through the same gather: `take`, `filter`, `slice`, `head`, `tail`, `sort_values`, `drop_nulls`. Nothing else changes and no operation aligns on labels yet. A default range index has to stay cheap, which means it is a start and a length rather than a materialized array, and it materializes only when something gathers it. That is what makes the field free on the path that does not use it, and it is the reason to do this stage separately: if a range index costs anything measurable on a filter of a million rows, the design is wrong and it is better to find that out before anything depends on it. This stage closes the 14 in basics.

**Stage two, the grouped result.** `groupby(...).agg(...)` returns a frame whose index is the key, named after the key column, with the aggregates as the only data columns, and `as_index=False` keeps the current flat shape. The keys are already computed and already named, so this is a question about what the frame layer assembles rather than about the kernel. Two keys become two index levels, which is a MultiIndex, and whether that lands in this stage or a third one depends on what stage one's `Index` turns out to look like. This stage closes the 59 in groupby.

Neither stage is firepanda issue 154, which is the `Index` type as an object with 73 callables of its own, the four set operations, `get_loc` and the `get_indexer` family. These two stages are the smaller thing that has to exist first for 154 to have somewhere to live, and the ordering argument in 154 already says the index API is not an add on to the index work, it is the index work. What these stages add is the other half of that sentence: there has to be an index for the API to be an API of.

## The one harness bug in the neighbourhood

Small, real, and worth fixing while the area is warm, because it actively misleads. `_label` in `fpcompat/compare.py` renders a non string label with its type in front, which is right for column labels and is what keeps the integer `1` apart from the string `"1"`. Applied to an index level name of `None` it produces the string `NoneType(None)`, so all 14 of the basics failures read `index names (), expected ('NoneType(None)',)`, which says pandas has a name called NoneType when pandas has no name at all. The difference that actually matters on those cases is the level count, which is reported second and truncated out of the summary.

The first attempt at this changed `_label` to render `None` as the empty string, and that was wrong and was reverted before it shipped. `_label` has to stay unambiguous, because a pandas label can genuinely be the string `None` and can genuinely be the empty string, and a renderer that folds any of those together is a renderer that can score a real difference as a pass. So `_label` is unchanged and the fix is at the message: the sentence a human reads says "unnamed", the level count check is reordered in front of the names check so that a missing index is reported as a missing index, and every place that is comparing a value rather than describing one still gets the unambiguous form. The lesson is that a readability problem in a message is a message problem, and reaching back into the renderer to solve it trades correctness for prose.

This turns nothing green. It changes 14 misleading messages into 14 accurate ones, which is the whole of its value and is enough.

## What stage one actually moved

Stage one shipped in firepanda pull request #192 and the driver reports the index in this repository's matching change. Basics goes from 145 pass and 32 fail to 155 and 22, which is the 14 in this document minus four that fail for a second reason as well. Stats is unchanged at 52 and 4, as predicted, because a scalar has no index. Groupby is unchanged at 54 and 62 and the shape of that 62 is the number worth reading. 57 of them read "0 index levels, expected 1", where before the reorder they read "index names (), expected ('key',)" and named a level that was not missing a name but missing entirely. Two say "expected 2", which is the two key grouping and wants a MultiIndex. The last three are the Arrow reader refusing a dictionary encoded column, which is firepanda issue #159 and is nothing to do with an index. So the whole of this section is one sentence now: a grouped result should be indexed by its key. That is exactly stage two and nothing else.

The design that came out of stage one is worth recording here because it decides what stage two can assume. An index is either an arithmetic range, a start and a length with no memory at all, or an array of labels, and a frame only pays for the second form when something takes it apart. Measured on a three column frame of a million rows against an untouched control that moved 4.6 per cent between binaries, a gather moved 2.8 per cent and a filter moved 16.8 per cent, so the range form did what it was supposed to and the question in the stage one paragraph above is answered: it costs about one more column on a filter and near nothing on a gather.

One thing was learned that the plan did not anticipate. Carrying labels through `take`, `filter` and `slice` is necessary but it is not sufficient, because an operation that produces new rows rather than selecting existing ones has to reset the labels instead of carrying them. `group_by` is built out of a filter and a sort, so it inherited both, and it came back labelled with the permutation of its group ordinals. Nothing in firepanda's own 47 test files noticed. This suite noticed immediately, because groupby fell from 54 pass to 14 the moment the driver started reporting the index. The general rule, and the one stage two has to hold to, is that selecting rows carries labels and producing rows makes them.

## What stage two actually moved

Stage two shipped in firepanda pull request #194 and the driver asks for the indexed shape here. The groupby section goes from 54 pass and 62 fail to 111 and 5, so 57 cases close in one change and it is the largest single move this suite has ever recorded.

The 5 that are left are worth naming, because none of them is an index. 3 are the Arrow reader refusing a dictionary encoded column, which is firepanda issue #159. 2 are the two key grouping, which pandas answers with a MultiIndex and firepanda has none, so `as_index` raises there rather than handing back one of the two levels, and those cases stay in the flat spelling and keep failing on the level count. That is the truthful report: the answer really does have the wrong shape, and pretending otherwise in the driver is the lie this document warned about in its second section.

So the estimate at the top of this document was 76 cases across two stages and the two stages closed 67 of them. The gap between those numbers is the 3 dictionary reader cases and the 2 MultiIndex cases, plus 4 basics cases that were failing for a second reason as well as the index. None of the difference is an index that did not get built.

The `as_index` default is the one thing here that does not match pandas and it is deliberate. pandas defaults it on. firepanda defaults it off, because turning it on by default would make a two key group by raise by default, and a shape that differs is a better answer than an operation that refuses. It flips when the MultiIndex lands, which is the remaining item and is the smaller half of firepanda issue #155.

Across basics, stats and groupby the failure count is 31, down from 98 when this document was written. 22 are basics, 4 are stats and 5 are groupby, and the index is no longer the largest cause of any of them.
