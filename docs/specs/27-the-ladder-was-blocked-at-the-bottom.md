# 27. The ladder was blocked at the bottom

The previous document argued that a large part of the unimplemented count was a door problem rather than a library one, and it ended by saying the transformation family was the same shape of work and should be next. It was. This document is about what happened when the same fix was applied a second time, because the second time produced a measurement the first one did not, and the measurement says something about how this board should be read.

## The number

Fourteen names reached the Python extension: `dropna`, `isna`, `notna`, `ffill`, `bfill`, `shift`, `diff`, `pct_change`, `cumsum`, `cumprod`, `cummax`, `cummin` and the two `is_monotonic` properties on a series, the same list minus the monotonic pair on a frame, and a frame `dropna` that removes rows.

Passing runs went from 851 to 901. Unimplemented went from 3214 to 3164. The failure count and the divergence count did not move.

Now the part worth stopping on. Of those fifty runs, twenty six were names resolving and twenty four were signatures comparing. That is all fifty. Not one behaviour case changed its outcome, in either direction, on any of the fourteen names.

## Why not one behaviour case moved

Because they were already passing. `basics/shift` was passing over six frames before this work started. `basics/shift-negative` was passing over six. `basics/cumsum` was passing over three, `stats/cumsum-tall` and `stats/cumsum-float-edges` were passing, `basics/frame-dropna` was passing over two. Every one of them was passing while `s.shift` did not exist as far as a Python program was concerned.

This is not a bug in the board. It is the board having two instruments and being honest about which one answered which question. The L0 and L1 cases ask a Python module whether a name is there and whether its signature agrees with the pandas one, so they go to the importable `firepanda` package. The L2 and L3 cases ask what a computation answers, so they go to the compiled Mojo driver, which is where the corpus is loaded and the operations are run. Both are firepanda. They are not the same door, and until this change one of them was locked.

So the board was in a position that reads as a contradiction and is not: `Series.shift` had thirteen passing behaviour cases and could not be called from Python. A conformance score has to be able to represent that, because it is the real state of the software.

## What the ladder does with it, and why that is right

A callable's attained level is the highest rung with a passing case and nothing failing or missing at or below it. `Series.shift` therefore attained nothing at all. Thirteen passing behaviour cases counted for zero, because the name did not resolve.

That rule was written before any of this happened and it earned itself here. A name that computes the right answer and cannot be called is not compatible with pandas in any sense a user cares about, and a score that awarded it partial credit would be measuring the library's internals rather than what a pandas program can do. The cumulative reading is the whole value of the ladder: it refuses to let a strong middle hide a missing bottom.

The visible consequence is small and precise. `Series.shift` is now the third callable on the entire board with a complete L0 to L3 ladder, after `DataFrame.tail` and `Series.std`, and it got there without a single line of behaviour changing. The `stats` L3 floor in `ratchet.json` moves from 1 to 2. Three fully conformant callables out of 1125 is not a good number, and it is the honest one, and the reason it is three rather than four or forty is worth reading rather than rounding.

## How much more of the board is in the same position

This is now a question the board can answer on its own rather than one that needs somebody to go and read Mojo, which is the thing document 26 said a black box comparison could not do. Cross reference the callables that have a passing L2 or L3 case against the callables that are failing or missing at L0 or L1, and thirty nine names come out.

Sixteen of them are the groupby surface: `DataFrame.groupby`, `Series.groupby` and the fourteen aggregations behind them, every one of which has passing behaviour cases and no Python door. Twelve are the `.dt` accessor and its methods, which is the family document 14 costed at 108 runs. Four are sorting and renaming: `DataFrame.sort_values`, `Series.sort_values`, `Series.argsort`, `DataFrame.rename`. The rest are `Series.corr`, `Series.cov`, `Series.astype`, `DataFrame.copy`, `DataFrame.drop`, `pandas.Timedelta` and `pandas.to_timedelta`.

Thirty nine callables with proven behaviour and no way to call them. That is the size of the next few slices, and it is a measurement rather than an estimate, which is the distinction document 26 was arguing for and could only make by hand.

## The doors, and the rule for how many there should be

One boundary method took all twelve column returning transformations. A transformation crosses as a word beside a whole number, because `shift`, `diff` and `pct_change` take a `periods`, `ffill` and `bfill` take a `limit`, both are always a whole number, and no transformation in the list takes two of them. The seven that take neither are handed a zero and ignore it.

There are three doors and not one, and the rule that decides is the shape of the answer rather than how similar the words are. Twelve hand back a column and share `transform`. The two monotonic questions hand back a bool. A frame `dropna` removes rows rather than transforming columns, so no per column loop produces it and it needs its own. The boundary refuses a frame `dropna` sent through `transform` rather than trusting the layer above to keep them apart, because two things separated by a convention that nothing checks are separated by a comment.

This is the same rule the reductions arrived at from the other side. A reduction folds a column to one number, a transformation gives back a column, and they are different doors for that reason and not because the words are different.

## Two decisions a reader should be able to argue with

`limit=0` is refused. The core reads `fill_forward(limit=0)` as no limit and pandas reads `ffill(limit=None)` as no limit, so the Python layer turns a `None` into a zero on the way in and a caller never has to know the two spellings exist. A caller who types a literal zero, though, means zero, and reading it as its exact opposite would be a silent wrong answer of the worst kind. So it raises with the message pandas raises.

pandas defaults `shift(fill_value=)`, `dropna(how=)` and `dropna(thresh=)` to a private sentinel, `lib.no_default`, and firepanda now has a `NO_DEFAULT` of its own for the same three places. Not the pandas object, because importing a private name out of pandas to spell a default would make pandas a hard dependency of a library that does not otherwise need one, and this repository is the place that would notice. Not `None`, because `None` is a value a caller can legitimately pass to `fill_value` and has to stay distinguishable from not passing anything. The L1 case compares the ordered parameter names and kinds and is satisfied. The parity test inside firepanda compares defaults too, and it now knows the two sentinels are the same idea, with the reason written where the test is rather than in a commit message.

## Refusals, again as policy

Eleven declared arguments raise by name with the reason in the message: `axis=1` on a frame transformation, `skipna=False` on a scan, `numeric_only`, `limit_area`, `freq`, `suffix`, `fill_value`, `how="all"`, `thresh`, `inplace` and `ignore_index`.

Document 26 made this argument for the reductions and it is the same argument. A declared parameter that is accepted and ignored is correct at its default, which is where every quick check uses it, and wrong everywhere else, which is where a real program uses it. There is a test per refusal so none of them can be dropped later by somebody tidying up. The signature still declares every one of them, because that is what the L1 board compares and what a caller reading the help should see.

## What this argues for next

Take the thirty nine. Groupby is sixteen of them and is the largest single block of proven behaviour with no door anywhere on the board, so it is next. The `.dt` accessor is twelve and has a document already. Sorting and renaming is four and is small.

The wider point is a refinement of the one document 26 made. That document said to go and read what is behind an unimplemented count before treating it as a work estimate. This one says that once enough of the ladder is populated, the board will tell you which part of the count is a door problem without anybody reading anything, because a callable that passes at L2 and fails at L0 has already declared itself. The instrument that could not tell a missing implementation from a missing door can tell them apart as soon as the behaviour cases exist to speak for the implementation. That is worth knowing before the next slice, and it is the reason this document has a number in it that document 26 could not have produced.
