# 35. The forty six questions nobody was asking

The board has a whole level about failure. L4 asks which exception class comes out of an operation that cannot be done, because a library that computes the right answer and raises the wrong exception is not a drop in replacement: the code around it catches by type, and an `except KeyError` that stops firing is a process that dies.

There are 46 L4 cases and 51 runs of them. Every single one of them has been scored `unimplemented` since the level was written, and the reason was the harness.

## The driver cannot answer the question

`fpcompat.driver` has said so in its own docstring the whole time, which is the part that makes this embarrassing rather than surprising. Mojo has one `Error` type carrying one string. A raise in the driver comes back over the wire as a status and a message with no class attached to it, and there is nowhere in the protocol for a class to go. Guessing a pandas type name out of the message text was considered and refused, because it would have been an invisible invention: nothing downstream of `compare` could tell a guessed class from a real one, and a suite that invents results is worse than a suite that has none.

So the driver was never going to answer an L4 case, and it shows. There is not one `errors/` entry in the eleven hundred lines of `drivers/firepanda/main.mojo`. Nobody was ever going to write one.

The result was a level of the board reporting on the harness while looking like it was reporting on the library. `unimplemented` counts against the score exactly as hard as a failure does, per document 01, so nothing was being excused, but nothing was being learned either. Forty six questions were being asked every run and none of them was reaching firepanda.

## The module has the classes

Document 25 is the correction that let the two forms of the subject coexist, and its rule is the level: L0 and L1 go to the module because only reflection answers a reflection question, and everything else goes to the driver.

L4 belongs on the module's side of that line for the same kind of reason. An L4 case asks which class comes out, the module raises real Python classes, and `firepanda.errors` exists precisely so that a tagged message from Mojo becomes a `ColumnNotFoundError` that is a `KeyError` before a user ever sees it. The question and the instrument fit. One line changed.

## What it found

Fifty one runs that were `unimplemented` became 7 passing, 15 failing and 29 still unimplemented. The 29 are names firepanda does not have yet and are the same answer as before by a better route. The 22 that are now real results are the point.

The 7 that pass are the ones that were already right and nobody could see: a missing column is a `KeyError` with the name in it, a missing key in a groupby is a `KeyError`, `astype` on text that will not read is now a `ValueError` saying `invalid literal for int() with base 10`, and a comparison between two different lengths raises the way pandas raises.

The 15 failures come in three shapes and it is worth separating them, because they cost very different amounts to fix.

Six are the right class and the wrong words. `tz_convert` on a naive column says `tz_convert has nothing to convert from` where pandas says `tz-naive`, and a user searching a traceback for the phrase in the pandas documentation finds nothing. These are one line each and they are not judgement calls.

Six are the wrong class. Comparing unordered categoricals raises `InvalidArgumentError`, which is a `ValueError`, where pandas raises `TypeError`; a nonexistent local time raises `DTypeError` where pandas raises `ValueError`; `quantile` on a boolean column and an unknown interpolation method both raise `NotImplementedError`, which is the honest answer for a method that is not written and the wrong answer for a method that is written and is refusing an argument. Each of these is a decision about which kind a raise carries and the fix is a word.

Two are neither, and they are the interesting ones. Three of the categorical messages come out of the Arrow import layer saying `arrow: a dictionary encoded column arrives ...`, which means the operation never got as far as the categorical code at all. The message is not wrong about what happened, it is answering a different question than the one the caller asked, and no rewording fixes that.

## The one that cannot be fixed the obvious way

`errors/astype-null-to-int` fails like this:

```
raised firepanda.errors.IntCastingNaNError, which is not a
pandas.errors.IntCastingNaNError, so an except clause written against pandas does
not catch it
```

That reads like a harness bug and it is not one. firepanda has a class of that name because pandas has a class of that name, the two are unrelated classes, and they are unrelated because firepanda has no dependencies and is not going to import pandas in order to inherit from it.

The verdict is telling the truth. A user who writes `except pandas.errors.IntCastingNaNError` around a firepanda call is not caught, and there is no version of firepanda in which they are. What that user should write is `except ValueError`, which both libraries satisfy, and both `IntCastingNaNError` classes are `ValueError` subclasses on purpose so that it works.

The obvious move is to relax the rule: accept a subject class as a match when its name and its builtin ancestry both agree with the declared one. That would turn this failure into a pass and it would do the same for all 46 pandas error types at once, which is exactly why it is not being done in the same change that found it. A rule that decides what compatibility means for an exception class is a decision to make on its own, in writing, with the reason attached, and not a side effect of a routing fix. The failure stays on the board until then, where it is doing its job, which is being a question somebody has to answer.

Something similar had to be fixed to even read the failure. The verdict printed bare class names, so a mismatch between two classes of the same name printed `raised IntCastingNaNError, which is not a IntCastingNaNError`. `_qualified` in `fpcompat/compare.py` now puts the module in front of anything that is not a builtin, because a verdict that reads as a harness bug will be treated as one.

## The ratchet went up and that is the board working

The recorded failure floor moves from 8 to 20, which is the largest single jump this repository has recorded. Not one line of firepanda changed to cause it. Twenty two results that were being reported as absences are now being reported as what they are, and twelve of them are bugs.

Document 32 made this argument once already about a differential sweep that opened at a hundred and ninety six differences, and it is the same argument. A conformance number that only ever goes up is a number that has stopped measuring anything. The failure count going up because the harness started asking a question it had been skipping is the single healthiest thing a board can do, and the alternative, which is a level that reports `unimplemented` forever while looking like a measurement, is the thing this repository exists to refuse.

The L3 temporal floor also moves from 7 to 8, which is not related to any of this. It had not been recorded since the file was last touched and this is the run that noticed.
