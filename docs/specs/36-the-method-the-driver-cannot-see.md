# 36. The method the driver cannot see

Document 25 settled how the two forms of the subject split the board, and the rule it settled on is the level. L0 and L1 ask whether a name resolves and whether its signature matches, which are questions about a module, so they go to the module. Document 35 moved L4 across for the same kind of reason, since the class an exception has is not something a message can carry. Everything else asks whether an answer is right, which is a question about data, and data is what the driver produces.

That rule has a reason in it rather than a lookup table behind it, which is why it survived two sections being added without anybody touching it. This document is about the one case it gets wrong.

## Three methods that arrived with nowhere to be measured

firepanda grew `DataFrame.filter`, `DataFrame.select_dtypes` and `DataFrame.truncate`. All three name a set of columns by a rule instead of by a list, and all three work out a list of labels and hand it to `select`. There are five cases waiting for them in `fpcompat/cases/indexing.py`, seven runs between them, and every one of those runs is L3, because each of the five names a parameter and a rule you have to name a parameter to reach is a parameter space question.

L3 goes to the driver, as does L2. The driver is `drivers/firepanda/main.mojo`, a Mojo binary that takes a case id, runs the firepanda spelling of that case against the core, and writes the answer out as Arrow. To arm `indexing/select-dtypes` the driver would need a branch that answers it, and the branch would have to do what `select_dtypes` does.

It cannot, and the interesting part is why not.

## The rules are not in the part of firepanda the driver can call

`select_dtypes` matches a frame's column types against numpy's type tree. In that tree a duration column is a signed integer, because `np.timedelta64` subclasses `np.signedinteger`, and a timestamp column is not a number, because `np.datetime64` does not subclass `np.number`. A bare `int` means int32 and int64 only while the abstract `integer` also means the unsigned widths, so the builtin is narrower than the class, which reads backwards and is what every caller's pandas does. And `include` and `exclude` are checked for overlap on the words the caller wrote rather than on the types those words expand to, so `include="number", exclude="floating"` is a legal call.

None of that is a dataframe operation. It is a pandas compatibility rule, four tables and three small functions of it, and it lives in `python/firepanda/_pandas.py` because that is where firepanda keeps the things it does in order to be pandas rather than the things it does in order to be a dataframe. The core has no opinion about numpy's type hierarchy and should not grow one.

So a driver branch for `indexing/select-dtypes` would have to carry a second copy of that tree, written in Mojo, next to the first one. `drivers/firepanda/README.md` forbids it in one sentence: the driver writes down what firepanda does and never what pandas does, and a driver that reimplements the operation it is testing is scoring itself. A second copy would be worse than that, because the two copies would drift, and the run that found the drift would report it as a firepanda bug.

`filter` and `truncate` are the same shape for smaller reasons. `filter` has a regex rule that is Python's `re` module, and `truncate` reads `Index.slice_indexer`, which is a Python layer method.

## The escape is per case and it has to say why

A case can now declare `in_process=True`, which says that its answer can only be reached through the subject's own module. `FirepandaEngine.out_of_process_for` reads it beside the level, an engine with no driver ignores it because it was going to run everything in process anyway, and `describe()` carries it into the result file.

Three things about the shape of that.

It is a property of the case and not of the section or of the name. `DataFrame.filter` could perfectly well have a case that the driver answers, and a rule that took a whole section away from the driver the moment one method in it needed the module would be the same mistake document 25 corrected, where an importable firepanda took the whole board away from the driver at once.

It requires a note, checked at declaration time and fatal like every other check in `case()`. Nothing in a case expression shows why one case is measured differently from every other case at its level. Without the note, the next reader has to work out from scratch whether the method really is unreachable from the driver or whether somebody wrote the flag because the driver branch was failing, and those two look identical from the outside. The note makes the second one something a person has to write down a false sentence to do.

It is serialized. A reader looking at a passing L3 case in a result file has no other way to know the driver never saw it, and that is precisely the thing they would want to know before trusting the outcome. The declaration belongs where the outcome is.

## What it does not excuse

An `in_process` case still needs an importable firepanda. When there is none, `FirepandaEngine.module()` raises `EngineUnavailable` and the runner scores the case `unimplemented`, exactly as it does for every L0 and L1 case on a machine with no staged extension. The flag moves a case to the other instrument, it does not give it a second chance.

And it is not a way around a driver branch that is merely tedious to write. The test that keeps that honest is the note, read by a person at review time, which is a weaker gate than a check and is the only kind of gate available for a question about intent.

## The cost

Seven runs come onto the board from this, which is small. The reason it is worth its own document is that the routing rule was stated as being about levels and is now about levels with one exception, and an exception to a rule with a reason in it needs its own reason written down beside it or the rule stops being a rule.
