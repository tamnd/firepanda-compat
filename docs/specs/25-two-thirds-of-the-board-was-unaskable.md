# Two thirds of the board was unaskable

## The number that was not the failures

The board had four failing records and 3441 unimplemented runs. Every document in this folder since the twentieth has been about the failures, because a failure is a wrong answer and a wrong answer is interesting. That was the wrong place to be looking. Four is not the gap. 3441 out of 4081 is the gap, and it had never been bucketed.

So it got the treatment document 20 gave the failures, which is to sort by reason before reading any of them as work.

## All 3441 said the same thing

Every one of them: `Absent: the firepanda driver has no entry for X`. One message, no distribution to read, which is why nobody had bucketed it before. The interesting split is not in the message, it is in the sections.

| section | absent runs |
| --- | --- |
| resolution | 1413 |
| signature | 1034 |
| strings | 228 |
| basics | 132 |
| windows | 117 |
| indexing | 104 |
| everything else | 413 |

`resolution` and `signature` are 2447 of the 3441, which is 71 per cent of what is missing and 60 per cent of the whole board. They are also the two sections nobody writes by hand. They are generated, one case per public pandas name, from the committed inventory: `resolution` asks whether the name resolves and whether you have to call it, `signature` asks whether its parameters match.

## Nobody was ever going to write 2447 driver entries

And it would not have helped if they had. The driver is a Mojo program that takes a case id and a corpus directory, runs the firepanda spelling of that case and writes the answer as an Arrow file. Ask it `resolution/dataframe.pivot` and there is nothing for it to do. "Does `DataFrame.pivot` exist" is not a question with an answer to compute, it is a question about a module, and the only thing that can answer a question about a module is reflection.

So 2447 runs on this board were not absent because the work had not been done. They were absent because the instrument could not be pointed at them, and they were going to stay absent no matter how much firepanda got written.

## The engine had two forms and used one

`FirepandaEngine` has held both forms since it was written. The driver, for a library with no Python bindings, and an importable `firepanda` module, for direct comparison in one interpreter. The docstring called the driver a scaffold, meant to be deleted the day the module arrived.

The constructor is an `elif`. If the module imports, the driver is never even constructed, and `out_of_process` is a property with no case in it, read once per run:

```python
if firepanda is not None:
    self._module = firepanda
elif DRIVER.exists():
    self._driver = DRIVER
```

That is a trap and it was one line from being sprung. firepanda has had a working Python extension for a while, and `tools/build_extension.sh` builds it and 177 tests run against it. The only reason the board was not already broken is that nothing put the built extension on this repository's path. The first person to do that would have seen every hand written case leave the driver and go to a binding that answers far fewer of them, the board would have fallen off a cliff, and there would have been no reason at all to suspect the harness.

## They are two instruments

The framing was wrong rather than the code. The driver is not a worse module. They answer different questions and neither one covers the other:

- The driver runs a case the library can do and the binding cannot yet spell. It is where the firepanda spelling of every hand written case lives.
- The module answers a question about the API itself, which the driver structurally cannot.

So the engine holds both at once and routes per case. The routing rule is the level, not a list of section names:

| level | question | instrument |
| --- | --- | --- |
| L0 | does this name resolve, and do you have to call it | module |
| L1 | do its parameters match | module |
| L2, L3, L4 | is the answer right | driver |

That is a rule with a reason in it. L0 and L1 are questions about the API and reflection is the only thing that answers them. L2 and above are questions about data and the driver is what runs those today. The rule stays correct when a section is added, which a list of section names would not. It happens to partition exactly: all 1413 `resolution` cases are L0, all 1034 `signature` cases are L1, and the five L0 cases in `divergences` are the plotting accessors, which are the same reflection question wearing a different section.

The mechanical change is small. The constructor stops being an `elif`. `out_of_process` becomes `out_of_process_for(case)`. The runner asks per case through a helper that falls back to the old property, so an engine with one form is unaffected and the pandas engine is untouched.

## The module has to be the one the driver was built from

An engine that imported whatever `firepanda` happened to be installed and ran the driver built from a branch would report one version and measure two. So `drivers/firepanda/build.sh` stages the extension beside the driver, in the same build, from the same checkout, and the engine looks there first. An ambient `firepanda` is accepted only when nothing is staged.

A failed extension build is not a failed driver build. A firepanda too old to have one, a toolchain that cannot link one today and a machine with no firepanda at all all mean the same thing to the board, which is that no reflection question has been answered, and that reads as unimplemented rather than as a broken run.

## What it found

The board goes from 629 passing runs to 805. Nothing in firepanda changed. 176 runs that were reported as absent were being answered correctly by a library that had already implemented them, and the suite had no way to ask.

It also found twelve failures that had never been visible, which is the part worth having. Seven of them were the harness and five of them are real.

## The seven that were the harness

Seven `resolution` cases came back `property, expected attribute`, on `DataFrame.columns`, `DataFrame.index`, `Series.index`, `Index.dtype`, `Index.hasnans`, `Index.inferred_type` and `Index.is_unique`.

The kind classifier had three values and split a `property` from an ordinary attribute by asking `isinstance(descriptor, property)`. Measured against pandas:

| name | descriptor | `isinstance(..., property)` |
| --- | --- | --- |
| `DataFrame.columns` | `AxisProperty` | False |
| `DataFrame.index` | `AxisProperty` | False |
| `Index.dtype` | `CachedProperty` | False |
| `DataFrame.shape` | `property` | True |

So pandas puts three of those on one side and one on the other, for reasons entirely internal to pandas, and firepanda uses a plain `property` throughout and lands on the other side of a line that describes how a descriptor was written. No user program can see the difference. `df.columns` and `df.shape` are read identically.

The distinction the module set out to catch is the one in its own docstring, that `frame.shape` and `frame.shape()` are different programs. That is callable against not callable, and it is worth having. Anything finer is not. The kind is now two values, `callable` and `value`, and the seven false failures are gone.

This is document 23's rule again in a different section: a case has to ask a question with one answer, and a case comparing something no user program can observe can fail without anybody being wrong. It is worth noticing that the first two documents written after the board could see a new dimension both found a case measuring the wrong thing. A generated case is not exempt from that, and being generated means the mistake arrives 1413 times at once.

## The five that are real

| case | firepanda has | pandas also has |
| --- | --- | --- |
| `signature/index.rename` | `name` | `inplace` |
| `signature/index.slice_locs` | `start`, `end` | `step` |
| `signature/index.take` | `indices` | `axis`, `allow_fill`, `fill_value` |
| `signature/index.unique` | nothing | `level` |
| `signature/pandas.read_csv` | `filepath_or_buffer` | the rest of it |

These are real gaps in the Python binding and they are the first ones this suite has ever been able to see, because a driver entry is hand written against a case rather than against a signature and a missing keyword argument is invisible to it.

They are not fixed here, and the reason is worth stating rather than deferring quietly. Adding a parameter name to match a signature, with nothing behind it, would turn five honest failures into five passes and one lie, and this board's only value is that a number on it is true. What an unimplemented keyword should do when somebody passes it is the error model question again, and it gets decided once for all of them rather than five times in a row here.

## The ratchet went up and that is the right answer

The failure floor goes from 3 callables to 8. A ratchet rising normally means a regression, and this is the case it exists to be overridden for: the board did not get worse, it got 176 runs better, and the failure count rose only because 2452 runs that could never be asked became askable and five of them found something.

A ratchet that could not record this would be a ratchet that discouraged pointing the instrument at new ground, which is the opposite of what it is for. The rule it keeps is that a number does not get worse silently. This one got worse loudly, in a document, with the five names written down.

## What is left

3260 runs are still unimplemented, and after this the shape of that number is different. 1309 of the 1413 names pandas has do not resolve in firepanda and 957 of the 1034 signatures are not there to compare, and those are now a work list with names on it rather than a wall. That list is the honest measure of how far from pandas this library is, and until today the suite could not produce it.
