# The firepanda driver

firepanda is a Mojo library with no Python module yet, so a case cannot be handed the library the way a case is handed pandas. This directory is the way around that. `main.mojo` is a program that takes a case id and a corpus frame, runs the firepanda spelling of that case, prints one line of JSON saying what shape came back, and writes the answer as an Arrow IPC file. `fpcompat/driver.py` is the other end of it.

The protocol is specified in [docs/specs/03-harness.md](../../docs/specs/03-harness.md). Read that before changing anything here, because there is no type checker spanning a Mojo program and a Python one and the specification is the only thing keeping the two sides agreeing.

## Building

```
pixi run driver                              # a sibling firepanda checkout
drivers/firepanda/build.sh /path/to/firepanda # or somewhere else
```

The Mojo toolchain comes from the firepanda checkout and not from this repository, because two pins for one toolchain is how a driver ends up compiled against a library it does not match. The script writes `stamp.json` next to the binary with the firepanda version, the commit, whether that checkout was dirty, and the Mojo version. The harness reads it into every result file. firepanda has no version constant in Mojo, so the checkout is the only place that answer exists, and a conformance number that cannot say which firepanda produced it is not a number anybody can act on.

Neither the binary nor the stamp is committed.

## Running it by hand

```
drivers/firepanda/firepanda-compat-driver \
    --case=basics/head --frame=two --corpus=corpus --out=/tmp/answer.arrow
```

Then `python -m fpcompat.runner --engine firepanda --filter basics` for the whole thing.

## Adding a case

Find the case in `fpcompat/cases/`, read the expression, and write the firepanda spelling of the same thing in the `if` chain in `main()`. A case id with no entry reports `absent`, which the harness scores as unimplemented, which counts against the score exactly as hard as a failure does, so the work list is whatever `pixi run conformance` currently calls unimplemented.

The one rule, and everything else here follows from it: **this program writes down what firepanda does and never what pandas does.** It would be easy to make the numbers look better. `DataFrame.tail` in pandas keeps the original row labels, firepanda has no index at all, and five lines here could manufacture an index column and turn a failure into a pass. That would be a lie, and worse than a lie it would be an invisible one, since nothing downstream can tell a real pass from a manufactured one. The cases that need an index fail until firepanda has an index, and that failure is the point of measuring.

Use the method a firepanda user would call. A reduction goes through `DataFrame.agg` rather than through a loop written here, because the suite is measuring firepanda's methods and not this file's arithmetic. A driver that reimplements the operation it is testing scores itself.
