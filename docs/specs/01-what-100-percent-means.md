# What 100 percent compatible means

The goal on the front of the project is that a pandas program keeps working when the import line changes. That sentence is not testable as written, so this document turns it into something that is.

## The five levels

Every pandas name is scored at one of five levels, and a name is at level N only if it passes every level below it.

**L0, the name resolves.** `firepanda.DataFrame.pivot_table` exists and is callable. This is the level a checkbox in document 06 actually measures, and on its own it is worth very little, because a name that raises `NotImplementedError` resolves.

**L1, the signature accepts what pandas accepts.** Same parameter names, same order, same defaults, same acceptance of positional and keyword forms. A user's call site is a signature, not a name, and `df.sort_values("a", kind="stable")` fails at the call site if `kind` was not implemented, no matter how good the sort is. L1 is checked mechanically by comparing our signature to `inspect.signature` of the pandas one, so it costs nothing to check and it catches the largest class of porting failure.

**L2, the default behaviour matches.** The call with only the required arguments returns what pandas returns, on every frame in the corpus, under the comparison rules in document 05. This is where most conformance work is and it is the level most projects stop at.

**L3, the parameter space matches.** Every enumerated parameter takes every one of its values, every boolean takes both, every numeric parameter takes a boundary value and an interior one, and the combinations that interact are enumerated explicitly rather than sampled. `rolling` has `window`, `min_periods`, `center`, `closed` and `step`, and `closed` crossed with `center` crossed with a window longer than the frame is where the answers stop matching. A name at L3 is a name a user can call the way the pandas documentation says they can.

**L4, the failures match.** Bad input raises the same exception type from `pandas.errors` or the builtins, and the message names the same column, the same dtype or the same value. pandas has 46 public exception and warning types and programs catch them. `MergeError`, `IntCastingNaNError`, `SpecificationError` and `OutOfBoundsDatetime` all appear in real error handling code. A library that raises a bare `Error` where pandas raises `MergeError` has broken a program that was handling the failure correctly.

L4 also covers the warnings, because pandas 3.0 signals deprecations through `Pandas4Warning` and `PandasChangeWarning`, and a test suite that runs with warnings as errors will fail on the wrong warning as loudly as on the wrong exception.

## The number

The published number is the fraction of pandas callables at L3 or better, per namespace, with L4 reported separately because the error surface is a different job with a different completion date.

There are 1125 public callables in pandas 3.0.3 across the 21 namespaces listed in document 02. That is the denominator, and it does not shrink. Anything firepanda deliberately does not do is a registered divergence under document 06, and a divergence is displayed in the score line rather than removed from it:

```
GroupBy   58 callables   L3 41   L2 9   L1 4   L0 2   divergent 2
```

Reading that line takes a second and it cannot be gamed. A score computed over a denominator that the scorer chooses is an advertisement.

## Why signature conformance is not pedantry

pandas parameters carry semantics that no reasonable person would guess. `dropna=True` on `groupby` is a filter on the key, not on the values. `observed=True` changes which groups exist for a categorical key. `na_position` on `sort_values` is independent of `ascending`, which is not what a naive implementation does. `min_periods=0` on `rolling` is not the same as `min_periods=1`. `keep="last"` on `drop_duplicates` and on `nlargest` mean different things.

Every one of these is a parameter with a default, which means a user's program depends on it without the user having typed it. L1 catches the missing spelling and L3 catches the wrong meaning, and only the two together mean the program still runs.

## What is deliberately not in the number

Three things, and each is a decision rather than a shortfall.

**Repr equality.** Frame rendering matches pandas in shape and not byte for byte, and no case compares `repr` output except the handful in the display section that pin column truncation and the null marker. Chasing byte equality on repr is a large amount of work that no program depends on, and where a program does depend on it, it is depending on something pandas itself changes between minor releases.

**The plotting and styling surface.** `plot`, `hist`, `boxplot` and `style` are registered divergences under document 06 in the parent folder. They stay in the denominator and they are reported as divergent forever, which is the honest presentation of a decision not to build a plotting stack.

**Timing.** Whether an operation is fast is document 09, and it is scored on its own axis. A correct answer that arrives ten times slower than pandas is L3 and is a performance bug. A wrong answer that arrives instantly is L0 with a failure attached. Mixing the two into one score hides both.

## The rule that makes the rest of it work

**A case that has never run is not a pass.** The runner reports four outcomes and there are exactly four: pass, fail, divergent, and unimplemented. There is no skip. A case that cannot run because the driver has no entry for it is unimplemented, which counts against the score exactly as hard as a failure does, and the report prints the count in the summary line rather than in a footnote.

This is the whole discipline. Every conformance suite that ever lied did it by turning failures into skips, and the way to not do that is to have no skip outcome at all.
