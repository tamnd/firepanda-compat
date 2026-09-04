<h1>firepanda-compat</h1>

<p>
  <a href="https://github.com/tamnd/firepanda-compat/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/tamnd/firepanda-compat/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg"></a>
  <a href="docs/specs"><img alt="Specification" src="https://img.shields.io/badge/spec-12%20documents-informational"></a>
  <img alt="Status" src="https://img.shields.io/badge/status-C0%20instrument-orange">
</p>

The pandas conformance suite for [firepanda](https://github.com/tamnd/firepanda). It runs the pandas API against pandas and against firepanda, compares the answers under written rules, and publishes a pass rate that is allowed to be low.

## Why this exists

The claim on the front of firepanda is that a pandas program keeps working when the import line changes. The parity checklist in the library repository is 205 checkboxes over 444 pandas names, and a checkbox is ticked by a person who believes the thing works.

Here is what that checklist is measuring against, counted this morning by `pixi run surface` against the pandas that is actually installed rather than from the documentation:

| | pandas 3.0.3 |
|---|---|
| Public names across 21 namespaces | 1413 |
| Public callables | 1125 |
| Parameters on those callables | 3267 |
| Exception and warning types in `pandas.errors` | 46 |
| Option keys under `describe_option` | 70 |

One line of that checklist reads `rolling with window, min_periods, center, closed, step`. That is one checkbox over five parameters whose interactions are the entire difficulty of the feature. Ticking it is a judgement call. Running two hundred generated cases over it and comparing every answer to pandas is not.

## The five levels

A name is scored at one of five levels and reaches a level only by passing every level below it.

| | |
|---|---|
| **L0** | the name resolves |
| **L1** | the signature accepts what pandas accepts, checked against `inspect.signature` |
| **L2** | the default call returns what pandas returns, on every frame in the corpus |
| **L3** | every parameter takes every one of its values, and the combinations that interact are enumerated |
| **L4** | bad input raises the same `pandas.errors` type with a message naming the same column, dtype or value |

The published number is the fraction of the 1125 callables at L3 or better, per section. Anything firepanda deliberately does not do is a registered divergence, and a divergence is displayed in the score line rather than removed from the denominator.

## The rules

**There is no skip outcome.** A case that has never run is not a pass. The runner reports pass, fail, divergent and unimplemented, and there is no fifth. Every conformance suite that ever lied did it by turning failures into skips.

**The denominator is the pandas surface, not our case list.** A suite that reports a pass rate over its own cases is reporting how good it is at writing cases it passes.

**A registered divergence must diverge.** The runner does not skip a case in the divergence registry, it runs it and requires the declared outcome. The day somebody implements the missing behaviour, the registry tells them to come and delete the entry.

**Publish the run that lost.** Same rule as [firepanda-bench](https://github.com/tamnd/firepanda-bench). A conformance number that got worse is published as prominently as one that got better.

**The harness is checked against itself first.** `pixi run oracle` runs the whole registry with pandas on both sides and must be a perfect score. Anything else means the harness is wrong, and no result from it is publishable until it is fixed. The first version of any normalizer has a bug in the index handling, and without this every one of those bugs is published as a firepanda failure.

## Status

C0, the instrument. The specification is 12 documents in [`docs/specs/`](docs/specs) and the tools land one pull request at a time behind it. There is no conformance number yet, and the first one published will be a low number, because a project that waits until its score is respectable before publishing one has learned to hide the number.

## Layout

| | |
|---|---|
| `fpcompat/` | the harness: surface, corpus, comparison, cases, runner, report |
| `surface/` | the committed inventory of the pandas surface, one file per pandas version |
| `corpus/` | the manifest describing the frames, digested; the frames themselves are generated |
| `baselines/` | one committed cost matrix baseline per machine, which the budget gate reads |
| `drivers/firepanda/` | the Mojo side, until firepanda is importable from Python at M3 |
| `results/` | one JSON per run, not committed |
| `docs/specs/` | the specification, mirrored from the author's notes |
| `tests/` | pytest over the harness, which is not the same thing as running the harness |

## Running it

```
pixi run surface       # rewrite the inventory from the installed pandas
pixi run corpus        # regenerate the frames and verify the manifest
pixi run oracle        # pandas against pandas, must be perfect
pixi run conformance   # pandas against firepanda
pixi run report        # the scoreboard
pixi run coverage      # which pandas names and parameters no case touches
pixi run site          # the three pages, into a gitignored directory
pixi run ratchet       # fail if a section went backwards since the recorded floor
pixi run test          # pytest over the harness itself
```

The cost matrix is its own set of commands, because it runs on its own corpus:

```
pixi run budget-corpus   # build the budget frames and reconcile the manifest
pixi run budget          # measure every operation, one process each
pixi run budget-matrix   # the table
pixi run budget-baseline # record this machine's floor from the last sweep
pixi run budget-gate     # fail if a row got 10% slower or heavier than that floor
```

## What the cost matrix is for

The goal is ten times the speed on a tenth of the resources. firepanda-bench checks that on 37 queries, which means it is not checked on `str.extract`, on `reindex`, or on any of the other thousand callables a real program calls. This suite already calls every operation on known inputs with the answers already verified, so `pixi run budget` puts a timer and a memory high water mark on the same operations and produces a row per operation instead of a row per query. Bench answers whether a query is fast. This answers which operation is slow, which is the question you need answered before you can fix anything.

53 operations, 20 of them chains like filter then group or merge then aggregate, because peak memory in pandas is dominated by intermediates and a single reduction cannot use much less memory than its input however good the engine is. One process per engine per operation, since a peak resident set is a property of a process. Seven repeats with the median published and the interquartile range beside it. The answer is consumed before the timer stops, which matters here because firepanda is lazy underneath after M4 and every row would otherwise read as instant.

The budget corpus is not the correctness corpus. That one is 64 rows and mean by design, and timing a call on 64 rows measures interpreter overhead. This one is the same generator, the same constants and the same seed at one million rows, with no edge value placement. It is generated rather than committed and `corpus/budget-manifest.json` is what makes a change to the inputs show up as a diff.

The rows we lose go in the same table as the rows we win, with no separate section and no footnote. A matrix where firepanda wins every row has either been curated or is measuring the wrong thing, and the first person to notice will be somebody deciding whether to trust the project. A row below one is a performance bug with a name and an input size, and the table is where it gets its name.

## What the scoreboard prints

Two lines and a table. The first line is the level counts, the second is everything that keeps the first one honest, and the table is the same thing per section, worst first.

```
firepanda 0.1.0 vs pandas 3.0.3   L3 179/1125 (15.9%)   L2 326   L1 1034   L0 1125
divergent 0   unimplemented 0   fail 0   untested 0   parameters 9.1%   cases 3138 in 167.6s
```

The last two numbers on the second line are there because a pass rate on its own is easy to make look good. `untested` counts the pandas callables no case in the suite has ever named, and `parameters` is the share of the 3267 pandas parameters that any case has exercised. A high L3 over a low parameter coverage is a suite that is not finished, so neither number is quotable without the other, and `pixi run coverage` prints the individual parameters nobody has touched as a work list.

Three rules the report will not bend. The denominator is the pandas surface and not our case list, because a suite that reports a pass rate over its own cases is reporting how good it is at writing cases it passes. A divergence gets its own column and is never folded into either the passing or the failing count. And the run that lost still gets printed, since the only thing that fails a build here is the ratchet and an imperfect oracle.

## The three repositories

[`firepanda`](https://github.com/tamnd/firepanda) is the library, and it builds with a Mojo toolchain and nothing else.

[`firepanda-bench`](https://github.com/tamnd/firepanda-bench) is the performance comparison against pandas, Polars, DuckDB, cuDF and MojoFrame. It answers how fast.

This is the third, and it answers whether the answer is right, across the whole API rather than across 37 queries. It is separate from the bench repository because a conformance run needs pandas and pyarrow and must run in minutes on every pull request, while a bench run needs Docker, cuDF and 50 GB of generated data. Putting them together means the conformance suite runs nightly, which is the outcome that makes it useless.
