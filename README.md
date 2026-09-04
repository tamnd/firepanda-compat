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
pixi run budget        # the operation level cost matrix
pixi run test          # pytest over the harness itself
```

## The three repositories

[`firepanda`](https://github.com/tamnd/firepanda) is the library, and it builds with a Mojo toolchain and nothing else.

[`firepanda-bench`](https://github.com/tamnd/firepanda-bench) is the performance comparison against pandas, Polars, DuckDB, cuDF and MojoFrame. It answers how fast.

This is the third, and it answers whether the answer is right, across the whole API rather than across 37 queries. It is separate from the bench repository because a conformance run needs pandas and pyarrow and must run in minutes on every pull request, while a bench run needs Docker, cuDF and 50 GB of generated data. Putting them together means the conformance suite runs nightly, which is the outcome that makes it useless.
