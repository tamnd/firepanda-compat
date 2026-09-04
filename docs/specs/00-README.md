# The compat folder

Written 4 September 2026, against pandas 3.0.3, pyarrow 25.0.0 and firepanda 0.6.40.

## Why this exists

Document 06 in the parent folder is the pandas conformance checklist. It is 205 checkboxes over 444 distinct pandas names, it is the best short description of the goal that exists, and it is not a measurement of anything. A checkbox is ticked by a person who believes the thing works. The claim on the front of this project is stronger than that: `import firepanda as pd` and your program keeps running. Nobody believes that claim because a maintainer ticked a box, and nobody should.

So this folder specifies the thing that turns the claim into a number. `tamnd/firepanda-compat` is a separate repository that holds the pandas surface inventory, the conformance case registry, the frame corpus, the comparison rules, the divergence registry and the scoreboard. It runs every case against real pandas and against firepanda in separate processes, compares the answers under written rules, and publishes a pass rate per section that is allowed to be low.

The short version of the argument is that the parity checklist counts names and the product is behaviour. Here is the gap, measured this morning against the pandas that is actually installed:

| | pandas 3.0.3 | in the parity checklist |
|---|---|---|
| Public names across 21 namespaces | 1413 | 444 |
| Public callables | 1125 | not counted |
| Parameters on those callables | 3267 | not counted |

A single row of document 06 reads `- [ ] rolling with window, min_periods, center, closed, step (M6)`. That is one checkbox over five parameters whose interactions are the entire difficulty of the feature, and `closed` alone has four values that each change the answer. Ticking it is a judgement call. Running two hundred generated cases over it and comparing every answer to pandas is not.

## What is in here

| | |
|---|---|
| `01-what-100-percent-means.md` | the definition being measured, five levels of conformance, and the scoring rule |
| `02-the-surface.md` | the measured inventory namespace by namespace, and what document 06 does not mention at all |
| `03-harness.md` | the repository, the case registry, the engine adapters, and how firepanda is driven before it is importable |
| `04-corpus.md` | the frames every case runs on, how they are generated, and why they are committed |
| `05-comparison.md` | equality semantics, tolerances, dtype rules, and the oracle self test |
| `06-divergences.md` | the registry of things that are allowed to differ, and the rule that keeps it honest |
| `07-scoreboard.md` | the report, the site, the CI gate, and the rule against green washing |
| `08-m6.md` | the first tier plan, which is what milestone issue 8 becomes |
| `09-resources.md` | the operation level cost matrix, and the ten times faster on a tenth of the memory goal |
| `10-bench-and-compat.md` | why this is not in firepanda-bench, and what firepanda-bench gains from it |
| `11-milestones.md` | C0 to C5 for the compat repository, with exit criteria |

Read `01` first. It is the only one that says what the project is claiming, and every other document in the folder is machinery in service of it.

## The three repositories

`firepanda` is the library. It builds with a Mojo toolchain and nothing else, and that must stay true.

`firepanda-bench` is the performance comparison against pandas, Polars, DuckDB, cuDF and MojoFrame. Fifteen db-benchmark queries, twenty two TPC-H queries, four ingestion files. It answers how fast.

`firepanda-compat` is this. It answers whether the answer is right, over the whole API surface rather than over thirty seven queries. The two are separate because they fail for different reasons and on different schedules, because a conformance run is minutes and a benchmark run is hours, and because a benchmark result that regresses is a discussion while a conformance result that regresses is a bug.

## What this folder does not do

It does not implement anything in firepanda. Every case here is a test, and a failing case is an issue in the library repository, not a patch in this one.

It does not decide what firepanda should do when pandas is wrong. That decision goes in `06-divergences.md` and it is made by a person, once, in writing, with the reason attached.

It does not measure performance against Polars or DuckDB. That is firepanda-bench and it stays there. What it does measure is firepanda against pandas on the same operation, because that number is a compatibility fact as much as a performance one. A user whose program runs and takes four times longer has not been given a working library.
