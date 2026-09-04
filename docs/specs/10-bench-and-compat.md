# firepanda-bench and firepanda-compat

Two repositories that both run pandas and both produce a number. Here is the line between them, and what each gains from the other.

## The split

`firepanda-bench` answers **how fast, against everyone**. Four competing engines, 37 published queries, two data sizes, two io modes, two machines, hours per run, results that are a discussion when they move.

`firepanda-compat` answers **is the answer right, across the whole API**. One competing implementation, which is pandas, because pandas is the specification. Thousands of small cases, minutes per run, results that are a bug when they move.

The rule for deciding where something goes: if it needs Polars, DuckDB, cuDF or a gigabyte of data, it is bench. If it needs pandas and a nasty little frame, it is compat.

## What compat takes from bench

**The result file discipline.** Version pins for every engine, machine identity, io mode, and a validator that refuses a result file missing any of them. That machinery in `tools/validate_results.py` is the reason bench numbers are quotable, and compat copies it rather than inventing a second version.

**One process per engine.** Non negotiable for memory measurement and bench already learned it.

**The splitmix64 generator.** Same constants, same seed, so that a Mojo driver can produce an identical frame without reading a file, and so that a person reading both repositories does not have to check whether the difference is meaningful.

**The publish what we lose rule.** Stated in both repositories, in the same words, because it is the same rule.

## What bench gains from compat

Four changes, and they are the enhancements this work adds to that repository.

**A cross engine answer check that is not a fingerprint.** Bench compares answers by row count, numeric column sums and an order independent FNV-1a digest of the text columns. That was the right call for a suite where a Mojo driver must compute the check without an Arrow sort, and it is a weak check: two answers with the same sums and the same multiset of strings can differ in which row each string is on. Compat's comparison layer is exact and it now exists, so bench gains an optional `--verify exact` mode that writes each engine's answer to Arrow IPC and hands both to `fpcompat.compare`. It is too slow for the timed path and it is not on the timed path. It runs once per query per release, which is enough to catch a fingerprint collision that would otherwise be invisible forever.

**Peak memory as a first class published number.** Bench already records peak resident set per run and the report treats it as a secondary column. The tenth of the resources goal makes it a headline, so the report grows a memory table beside the time table and the site plots both. Nothing new is measured, the data has been in the result files since the first run, and it is simply not published today.

**The operation level matrix linked from the front page.** Bench answers queries and compat answers operations, and a reader who sees a query where firepanda loses wants to know which operation inside it lost. The compat cost matrix from document 09 is that answer, so bench's report links to it per query, listing the operations that query is made of.

**A firepanda column that says why it cannot run something, from the compat data.** Bench's TPC-H table today is 22 explicit refusals, and each one is a string a person wrote. The compat scoreboard already knows which callables are unimplemented, so a refusal can name the specific missing operation rather than saying "not supported yet", and it stops being a string a person has to remember to update.

## What stays separate forever

**Compat never installs Polars, DuckDB or cuDF.** The oracle is pandas. Adding a second oracle doubles the divergence registry and answers a question nobody asked, since firepanda does not claim to be Polars compatible.

**Bench never gates on conformance.** A benchmark run that finds a wrong answer reports a disagreement and stops timing that query, which is what it does today. It does not become a correctness suite, because a suite that is both is slow enough that it runs neither often enough.

**Two result formats.** They look similar and they are not the same, and unifying them would mean one schema serving two purposes and being awkward at both. What is shared is the validator's discipline, not the schema.

## The third repository question

Somebody will ask why compat is not a directory inside bench, so the answer is written down.

A conformance run needs pandas, pyarrow and nothing else, and it needs to run in minutes on every pull request to the library. A bench environment needs Polars, DuckDB, cuDF on Linux, Docker and 50 GB of generated data. Putting them together means every conformance run in CI resolves a dependency set that includes cuDF, and the practical result is that the conformance suite runs nightly instead of per commit, which is exactly the outcome that makes it useless.

The parent folder's document 00 said two repositories and put the conformance suite in bench. That was written before the surface was counted and before the environment for a bench run had grown a GPU feature. Three repositories is the correct number and this is the change of mind, recorded here rather than quietly implemented.
