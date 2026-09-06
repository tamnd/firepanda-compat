# The harness

`github.com/tamnd/firepanda-compat`. Python, because pandas is Python and the reference implementation has to be the real thing running in a real interpreter. The library it tests is Mojo, and how that gap is crossed is the only structurally interesting decision in this document.

## The layout

```
fpcompat/               the harness, importable, no side effects at import
  surface.py            walks pandas, writes the inventory
  corpus.py             generates the frames, deterministically
  cases/                the case registry, one module per parity section
    strings.py  reshape.py  windows.py  groupby.py  indexing.py  stats.py  nested.py  categorical.py
  compare.py            answer equality, tolerances, dtype rules
  divergences.py        the registry loader and the assertions on it
  divergences.toml      the registry itself, edited by people
  engines/
    pandas_engine.py    the oracle
    firepanda_engine.py the subject, in either of its two forms
  runner.py             runs a selection of cases on one engine, one process
  report.py             the scoreboard
  budget.py             wall clock and peak resident set per case
surface/pandas-3.0.3.json   committed inventory snapshot
corpus/manifest.json        committed corpus description and digests
drivers/firepanda/          the Mojo side, built against a firepanda checkout
results/                    one JSON per run, not committed
docs/specs/                 this folder, mirrored
tests/                      pytest over the harness itself
```

A run is `pixi run conformance`, which is a few minutes, and `pixi run report`, which is a second.

## A case

A case is a declaration, not a function body, because everything except the expression has to be machine readable for the scoreboard to say anything.

```python
case(
    id="str/pad/both-fillchar",
    api="Series.str.pad",
    section="4",
    milestone="M6",
    covers=["width", "side", "fillchar"],
    frames=["strings_ascii", "strings_unicode", "strings_null_heavy"],
    expr=lambda pd, f: f["s"].str.pad(12, side="both", fillchar="."),
)
```

`expr` takes the module and a frame and returns an answer. The same lambda runs on pandas and on firepanda because the module is a parameter, which is the whole point of an API that is a copy of another API, and it means a case cannot accidentally be written against one engine's spelling.

`covers` is what makes L3 measurable. The scoreboard joins `covers` against the parameter list the surface tool read from `inspect.signature`, and a parameter that no case names is reported as uncovered. That report is the work list for the next milestone, generated rather than written.

`id` is stable forever. A divergence registry entry, an expected failure, a performance measurement and a bug report all refer to a case by id, and renaming one is a breaking change to the repository.

## Outcomes

Four, per document 01: pass, fail, divergent, unimplemented. Written down again here because the implementation is where the fifth one gets added by accident.

`unimplemented` is produced by exactly one condition, which is the subject raising `AttributeError` or `NotImplementedError` at the top of the call, or the driver replying that it has no entry for the case id. Anything else that raises is a `fail`. An implementation that raises `NotImplementedError` from inside a branch is a fail, deliberately, because a half implemented method is worse than an absent one and the score should say so.

## The two engines

The oracle is pandas, imported in the runner's process, pinned in `pixi.toml`, and its version recorded in every result file. Nothing about it is clever.

The subject is firepanda, and it has two forms because the project is not finished.

**After M3, in process.** `import firepanda` and bind it as the module the case lambda receives. This is the form the whole design is aimed at, it makes every case a direct comparison in one interpreter, and it is three lines of code.

**Before M3, over a driver.** firepanda is a Mojo library with no Python module yet, so `drivers/firepanda/main.mojo` is a program that takes a case id and a corpus directory, runs the firepanda spelling of that case, and writes the answer as an Arrow IPC file. The harness reads that file with pyarrow and compares it to the pandas answer converted to Arrow. This is the same shape firepanda-bench already uses for its query drivers, it works today, and it costs a second implementation of every case in Mojo.

That second implementation is a real cost and it is worth paying for one reason: it is the only way the M6 conformance number exists before M3 lands, and M6 is where the library either becomes usable or does not. The driver cases are written in the same order as the Python ones, keyed by the same ids, and the harness reports every id the driver does not know as unimplemented, so the two sides drifting apart shows up as a falling score rather than as a silent skip.

When the in process form lands, the driver stays for one release as a cross check and is then deleted. Two implementations of a test suite is a state to pass through and not to live in.

## The driver protocol

Written down because it is a contract between a Mojo program and a Python one, with no type checker spanning the two of them, and the only thing that keeps them agreeing is that both sides were written against this section.

One invocation is one case on one frame. The driver takes `--case`, `--frame`, `--corpus` and `--out`, prints one line of JSON on stdout, and writes the answer itself to `--out` as an Arrow IPC file. stdout before that line is ignored, which is so that a `print` left in during a debugging session costs that person an afternoon rather than costing the project a run. stderr is free. The exit status is not the protocol: zero means a line was printed, whatever that line said.

The line has a `status` and there are four of them.

`ok` carries a `kind`, which is `scalar`, `frame`, `series`, `index` or `tuple`, and that is the vocabulary of the normalized answer in document 05. The shape is part of the answer, so a driver that returns a one column frame where pandas returns a Series has failed the case and has not almost passed it. A frame carries `columns`, the labels in order. A series carries `name`. Every kind carries `index`, the number of leading columns of the written table that came from an index, which is zero on every answer today and is in the protocol because it will not always be. A scalar is written as a table of one row and one column rather than as a number in the JSON line, so that its type travels as an Arrow type: a float through JSON is a decimal string that loses its last bits, and an integer through JSON has no width at all, while an int32 answer where pandas gives int64 is a conformance failure.

`index` and `tuple` are on that list because pandas returns both of them from ordinary calls and neither of them is a Series. `df.columns` is an Index, and a library that hands back a Series of the same labels has returned the wrong thing even when every label matches, so folding an Index into a Series on the way through the protocol would score a real difference as a pass. An index answer is one column, written under the value name, and it carries `name` the way a series does. `df.shape` is a tuple, and it goes through as one row of one column per part, each part keeping its own Arrow type for the same reason a scalar does not travel as a number: a pair of integers through JSON has no width, while an int32 row count where pandas gives int64 is a result somebody should see.

`absent` means the driver has no entry for the case id, and the harness scores it `unimplemented`. That is the one status a suite is tempted to make cheap and it is not cheap here, because unimplemented counts against the score exactly as hard as a failure does. A driver that quietly skipped what it did not know would produce a suite whose score goes up when you delete cases.

`raised` means firepanda raised, and it carries a `type` and a `message`. The type is `Error` for everything today, because Mojo has one exception type carrying a string, so every L4 case naming a pandas exception type fails against the driver. Guessing a pandas type name out of the message text would be inventing a result, and it would be an invisible invention, since nothing downstream could tell a guessed type from a real one.

`broken` means the driver failed at its own job, which is a bug in this repository or in the build rather than a fact about anything. A missing corpus file is broken. A corpus file that is there and that firepanda refuses to read is not broken, it is `raised`, because firepanda having no reader for a dictionary encoded column is a real gap and hiding it behind a message about the driver would lose it. That distinction is one line in the driver, a check that the file exists before trying to read it, and it is the difference between a suite that reports firepanda's limits and one that reports its own.

Three things the protocol deliberately cannot express, all for the same reason, which is that a harness papering over the subject's limits is measuring the harness. There is no index, so a pandas answer whose index is a plain zero to n-1 range compares equal to a firepanda answer with no index, per the global rule in document 05, and a pandas answer whose index is anything else fails. `DataFrame.tail` is the first case that fails that way and it is supposed to. There is no exception type, as above. There are no warnings, because a separate process has no way to hand a warning object back, so a case declaring a warning fails on the subject side until firepanda has somewhere to report one from.

The driver is built by `drivers/firepanda/build.sh`, which uses the Mojo toolchain pinned by the firepanda checkout it is pointed at rather than one pinned here. Two pins for one toolchain is how a driver ends up compiled against a library it does not match. The library and the toolchain are separate arguments, defaulting to the same checkout, because the loop this repository exists to support is to find a failure, fix it on a firepanda branch, rebuild and see whether the number moved, and the sane place to hold that branch is a git worktree, which has no pixi environment of its own. Installing a second copy of the same pinned toolchain to compile a branch of the same library is minutes of downloading for nothing. The build runs with the current directory set to the library checkout and the pixi manifest named by an absolute path, and not the other way around, because Mojo resolves an import against the current directory before it looks at `-I`. Building from the toolchain checkout with the library on `-I` compiles the toolchain checkout's library and reaches the `-I` path for nothing that exists in both, which is the separable case failing in the one way that leaves no trace. The script also writes `stamp.json` next to the binary, recording the firepanda version, the commit, whether the checkout was dirty, and the Mojo version, and the harness reads that into the result file. firepanda has no version constant in Mojo, so the checkout it was compiled against is the only place that answer exists, and a conformance number that cannot say which firepanda produced it is not a number anybody can act on.

## Process isolation

One engine per process, always, even when both could import. pandas leaves global state behind, options are process wide, and a firepanda extension module and pandas in one interpreter share an allocator and a set of Arrow symbols. Isolation also means a segfault in the subject is a failed case with a captured signal rather than a lost run, which matters more than it should while the subject is a young library in a systems language.

The runner therefore forks one worker per engine, hands it a case selection over a pipe, and collects a JSON result per case. Crash recovery re-runs the remaining selection in a fresh worker and marks the crashed case as a fail with `signal` in the reason. A crash is a conformance failure of the loudest kind and it is never a skip.

## Answers

Every answer is normalized to one of three things before comparison: an Arrow table, an Arrow array, or a scalar with its type. A pandas answer becomes a table through `pyarrow.Table.from_pandas` with the index preserved as columns under reserved names, and a firepanda answer arrives as Arrow already, which is the whole reason the Arrow C interface is at M2 rather than at M9.

Normalization is where an index becomes data. `df.groupby("k").sum()` returns a frame with `k` in the index in pandas and a frame with `k` as a column in firepanda, which is not a difference in the answer and would be an avalanche of false failures if it were compared naively. The normalizer moves a named index into columns named `__index__0` and so on, and the comparison rules in document 05 say when that is allowed and when a case is specifically testing the index and turns it off.

## Determinism

The corpus is generated from a fixed seed with an algorithm written down in document 04, and the manifest carries a digest per frame. The runner asserts the digest before running anything. A conformance result computed against a corpus nobody can reproduce is not a result, and the failure mode this prevents is the one where a case passes on a laptop and fails in CI because the frames were not the same frames.

Case selection order does not affect outcomes. Cases share nothing, no case mutates a corpus frame, and the harness asserts that by re-checking the digest after each module of cases rather than trusting it.

## The tasks

| | |
|---|---|
| `pixi run surface` | rewrites `surface/pandas-3.0.3.json` from the installed pandas |
| `pixi run corpus` | regenerates the corpus and verifies the manifest |
| `pixi run conformance` | runs the case registry against both engines |
| `pixi run oracle` | runs the registry against pandas twice, which must be a perfect score |
| `pixi run report` | the scoreboard, markdown and JSON |
| `pixi run budget` | the cost matrix from document 09 |
| `pixi run coverage` | which pandas names and parameters no case touches |
| `pixi run test` | pytest over the harness, which is not the same thing as running the harness |
| `pixi run driver` | builds `drivers/firepanda` against a sibling firepanda checkout and stamps it |

`pixi run oracle` is the one people forget to build. It runs pandas against pandas through the entire pipeline, and any result other than a perfect score is a bug in the harness rather than in anything it measures. It is what stops a normalization mistake from being published as a firepanda failure, and it runs on every commit to this repository.
