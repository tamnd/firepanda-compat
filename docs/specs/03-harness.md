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

`pixi run oracle` is the one people forget to build. It runs pandas against pandas through the entire pipeline, and any result other than a perfect score is a bug in the harness rather than in anything it measures. It is what stops a normalization mistake from being published as a firepanda failure, and it runs on every commit to this repository.
