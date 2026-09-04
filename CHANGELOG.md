# Changelog

All notable changes to this repository are recorded here. The format is
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this repository
follows [semantic versioning](https://semver.org/spec/v2.0.0.html), where the
public interface is the case id namespace, the result file schema and the
divergence registry format.

## [Unreleased]

### Added

- A comparison that sorts in Arrow rather than in the interpreter, and a second clock on every result record. The oracle took 195 seconds while the per case timings in the file it wrote summed to 5.1, because the clock stopped before the comparison started and roughly 160 seconds were going into a sort that rendered every value to a string and sorted Python tuples, on six merge answers of ten million rows each. Arrow sorts the key columns now and the rendered sort remains for the dictionary encoded and nested columns Arrow refuses, with both sides of a comparison always taking the same path. The oracle is 195 seconds down to 14 and the relaxation sweep 139 down to 39. `compare_seconds` is on every record and totalled on the document, so the next time the two numbers diverge the file says so.
- The relaxation sweep, `pixi run sweep`, and the correction to the specification that came with it. `05-comparison.md` said the oracle would be run a second time with each relaxation individually disabled and the cases that then failed would be the ones needing it. Measured: 18 case and relaxation pairs, run with their relaxation off, zero failures, because pandas against pandas returns both sides in the same order and an order relaxation has nothing to absorb. Under the specified rule every declaration in the repository was unnecessary and would have been deleted. The sweep reorders the oracle's own answer instead and requires each declaration to be what makes the case pass against it. All 18 declarations are load bearing.
- The specification, 12 documents in `docs/specs/`, mirrored from the author's notes.
- The scaffold: pixi environment pinned to pandas 3.0 and pyarrow 25, CI, licence.
- The surface tool, which counts the pandas API rather than remembering it, and the committed inventory of pandas 3.0.3 that every coverage number is computed against.
- The corpus, 56 frames generated from one seed, described by a committed manifest and never committed as data.
- The comparison layer, which decides what the same answer means, with four tolerance classes and a closed list of relaxations that each have to say why.
- The case registry and the runner, 623 cases over 1429 runs, four outcomes and no skip, one engine per process, and a worker crash attributed to the case that caused it.
- The surface sweep: 1413 generated L0 cases and 1034 generated L1 cases, one per name and one per readable signature, so that the two cheapest levels are measured over the whole of pandas rather than over the names somebody thought to check.
- The scoreboard, which reports against the pandas surface rather than against our own case list, prints the parameter coverage next to the score so that neither number can be quoted alone, and refuses a pull request that lowers a section without editing the recorded floor.
- The divergence registry, where a registered divergence has to actually diverge: 7 entries, 68 cases asserting them, a generated public page, and a build that fails when an entry stops being true.
- The cost matrix, 65 operations of which 21 are chains, one process per engine per operation, seven repeats with the median and the interquartile range, and a per machine baseline with a ten percent regression gate. It runs on its own corpus, which is the correctness generator and seed at one million rows.
- `operations.json`, the cost matrix operation table with the numbers taken out, committed and checked in CI so that firepanda-bench can link a query to the operations it is made of without depending on this package.
- Eleven cost matrix rows that came from firepanda-bench rather than from the parity sections, because linking each benchmark query to its operations found eleven that a published query runs and the matrix had no row for. `GroupBy.median` is the one that mattered: it keeps its values per group, and until it existed the matrix measured no reduction with per group memory in it at all.
