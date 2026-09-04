# Changelog

All notable changes to this repository are recorded here. The format is
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this repository
follows [semantic versioning](https://semver.org/spec/v2.0.0.html), where the
public interface is the case id namespace, the result file schema and the
divergence registry format.

## [Unreleased]

### Added

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
