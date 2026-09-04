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
- The divergence registry, where a registered divergence has to actually diverge: 7 entries, 68 cases asserting them, a generated public page, and a build that fails when an entry stops being true.
