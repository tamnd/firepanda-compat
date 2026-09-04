# The scoreboard

What a run produces, who reads it, and the rules that stop it from becoming an advertisement.

## The summary line

One line, printed at the end of every run and pasted into the milestone issue after every merge.

```
firepanda 0.6.40 vs pandas 3.0.3   L3 312/1125 (27.7%)   L2 401   L1 486   L0 1125
divergent 50   unimplemented 641   fail 44   cases 2183 in 214s
```

Nobody has to interpret it and it fits in a commit message. The four level counts are cumulative, so L1 486 means 486 callables have a signature that accepts what pandas accepts, of which 401 also produce the right default answer, of which 312 also produce the right answer across their parameters.

`fail` and `unimplemented` are separate because they mean different things to a person planning work. 641 unimplemented is a schedule. 44 failures is a bug list, and the bug list is the number that must go to zero before a release, whatever the schedule says.

## Per section

The report is a table keyed by the sections of the parity checklist, so a reader who knows document 06 can find their way around it, and so that a section can be reported as done independently.

```
| Section              | Callables | L3  |  %   | Divergent | Fail |
|----------------------|-----------|-----|------|-----------|------|
| 4  .str              |        57 |  57 | 100% |         3 |    0 |
| 6  .cat              |         8 |   8 | 100% |         0 |    0 |
| 8  GroupBy           |        58 |  41 |  71% |         2 |    3 |
| 9  Windows           |        53 |  12 |  23% |         0 |    7 |
| 2  DataFrame         |       186 |  90 |  48% |        14 |   11 |
```

**A section at 90 percent is reported as 90 percent, not as done.** That sentence is already in the parent folder's document 06 and it is repeated here because this is the document where breaking it would be convenient.

## Parameter coverage

The second table, and the one that generates the work list. The surface tool knows every parameter of every callable, the case registry knows which parameters its cases name, and the join is the list of parameters no case has ever exercised.

```
Series.str.split       covered: pat, n, expand      uncovered: regex
DataFrame.sort_values  covered: by, ascending       uncovered: axis, kind, na_position, key
```

An uncovered parameter is not a failure and it is not a pass either, it is a hole in the measurement, and the honest reading of a 100 percent L3 score with 40 percent parameter coverage is that the suite is not finished. So the front page carries both numbers next to each other and neither is quotable without the other.

## The site

Static, generated from the JSON results, published per commit on the default branch. Three pages and no more.

The front page is the summary line, the section table, the coverage percentage and the divergence count, with the pandas and firepanda versions and the commit at the top.

The divergence page is generated from `divergences.toml`, one entry per section, with the reason as written by the person who made the decision. This is the page a user lands on from a search for why their program behaves differently, so it is written for them and not for us.

The history page is the summary line over time, one point per commit on the default branch. A conformance number that only exists for today is a claim. A conformance number with six months behind it is evidence, and the shape of the curve is the most useful single thing this repository produces for anyone deciding whether to bet on the project.

## The CI gate

On every pull request to the library, the conformance run must not lower L3 for any section and must not raise the failure count. That is a ratchet rather than a threshold, and it is the only gate, because a threshold on a young project is either so low it means nothing or so high it blocks the work that would raise it.

The ratchet is stored as a committed file of per section counts, and raising it is part of the pull request that earns it. A pull request that lowers a section is not blocked if it says why in the body under a `Conformance:` line, which is then quoted in the release notes. Blocking a deliberate temporary regression is how people learn to route around the gate.

## The rules against green washing

**No skip outcome.** Document 01. It is the whole of the discipline and everything else here is detail.

**The denominator is the pandas surface.** Not the cases we wrote. A suite that reports pass rate over its own cases reports how good it is at writing cases it passes.

**Publish the run that lost.** Same rule the bench repository has. A conformance run that got worse is published with the same prominence as one that got better, and the history page makes that automatic rather than a matter of will.

**The oracle run is published too.** If the pandas against pandas self test is not perfect, the front page says so at the top, in place of the score, because a harness that disagrees with itself has no business publishing a number about anything else.

**Version pins in every artifact.** pandas version, pyarrow version, firepanda version, Mojo toolchain version, machine, and the corpus digest. A conformance claim without the pandas version attached is not a claim, since the thing being conformed to moves.
