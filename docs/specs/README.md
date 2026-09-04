# The specification

Twelve documents. They are mirrored from the author's notes at `Spec/2126/compat`
and they are the reason every design decision in this repository looks the way it
does.

| | |
|---|---|
| [00](00-README.md) | why this repository exists, and the three repositories |
| [01](01-what-100-percent-means.md) | the definition being measured, five levels, the scoring rule |
| [02](02-the-surface.md) | the measured pandas inventory, and what the parity checklist misses |
| [03](03-harness.md) | the repository, the case registry, the engines, the driver |
| [04](04-corpus.md) | the frames, how they are generated, why they are committed |
| [05](05-comparison.md) | equality semantics, tolerances, dtype rules, the oracle self test |
| [06](06-divergences.md) | the registry of things allowed to differ, and the rule that keeps it honest |
| [07](07-scoreboard.md) | the report, the site, the CI gate, the rules against green washing |
| [08](08-m6.md) | the first tier parity plan, which is milestone issue 8 in the library |
| [09](09-resources.md) | the operation level cost matrix, and the ten times on a tenth goal |
| [10](10-bench-and-compat.md) | the line between this and firepanda-bench |
| [11](11-milestones.md) | C0 to C5, with exit criteria |

Read [01](01-what-100-percent-means.md) first. It is the only one that says what
the project is claiming, and everything else is machinery in service of it.
