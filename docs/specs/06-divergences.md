# Divergences

Places where firepanda deliberately does not do what pandas does. Every one is a decision, every decision is written down before the case is allowed to fail, and the registry asserts the divergence rather than excusing it.

## The registry

`fpcompat/divergences.toml`, edited by people, read by the runner.

```toml
[[divergence]]
id = "str/replace/backreference"
cases = ["str/replace/regex-backref-*", "str/extract/backref-*"]
kind = "engine"
reason = "RE2 has no backreferences. A pattern using one raises rather than matching, and the message names the pattern."
spec = "06-pandas-parity.md#the-genuine-omissions"
expect = "raises"
since = "2026-09-04"
```

`kind` is one of four. `engine` means a deliberate design difference, of which the seven in the parent folder's document 06 are the whole list today. `upstream` means pandas is doing something we consider a bug and we are matching the corrected behaviour, which requires a link to the pandas issue and is the rarest and most suspicious kind. `unsupported` means the operation is not implemented and is not planned for the current milestone, which is a scheduling fact rather than a design one and carries a milestone tag. `pending` is a divergence with an expiry date, which is how a temporary difference is allowed to exist without becoming permanent by inattention.

## The rule that keeps it honest

**A registered divergence must diverge.** The runner does not skip a case that matches a registry entry, it runs it, and it requires the outcome the entry declares. If `expect = "raises"` and the case now returns an answer, the case fails, loudly, with a message saying the registry is out of date.

This inverts the usual relationship. In most suites a known failure list is where cases go to be forgotten, and it grows because adding an entry is easier than fixing a bug. Here an entry is an assertion with a maintenance cost, and the day somebody implements backreference support by switching the regex engine, the registry tells them to come and delete the entry.

`pending` entries carry an `expires` date, and an expired entry fails the run whatever the case does. That is the only mechanism in this repository with a calendar in it, and it exists because "temporarily" is the most load bearing word in software.

## The seven

The parent folder's document 06 lists seven genuine omissions and they are the founding entries of this registry. Restated here with what the compat suite actually does about each.

**`plot`, `hist`, `boxplot` and `.style`.** `kind = "engine"`, `expect = "raises"`, and the message points at `to_pandas()`. Cases exist for each name, they assert the raise and the message, and they stay in the denominator forever. Roughly a dozen callables.

**Pickle IO.** `read_pickle` and `to_pickle` raise with a pointer to Parquet and IPC. Two callables, two cases, and the message is checked because the message is the entire value of the feature.

**`dtype=object`.** Raises at construction naming the column and the offending Python type. The case builds a frame from a list of `datetime.timedelta` objects and asserts both halves of the message. This one is worth its case because it is the divergence a beginner hits first.

**`inplace=`.** Raises `TypeError` with the reassignment form in the message. This affects around 30 pandas callables that take the parameter, so it is 30 cases generated from the surface inventory rather than 30 cases written by hand, which is the pattern for any divergence that touches a parameter rather than a name.

**Automatic index alignment.** The largest behavioural divergence in the project and the one most likely to change a user's answer silently rather than raise. The cases are explicit: two frames with different indexes, added, where pandas produces the union of the indexes with nulls and firepanda produces a positional result or raises on a length mismatch. `expect = "differs"`, with the expected firepanda answer written into the case, because "differs" without saying how is not a specification.

This entry is the one to read twice before shipping. A user porting a program that relied on alignment gets a wrong answer rather than an error, and the migration guide has to say so in the first screen rather than in a caveats section.

**Regex lookaround and backreferences.** RE2 semantics. `expect = "raises"` and the message contains the offending construct, so that a user who wrote `(?<=foo)bar` is told which part is the problem rather than being told their pattern is invalid.

**No implicit index.** The other half of the alignment entry. Operations that pandas performs against an index that was never declared behave positionally, and the cases enumerate where that is visible, which is fewer places than it sounds: `reindex`, `align`, `loc` on a frame with no declared index, and arithmetic between misaligned frames.

## What is not allowed in here

**A divergence added in the same pull request that makes a case fail.** The registry is edited in its own pull request, reviewed on its own, and merged before the change that needs it. This is slower on purpose. A registry entry is a promise to users about what the library does not do, and it should never be possible to add one while trying to get a build green.

**A divergence without a `reason` a user would accept.** "Hard to implement" is not a reason for `kind = "engine"`, it is `kind = "unsupported"` with a milestone tag, and the difference between those two words is the difference between a design and an excuse.

**A blanket entry.** `cases = ["str/*"]` is not allowed. Patterns may cover a family of generated cases, and a pattern that would match an entire section is rejected by the loader, because at that point the score has been redefined rather than measured.

## Reporting

The scoreboard prints divergences as their own column and the report has a page listing every entry with its reason, generated from the TOML, which is the public document users read when their program does something different. That page is the honest version of the migration guide, and it is generated rather than written so it cannot fall behind the code.

Counts as of today: 7 entries, covering approximately 50 of the 1125 callables, or 4.4 percent of the surface. That number goes on the front page of the report next to the pass rate, because a 95 percent pass rate over a surface with 5 percent registered divergences is a different claim from a 95 percent pass rate with none.
