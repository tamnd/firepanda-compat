# Milestones for the compat repository

Six, C0 through C5. They are small because the repository is a measuring instrument and an instrument that takes six months to build measures nothing for six months.

The ordering constraint that shapes all of them: the harness has to produce a number before M6 work starts in the library, because the point of a number is to steer the work rather than to grade it afterwards.

## C0, the instrument

The scaffold, and it is worth doing properly once. Repository, licence, pixi environment pinned to pandas 3.0 and pyarrow 25, CI, the specification mirrored from `Spec/2126/compat`, and the four pieces that everything else sits on.

- The surface tool, walking 21 namespaces and writing a committed JSON inventory
- The corpus generator and its manifest
- The comparison layer, with the per dtype rules from document 05
- The case registry and the runner, with the four outcomes and no skip

Exit: `pixi run oracle` is a perfect score over a registry of at least 200 cases, run on every commit. Nothing about firepanda is involved yet, and that is the point. An instrument is calibrated against a known input before it is pointed at an unknown one.

## C1, the subject

Connecting firepanda. Two engine adapters, one for the in process import that does not exist yet and one for the Mojo driver that can exist today, behind one interface so that the second can be deleted without touching a case.

- The driver protocol, a case id and a corpus directory in, an Arrow IPC answer out
- `drivers/firepanda/main.mojo`, built against a firepanda checkout, with the M1 operations implemented
- The scoreboard and the first published number
- The divergence registry, loaded and asserted, with the seven founding entries

Exit: a summary line exists for firepanda 0.6.x and is published. It will be a low number and publishing it is the whole exercise. A project that waits until its conformance score is respectable before publishing one has learned to hide the number, and it will keep doing that.

## C2, coverage of the M6 surface

Cases for the eleven workstreams of document 08, written ahead of the library work rather than after it, so that every workstream starts with a failing measurement and finishes when the measurement passes.

- Around 2000 cases across the eleven workstreams
- Parameter coverage above 80 percent for each
- The generated case families: one case per value of every enumerated parameter, produced from the surface inventory rather than typed out
- Error cases for the 46 exception types

Exit: parameter coverage above 80 percent for every M6 workstream, and the uncovered parameter report is empty of anything a user would call.

## C3, the cost matrix

Document 09. The budget corpus, the per operation timing and memory measurement, the chained cases, the two machines, the nightly run and the per pull request subset.

Exit: a matrix row for every operation with a case, published, including the rows firepanda loses.

## C4, the in process engine

When M3 lands the Python front door, the driver becomes unnecessary. This milestone is the adapter swap, one release of running both to check they agree, and then deleting the Mojo driver and roughly 3000 lines of duplicated case implementations.

Exit: the driver is deleted, the scoreboard is unchanged across the swap, and the run takes half as long.

## C5, the release gate

Conformance becomes a condition of a firepanda release rather than a report about one.

- The ratchet enforced on every library pull request
- The release workflow refusing to publish if the score fell or the oracle is not perfect
- The divergence page generated into the library's documentation site
- The history page, six months deep by then, on the front page of the library README

Exit: the library README carries a conformance badge whose number is generated, and no human can change it without changing the code.

## What is not a milestone

**Getting to 100 percent.** That is the library's job and this repository cannot do anything about it. The milestones here are about the instrument being trustworthy, complete in coverage, and wired into the process. A perfect score with an instrument nobody believes is worth nothing, and a low score with an instrument everyone believes is a work plan.

**Supporting pandas 2.x.** The target is pandas 3.0 and there is one target. A user on 2.x is a user who will be on 3.x, and a compatibility suite that straddles a major version of the thing it is conforming to spends its life on the differences between the two rather than on the differences that matter.

**A pandas 3.1 upgrade path.** It will happen, the surface tool will produce a diff, the diff will be a work list, and that is the whole plan. Writing more of it down now would be guessing about a release that does not exist.
