# Cost matrix baselines

One file per machine per engine per corpus size, holding two numbers per operation: the median wall clock and the peak resident set. `pixi run budget-gate` reads the file for the machine it is running on and fails when any row is more than ten percent slower or heavier than what is recorded here.

## Why this is per machine and not per project

A timing is a property of a machine, so a baseline recorded on a four core runner is not a baseline for a sixteen core desktop whatever else the two have in common. The filename carries the system, the architecture and the core count, and the full machine description including the processor goes inside the file. The gate refuses to run when the processor in the baseline is not the processor in the run, because two machines that happen to share a filename key would produce a comparison that is about the hardware.

The hostname is deliberately not in the key. It is somebody's laptop name and it does not belong in a public repository.

## Which machines these are

`docs/specs/09-resources.md` names two, the AMD EPYC VPS and the i9-13900K desktop that firepanda-bench already uses, because a second set of machines would mean a second set of noise to characterize. Those are the published floors and neither is here yet.

`darwin-arm64-10core-pandas-1000000.json` is a development laptop, an M-series MacBook. It is committed because a mechanism with no file in it has never been exercised, and it is useful locally: run `pixi run budget` then `pixi run budget-gate` on the same laptop and a change that made an operation twice as slow shows up before it is pushed. It is not a published floor. A laptop throttles, and a number measured on one is worth what it cost to measure.

## Raising a floor

The gate is a floor and not a threshold, so a row that legitimately costs more now is not a bug in the gate. Rerun the sweep, run `pixi run budget-baseline`, commit the file in the same pull request as the change, and say in the pull request body which rows moved and why. A baseline that gets regenerated with no explanation is a baseline nobody reads, which is a gate that does nothing.

## What is not in here

Page faults, CPU time and thread counts are measured and they go in the result file, but they are not in the baseline. A baseline carrying page fault counts is a baseline that changes when the allocator changes, and then it gets regenerated on every pull request until nobody reads the diff.
