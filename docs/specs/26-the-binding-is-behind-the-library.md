# 26. The binding is behind the library

The previous document ended with a number that was meant to be uncomfortable. Once the harness could actually point itself at every pandas name, 1309 of the 1413 names in the surface did not resolve against firepanda at all. Not wrong, not divergent, not there. This document is about what happened when somebody went and looked at what was behind those names, because the answer was not what the board implied.

## The measurement that changed the plan

The obvious reading of 1309 unresolved names is that firepanda has not implemented 1309 things and the work ahead is 1309 pieces of work. So the unresolved list was cross referenced against the method definitions in the Mojo source, and the reading was wrong in a way that matters.

Take the reductions. pandas has seventeen whole column reductions that firepanda's `AggKind` also has: sum, mean, min, max, count, first, last, size, var, std, median, quantile, nunique, corr, cov, sem and skew. Every one of them is implemented in the kernel, has been for a while, is tested in Mojo over more dtypes than are reachable from Python, and is fast. The Python extension exposed none of them. Zero of seventeen.

That means `s.sum()` did not work. `s.mean()` did not work. `df.count()` did not work. These are, after `df["a"]`, the most written expressions in pandas, and the library computed all of them correctly and had no door a pandas program could walk through to ask.

So the gap the board was reporting was not a library gap. It was a binding gap, and a binding gap is a different size of problem with a different shape of fix.

## Why this distinction is the whole point of the board

A conformance board that says "1309 names do not resolve" is telling you something true and useless. What you need to know is how many of those are missing implementations, which are months, and how many are missing doors, which are days. Those two numbers get you to a plan and the total does not.

The board cannot tell them apart on its own, because from outside a missing method and an unexposed method look identical. A person has to go and read the Mojo. That is not a defect in the board. It is the limit of what a black box comparison can see, and the useful response is to write down what the reading found so the next person does not have to do it again.

The reading, for the record: the reductions were seventeen implemented and zero exposed. The transformations look similar, with `drop_nulls`, `fill_null`, `fill_forward`, `fill_backward`, `is_null`, `shift`, `diff`, `pct_change` and the four cumulative scans all present in the core under their own names rather than under the pandas names. So do the fifteen `dt_*` methods that a `.dt` accessor would surface. The unimplemented count on the board is a real number and it is not a measurement of how much library is missing.

## What was built

Twelve of the seventeen are now reachable from Python on both a series and a frame, plus `count` on a frame: sum, mean, min, max, median, skew, std, var, sem, quantile, nunique, count.

The mechanism is deliberately one door rather than twelve. There is one boundary method on each side that takes a reduction name and a float parameter, a small module that turns that pair into an `AggKind`, and a table of generated Python members that spell out each pandas signature and call through. The parameter rides beside the name because `ddof` and `q` are the only two numbers any of these reductions take, and giving each reduction its own entry point across the language boundary would have been twelve doors into one room, each of which could drift from the others.

Five are left out on purpose. `corr` and `cov` need a second column and the index alignment that goes with reading it, which is a real piece of work rather than a table entry. `size`, `first` and `last` are grouped shapes rather than whole column answers, and belong with the groupby surface.

## The one decision that was not mechanical

`DataFrame.agg_all` in the core answers a one row frame. That is the right Arrow answer, because a frame lets every column keep its own type and a sum over an int column and a sum over a float column genuinely have different types.

pandas answers a series. A series has one dtype, so the answers have to agree on a type or be forced into one.

Three rules close that gap. If every column reduces to the same type, that type stays. If they are all numbers and differ, they widen to float64, which is what pandas does. If they have nothing in common, the call is refused with that sentence.

The third rule is the interesting one, because pandas does not refuse. pandas answers an object series holding a number and a string side by side. Arrow has no object column and firepanda is not going to grow one, so the choices were to refuse with a reason or to widen everything to text. Widening to text would produce an answer that looks right, prints plausibly, and is wrong in a way the caller finds three steps later. A refusal that names the reason is worse to receive and better to have received. It goes in the divergence register rather than being smuggled in as a feature.

## Refusals as a policy, not an accident

Every pandas argument that is declared on these signatures and not implemented raises by name with the reason in the message: `skipna=False`, `min_count`, `numeric_only`, a non-linear `interpolation`, a list of quantiles, `nunique(dropna=False)`, and a frame reduction over `axis=1`.

This is worth stating as policy because the alternative is tempting. A declared parameter that is accepted and quietly ignored gives the correct answer at its default value, which is where every quick test uses it, and the wrong answer everywhere else, which is where a real program uses it. It is the single most expensive kind of compatibility bug, because the library reports success and the caller has no reason to look. Each refusal has a test, so that none of them can be dropped later by somebody tidying up.

The signature still declares the parameter, because the signature is what the L1 board compares and because a caller reading the help should see the pandas shape. Declaring it and refusing it is honest. Declaring it and ignoring it is not.

## What it moved

Passing runs went from 805 to 851. Unimplemented went from 3260 to 3214. Twenty four pandas names now resolve that did not, and twenty four signatures now match that could not be compared.

More interesting than the totals, `Series.std` and `Series.var` are now the first callables in the statistics section to pass at every level of the ladder, from the name resolving through the signature matching to the values agreeing with pandas over the corpus. The board's count of fully conformant callables went from one to two. Two is not a good number. It is the first honest one, and it is two because the ladder is strict rather than because the library computes two things.

## What this argues for next

The transformation family is the same shape of work and should be next: `dropna`, `fillna`, `ffill`, `bfill`, `astype`, `isna`, `notna`, `shift`, `diff`, `pct_change` and the four cumulative scans, on both a series and a frame. Every one of them exists in the core under a different name. A `.dt` accessor would make fifteen of the forty two `dt` names resolve for the same reason.

The general rule this document is arguing for: before treating an unimplemented count as a work estimate, go and read what is behind it. On this board a large part of the number was a naming and exposure problem rather than a computing problem, and those get fixed at a completely different rate.
