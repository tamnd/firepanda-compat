# The sort that was measuring numpy

## Three failures that outlived twenty documents

`basics/sort-values [keys_10]`, `basics/sort-values [keys_1000]` and `basics/sort-values-descending [keys_10]` had been failing since the board first had numbers on it. They were the oldest entries on it and they were also the worst kind, because every other failure was an absence and these three were wrong answers. A program that calls a function firepanda does not have gets an error and knows. A program that calls `sort_values` gets rows back.

They are fixed now, and the fix is in this repository rather than in the library, which is the part worth writing down.

## The failure message was pointing at the answer

The message was `index NoneType(None) row 0: 46, expected 7932`. It names the index and not the values, and that is the whole of the clue.

The driver writes the frame's row labels into a column called `__index__0`, per document 03. Document 05 drops an index when both sides carry a plain 0 to n-1 range and compares it otherwise, and the index a sort produces is not a plain range, so it was compared. Running the two engines by hand showed the key column equal, the value column equal, and the labels different. Both libraries had put the same rows in the same order by key, and had disagreed about which of two rows with the same key came first.

## pandas does not have one answer here

Measured, on the frames the case runs on:

```
df.sort_values("key") != df.sort_values("key", kind="stable")
```

The two disagree on `keys_10` and on `keys_1000`, with the key columns equal in both. `sort_values` defaults to `kind="quicksort"`, which is numpy's introsort, which is not stable, so the order inside a group of equal keys is whatever the partitioning happened to leave behind.

That makes `kind` a different kind of argument from what its name suggests. It reads like a performance knob, three algorithms with the same result and different constants. It is the only argument `sort_values` has that decides what the answer is, because it is the only one that says anything about a group of equal keys, and on a frame with ten distinct keys and ten thousand rows a group of equal keys is a thousand rows.

## The probe

The general question is not whether a case has ties. It is whether the case's answer moves when the tie order changes. So every sort shaped case in the registry was run five times, once with the default and once each under `stable`, `quicksort`, `heapsort` and `mergesort`, and the answers were compared against each other.

Thirty runs across eighteen cases. One of them is an error case with no ordering to compare. Twenty four give one answer under all five kinds. Two are `basics/sort-values-stable`, which the probe can only move by overriding the kind the case names. Three move, and they are exactly the three that were failing:

| Case | Frame | Under five kinds |
|---|---|---|
| `basics/sort-values` | `keys_10` | moves |
| `basics/sort-values` | `keys_1000` | moves |
| `basics/sort-values` | `keys_awkward` | one answer |
| `basics/sort-values-descending` | `keys_10` | moves |
| `basics/sort-values-descending` | `keys_awkward` | one answer |
| `basics/sort-values-na-first` | both frames | one answer |
| `basics/sort-values-stable` | both frames | moves only when the probe overrides the kind the case names |
| `basics/sort-two-columns`, `basics/sort-index`, `indexing/sort-index-multi` | all frames | one answer |
| `strings/sort`, `categorical/sort-*`, `temporal/sort-timestamps`, `nested/list-sort-by-other` | all frames | one answer |
| the four `divergences/inplace` sort entries | all frames | one answer |

Two rows in that table need reading carefully.

`basics/sort-values-stable` moves under the probe because the probe replaces the `kind` the case passes. That is the probe confirming the case rather than accusing it. The case names `stable`, the three unstable kinds give a different answer, so the case is pinning an order that would otherwise be loose, which is what it is for.

`keys_awkward` does not move, and it does not move for a reason that is not reassuring. It has ten rows. numpy's introsort falls back to insertion sort below sixteen elements and insertion sort is stable, so on that frame every kind is stable and the case passes for a reason that has nothing to do with what the case says. Two more runs were sitting in that position: `basics/sort-values-na-first` uses the default kind on `keys_awkward` and on the sixty four row `strings_null_heavy` and passes today for the same accident.

## What firepanda answers

Run through the driver, for all three failing runs: the labels equal pandas' stable answer exactly, the key column equal, the value column equal. Not close, not equal after a sort, equal row for row and label for label.

It is not luck either. `firepanda/kernel/sort.mojo` says so in four separate places, and the one that matters most is the descending path, which is where an implementation usually loses stability without noticing:

> Complementing the key reverses the order without reversing the array, which is the difference between a stable descending sort and one that scrambles the rows it considers equal.

firepanda has no `kind` argument, and after this it is clear it does not need one. Its sort is a stable merge sort with an insertion sort underneath, so stable is the only answer it can give, and stable is the answer pandas gives when it is asked for a reproducible one.

## Three ways to fix it

A relaxation. Rejected. Document 05 opens with the sentence that the default is strict and says row order is compared exactly outside the grouped case. A relaxed version of `basics/sort-values` would be `basics/sort-values-stable` with the guarantee taken out, which is not a case, and a relaxation that says the answer is allowed to be anything is a case that has stopped measuring.

A divergence entry. Rejected, and this is the more interesting rejection. A divergence in this repository asserts that firepanda deliberately does not do what pandas does. firepanda does do what pandas does. It does the thing pandas does when pandas is asked to be reproducible, which is one of pandas' own answers and the only one pandas will repeat. Writing an entry would put a false statement into a registry whose entire value is that every entry in it is true and is checked.

Changing the case. Chosen. If the answer is not specified, the case has to ask a question that has one.

## Naming a second key

The second column of each frame is a tiebreaker, and it is a total one. Measured: `(key, value)` is distinct for ten thousand of ten thousand rows on `keys_10`, ten thousand of ten thousand on `keys_1000`, and ten of ten on `keys_awkward`. So `df.sort_values(["key", "value"])` has exactly one correct answer, every sort algorithm gives it, and firepanda gives it.

The case did not lose anything it was measuring. It is still the default call with the default kind and the default null position, still ten thousand rows over ten distinct keys, and every group of equal keys still has a thousand rows in it. The ties are all still there. The case now says how they are resolved instead of comparing whatever numpy left.

It is also what the API asks of a user. Somebody who sorts on a column with duplicates and expects the same file out twice has to name a tiebreaker or name the kind, and the two cases now show one of each. `basics/sort-values-na-first` got the same edit for the latent reason above, which is a case that says what it means rather than a fix to a failure that was showing.

## Four runs that were absent and should not have been

`basics/sort-values-stable` had no driver entry. That is backwards: of all the sort cases it is the one whose pandas answer is defined, and it was the one not being measured. It has an entry now and passes, which is a statement about firepanda's sort that the suite was previously taking on trust.

`basics/sort-values-na-first` had no entry either. Its driver line takes the first two column names off the frame rather than naming them, because the column carrying the nulls is the key in one of its frames and the value in the other, and that difference is the point of the case.

`basics/sort-index` stays absent. firepanda has no `sort_index` on a frame, which is a real gap and belongs on the gap list rather than in this document.

## The rule

A case must not compare an answer the API does not specify.

Document 05 already had a version of this. Its row order paragraph exempts the grouped case and the four cases where pandas documents the order as undefined. The hole was the third possibility, which is where these three runs lived: pandas does not document the order and does not define it, it just returns one. An undocumented order is not a promise, and a suite that compares one is reporting on the sort numpy shipped rather than on either library. Document 05 now says so.

This is the fourth time this project has caught its own instrument rather than the library, and it is the one that took longest, which fits. A differential suite produces evidence about two things at once, the thing under test and the test, and a failure that survives twenty documents of attention is far more likely to be the second than the first. The check is cheap: when a case's answer depends on an order, run it under every ordering the library will give you and see whether it moves.

## The board after

626 passing runs, up from 619. Three of those are the failures fixed and four are the cases that were absent. Failing runs eight to five, and the ratchet's callable count five to four, since `DataFrame.sort_values` is off it.

The five that are left are two missing index levels on a two key group by, two skew values differing in the twelfth significant figure, and one cast to string that loses a null. All five are the library, and the astype one is the smallest.
