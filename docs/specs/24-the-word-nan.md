# The word nan

## The last failure in basics

`basics/astype-string [float64_no_nulls]` was the only failing run left in the `basics` section after document 23 cleared the sort. It is one expression, `df["value"].astype("str")`, on a frame of sixty four float values, and the report said the two answers differ. That is the whole of the failure message, which is why it sat on the board for as long as it did: a cast to text on a float column has a hundred places it could go wrong and the report names none of them.

So the first thing was to stop guessing and put the two columns side by side. Sixty four values, rendered by pandas 3.0.5 and by firepanda, compared one at a time.

## Sixty four values and one of them

Sixty three of them were byte for byte identical. That includes every value in the frame that was put there to be difficult. `5e-324`, which is the smallest positive double and the one a naive renderer prints as zero. `-0.0`, which is the one a renderer prints as `0.0` after normalising a sign it should have kept. `1.7976931348623157e+308`, which needs all seventeen significant digits and the exponent. `8176596941441788.0`, which is large enough that the shortest round tripping representation is not the obvious one. `inf` and `-inf`, which are values rather than errors and are spelled as words. firepanda's float to text rendering was already right, on the hard cases, and nobody had ever measured that.

The sixty fourth value is row zero, which is a NaN. pandas gives a missing row. firepanda gave the three letter string `nan`.

That is the entire failure. One value out of sixty four, and it is not a rendering difference at all, it is a question about where a missing row lives.

## A missing row has two spellings

A float column says a row is missing by putting a NaN in it. A text column cannot, because there is no NaN in a sequence of bytes, so it says it by clearing a validity bit. Both are ways of saying the same thing and the library reads both, and a cast that crosses between the two types has to carry the missing row from one spelling to the other or it stops being missing.

Writing the word `nan` is what happens when the cast does not carry it. The row is no longer absent, it is present and holds a three character string, and every count, every `isna`, every `dropna` and every group by downstream now sees a value where there was a gap. It is not a cosmetic difference in one cell, it is the point at which the column stops being the column.

pandas used to do exactly this, because its old object backed string column had no missing value of its own and `str(float("nan"))` is `"nan"`. pandas 3 gave the `str` dtype a real missing value and the behaviour changed with it. So the rule below is not an old habit that firepanda inherited by accident, it is a rule pandas arrived at on purpose and recently.

## What pandas does at the boundary

Measured on pandas 3.0.5, one line each.

| source | `astype("str")` | missing rows carried |
| --- | --- | --- |
| `float64` with a NaN | dtype `str` | 1 |
| `float64` with a NaN, to `string` | dtype `string` | 1 |
| `datetime64` with a `NaT` | dtype `str` | 1 |
| `timedelta64` with a `NaT` | dtype `str` | 1 |
| `int64` | dtype `str` | 0, there were none |
| `float64` holding both infinities | dtype `str` | 0, they are values and spell `inf` and `-inf` |

The rule is one sentence. Casting to text preserves missingness, across every source type that has a missing value, and it does not touch the infinities. The infinities matter here because the tempting shortcut is a check for "is this float strange", and two of the three strange floats are ordinary values that a column holds on purpose.

The reverse direction is the same rule read backwards. `pd.Series(["1.5", None, "nan"]).astype("float64")` gives two missing rows out of three, so the missing string and the literal word both come back as a NaN. That second one is worth noticing: pandas will read the word back even though it will not write it.

## Where the translation belongs

There were two candidates for where the NaN becomes a null, and they are not equivalent.

It could go in the pandas facing layer, on the grounds that the missing value spelling is a pandas question and the kernels speak Arrow. That reading is wrong here, and the reason is that the kernel package already made this decision and made it the other way. `present_bitmap_any` reads a float NaN as an absent row. `missing_count_any` counts it. The mask that `dropna` is built on drops it. Three kernels in `firepanda/kernel/nulls.mojo` already treat a float NaN as missing, so a fourth one doing the same thing is not a new policy leaking down a layer, it is an existing shared reading being applied at the one boundary in the package where the missing row has somewhere else to go.

So the outbound half is in `firepanda/kernel/cast.mojo`, inside `cast_to_strings`, and it is one branch: a floating point source whose value is a NaN appends a null instead of appending its rendering. Everything else about the function is unchanged, which is what the sixty three identical values earned it.

## The half the kernel cannot do

The inbound half is not symmetrical and does not belong in the same place. A cast out of text produces a cleared validity bit, because that is the only thing a text column has to hand, and whether that bit should become a NaN depends on what the answer's type is and on who is asking. An Arrow consumer wants the bit. A pandas program must never be handed a float column carrying a null, because pandas has no such column and the whole widening rule this suite scores against exists to make sure it never sees one.

That is a question about what pandas would have said rather than about what is in the buffers, and the library already has a line drawn exactly there. `Array.null_count` counts cleared bits and nothing else. `Series.null_count` counts the cleared bits plus the NaNs. An `Array` is Arrow and reports the buffers, a `Series` is pandas and reports what pandas would report. The inbound translation goes on the `Series` side of that line, so `Series.cast` and `DataFrame.cast` put a float answer's missing rows back into its values and the kernel is left alone.

It applies only when the column that was read was text, and the first draft got that wrong by applying it to every cast that answers a float. A float64 narrowing to a float32 is Arrow at both ends, nothing about how it spells a missing row is changing, and normalising its bitmap away would drop information the caller handed in and would change what the Arrow writer emits. The existing test that caught it is one line, `df.cast("score", DType.float32)` asserting the null count is still one, and it is the reason the rule in this document says "crosses between text and a number" rather than "answers a float". The move is a change of spelling, so it happens exactly where the spelling changes.

An integer answer keeps its missing row in the bitmap in both directions, because an integer has no NaN to hold it. pandas widens to float64 instead. Choosing between widening and raising there is the error model milestone rather than this one, and the honest thing is to say so in a docstring rather than to pick one quietly inside a cast.

## Naming the case that measures the other half

The failing case only ever exercised the outbound leg, and a fix to a round trip that is only tested in one direction is a fix that is half asserted. The suite has no string frame of numerals with a missing row in it, so the text to number direction cannot be reached by loading a frame. It can be reached by going out and coming back.

`basics/astype-round-trip` is `df["value"].astype("str").astype(df["value"].dtype)` over `int64_no_nulls` and `float64_no_nulls`. In pandas both come back identical to the column that went out, NaN included, which was measured before the case was written rather than after. The integer frame is the formality and the float frame is the question, because its NaN has to leave the values on the way out and arrive back in them on the way in. A library that writes the word `nan` passes the outbound leg of this case and then reads a word back as a number, so the case fails in both directions at once, which is what makes it worth having.

The driver reads the target type off the column instead of naming it, since the case runs on two frames of different types and asking for the wrong one would turn a round trip into a conversion.

## The corpus gap

There is still no frame of numerals as text with a missing row. Nothing in the suite loads one, so the whole text to number direction is only reachable through a round trip, and a round trip cannot see a text column that firepanda did not write. A frame that arrives from a CSV with an empty field in a numeric column is the ordinary case a user hits on their first afternoon and this suite cannot currently ask about it. That is a corpus item rather than a case item and it is written down here so it is not rediscovered a third time.

## The rule

A cast that crosses between text and a number carries the missing row across, in both directions, and the two directions live in different places for a reason that is worth stating rather than a reason of convenience. Going out, the missing row moves into the bitmap in the kernel, because the kernels already agree that a float NaN is an absent row. Coming back, it moves into the values in the frame layer, because whether a null should become a NaN is a question about pandas and the kernel does not answer those.

The infinities are not part of this rule. They are values.

## The board after

The `basics` section has no failing runs left, which it has not been able to say since the board was first scored. Two runs were added by the round trip case and both pass, and the run that was failing now passes, so the board goes from 626 passing runs to 629 and the failure count from 5 records to 4, with the ratchet floor going from 4 callables to 3 and `Series.astype` coming off it. The failures that remain on the whole board are two missing index levels on a two key group by and two skew values differing in the twelfth significant figure. Both are the library rather than the harness, which is the first time in this suite's life that every remaining failure has been.
