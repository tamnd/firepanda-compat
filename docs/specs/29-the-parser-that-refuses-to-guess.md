# 29. The parser that refuses to guess

Document 28 measured what `pandas.to_datetime` would be worth before it existed. Forty eight runs on the `.dt` accessor alone, plus its own two cases, plus whatever the temporal behaviour cases turn out to hold. This document is what happened when it was built, and it is unusual among these documents in that the prediction was exactly right and the interesting part is somewhere else entirely.

## The number

Passing runs went from 903 to 955 out of 4081. Failures stayed at nine and divergences at seven.

Fifty two, made of three parts. Forty eight are the `.dt` accessor rows document 28 measured in advance, unblocked without a line of accessor code changing. Two are `resolution/pandas.to_datetime` and `signature/pandas.to_datetime`, the constructor's own cases. Two are L3 cases the corpus has had since the rounding slice and no engine here could answer until now, which are the subject of half of this document.

The temporal section is the row that moved. It was zero at every rung, all 236 callables reporting unimplemented behind one missing constructor. It now reads L0 13, L1 13, L2 12, L3 7, and 22 of its runs pass. That is still 227 callables to go, but the section is now measuring the section rather than measuring a door, which is what document 28 said it could not do.

## What pandas actually does, which is not what anybody remembers

Every expected value in this slice was read off a running pandas 3.0.5. Three of the answers were surprises, and each one is a place where the obvious implementation is wrong and passes every test somebody would write from memory.

The result unit is microseconds. pandas moved off nanoseconds by default in version 3, so a column of ordinary dates is `datetime64[us]`, and it goes to nanoseconds only when some row carried more than six digits after the decimal point. The unit is therefore a property of the whole column decided by its single most precise row, and that row can be anywhere, so the unit is not known until every row has been read. An implementation that normalises to nanoseconds on the way in and rescales at the end gives numbers a thousand times too large and passes any test whose column has one row in it.

The format is guessed once, from the first row that is not missing, and every other row is held to it. A column holding `2026-01-01` and `2026-01-02T03:04:05` is a ValueError in pandas rather than a column with two shapes in it. The friendly behaviour, reading each row on its own terms, is the one a person would build, and pandas hides it behind `format='mixed'` on purpose.

`nan` is missing and `null` is not. The list pandas treats as missing is the empty string, `NaT` and `nan`, and it refuses `None`, `null`, `NA` and a lone hyphen. A reader who is generous with that list silently accepts files pandas rejects, which is the worst kind of incompatibility because nothing reports it.

The guesser also refuses everything that is not ISO 8601. `01/02/2026` raises rather than being read as either ordering. That is a refusal firepanda copies deliberately, because guessing an ordering is guessing a continent.

## The bug the tests did not have

The firepanda driver gained entries for two temporal cases the corpus had been carrying since the rounding slice with nothing able to answer them. Both are round trips: render a timestamp column to text with `dt.strftime`, read it back with `to_datetime`, and require the column that comes out to be the column that went in. One uses the guesser and one passes `%d/%m/%Y` to exercise a shape the guesser will not touch.

They failed on the first run, at row five of the standard `temporal_range` frame.

    1 differences: column __value__ row 5: datetime.datetime(1715, 6, 13, 0, 25, 26, 290448), expected datetime.datetime(2300, 1, 1, 0, 0)

Every row was being folded to nanoseconds and rescaled to the column's unit afterwards. Nanoseconds is the finest unit there is and it is also the narrowest range: an Int64 of them reaches 1677 to 2262 and nothing further. 2300-01-01 is 1.04e19 nanoseconds, past the Int64 ceiling of 9.22e18, and the wrap lands in the summer of 1715. Nothing raised. The column looked fine.

The fix is to fold at the column's own unit rather than at nanoseconds, which means the fields have to be carried until the unit is known, which is the same fact from the section above arriving a second time from the other direction. Once folded at microseconds, 2300 fits with room for another two hundred and ninety two thousand years, and a column that genuinely is nanoseconds because some row had nine digits raises with the year and the unit in the message instead of wrapping.

This matters because of what did not find it. Twenty three Mojo tests written specifically for the parser, and thirty three Python tests comparing against a live pandas, all passed. Every one of those tests used a date somebody chose while thinking about parsing, and nobody thinking about parsing picks the year 2300. The corpus frame picked it, because the corpus frame was built to span resolutions and ranges and was built by somebody else for another purpose entirely.

That is the argument for the corpus in one sentence. A unit test runs your code over the inputs you thought of, and a corpus case runs your code over inputs chosen without reference to your implementation. The second one is the only one that can surprise you, and this slice is the clearest instance so far: two cases written months before the parser was, twenty lines of driver code to point them at it, one silent correctness bug in a code path that already had fifty six passing tests over it.

It is also the second time in three documents that the board found a defect the feature's own tests could not. Document 28's was a property that answered itself when read off the class. This one is an overflow. Neither is a compatibility question in the narrow sense, and both were found because a conformance suite runs your code somewhere its author was not looking.

## The second door

The forty eight accessor runs did not arrive when `to_datetime` shipped. The first board run after it moved by two.

Every `dt` case still reported that a column is built from a sequence of values. The reason is that the board builds the namespace by writing `pd.Series(pd.to_datetime([...]))`, so a series goes straight back into the series constructor, and the constructor was reading it as a sequence, iterating it out to Python values and inferring a type off them. For a column of instants that inference gives whole numbers, so the timestamp column became an integer column and the accessor could not attach.

The fix is that `Series(a_series)` copies the column rather than iterating it, which is what pandas does. Two things were wrong before, not one. It was slow for the columns it got right, and it lost the type of the columns it did not. After the fix the board went to 953, exactly the forty eight plus two.

Document 28's rule was that when a whole namespace reports one sentence, the sentence is the work item. The refinement this adds is that a door can have a second door behind it, and that the board is what tells you, within seconds, that you have opened only the first one. A count that moves by two when it was predicted to move by fifty is not a disappointment, it is a diagnostic, and it is only available because the prediction was written down first.

## The parameters that are refused by name

`pandas.to_datetime` has ten parameters. Five are declared and refused: `dayfirst` and `yearfirst`, because the guesser reads ISO 8601 and nothing else, `origin`, because an epoch other than 1970 has to move every value, `exact`, because it asks whether a format may match part of a value and this parser reads all of a value or none of it, and `format='mixed'` and `format='ISO8601'`, because they ask for the format to be worked out per row. `cache` is accepted and changes nothing, honestly, because it is a performance hint and there is nothing to cache.

That shape is why this is the first module level function in the pandas surface written by hand rather than generated. The generator writes one call per table entry and cannot express five refusals with five different reasons. Being hand written is also why it is named in the signature parity test explicitly: a signature nothing generates is a signature nothing keeps in step, and the parity walk against a running pandas is what keeps it.

The general point is one document 26 made about bindings and is worth restating for parameters. A parameter that is silently ignored is worse than a parameter that is missing, because a program that passes `dayfirst=True` and gets American ordering back has been given a wrong answer rather than an error. Each of the five raises with its own name in the message.

## What this argues for next

`pandas.date_range` in its fixed frequency form, which is the other half of the constructor work document 28 named, and the last thing standing between the temporal section and its remaining 227 callables.

One case is worth filing before that. The board's pandas corpus contains `errors/out-of-bounds-datetime`, which asks for a pandas style `OutOfBoundsDatetime` raised out of a nanosecond column that cannot hold the value. firepanda now raises the right refusal in the right place with the year and the unit in the message, but it raises a plain error, so the case needs an error type rather than any new behaviour. That is a small piece of work whose value is that it is the first case on the board asking not what the answer is but what the exception is called, and there will be many more of those.

The report ask from documents 13 and 28 is now asked a third time. Fifty five rows carrying one sentence was the example then, and this slice turned it into fifty two rows that moved together for one reason. Message bucketing would have shown both as a single line, and would have shown the second door as a single line the moment the first one opened.
