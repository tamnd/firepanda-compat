# The frequency string

`dt.floor`, `dt.ceil` and `dt.round` each take a frequency string, and every one of them is one division and one multiplication once that string has become a number. So the whole of the rounding trio is a parser and three lines of arithmetic, and all of the difficulty is in the parser and in one rule about ties. This document is what a running pandas 3.0.3 answered when it was asked, which is not what the pandas documentation says and is not what a reimplementation would guess. It follows `14-the-dt-accessor.md`, which scoped the group.

## The grammar, measured

A frequency is an optional sign, an optional count, and an alias. The count may carry a decimal point. Space at either end is trimmed and space in the middle is not allowed.

Seven aliases round, and they are `D`, `h`, `min`, `s`, `ms`, `us` and `ns`. That is the whole list. Everything else pandas knows about is a non fixed frequency, meaning one whose length in seconds depends on where in the calendar it lands, and pandas refuses those rather than picking an average. `W` and `ME` and `QE` and `YS` and `B` all raise, with `<Week: weekday=6> is a non-fixed frequency` and the like.

The long spellings do not work. `day`, `hour`, `minute`, `second`, `seconds`, `millisecond`, `micro` and `nano` all raise. So do the single letter forms `S`, `L`, `U` and `N`, and so do `T` for a minute and `H` for an hour. Those four were the pandas 2 spellings and pandas 3 removed them. A library that accepts them is a library that takes input pandas rejects, which is a smaller problem than rejecting input pandas takes and is still a difference, so they are refused here too.

## Four answers that are not what anybody would guess

**A frequency finer than the column is not an error.** `dt.floor('ms')` on a column of whole seconds answers the column unchanged. So does `dt.floor('1500ms')`, because a period of one and a half seconds becomes a period of one second when it is expressed in the column's own unit, and flooring to a period of one does nothing. `dt.floor('2500ms')` on the same column is a real two second rounding, so the truncation is into the column unit and not to the nearest alias.

**A count of zero is not an error either.** `dt.floor('0h')` answers the column unchanged, which is the same answer the finer case gives and by the same route: the period is zero and there is nothing to divide by. Both of those have to be handled before the division rather than inside it, because a division by zero is a trap and not an answer.

**A decimal count is exact and not floating point.** `dt.floor('1.5h')` on a second column is a period of exactly 5400 seconds. Carrying the count as fifteen over ten rather than as a float is what makes that true for any number of digits, and it also makes `1500ms` and `2500ms` come out at one and two rather than at whatever the float rounds to.

**A negative frequency is accepted, and pandas' own answer for it does not follow from its answer for the positive one.** `dt.floor('-1h')` is the floored quotient by a period of minus one hour, which moves an instant forward in time rather than back. `dt.round('-1h')` sends the epoch itself to the hour before it and sends the hour before the epoch to two hours before it. Neither of those is what anybody wants and both of them are what pandas does, measured on twelve rows.

## The tie rule, which is one rule and not two

`dt.round` breaks a tie towards the even multiple, the same way the numeric round does. Half past midnight rounds back to midnight and half past one rounds forward to two. That much is documented. What is not documented is how it behaves when the period is negative, and the naive form of the rule, comparing the size of the remainder against half the size of the period, gets the positive case right and the negative case wrong.

The rule that reproduces both is stated on the signed period. Take the floored quotient `q` and the floored remainder `r`. If twice `r` is greater than the period, step `q` on by one. If twice `r` equals the period, step `q` on only when `q` is odd. Otherwise leave it. With a positive period `r` is in `[0, p)` and this is the ordinary half to even. With a negative period `r` is in `(p, 0]` and the comparison flips with it, which is exactly what produces the answers above. It was checked against all twelve rows at both signs before it was written down.

Ceiling has the same shape. Take the floored quotient and step it on by one whenever the multiple it names is not the value itself. That is right for both signs too, and it is right for a row that is already on a period, which is the row an implementation that adds a period and then floors gets wrong while passing everything else.

## Resolution changes

`dt.as_unit` is the other half of this group and has no frequency in it. Going down in precision floors rather than truncating towards zero, so half a second before the epoch is minus one second and not zero, which is the same hazard the calendar fields have and is a different division. Going up in precision is a multiply, and pandas raises `OutOfBoundsDatetime` when the result will not fit in an int64, which is reachable: the year 2300 is a perfectly ordinary second column and is outside what an int64 of nanoseconds holds.

That last check has to read the rows that are really there. A null row holds whatever the file that produced it left in the buffer, so a cheap vector scan over the whole values buffer is allowed to say yes when the only large value is under a cleared validity bit, and a row by row pass then settles it. Doing it the other way round, one row at a time from the start, would put a bitmap lookup in front of every row of a kernel that is otherwise a multiply.

`dt.as_unit` is also the one thing on the accessor a zoned column is allowed to do. Which second an instant is does not depend on the clock it is being read against, so there is no zone database standing in front of it, and the zone comes through untouched. Everything else in the group is refused for a zoned column, because pandas rounds the local reading and the stored instants are UTC, and in a zone whose offset is not a whole number of hours those are two different answers rather than one answer written twice.

## What this is worth

Six cases and fourteen runs: `temporal/floor`, `temporal/ceil` and `temporal/round` over the range frame and the resolutions frame, `temporal/round-minute` over the range frame, and `temporal/as-unit` and `temporal/as-unit-up` over the resolutions frame.

The frequency parser is worth more than that, because M7 needs it. `resample` and `asfreq` take the same strings and the offset frequencies extend the same grammar, so the parser written here is the one those use, and the seven fixed aliases are the part of it that does not change when the offsets arrive.

## How it was checked

Five hundred instants drawn from a fixed seed and spread over two hundred and fifty years, at all four resolutions, through nine frequencies and all three modes, plus all sixteen resolution conversions. That is 62000 answers, generated once by pandas and once by firepanda and diffed. They were identical.

Twenty seven tests hold the parts of that a reader can check by eye, and the fuzzer runs the vector kernel against the one row at a time twin on a random column with a random period one case in eight. The twin divides with a truncating divide and a correction rather than with `//`, so agreement between the two is evidence about the answer rather than about the operator.

## What stays out

A frequency with an anchor on it, so `W-MON` and `QE-JAN`, which are offsets and are M7. `resample` and `asfreq`, which take these strings and do something else with them. The `ambiguous` and `nonexistent` keywords on the rounding methods, which only mean anything once there is a zone database.
