# An hour that is missing or doubled

## What 37 left

[37](37-the-database-that-was-already-there.md) put the zone database in and stopped at the default. A reading the clock skipped or repeated raised pandas' own `ValueError`, and every other value of `ambiguous` and `nonexistent` was refused by name. That was the right place to stop, because the default is what the two cases on the board ask about. But the other values are what anybody actually writes once they have seen the default fail, and the refusal covered `tz_localize`, `floor`, `ceil` and `round` on a series and on a `DatetimeIndex`, which is eight doors.

This document is what pandas 3.0.3 does with each value, measured rather than read off the docstring, and the one place where what it does is not what the docstring suggests.

## The two lists

`ambiguous` is about a reading the clock showed twice, which happens once a year when it goes back. pandas takes `"raise"`, `"NaT"`, `"infer"`, a single `True` or `False`, and a list with one flag per row. `True` means the first of the two instants, the one still on daylight saving time, and that holds in both hemispheres, since a clock going back is always leaving daylight time whichever month it happens in. A string pandas does not know is not refused. It is carried down to the repeated hour and raises the ordinary error about the hour, so `ambiguous="bogus"` on a column with no repeated reading answers normally, and firepanda does the same.

`nonexistent` is about a reading the clock skipped, which happens once a year when it goes forward. pandas takes `"raise"`, `"NaT"`, `"shift_forward"`, `"shift_backward"` and a timedelta, and it does check this one: a string outside the four is a `ValueError` listing them, which firepanda already matched.

A list of flags that is the wrong length raises `Length of ambiguous bool-array must be the same size as vals`, before any reading is looked at.

## The shifts go by the hour

The docstring says `shift_forward` moves a skipped reading "forward to the closest existing time". For New York that is three o'clock, and three o'clock is also the next whole hour, so the two readings agree. Lord Howe Island tells them apart, because its clock goes forward by thirty minutes, from two to half past. The closest existing time after a quarter past two is half past two. pandas answers three.

The reason is in how pandas computes it. It takes the reading, rounds it up to the next whole hour of the clock, and reads the result on the offset after the change. `shift_backward` rounds down to the start of the hour and takes one unit off, so a quarter past two on Lord Howe goes to one second before two, or one microsecond or one nanosecond before, depending on the column's resolution. A timedelta is added to the reading as it is, and pandas then checks that the result has left the hour the reading was in, refusing with `The provided timedelta will relocalize on a nonexistent time: 0 days 00:20:00` if it has not. That check is against the hour too and not against the gap, so on Lord Howe a shift of forty minutes from a quarter past two, which would land on five to three and outside the gap, is still refused.

firepanda follows pandas here rather than the docstring. The answer pandas gives is the one a program written against pandas already has, and a library that answered half past two on Lord Howe would be more right and would break the comparison for the one zone where anybody would notice.

## A list of flags is two questions

The kernel takes the policy as a value, which is two small integers and a shift, and it runs the same parallel pass it ran for the default. A list of flags does not fit in that value, and carrying a column of flags across the boundary for the one case that needs it would make every other call pay for it. So a list is answered in Python by asking twice, once as though every flag were `True` and once as though every one were `False`, and taking each row from the side its flag names with `where`. A row that is not a repeated reading answers the same either way, so only the rows the flags are about can tell the two apart. When every flag agrees, it is one call.

## What is still refused

`ambiguous="infer"` is the one value still refused. It does not take a decision from the caller. It reads the order of the rows to find the point where the clock went back, which is a scan across rows rather than a question about each one, and it has failure modes of its own that want measuring first. The scalar `Timestamp` on a rule zone is still refused as well, since it is pure Python and the question there is whether to hand it to Python's own `zoneinfo`.
