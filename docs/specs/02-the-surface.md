# The surface being matched

Every number in this document was produced by `pixi run surface` in the compat repository against pandas 3.0.3 and pyarrow 25.0.0 on 4 September 2026, not from reading the pandas documentation. The tool walks each namespace, keeps the public names, counts the callables, and reads `inspect.signature` for the parameter count. Its output is committed as `surface/pandas-3.0.3.json` so that a future pandas release shows up as a diff rather than as a surprise.

## The inventory

| Namespace | Public names | Callables | Parameters |
|---|---|---|---|
| top level `pandas` | 119 | 102 | 616 |
| `DataFrame` | 203 | 186 | 1037 |
| `Series` | 203 | 180 | 849 |
| `Index` | 91 | 73 | 210 |
| `MultiIndex` | 108 | 86 | 252 |
| `DatetimeIndex` | 144 | 91 | 261 |
| `.str` | 57 | 57 | 104 |
| `.dt` | 42 | 13 | 19 |
| `.cat` | 11 | 8 | 8 |
| `.list` | 2 | 2 | 0 |
| `.struct` | 3 | 2 | 1 |
| `GroupBy` | 64 | 58 | 191 |
| `Rolling` | 33 | 22 | 63 |
| `Expanding` | 33 | 22 | 61 |
| `ExponentialMovingWindow` | 27 | 9 | 25 |
| `Resampler` | 34 | 27 | 50 |
| `Timestamp` | 75 | 42 | 47 |
| `Timedelta` | 22 | 10 | 8 |
| `offsets` | 47 | 46 | 3 |
| `errors` | 50 | 46 | 5 |
| `api.types` | 45 | 45 | 54 |
| **Total** | **1413** | **1127** | **3864** |

`DataFrame` and `Series` share 174 names and have 29 each of their own, so the two together are 232 distinct names rather than 406. The totals above are per namespace and are not deduplicated, because a name that behaves differently on a frame and on a series is two behaviours to match and `apply` is the obvious example.

Two rows deserve a second look. `.dt` has 42 public names and only 13 callables, because 29 of them are properties, and a property is not less work to implement than a method. `offsets` has 46 callables carrying 3 parameters between them, because the parameters live on the constructors of 46 separate classes that this counting method does not open. Both numbers are lower bounds, and where a count is a lower bound the tool says so in the JSON rather than in a comment here.

## What document 06 does not mention

Document 06 in the parent folder names 444 distinct symbols. The tool subtracts those from the inventory, and what is left is the honest gap list. It is not a criticism of that document, which was written as a plan rather than as an inventory, and the whole reason this repository exists is that the difference matters.

**The `Index` object, entirely.** 91 public names, of which 30 are not mentioned anywhere in document 06: `union`, `intersection`, `difference`, `symmetric_difference`, `get_loc`, `get_indexer`, `get_indexer_non_unique`, `get_level_values`, `slice_locs`, `slice_indexer`, `asof_locs`, `append`, `delete`, `putmask`, `has_duplicates`, `identical`, `is_`, `inferred_type`, `nlevels`, `names`, `set_names`, `sortlevel`, `to_flat_index`, `to_series`, `ravel`, `view`, `nbytes`, `drop`, `get_slice_bound`, `get_indexer_for`. Document 06 says the index is optional and explicit, and it then describes only the frame methods that use one. `loc` is built on `get_loc` and `get_indexer`, `reindex` is built on `get_indexer`, and a merge on an index is built on `get_indexer_non_unique`. The index API is not an add on to the index work, it is the index work.

`MultiIndex` adds 108 names on top and document 06 mentions three of them.

**The arithmetic method spellings.** Document 06 names `add`, `sub`, `mul`, `div`, `mod`, `pow` and says "and the `r`-prefixed reflected forms". The actual surface also has `truediv`, `floordiv`, `divide`, `multiply`, `subtract`, `radd`, `rsub`, `rmul`, `rdiv`, `rtruediv`, `rfloordiv`, `rmod`, `rpow`, `rdivmod`, and the six comparison methods `eq`, `ne`, `lt`, `le`, `gt`, `ge`. That is 20 names for six operations, they all take `axis`, `level` and `fill_value`, and they are cheap to implement and cheap to forget.

**The error and warning types.** 46 of them, and document 06 does not name one. Covered in document 01 section on L4.

**`pandas.api.types`.** 45 predicates, `is_numeric_dtype` through `is_datetime64tz_dtype`, which every piece of library code that accepts a dataframe calls. A program that consumes a firepanda frame and asks `pd.api.types.is_string_dtype(col)` is asking a question our frame has to be able to answer.

**The options system.** 70 option keys under `describe_option`. Document 06 has one line for `set_option` and friends. Most of the 70 are display options and are not load bearing, and `mode.chained_assignment`, `future.*` and `compute.use_numexpr` are read by real code.

**Named individual gaps.** `case_when` on `Series`, which is new and is the pandas answer to a chained where. `from_arrow` on both `DataFrame` and `Series`, which is the pandas 3.0 constructor from anything exporting the Arrow C interface and is exactly the interop path this project cares about. `read_iceberg` and `to_iceberg`. `iterrows` and `itertuples`, which are slow, which are in every pandas program written by a beginner, and which therefore have to exist. `add_prefix`, `add_suffix`, `isetitem`, `first_valid_index`, `last_valid_index`, `keys`, `set_flags`, `drop`, and the aliases `kurtosis` for `kurt` and `product` for `prod`. On the accessors, `str.isascii` and `groupby.ohlc`. On the windows, `rolling.method` and `rolling.on`, the second of which is how a time based window names its column.

None of these is hard. All of them are the difference between a program running and a program raising `AttributeError` on line 40.

## What this does to the plan

Three conclusions, and they are the reason document 08 in this folder rewrites the M6 checklist rather than ticking it.

**The M6 index work is bigger than one bullet.** `loc`, `iloc`, `at` and `iat` sit on top of an `Index` type with 73 callables and a `MultiIndex` with 86 more. The parity checklist has this as one line. It is a milestone sized piece of work on its own and it is the thing every other M6 item leans on, because `groupby.apply`, `unstack`, `value_counts` and `describe` all produce indexed output.

**The cheap names should land first and together.** The 20 arithmetic spellings, the 10 forgotten frame methods and the alias pairs are a day of work between them and they move the L0 and L1 numbers more than anything else on the list. They are exactly the kind of work that never gets prioritised because each one alone feels too small to be worth an issue, which is why they are one issue.

**The error types are a milestone of their own and should start now.** Every operation that can fail should raise the pandas type from the first day it exists, because retrofitting error types across 1127 callables afterwards is the kind of task that never gets done. The cost per operation is one line at the point of raising, if the exception types exist to be raised. They should exist before M6 lands anything, which makes them a prerequisite rather than a follow up.
