"""When two answers count as the same answer.

Implements `docs/specs/05-comparison.md`.

This is the part of a differential suite that decides whether it produces bugs or
noise, and it is the part usually left to whatever `assert_frame_equal` does by
default. It is not left to that here. `assert_frame_equal` compares floats at a
relative 1e-5, which is loose enough to hide a genuinely wrong reduction, it has a
`check_like` that will reorder columns for you, and it knows nothing about which of
two answers came from the implementation under test.

Three ideas hold the whole module up.

The default is strict, and everything that relaxes it is opt in per case and
recorded. A suite where relaxations are invisible drifts, because every one of them
is reasonable on the day it is added. A suite that prints "1841 pass, of which 63
under a declared relaxation" does not drift, because the second number is
embarrassing enough to be looked at.

A case picks a tolerance class and never a number. There are three classes, they
correspond to three real reasons that two correct implementations differ in the last
bits, and a case that needs more room picks the next class up and says why. Letting a
case name a float is how a suite ends up with one case at 1e-3 that nobody remembers
agreeing to.

Both answers are normalized to Arrow and compared there, so the comparison is about
values and not about two libraries' `repr`. Normalization is also where an index
becomes data, which is the single biggest source of false failures in a suite like
this one, because `df.groupby("k").sum()` puts `k` in the index and a frame library
that has not built an index yet puts it in a column, and that is not a difference in
the answer.

Usage:
    from fpcompat.compare import Rules, Tolerance, compare

    verdict = compare(pandas_answer, firepanda_answer, Rules(tolerance=Tolerance.ACCUMULATION))
    if not verdict:
        print("\\n".join(verdict.differences))
"""

from __future__ import annotations

import builtins
import enum
import math
import warnings
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

# How many differences a verdict carries before it stops collecting. A failing case
# with 10000 wrong rows is one bug, and a report that prints all of them buries the
# next case. The count of the rest is kept, because "the first 8 of 10000" and "the
# first 8 of 8" are very different bugs.
MAX_DIFFERENCES = 8


class Tolerance(enum.Enum):
    """The float tolerance classes. A case picks one of these and never a number.

    `EXACT` is not the default for floats and is not reachable by picking a smaller
    number, it is its own class, used by cases about values that must round trip
    unchanged rather than be computed.

    `SINGLE` is one operation on one value: a cast, an add, a comparison, a string
    to float conversion. 1e-12 is far tighter than the 1e-5 `assert_frame_equal`
    uses and looser than exact, which is the right place for something that has been
    through one rounding.

    `ACCUMULATION` is a sum, a mean, a cumulative anything, a dot product. A tree
    reduction and a linear one differ in the last bits and that is expected rather
    than wrong, because a parallel sum that reproduced the serial answer exactly
    would not be a parallel sum.

    `STATISTICAL` is variance, standard deviation, skew, kurtosis and correlation.
    These subtract quantities of similar magnitude, so a two pass implementation and
    a Welford one genuinely differ by more than the inputs suggest, and pretending
    otherwise means either a false failure or an implementation forced to be slow to
    match an arbitrary bit pattern.
    """

    EXACT = None
    SINGLE = 1e-12
    ACCUMULATION = 1e-9
    STATISTICAL = 1e-7

    @property
    def relative(self) -> float | None:
        """The relative tolerance, or None when the class is exact."""
        return self.value


# The relaxations, with the reason each one exists. A name not in here cannot be
# declared, which is the mechanism that stops the set from growing quietly: adding
# one means editing this dictionary and writing the sentence, in a diff somebody
# reviews.
RELAXATIONS: dict[str, str] = {
    "grouped_order": (
        "pandas sorts group keys by default and an engine whose group by is a hash "
        "aggregation does not have to, so both answers are sorted by the key columns "
        "before comparing. A case specifically about sort= must not declare this, "
        "because it would be testing nothing."
    ),
    "row_order": (
        "The four places pandas itself documents the output order as undefined. "
        "Everywhere else row order is compared exactly, because an engine that "
        "returns the right rows in the wrong order has a bug a user will hit, and "
        "sorting here to be safe means they hit it in their own program instead."
    ),
}

# There is deliberately no `column_order` here. The first draft of this module had
# one, on the reasoning that pandas sometimes builds a column set from a dictionary
# and its order is an implementation detail. No case needed it, and a relaxation
# added because it seemed reasonable is exactly the thing the design of this module
# exists to prevent. If a case turns out to need it, adding it means editing this
# dictionary and document 05 in the same pull request, which is the point.


class RelaxationError(ValueError):
    """A case declared a relaxation that does not exist."""


@dataclass(frozen=True)
class Rules:
    """What a case declares about how its answer should be compared.

    Everything here defaults to the strict reading. A case that leaves this alone
    gets an exact structural comparison with a 1e-12 relative tolerance on floats,
    which is what almost every case should want.

    Attributes:
        tolerance: The float tolerance class. Integers and booleans ignore it,
            because an integer answer that is off by one is a bug and there is no
            reading of that sentence where it is not.
        relaxations: Names from `RELAXATIONS`. Anything else raises.
        reason: Why this case needs a class looser than `SINGLE` or a relaxation. It
            is required in that case, because the report prints it and a relaxation
            nobody can explain is a relaxation to delete.
        strict_index: Compare a default `RangeIndex` instead of dropping it. Set by
            cases that are specifically about the index, which are the cases where
            the global drop would be hiding the thing under test.
        signed_zero: Compare zeroes bitwise, so that -0.0 and 0.0 differ. Four cases
            use this, all of them about sign preservation: `abs`, `min`, `max` and
            `sum` over a column of zeroes of both signs.
    """

    tolerance: Tolerance = Tolerance.SINGLE
    relaxations: frozenset[str] = frozenset()
    reason: str = ""
    strict_index: bool = False
    signed_zero: bool = False

    def __post_init__(self) -> None:
        unknown = sorted(set(self.relaxations) - set(RELAXATIONS))
        if unknown:
            raise RelaxationError(
                f"no such relaxation: {', '.join(unknown)}. The set is "
                f"{', '.join(sorted(RELAXATIONS))} and it is deliberately closed, so "
                "adding one means editing RELAXATIONS in compare.py and writing down "
                "why it exists."
            )
        if self.needs_reason and not self.reason:
            raise RelaxationError(
                "a case that declares a relaxation or a tolerance looser than SINGLE "
                "has to say why, because the scoreboard prints the reason and a "
                "relaxation nobody can explain is a relaxation to delete"
            )

    @property
    def needs_reason(self) -> bool:
        """Whether this case has departed from the strict default at all."""
        looser = self.tolerance in (Tolerance.ACCUMULATION, Tolerance.STATISTICAL)
        return looser or bool(self.relaxations)

    def without(self, name: str) -> Rules:
        """Returns these rules with one relaxation turned off.

        This is what the "is every declared relaxation actually needed" check runs.
        A case that still passes without one of its relaxations declared a relaxation
        it does not need, and the tool that finds those deletes them.

        Args:
            name: The relaxation to drop.

        Returns:
            A new `Rules`.
        """
        return replace(self, relaxations=frozenset(self.relaxations) - {name}, reason=self.reason)


@dataclass
class Verdict:
    """The result of one comparison.

    Attributes:
        equal: Whether the two answers are the same answer under the rules.
        differences: Up to `MAX_DIFFERENCES` readable lines, most structural first.
        extra: How many differences were found beyond the ones listed.
        relaxations_used: Which declared relaxations actually changed something. A
            relaxation that is declared and never used is caught by this without
            having to run the case a second time with it disabled.
    """

    equal: bool = True
    differences: list[str] = field(default_factory=list)
    extra: int = 0
    relaxations_used: frozenset[str] = frozenset()

    def __bool__(self) -> bool:
        return self.equal

    def note(self, line: str) -> None:
        """Records one difference and marks the verdict unequal.

        Args:
            line: The difference, phrased so that it makes sense on its own in a
                report next to a case id.
        """
        self.equal = False
        if len(self.differences) < MAX_DIFFERENCES:
            self.differences.append(line)
        else:
            self.extra += 1

    def summary(self) -> str:
        """One line for a report."""
        if self.equal:
            used = ", ".join(sorted(self.relaxations_used))
            return f"equal (under {used})" if used else "equal"
        tail = f", and {self.extra} more" if self.extra else ""
        return f"{len(self.differences) + self.extra} differences: {self.differences[0]}{tail}"


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


def canonical_type(kind: pa.DataType) -> str:
    """Renders an Arrow type as the thing conformance actually cares about.

    Exact type equality would be wrong in three places, all of them cases where Arrow
    is expressing a storage decision that pandas does not have a dtype for, so a
    difference there is not something a user can observe.

    A string is a string. Arrow has `string`, `large_string` and `string_view` and
    the difference between them is a 32 bit offset, a 64 bit offset and an inline
    prefix. pandas has one string dtype. firepanda's strings are views, so comparing
    the physical width would fail every string case in the suite for a reason no user
    could see. Binary and list widths are folded for the same reason.

    Dictionary index width is folded too, and that one is in the specification
    directly: two libraries can represent the same categorical with different code
    assignments and both be right, so the codes are compared through the categories
    rather than as integers, and int8 codes against int32 codes is not a difference.

    Everything else is exact. Integer width is exact, float width is exact, timestamp
    resolution is exact because pandas 3.0 carries it on the dtype and a microsecond
    answer where pandas gives nanoseconds is a failure rather than a rounding
    question, timezone is exact and is compared by name, and struct field order is
    exact because it is part of the Arrow type.

    Args:
        kind: The Arrow type.

    Returns:
        A canonical rendering, so that comparison is string comparison and the
        difference message is something a person can read.
    """
    if pa.types.is_string(kind) or pa.types.is_large_string(kind) or kind == pa.string_view():
        return "string"
    if pa.types.is_binary(kind) or pa.types.is_large_binary(kind) or kind == pa.binary_view():
        return "binary"
    if pa.types.is_dictionary(kind):
        return f"categorical<{canonical_type(kind.value_type)},ordered={kind.ordered}>"
    if pa.types.is_fixed_size_list(kind):
        return f"list[{kind.list_size}]<{canonical_type(kind.value_type)}>"
    if pa.types.is_list(kind) or pa.types.is_large_list(kind) or pa.types.is_list_view(kind):
        return f"list<{canonical_type(kind.value_type)}>"
    if pa.types.is_struct(kind):
        fields = ",".join(f"{f.name}:{canonical_type(f.type)}" for f in kind)
        return f"struct<{fields}>"
    if pa.types.is_map(kind):
        return f"map<{canonical_type(kind.key_type)},{canonical_type(kind.item_type)}>"
    return str(kind)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

# Arrow column names are positional, which sidesteps every problem pandas column
# labels cause: they can be integers, they can be tuples, and they can repeat. The
# labels themselves are compared separately, as rendered strings, so `1` and `"1"`
# are still different column names.
INDEX_PREFIX = "__index__"
VALUE = "__value__"


@dataclass
class Answer:
    """One side of a comparison, normalized.

    Attributes:
        kind: `frame`, `series`, `index`, `array`, `scalar`, `tuple`, `mapping` or
            `python`. Kinds never compare equal across kinds: a Series and a one
            column DataFrame are different answers and a suite that treated them as
            the same would pass a library that returned the wrong one.
        table: Index columns first, then data columns, named positionally.
        n_index: How many leading columns of `table` came from the index.
        columns: The rendered data column labels, in order.
        index_names: The rendered index level names.
        default_index: Whether the index was a plain 0 to n-1 `RangeIndex` with no
            name, which is the one thing that gets dropped globally.
        name: The `name` of a Series or an Index, rendered.
        parts: The elements, for a tuple answer.
        value: The value, for a scalar or a python answer.
        type_name: The canonical type, for a scalar.
    """

    kind: str
    table: pa.Table | None = None
    n_index: int = 0
    columns: tuple[str, ...] = ()
    index_names: tuple[str, ...] = ()
    default_index: bool = False
    name: str | None = None
    parts: tuple[Answer, ...] = ()
    value: Any = None
    type_name: str | None = None


def _label(value: Any) -> str:
    """Renders a pandas label so two labels compare as strings.

    A string renders as itself and everything else renders with its type in front,
    which is what keeps the integer `1` and the string `"1"` apart. They are different
    column names in pandas, and `repr` alone is not enough to tell them apart because
    `repr(1)` is the string `"1"`, which is the first version of this function and
    which a test caught.

    Args:
        value: The label.

    Returns:
        The rendering.
    """
    return value if isinstance(value, str) else f"{type(value).__name__}({value!r})"


def _is_default_index(index: pd.Index) -> bool:
    """Whether an index is a plain 0 to n-1 RangeIndex carrying no information.

    Args:
        index: The index.

    Returns:
        True when the index is not data.
    """
    return (
        isinstance(index, pd.RangeIndex)
        and index.name is None
        and index.start == 0
        and index.step == 1
    )


def _keeps_its_nans(values: Any) -> bool:
    """Reports whether this is a float column whose NaNs are values and not nulls.

    pyarrow converts a pandas object using pandas' own idea of missing, which for a
    numpy backed float column means every NaN becomes an Arrow null. That is the
    right answer for somebody moving data between the two libraries and the wrong
    one for a conformance harness, because it throws away the distinction on the
    oracle side of a comparison that is about to insist on it. A subject engine that
    answered with a null where pandas answered with a NaN would be handed a pass.

    So a numpy float column is converted with pandas' rules switched off and its
    NaNs arrive as NaNs. Nothing else changes. An extension dtype carries a real
    mask, its `pd.NA` is a null under either rule, and it keeps its NaNs already.

    Args:
        values: The sequence about to be converted.

    Returns:
        True when the conversion has to keep pandas out of it.
    """
    dtype = getattr(values, "dtype", None)
    if dtype is None:
        return False
    # An extension dtype has no `.kind` reachable this way in every pandas version,
    # and it is not what this is about in any of them, so it is asked directly.
    if isinstance(dtype, pd.api.extensions.ExtensionDtype):
        return False
    return getattr(dtype, "kind", "") == "f"


def _array(values: Any) -> pa.Array:
    """Converts a pandas or numpy sequence to an Arrow array.

    The fallback matters. A pandas object column can hold anything, including things
    Arrow has no type for, and a conversion failure there must not become a harness
    crash that gets attributed to the library under test. Those columns are rendered
    to strings and compared as strings, which is weaker than comparing values and is
    still enough to catch a wrong answer.

    Args:
        values: The sequence.

    Returns:
        An Arrow array.
    """
    try:
        if _keeps_its_nans(values):
            return pa.array(values, from_pandas=False)
        return pa.array(values)
    except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError, ValueError):
        rendered = [None if value is None or value is pd.NA else repr(value) for value in values]
        return pa.array(rendered, type=pa.large_string())


def _index_arrays(index: pd.Index) -> tuple[list[pa.Array], tuple[str, ...]]:
    """Splits an index, including a MultiIndex, into columns and level names.

    Args:
        index: The index.

    Returns:
        The arrays and the rendered level names.
    """
    if isinstance(index, pd.MultiIndex):
        arrays = [_array(index.get_level_values(level)) for level in range(index.nlevels)]
        names = tuple(_label(name) for name in index.names)
        return arrays, names
    return [_array(index)], (_label(index.name),)


def _from_frame(frame: pd.DataFrame) -> Answer:
    """Normalizes a DataFrame, moving the index into leading columns."""
    index_arrays, index_names = _index_arrays(frame.index)
    arrays = list(index_arrays)
    names = [f"{INDEX_PREFIX}{i}" for i in range(len(index_arrays))]
    for position in range(frame.shape[1]):
        arrays.append(_array(frame.iloc[:, position]))
        names.append(f"c{position}")
    return Answer(
        kind="frame",
        table=pa.Table.from_arrays(arrays, names=names),
        n_index=len(index_arrays),
        columns=tuple(_label(label) for label in frame.columns),
        index_names=index_names,
        default_index=_is_default_index(frame.index),
    )


def _from_series(series: pd.Series) -> Answer:
    """Normalizes a Series the same way, with one data column."""
    index_arrays, index_names = _index_arrays(series.index)
    arrays = [*index_arrays, _array(series)]
    names = [f"{INDEX_PREFIX}{i}" for i in range(len(index_arrays))] + [VALUE]
    return Answer(
        kind="series",
        table=pa.Table.from_arrays(arrays, names=names),
        n_index=len(index_arrays),
        columns=(VALUE,),
        index_names=index_names,
        default_index=_is_default_index(series.index),
        name=None if series.name is None else _label(series.name),
    )


def _scalar_type(value: Any) -> str:
    """The canonical type of a scalar answer.

    pandas has three spellings of missing and Arrow can build a scalar from none of
    them, so they are named here rather than crashing the conversion. `NaT` and `NA`
    are kept apart from each other and from `None`, because a library that returns
    one where pandas returns another has changed what the user sees.

    Args:
        value: The scalar.

    Returns:
        The rendering.
    """
    if value is pd.NaT:
        return "null[NaT]"
    if value is pd.NA:
        return "null[NA]"
    if value is None:
        return "null"
    try:
        return canonical_type(pa.scalar(value).type)
    except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError, ValueError):
        return f"python[{type(value).__name__}]"


def normalize(answer: Any) -> Answer:
    """Normalizes any answer an engine can return.

    Args:
        answer: A DataFrame, Series, Index, array, scalar, tuple or anything else.

    Returns:
        The normalized answer.
    """
    if isinstance(answer, Answer):
        return answer
    if isinstance(answer, pd.DataFrame):
        return _from_frame(answer)
    if isinstance(answer, pd.Series):
        return _from_series(answer)
    if isinstance(answer, pd.Index):
        return Answer(
            kind="index",
            table=pa.Table.from_arrays([_array(answer)], names=[VALUE]),
            columns=(VALUE,),
            name=None if answer.name is None else _label(answer.name),
        )
    if isinstance(answer, pa.Table):
        return Answer(
            kind="frame",
            table=answer,
            columns=tuple(answer.column_names),
            index_names=(),
            default_index=True,
        )
    if isinstance(answer, pa.Array | pa.ChunkedArray):
        return Answer(
            kind="array",
            table=pa.Table.from_arrays([answer.combine_chunks()], names=[VALUE]),
            columns=(VALUE,),
        )
    if isinstance(answer, pd.api.extensions.ExtensionArray):
        # A Categorical, an ArrowStringArray, a DatetimeArray and everything else
        # pandas returns from `unique` and from `Series.array`. These have to be an
        # array and not a scalar, and going through `_array` keeps a Categorical's
        # categories and its ordered flag, which the dictionary comparison then reads.
        return Answer(
            kind="array",
            table=pa.Table.from_arrays([_array(answer)], names=[VALUE]),
            columns=(VALUE,),
        )
    if isinstance(answer, np.ndarray):
        if answer.ndim == 1:
            return Answer(
                kind="array",
                table=pa.Table.from_arrays([_array(answer)], names=[VALUE]),
                columns=(VALUE,),
            )
        return Answer(kind="python", value=answer.tolist(), type_name="ndarray")
    if isinstance(answer, tuple):
        return Answer(kind="tuple", parts=tuple(normalize(part) for part in answer))
    if isinstance(answer, dict):
        return Answer(
            kind="mapping",
            value={_label(key): normalize(value) for key, value in answer.items()},
        )
    return Answer(kind="scalar", value=answer, type_name=_scalar_type(answer))


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


def _floats_equal(left: float, right: float, rules: Rules) -> bool:
    """Compares two floats under a tolerance class.

    NaN equals NaN, which `==` does not do and which every differential suite needs,
    since a column of NaN is a perfectly good answer. The infinities compare exactly
    and are not equal to each other, and an infinity is never within a relative
    tolerance of a finite number no matter how large that number is.

    Args:
        left: One value.
        right: The other.
        rules: The case rules.

    Returns:
        Whether they are the same value.
    """
    if math.isnan(left) or math.isnan(right):
        return math.isnan(left) and math.isnan(right)
    if math.isinf(left) or math.isinf(right):
        return left == right
    if left == right:
        # -0.0 == 0.0 is true in IEEE and that is the answer for value comparison,
        # except in the four cases that are specifically about sign preservation.
        if rules.signed_zero and left == 0.0:
            return math.copysign(1.0, left) == math.copysign(1.0, right)
        return True
    relative = rules.tolerance.relative
    if relative is None:
        return False
    scale = max(abs(left), abs(right))
    return abs(left - right) <= relative * scale


def _values_equal(left: Any, right: Any, rules: Rules) -> bool:
    """Compares two decoded Arrow values, recursing into lists and structs.

    A null is not a NaN and an empty list is not a null list, at every level. Those
    two are the distinctions a library conflates first and they are the two that
    silently corrupt somebody's data, so they are checked before anything else here.

    Args:
        left: One value.
        right: The other.
        rules: The case rules.

    Returns:
        Whether they are the same value.
    """
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, float) or isinstance(right, float):
        if not isinstance(left, float) or not isinstance(right, float):
            return False
        return _floats_equal(left, right, rules)
    if isinstance(left, list) or isinstance(right, list):
        if not isinstance(left, list) or not isinstance(right, list):
            return False
        if len(left) != len(right):
            return False
        return all(_values_equal(a, b, rules) for a, b in zip(left, right, strict=True))
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            return False
        if list(left) != list(right):
            return False
        return all(_values_equal(left[key], right[key], rules) for key in left)
    return bool(left == right)


def _render(value: Any) -> str:
    """Renders one value for a difference message.

    Args:
        value: The value.

    Returns:
        A short rendering, with `null` distinguishable from `nan` and from the string
        `"null"`, because a report that cannot tell those apart wastes the reader's
        time on exactly the bugs that are hardest to see.
    """
    if value is None:
        return "null"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if value == 0.0 and math.copysign(1.0, value) < 0:
            return "-0.0"
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def _has_float(kind: pa.DataType) -> bool:
    """Whether a type contains a float anywhere, including inside a list or a struct.

    Args:
        kind: The Arrow type.

    Returns:
        Whether a float is in there somewhere.
    """
    if pa.types.is_floating(kind):
        return True
    if pa.types.is_struct(kind):
        return any(_has_float(f.type) for f in kind)
    if pa.types.is_dictionary(kind):
        return _has_float(kind.value_type)
    if kind.num_fields == 1:
        return _has_float(kind.field(0).type)
    return False


def _compare_categories(
    left: pa.ChunkedArray, right: pa.ChunkedArray, label: str, verdict: Verdict
) -> None:
    """Compares the category sets of two dictionary columns.

    Ordered as a list, unordered as a set, because two libraries can assign codes in
    a different order and both be right when nothing depends on that order. Whether
    an unused category survived is compared either way, since that is a real semantic
    pandas users depend on and it is the whole of what `observed=` is about.

    Args:
        left: The pandas side.
        right: The other side.
        label: The column label for the message.
        verdict: Collects the differences.
    """
    left_categories = left.combine_chunks().dictionary.to_pylist()
    right_categories = right.combine_chunks().dictionary.to_pylist()
    if left.type.ordered:
        if left_categories != right_categories:
            verdict.note(
                f"{label}: the categorical is ordered and the categories are in a "
                f"different order, {left_categories} against {right_categories}"
            )
        return
    if set(left_categories) != set(right_categories):
        only_left = sorted(set(left_categories) - set(right_categories), key=repr)
        only_right = sorted(set(right_categories) - set(left_categories), key=repr)
        verdict.note(
            f"{label}: different categories, missing {only_right} and unexpected "
            f"{only_left}. An unused category that did not survive is a real "
            "difference and not a detail, since that is what observed= is about"
        )


def _compare_column(
    left: pa.ChunkedArray, right: pa.ChunkedArray, label: str, rules: Rules, verdict: Verdict
) -> None:
    """Compares one column, values and nulls and type.

    Args:
        left: The pandas side.
        right: The other side.
        label: The column label for the messages.
        rules: The case rules.
        verdict: Collects the differences.
    """
    left_type, right_type = canonical_type(left.type), canonical_type(right.type)
    if left_type != right_type:
        verdict.note(f"{label}: dtype {right_type}, expected {left_type}")
        return
    if left.null_count != right.null_count:
        verdict.note(
            f"{label}: {right.null_count} nulls, expected {left.null_count}. A null "
            "is not a NaN and this comparison keeps them apart"
        )
    if pa.types.is_dictionary(left.type):
        _compare_categories(left, right, label, verdict)

    # The fast path, which is what almost every passing case takes. It is skipped for
    # a signed zero case because `pa.array([0.0]).equals(pa.array([-0.0]))` is True:
    # Arrow compares the values and IEEE says those are equal, which is the right
    # answer everywhere except the four cases that exist to check the sign.
    if not (rules.signed_zero and _has_float(left.type)) and left.equals(right):
        return
    left_values = left.to_pylist()
    right_values = right.to_pylist()
    for row, (a, b) in enumerate(zip(left_values, right_values, strict=True)):
        if not _values_equal(a, b, rules):
            verdict.note(f"{label} row {row}: {_render(b)}, expected {_render(a)}")
            if verdict.extra:
                return


def _sort_key(table: pa.Table, columns: list[str]) -> np.ndarray:
    """Returns the permutation that sorts a table by some of its columns.

    Sorted on the rendered values rather than through `sort_indices`, because the key
    columns of a grouped answer can be dictionary encoded or nested and Arrow sorts
    neither. Nulls sort first and consistently on both sides, which is all this
    needs: the permutation only has to be the same permutation for equal data, not
    any particular one.

    This is the slow path and it is slow in the way an interpreter is slow, a Python
    tuple and several strings per row. `_arrow_sort_key` is tried first and this runs
    when Arrow refuses the key columns. Both sides of a comparison always take the
    same path, which is the property that matters: two permutations produced by two
    different rules would disagree about equal data and fail the case.

    Args:
        table: The table.
        columns: The column names to sort on.

    Returns:
        Row indices.
    """
    keys = [table.column(name).to_pylist() for name in columns]
    rows = range(table.num_rows)
    return np.fromiter(
        sorted(rows, key=lambda row: tuple(_render(key[row]) for key in keys)),
        dtype=np.int64,
        count=table.num_rows,
    )


def _arrow_sort_key(table: pa.Table, columns: list[str]) -> np.ndarray | None:
    """The same permutation, computed by Arrow, or nothing when Arrow will not.

    Arrow declines dictionary encoded, list and struct columns, which is why the
    rendered sort exists at all. It takes everything else, and everything else is
    almost every answer in the registry, including the merge answers with ten
    million rows in them that the rendered sort spends minutes on.

    Args:
        table: The table.
        columns: The column names to sort on.

    Returns:
        Row indices, or None when a key column is a type Arrow cannot sort.
    """
    # Null placement per key rather than as an option, which is what pyarrow 25
    # wants and what earlier versions deprecated their way towards.
    keys = [(name, "ascending", "at_start") for name in columns]
    try:
        indices = pc.sort_indices(table, sort_keys=keys)
    except (pa.ArrowNotImplementedError, pa.ArrowInvalid, pa.ArrowTypeError):
        return None
    return indices.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)


def _is_identity(order: np.ndarray) -> bool:
    """Whether a permutation leaves every row where it was.

    Which is how a relaxation gets recorded as used or not: an order relaxation that
    reordered nothing did nothing.

    Args:
        order: Row indices.

    Returns:
        True when the permutation is 0, 1, 2 and so on.
    """
    return bool(np.array_equal(order, np.arange(len(order), dtype=np.int64)))


def _widen_views(kind: pa.DataType) -> pa.DataType:
    """Returns the non-view type a view type holds the same values as.

    Arrow's `take` has no kernel for the view layouts. That is a gap in pyarrow 25 and
    not a fact about either engine, but it lands here rather than there, because
    firepanda's strings are views and an order relaxation on a string key is the
    ordinary case rather than an exotic one. A grouped comparison on a text key
    reported "the comparison itself raised ArrowNotImplementedError", which is the
    harness admitting it has a bug and is exactly what should not sit in a result
    file.

    Widening rather than narrowing, because a view holds 64 bit offsets internally and
    `string` does not, so a long column would overflow on the way in. Nothing is lost
    to the comparison either way: `canonical_type` already folds all three string
    widths to one name, on the grounds that pandas has one string dtype and the
    difference between a 32 bit offset, a 64 bit offset and an inline prefix is not
    something a user can observe.

    Args:
        kind: An Arrow type.

    Returns:
        The same type with any view layout replaced, recursing through lists and
        structs, since a view can be nested inside either.
    """
    if kind == pa.string_view():
        return pa.large_string()
    if kind == pa.binary_view():
        return pa.large_binary()
    if pa.types.is_list(kind) or pa.types.is_large_list(kind) or pa.types.is_list_view(kind):
        return pa.large_list(_widen_views(kind.value_type))
    if pa.types.is_struct(kind):
        return pa.struct([f.with_type(_widen_views(f.type)) for f in kind])
    return kind


def _take(table: pa.Table, order: np.ndarray) -> pa.Table:
    """Reorders a table's rows, widening any view column first so that Arrow can.

    Args:
        table: The table.
        order: Row indices, in output order.

    Returns:
        The reordered table.
    """
    widened = pa.schema([field.with_type(_widen_views(field.type)) for field in table.schema])
    if widened != table.schema:
        table = table.cast(widened)
    return table.take(pa.array(order))


def _apply_ordering(
    left: Answer, right: Answer, rules: Rules, verdict: Verdict
) -> tuple[pa.Table, pa.Table]:
    """Applies whichever order relaxations the case declared.

    Records which ones actually changed something, so a relaxation that was declared
    and never needed is visible without running the case twice.

    Args:
        left: The pandas side.
        right: The other side.
        rules: The case rules.
        verdict: Collects which relaxations were used.

    Returns:
        The two tables, possibly reordered.
    """
    left_table, right_table = left.table, right.table
    if left_table is None or right_table is None:
        return left_table, right_table
    used = set(verdict.relaxations_used)

    columns: list[str] | None = None
    if "grouped_order" in rules.relaxations:
        columns = list(left_table.column_names[: left.n_index]) or list(left_table.column_names)
        name = "grouped_order"
    elif "row_order" in rules.relaxations:
        columns = list(left_table.column_names)
        name = "row_order"

    if columns and left_table.num_rows == right_table.num_rows:
        right_columns = [c for c in columns if c in right_table.column_names]
        # One decision for both sides. If Arrow will not sort one of them, neither of
        # them goes through Arrow, because the two rules do not agree on where a null
        # sits or on whether 9 comes before 10, and a case whose sides were ordered by
        # two different rules fails on data that is equal.
        left_order = _arrow_sort_key(left_table, columns)
        right_order = None if left_order is None else _arrow_sort_key(right_table, right_columns)
        if left_order is None or right_order is None:
            left_order = _sort_key(left_table, columns)
            right_order = _sort_key(right_table, right_columns)
        if not _is_identity(left_order) or not _is_identity(right_order):
            used.add(name)
        left_table = _take(left_table, left_order)
        right_table = _take(right_table, right_order)

    verdict.relaxations_used = frozenset(used)
    return left_table, right_table


def _compare_tabular(left: Answer, right: Answer, rules: Rules, verdict: Verdict) -> None:
    """Compares two answers that both became tables.

    Args:
        left: The pandas side.
        right: The other side.
        rules: The case rules.
        verdict: Collects the differences.
    """
    if left.name != right.name:
        verdict.note(f"name {right.name!r}, expected {left.name!r}")

    drop_index = not rules.strict_index and left.default_index and right.default_index
    if drop_index:
        # The one relaxation applied globally rather than per case, and the reason a
        # frame carrying a plain 0 to n-1 index is comparable to one that has no
        # index at all. When either side carries anything else, both are compared.
        left = replace(
            left,
            table=left.table.drop(left.table.column_names[: left.n_index]),
            n_index=0,
            index_names=(),
        )
        right = replace(
            right,
            table=right.table.drop(right.table.column_names[: right.n_index]),
            n_index=0,
            index_names=(),
        )
    elif left.index_names != right.index_names:
        verdict.note(f"index names {right.index_names}, expected {left.index_names}")

    if left.n_index != right.n_index:
        # Different numbers of index levels, and the two tables therefore have
        # different numbers of columns, so there is nothing further to compare. This
        # has to return rather than fall through: the column loop below is indexed by
        # the left side's shape, and running it against a narrower right side is an
        # IndexError out of pyarrow rather than a difference, which the runner then
        # reports as a bug in this file. It was one, and this is it.
        #
        # The message is worth reading twice, because this is the shape of every
        # failure a library with no index produces against pandas. An answer with no
        # index is only comparable to a pandas answer whose index is a plain range,
        # and `DataFrame.tail` does not produce one of those.
        verdict.note(
            f"{right.n_index} index levels, expected {left.n_index}. A pandas answer "
            "whose index is a plain 0 to n-1 range compares equal to one with no "
            "index, and this index is not that, so the index is part of the answer"
        )
        return

    left_columns, right_columns = list(left.columns), list(right.columns)
    if left_columns != right_columns:
        missing = [c for c in left_columns if c not in right_columns]
        unexpected = [c for c in right_columns if c not in left_columns]
        detail = (
            f"missing {missing} and unexpected {unexpected}"
            if missing or unexpected
            else "the same columns in a different order, which is a difference "
            "because column order is something pandas promises"
        )
        verdict.note(f"columns: {detail}, got {right_columns}, expected {left_columns}")
        return

    if left.table.num_rows != right.table.num_rows:
        verdict.note(f"{right.table.num_rows} rows, expected {left.table.num_rows}")
        return

    left_table, right_table = _apply_ordering(left, right, rules, verdict)

    index_labels = [f"index {name}" for name in (left.index_names or ())]
    labels = index_labels[: left.n_index] + [f"column {label}" for label in left_columns]
    for position, label in enumerate(labels):
        _compare_column(
            left_table.column(position), right_table.column(position), label, rules, verdict
        )


def compare(left: Any, right: Any, rules: Rules | None = None) -> Verdict:
    """Compares two answers.

    The first argument is the oracle and the second is the subject, which only
    matters for the wording of the messages: they read as "got this, expected that"
    with pandas on the expected side.

    Args:
        left: The pandas answer.
        right: The other engine's answer.
        rules: What the case declared. Strict when omitted.

    Returns:
        The verdict.
    """
    rules = rules or Rules()
    verdict = Verdict()
    a, b = normalize(left), normalize(right)

    if a.kind != b.kind:
        verdict.note(
            f"{b.kind}, expected {a.kind}. A Series and a one column frame are "
            "different answers and this suite does not treat them as the same"
        )
        return verdict

    if a.kind in ("frame", "series", "index", "array"):
        _compare_tabular(a, b, rules, verdict)
        return verdict

    if a.kind == "tuple":
        if len(a.parts) != len(b.parts):
            verdict.note(f"{len(b.parts)} values in the tuple, expected {len(a.parts)}")
            return verdict
        for position, (one, other) in enumerate(zip(a.parts, b.parts, strict=True)):
            inner = compare(one, other, rules)
            if not inner:
                for line in inner.differences:
                    verdict.note(f"tuple[{position}] {line}")
        return verdict

    if a.kind == "mapping":
        if list(a.value) != list(b.value):
            verdict.note(f"keys {list(b.value)}, expected {list(a.value)}")
            return verdict
        for key in a.value:
            inner = compare(a.value[key], b.value[key], rules)
            if not inner:
                for line in inner.differences:
                    verdict.note(f"[{key}] {line}")
        return verdict

    if a.type_name != b.type_name:
        verdict.note(f"scalar type {b.type_name}, expected {a.type_name}")
        return verdict
    if not _values_equal(_scalar_value(a.value), _scalar_value(b.value), rules):
        verdict.note(f"{_render(b.value)}, expected {_render(a.value)}")
    return verdict


def _scalar_value(value: Any) -> Any:
    """Reduces a scalar to something the value comparison understands.

    Args:
        value: The scalar.

    Returns:
        A Python value, with the three pandas spellings of missing mapped to None
        after `_scalar_type` has already kept them apart.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def same(left: Any, right: Any, **kwargs: Any) -> bool:
    """Whether two answers are equal, for a test that does not want the detail.

    Args:
        left: The pandas answer.
        right: The other answer.
        **kwargs: Passed to `Rules`.

    Returns:
        Whether they are equal.
    """
    return bool(compare(left, right, Rules(**kwargs)))


# ---------------------------------------------------------------------------
# Errors and warnings
# ---------------------------------------------------------------------------


def resolve_error(name: str) -> type[BaseException]:
    """Looks up an exception type by the name a case declares.

    Args:
        name: `KeyError`, `pandas.errors.MergeError`, `MergeError`, and so on.

    Returns:
        The class.

    Raises:
        LookupError: When no such type exists, which is a broken case rather than a
            conformance failure and should stop the run rather than count as one.
    """
    short = name.rsplit(".", 1)[-1]
    # pandas first, because `pandas.errors` shadows nothing in builtins and the cases
    # that matter here are the 46 pandas types rather than `ValueError`. pyarrow is
    # last and it is there because the Arrow backed accessors let Arrow's own errors
    # through unchanged, so `ArrowInvalid` is part of what a caller sees and there is
    # no honest way to describe those cases without naming it.
    for namespace in (pd.errors, builtins, pa.lib):
        found = getattr(namespace, short, None)
        if isinstance(found, type) and issubclass(found, BaseException):
            return found
    raise LookupError(
        f"no exception type called {name}. The case names something that does not "
        "exist, which is a bug in the case and not a conformance failure"
    )


def check_error(raised: BaseException | None, name: str, substring: str) -> Verdict:
    """Checks a raised exception against what a case declared.

    The type has to match exactly, including which of the 46 `pandas.errors` types it
    is, because `MergeError` and `ValueError` are a different experience for anyone
    catching one. The message is only checked for a substring, and that substring is
    the thing a user would search for, which is a column name, a dtype name or a
    value. Everything past it is not compared, because pandas rewords its messages
    between releases and pinning them would turn a pandas upgrade into a hundred
    failing cases that are all the same non bug.

    Args:
        raised: What was raised, or None when nothing was.
        name: The expected type name.
        substring: What the message has to contain.

    Returns:
        The verdict.
    """
    verdict = Verdict()
    expected = resolve_error(name)
    if raised is None:
        verdict.note(f"nothing was raised, expected {expected.__name__}")
        return verdict
    if type(raised) is not expected:
        verdict.note(
            f"raised {type(raised).__name__}, expected exactly {expected.__name__}. "
            "A subclass is not a match, because the type is what a user catches"
        )
    if substring and substring not in str(raised):
        verdict.note(
            f"the message does not contain {substring!r}, which is what a user would "
            f"search for. The message was {str(raised)!r}"
        )
    return verdict


def check_warnings(
    caught: list[warnings.WarningMessage], expected: tuple[str, str] | None
) -> Verdict:
    """Checks the warnings raised against what a case declared.

    A case that expects no warning asserts that none was raised, which catches the
    opposite failure from the usual one: a library that warns where pandas does not
    breaks somebody's `-W error` build, and that user will report it as a crash.

    Args:
        caught: What `warnings.catch_warnings(record=True)` collected.
        expected: The type name and the message substring, or None for no warning.

    Returns:
        The verdict.
    """
    verdict = Verdict()
    if expected is None:
        if caught:
            names = sorted({type(item.message).__name__ for item in caught})
            verdict.note(
                f"warned {names} where pandas warns about nothing, which is a "
                "difference a user running with -W error finds as a crash"
            )
        return verdict

    name, substring = expected
    wanted = resolve_error(name)
    matches = [item for item in caught if type(item.message) is wanted]
    if not matches:
        raised = sorted({type(item.message).__name__ for item in caught}) or ["nothing"]
        verdict.note(f"warned {raised}, expected {wanted.__name__}")
        return verdict
    if substring and not any(substring in str(item.message) for item in matches):
        verdict.note(
            f"no {wanted.__name__} message contains {substring!r}, and the messages "
            f"were {[str(item.message) for item in matches]}"
        )
    return verdict
