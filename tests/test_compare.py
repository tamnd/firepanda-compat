"""Tests for the comparison layer.

The most important test in this file is the last one, which compares every frame in
the corpus against itself under zero relaxations and expects exact equality. That is
the smallest possible version of the oracle self test from document 05, and it is
what catches a normalizer that quietly drops a column, mangles an index, or decides
that a null and a NaN are the same thing. Without it, the first bug in the normalizer
gets published as ten firepanda failures, and every one of those is an hour of
somebody's day spent debugging the wrong repository.

The rest of the file is a test per rule. Each one is written as two answers that
differ in exactly one way, because a test that changes two things at once passes for
the wrong reason about half the time.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from fpcompat import corpus
from fpcompat.compare import (
    RELAXATIONS,
    Rules,
    Tolerance,
    canonical_type,
    check_error,
    check_warnings,
    compare,
    normalize,
    same,
)

RELAXED = Rules(relaxations=frozenset({"grouped_order"}), reason="a test")


# ---------------------------------------------------------------------------
# The rules a case declares
# ---------------------------------------------------------------------------


def test_a_relaxation_that_does_not_exist_is_refused():
    """The mechanism that keeps the set of relaxations closed."""
    with pytest.raises(ValueError, match="no such relaxation"):
        Rules(relaxations=frozenset({"close_enough"}), reason="it nearly passes")


def test_a_relaxation_without_a_reason_is_refused():
    with pytest.raises(ValueError, match="has to say why"):
        Rules(relaxations=frozenset({"row_order"}))


def test_a_looser_tolerance_class_needs_a_reason():
    with pytest.raises(ValueError, match="has to say why"):
        Rules(tolerance=Tolerance.STATISTICAL)
    Rules(tolerance=Tolerance.STATISTICAL, reason="Welford against two pass")


def test_the_strict_default_needs_no_reason():
    assert Rules().tolerance is Tolerance.SINGLE
    assert Rules().relaxations == frozenset()


def test_every_relaxation_is_written_down():
    """A name with no sentence beside it is a relaxation nobody has to justify."""
    for name, why in RELAXATIONS.items():
        assert len(why) > 80, name


def test_without_drops_one_relaxation():
    """What the "is this relaxation needed" check runs."""
    rules = Rules(relaxations=frozenset({"grouped_order", "row_order"}), reason="a test")
    assert rules.without("row_order").relaxations == frozenset({"grouped_order"})


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


def test_the_three_string_widths_are_one_dtype():
    """Arrow's string widths are a storage decision pandas has no dtype for."""
    widths = {canonical_type(t) for t in (pa.string(), pa.large_string(), pa.string_view())}
    assert widths == {"string"}


def test_integer_and_float_widths_are_not_folded():
    assert canonical_type(pa.int32()) != canonical_type(pa.int64())
    assert canonical_type(pa.float32()) != canonical_type(pa.float64())


def test_dictionary_index_width_is_folded_and_ordered_is_not():
    """Two libraries can assign codes differently and both be right, so the codes are
    compared through the categories. The `ordered` flag is a promise and is not."""
    narrow = pa.dictionary(pa.int8(), pa.large_string(), ordered=True)
    wide = pa.dictionary(pa.int32(), pa.string(), ordered=True)
    assert canonical_type(narrow) == canonical_type(wide)
    unordered = pa.dictionary(pa.int8(), pa.large_string(), ordered=False)
    assert canonical_type(narrow) != canonical_type(unordered)


def test_struct_field_order_is_part_of_the_type():
    one = pa.struct([("a", pa.int64()), ("b", pa.int64())])
    other = pa.struct([("b", pa.int64()), ("a", pa.int64())])
    assert canonical_type(one) != canonical_type(other)


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


def test_integers_get_no_tolerance_at_any_class():
    left = pd.DataFrame({"a": [1, 2, 3]})
    right = pd.DataFrame({"a": [1, 2, 4]})
    for tolerance in Tolerance:
        rules = Rules(tolerance=tolerance, reason="a test")
        assert not compare(left, right, rules)


def test_the_float_classes_are_the_widths_they_say():
    base = pd.DataFrame({"a": [1.0]})
    for tolerance, inside, outside in (
        (Tolerance.SINGLE, 1e-13, 1e-11),
        (Tolerance.ACCUMULATION, 1e-10, 1e-8),
        (Tolerance.STATISTICAL, 1e-8, 1e-6),
    ):
        rules = Rules(tolerance=tolerance, reason="a test")
        assert compare(base, pd.DataFrame({"a": [1.0 + inside]}), rules), tolerance
        assert not compare(base, pd.DataFrame({"a": [1.0 + outside]}), rules), tolerance


def test_exact_is_a_class_and_not_a_small_number():
    """One unit in the last place apart, which is as close as two float64 values get
    without being the same value. `1.0 + 1e-16` is not, it is `1.0`."""
    left = pd.DataFrame({"a": [1.0]})
    right = pd.DataFrame({"a": [np.nextafter(1.0, 2.0)]})
    assert compare(left, right)
    assert not compare(left, right, Rules(tolerance=Tolerance.EXACT))


def test_nan_equals_nan():
    """Which `==` does not do, and a column of NaN is a perfectly good answer."""
    nan = pd.DataFrame({"a": [float("nan"), 1.0]})
    assert compare(nan, nan.copy())


def test_the_infinities_compare_exactly():
    inf = pd.DataFrame({"a": [float("inf")]})
    assert compare(inf, inf.copy())
    assert not compare(inf, pd.DataFrame({"a": [float("-inf")]}))
    assert not compare(inf, pd.DataFrame({"a": [1e308]}))


def test_infinity_is_never_within_a_tolerance_of_a_finite_number():
    rules = Rules(tolerance=Tolerance.STATISTICAL, reason="a test")
    assert not compare(pd.DataFrame({"a": [float("inf")]}), pd.DataFrame({"a": [1e308]}), rules)


def test_negative_zero_is_a_value_difference_only_where_the_case_says_so():
    """Four cases in the suite are about sign preservation. Everywhere else IEEE is
    right that these are the same number."""
    zero = pd.DataFrame({"a": [0.0, 1.0]})
    negative = pd.DataFrame({"a": [-0.0, 1.0]})
    assert compare(zero, negative)
    assert not compare(zero, negative, Rules(signed_zero=True))


def test_signed_zero_is_seen_through_a_list_column():
    """The fast path is Arrow's own `equals`, which says these are equal because IEEE
    says so, and it has to be turned off rather than trusted for these four cases."""
    left = pa.table({"v": pa.array([[0.0]])})
    right = pa.table({"v": pa.array([[-0.0]])})
    assert compare(left, right)
    assert not compare(left, right, Rules(signed_zero=True))


def test_a_null_is_not_a_nan():
    """The distinction a library conflates first and the one that corrupts data."""
    nan = pa.table({"v": pa.array([float("nan")], type=pa.float64())})
    null = pa.table({"v": pa.array([None], type=pa.float64())})
    verdict = compare(nan, null)
    assert not verdict
    assert "null" in verdict.differences[0]


def test_pandas_turns_a_nan_into_a_null_in_an_arrow_backed_column():
    """Not a rule of this module and worth pinning anyway, because it explains why the
    test above is written against Arrow rather than against pandas. Constructing a
    `float64[pyarrow]` Series from a NaN produces a null, so a test written the
    obvious way would be comparing a null to a null and passing for no reason."""
    assert pa.array(pd.Series([float("nan")], dtype="float64[pyarrow]")).null_count == 1


def test_strings_are_byte_exact_and_neither_side_normalizes():
    """The corpus carries a combining sequence and its precomposed equivalent next to
    each other for exactly this. A library that silently normalizes unicode has
    changed the user's data."""
    values = corpus.load("strings_unicode").column("value").to_pylist()
    combining, precomposed = values[0], values[1]
    assert combining != precomposed
    left = pa.table({"v": pa.array([combining])})
    right = pa.table({"v": pa.array([precomposed])})
    assert not compare(left, right)


def test_an_empty_list_is_not_a_null_list():
    left = pa.table({"v": pa.array([[], None], type=pa.list_(pa.int64()))})
    right = pa.table({"v": pa.array([None, None], type=pa.list_(pa.int64()))})
    assert not compare(left, right)


def test_nested_floats_use_the_tolerance():
    left = pa.table({"v": pa.array([[1.0, 2.0]])})
    right = pa.table({"v": pa.array([[1.0, 2.0 + 1e-13]])})
    assert compare(left, right)
    assert not compare(left, right, Rules(tolerance=Tolerance.EXACT))


# ---------------------------------------------------------------------------
# Categoricals
# ---------------------------------------------------------------------------


def test_an_unused_category_that_did_not_survive_is_a_difference():
    """Which is the whole of what `observed=` is about."""
    left = pd.Series(pd.Categorical(["a", "b"], categories=["a", "b", "z"]))
    right = pd.Series(pd.Categorical(["a", "b"], categories=["a", "b"]))
    verdict = compare(left, right)
    assert not verdict
    assert "z" in verdict.differences[0]


def test_unordered_categories_compare_as_a_set():
    left = pd.Series(pd.Categorical(["a", "b"], categories=["a", "b"]))
    right = pd.Series(pd.Categorical(["a", "b"], categories=["b", "a"]))
    assert compare(left, right)


def test_ordered_categories_compare_as_a_list():
    left = pd.Series(pd.Categorical(["a", "b"], categories=["a", "b"], ordered=True))
    right = pd.Series(pd.Categorical(["a", "b"], categories=["b", "a"], ordered=True))
    assert not compare(left, right)


def test_the_ordered_flag_itself_is_compared():
    left = pd.Series(pd.Categorical(["a", "b"], categories=["a", "b"], ordered=True))
    right = pd.Series(pd.Categorical(["a", "b"], categories=["a", "b"], ordered=False))
    assert not compare(left, right)


def test_codes_are_compared_through_the_categories():
    """Two libraries can assign codes in a different order and both be right, so long
    as the values they decode to are the same."""
    left = pd.Series(pd.Categorical(["a", "b", "a"], categories=["a", "b"]))
    right = pd.Series(pd.Categorical(["a", "b", "a"], categories=["b", "a"]))
    assert list(left.cat.codes) != list(right.cat.codes)
    assert compare(left, right)


# ---------------------------------------------------------------------------
# Temporal
# ---------------------------------------------------------------------------


def test_the_resolution_is_compared_exactly():
    """A microsecond answer where pandas gives nanoseconds is a failure and not a
    rounding question, because pandas 3.0 carries the resolution on the dtype."""
    stamps = pd.to_datetime(["2024-01-01"])
    left = pd.Series(stamps.astype("datetime64[us]"))
    right = pd.Series(stamps.astype("datetime64[ns]"))
    assert not compare(left, right)


def test_a_timezone_is_compared_by_name_and_not_by_offset():
    """The same instant in `America/New_York` and in `UTC` is not the same answer,
    which matches what pandas does."""
    left = pd.Series(pd.date_range("2024-01-01", periods=2, tz="UTC"))
    right = pd.Series(left.dt.tz_convert("America/New_York"))
    assert not compare(left, right)


def test_a_naive_column_is_not_a_zoned_one():
    left = pd.Series(pd.date_range("2024-01-01", periods=2))
    right = pd.Series(pd.date_range("2024-01-01", periods=2, tz="UTC"))
    assert not compare(left, right)


# ---------------------------------------------------------------------------
# Shape, names and the index
# ---------------------------------------------------------------------------


def test_a_series_is_not_a_one_column_frame():
    frame = pd.DataFrame({"a": [1, 2]})
    assert not compare(frame["a"], frame)


def test_a_column_label_that_is_a_string_is_not_one_that_is_an_integer():
    """`repr(1)` is the string `"1"`, so rendering labels with `repr` alone folds
    these two together. Both frames print identically and they are not the same
    answer."""
    assert not compare(pd.DataFrame({1: [1]}), pd.DataFrame({"1": [1]}))
    assert not compare(pd.Series([1], name=1), pd.Series([1], name="1"))


def test_column_order_is_a_difference():
    left = pd.DataFrame({"a": [1], "b": [2]})
    assert not compare(left, left[["b", "a"]])


def test_a_default_range_index_is_dropped_on_both_sides():
    """The one relaxation applied globally, and the reason a frame carrying a plain 0
    to n-1 index is comparable to one that has no index at all."""
    left = pd.DataFrame({"a": [1, 2, 3]})
    assert compare(left, pa.table({"a": [1, 2, 3]}))


def test_an_index_that_carries_information_is_compared():
    left = pd.DataFrame({"a": [1, 2]}, index=["x", "y"])
    right = pd.DataFrame({"a": [1, 2]}, index=["x", "z"])
    assert not compare(left, right)


def test_a_named_range_index_is_not_a_default_one():
    left = pd.DataFrame({"a": [1, 2]})
    right = left.rename_axis("row")
    assert not compare(left, right)


def test_strict_index_compares_the_range_index_a_case_is_about():
    left = pd.DataFrame({"a": [1, 2]})
    right = pd.DataFrame({"a": [1, 2]}, index=pd.RangeIndex(2))
    assert compare(left, right, Rules(strict_index=True))
    shifted = pd.DataFrame({"a": [1, 2]}, index=[5, 6])
    assert not compare(left, shifted, Rules(strict_index=True))


def test_a_multiindex_becomes_one_column_per_level():
    frame = pd.DataFrame({"v": [1, 2]}, index=pd.MultiIndex.from_tuples([("a", 1), ("b", 2)]))
    answer = normalize(frame)
    assert answer.n_index == 2
    assert compare(frame, frame.copy())


def test_the_index_level_names_are_compared():
    frame = pd.DataFrame({"v": [1]}, index=pd.Index(["a"], name="k"))
    renamed = frame.rename_axis("key")
    assert not compare(frame, renamed)


def test_a_series_name_is_compared():
    assert not compare(pd.Series([1], name="a"), pd.Series([1], name="b"))


def test_a_row_count_difference_stops_before_the_values():
    verdict = compare(pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [1]}))
    assert not verdict
    assert "rows" in verdict.differences[0]
    assert len(verdict.differences) == 1


# ---------------------------------------------------------------------------
# The order relaxations
# ---------------------------------------------------------------------------


def test_row_order_is_a_difference_by_default():
    """Not "sorted before comparing to be safe". An engine that returns the right rows
    in the wrong order has a bug a user will hit."""
    left = pd.DataFrame({"a": [1, 2, 3]})
    assert not compare(left, left.iloc[::-1].reset_index(drop=True))


def test_grouped_order_sorts_by_the_key_columns():
    frame = pd.DataFrame({"k": ["b", "a", "c"], "v": [1, 2, 3]})
    grouped = frame.groupby("k").sum()
    shuffled = grouped.iloc[[2, 0, 1]]
    assert not compare(grouped, shuffled)
    assert compare(grouped, shuffled, RELAXED)


def test_a_relaxation_that_changed_nothing_is_reported_as_unused():
    """So a relaxation declared and never needed is visible without running the case a
    second time with it turned off."""
    grouped = pd.DataFrame({"k": ["a", "b"], "v": [1, 2]}).groupby("k").sum()
    verdict = compare(grouped, grouped.copy(), RELAXED)
    assert verdict.equal
    assert verdict.relaxations_used == frozenset()


def test_a_relaxation_that_was_used_is_reported_as_used():
    grouped = pd.DataFrame({"k": ["b", "a"], "v": [1, 2]}).groupby("k").sum()
    verdict = compare(grouped, grouped.iloc[::-1], RELAXED)
    assert verdict.equal
    assert verdict.relaxations_used == frozenset({"grouped_order"})


def test_grouped_order_does_not_hide_a_wrong_value():
    grouped = pd.DataFrame({"k": ["a", "b"], "v": [1, 2]}).groupby("k").sum()
    wrong = grouped.copy()
    wrong.iloc[0, 0] = 99
    assert not compare(grouped, wrong, RELAXED)


def test_row_order_sorts_everything():
    left = pd.DataFrame({"a": [3, 1, 2]})
    rules = Rules(relaxations=frozenset({"row_order"}), reason="pandas documents this as undefined")
    assert compare(left, left.sort_values("a").reset_index(drop=True), rules)


# ---------------------------------------------------------------------------
# Scalars, tuples and mappings
# ---------------------------------------------------------------------------


def test_a_scalar_carries_its_type():
    assert same(np.int64(5), 5)
    assert not same(np.int64(5), np.int32(5))
    assert not same(5, 5.0)


def test_the_three_spellings_of_missing_stay_apart():
    """A library that returns `None` where pandas returns `NaT` has changed what the
    user sees, and a suite that folded these together would never say so."""
    assert not same(pd.NaT, None)
    assert not same(pd.NA, None)
    assert not same(pd.NaT, pd.NA)
    assert same(pd.NaT, pd.NaT)


def test_a_scalar_float_uses_the_tolerance():
    assert same(1.0, 1.0 + 1e-15)
    assert not compare(1.0, 1.0 + 1e-15, Rules(tolerance=Tolerance.EXACT))


def test_a_tuple_answer_compares_element_wise():
    """Which `align`, `factorize` and `divmod` all return."""
    frame = pd.DataFrame({"a": [1]})
    assert same((frame, 1), (frame.copy(), 1))
    assert not same((frame, 1), (frame.copy(), 2))
    assert not same((frame,), (frame.copy(), 1))


def test_a_mapping_answer_compares_by_key():
    assert same({"a": pd.Series([1])}, {"a": pd.Series([1])})
    assert not same({"a": pd.Series([1])}, {"a": pd.Series([2])})
    assert not same({"a": pd.Series([1])}, {"b": pd.Series([1])})


# ---------------------------------------------------------------------------
# Errors and warnings
# ---------------------------------------------------------------------------


def test_an_error_matches_on_the_type_and_a_substring():
    raised = KeyError("column 'z' is not in the frame")
    assert check_error(raised, "KeyError", "z")
    assert not check_error(raised, "ValueError", "z")
    assert not check_error(raised, "KeyError", "not a substring of anything")


def test_a_subclass_is_not_a_match():
    """`MergeError` is a `ValueError`, and a case that expects one and gets the other
    is a difference, because the type is what a user catches."""
    assert issubclass(pd.errors.MergeError, ValueError)
    assert not check_error(pd.errors.MergeError("bad merge"), "ValueError", "bad")
    assert check_error(pd.errors.MergeError("bad merge"), "MergeError", "bad")


def test_the_pandas_error_types_resolve_by_their_short_name():
    for name in ("MergeError", "ParserError", "OutOfBoundsDatetime", "IntCastingNaNError"):
        assert check_error(getattr(pd.errors, name)("x"), name, "x")


def test_nothing_raised_where_something_was_expected():
    verdict = check_error(None, "KeyError", "z")
    assert not verdict
    assert "nothing was raised" in verdict.differences[0]


def test_a_case_that_names_a_type_that_does_not_exist_stops_the_run():
    """A broken case is not a conformance failure and must not be counted as one."""
    with pytest.raises(LookupError):
        check_error(KeyError("x"), "NoSuchErrorType", "x")


def test_a_warning_that_should_not_have_fired_is_a_difference():
    """The failure a user finds as a crash under `-W error`."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn("something", FutureWarning, stacklevel=1)
    assert not check_warnings(list(caught), None)
    assert check_warnings([], None)


def test_a_warning_matches_on_the_type_and_a_substring():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn("the frobnicator is deprecated", FutureWarning, stacklevel=1)
    caught = list(caught)
    assert check_warnings(caught, ("FutureWarning", "frobnicator"))
    assert not check_warnings(caught, ("DeprecationWarning", "frobnicator"))
    assert not check_warnings(caught, ("FutureWarning", "widget"))


# ---------------------------------------------------------------------------
# The self test
# ---------------------------------------------------------------------------


def test_every_corpus_frame_equals_itself_exactly():
    """The smallest version of the oracle, and the most valuable test in this file.

    Every frame in the corpus, through the whole normalizer, against a copy of
    itself, with zero relaxations and the exact tolerance class. Anything other than
    a perfect result here is a bug in this module, and it would otherwise be published
    as a firepanda failure and cost somebody a day in the wrong repository.
    """
    rules = Rules(tolerance=Tolerance.EXACT)
    failures = {}
    for name in corpus.frames():
        frame = corpus.load(name).to_pandas()
        verdict = compare(frame, frame.copy(), rules)
        if not verdict:
            failures[name] = verdict.summary()
    assert failures == {}


def test_every_corpus_frame_equals_itself_through_arrow_as_well():
    """The same run without pandas in the middle, so that a bug in `to_pandas` cannot
    make the test above pass by damaging both sides identically."""
    rules = Rules(tolerance=Tolerance.EXACT)
    for name in corpus.frames():
        table = corpus.load(name)
        assert compare(table, corpus.load(name), rules), name


def test_a_single_changed_value_is_caught_in_every_corpus_frame():
    """The other half of the test above. A comparison that says everything is equal is
    only worth something if it also says when something is not, and a normalizer that
    dropped every column would pass the self test perfectly."""
    for name in corpus.frames():
        frame = corpus.load(name).to_pandas()
        if frame.empty:
            continue
        damaged = frame.copy()
        column = damaged.columns[-1]
        # Overwriting with a string changes either the value or the dtype, and either
        # one has to be a difference, which is the point.
        damaged[column] = damaged[column].astype(object)
        damaged.iloc[0, damaged.columns.get_loc(column)] = "damaged"
        assert not compare(frame, damaged), name
