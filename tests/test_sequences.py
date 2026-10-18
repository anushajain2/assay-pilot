"""Sequence validation and mutation notation.

These are the checks that stop a whole campaign being analysed against the
wrong numbering. The important behaviours are: reject rather than repair, and
never shift a position to make a mutation fit.
"""

from __future__ import annotations

import pytest

from assaypilot.errors import SequenceError
from assaypilot.sequences import (
    hamming_distance,
    min_distance_to_set,
    mutated_positions,
    mutation_string,
    parse_mutation_string,
    sequence_hash,
    substitutions,
    validate_sequence,
)


def test_validate_accepts_standard_residues(reference):
    assert validate_sequence(reference.lower()) == reference


def test_validate_strips_whitespace(reference):
    spaced = reference[:10] + "\n " + reference[10:]
    assert validate_sequence(spaced) == reference


@pytest.mark.parametrize("bad", ["MKAILX", "MKAIL-", "MKAIL*", "MKAIL1"])
def test_validate_rejects_non_standard_characters(bad):
    with pytest.raises(SequenceError, match="invalid character"):
        validate_sequence(bad)


def test_validate_explains_ambiguous_characters():
    with pytest.raises(SequenceError, match="ambiguous or a gap"):
        validate_sequence("MKAILX")


def test_validate_rejects_length_mismatch(reference):
    with pytest.raises(SequenceError, match="substitution-only"):
        validate_sequence(reference[:-1], reference=reference)


def test_validate_rejects_empty():
    with pytest.raises(SequenceError, match="empty"):
        validate_sequence("   ")


def test_substitutions_and_mutation_string(reference, mutator):
    variant = mutator(reference, (3, "G"), (7, "D"))
    subs = substitutions(reference, variant)
    assert [str(s) for s in subs] == [f"{reference[2]}3G", f"{reference[6]}7D"]
    assert mutation_string(reference, variant) == f"{reference[2]}3G:{reference[6]}7D"
    assert mutation_string(reference, reference) == "WT"


def test_parse_mutation_string_round_trips(reference, mutator):
    variant = mutator(reference, (3, "G"), (7, "D"))
    notation = mutation_string(reference, variant)
    assert parse_mutation_string(notation, reference) == variant


@pytest.mark.parametrize("separator", [":", ",", ";", " ", "+", "/"])
def test_parse_mutation_string_accepts_separators(reference, separator):
    notation = f"{reference[2]}3G{separator}{reference[6]}7D"
    assert parse_mutation_string(notation, reference)[2] == "G"


def test_parse_mutation_string_accepts_wildtype_aliases(reference):
    for alias in ("WT", "wt", "wild-type", "reference"):
        assert parse_mutation_string(alias, reference) == reference


def test_parse_mutation_string_rejects_wrong_wildtype(reference):
    wrong = "Z" if reference[2] != "Z" else "A"
    with pytest.raises(SequenceError, match="will not renumber"):
        parse_mutation_string(f"{wrong}3G", reference)


def test_parse_mutation_string_rejects_out_of_range(reference):
    with pytest.raises(SequenceError, match="outside the reference"):
        parse_mutation_string(f"{reference[0]}9999G", reference)


def test_parse_mutation_string_rejects_conflicting_positions(reference):
    notation = f"{reference[2]}3G:{reference[2]}3W"
    with pytest.raises(SequenceError, match="substituted twice"):
        parse_mutation_string(notation, reference)


def test_parse_mutation_string_rejects_malformed_token(reference):
    with pytest.raises(SequenceError, match="not a substitution"):
        parse_mutation_string("hello", reference)


def test_sequence_hash_is_identity_independent(reference, mutator):
    variant = mutator(reference, (3, "G"))
    assert sequence_hash(variant) == sequence_hash(variant.lower())
    assert sequence_hash(variant) != sequence_hash(reference)


def test_distance_helpers(reference, mutator):
    a = mutator(reference, (3, "G"))
    b = mutator(reference, (3, "G"), (7, "D"))
    assert hamming_distance(a, b) == 1
    assert mutated_positions(reference, b) == frozenset({3, 7})
    assert min_distance_to_set(b, [a, reference]) == 1
    assert min_distance_to_set(b, []) == len(reference)


def test_hamming_distance_rejects_unequal_lengths(reference):
    with pytest.raises(SequenceError, match="different length"):
        hamming_distance(reference, reference[:-1])
