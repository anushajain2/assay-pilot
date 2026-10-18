"""Sequence handling: validation, substitution notation, and distances.

Version one of AssayPilot supports *substitution-only variants of the campaign
reference sequence*: same length, standard amino acids. Everything in this
module enforces or describes that contract. Nothing here silently repairs a
sequence; invalid input raises :class:`~assaypilot.errors.SequenceError` so the
caller can report it with a reason.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .errors import SequenceError

#: The twenty standard amino acids, in the order used for one-hot encoding.
AMINO_ACIDS: tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWY")

#: Characters that appear in FASTA files but are not supported as variant residues.
AMBIGUOUS_CHARACTERS: frozenset[str] = frozenset("BJOUXZ*-.")

_AA_INDEX: dict[str, int] = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

_MUTATION_RE = re.compile(r"^(?P<wt>[A-Z])(?P<pos>\d+)(?P<mt>[A-Z])$")

#: Separators accepted between substitutions in mutation notation ("A12G:L30P").
_MUTATION_SEPARATORS = re.compile(r"[:;,+/\s]+")


def aa_index(residue: str) -> int:
    """Return the one-hot column index of a standard amino acid."""
    try:
        return _AA_INDEX[residue]
    except KeyError:  # pragma: no cover - guarded by validate_sequence
        raise SequenceError(f"{residue!r} is not one of the 20 standard amino acids") from None


def sequence_hash(sequence: str) -> str:
    """Return a stable identifier for a sequence.

    Used as the key for embedding caches and for grouping replicates of the same
    sequence into the same evaluation partition. The hash covers the uppercase
    sequence only, so two identically-sequenced variants with different IDs
    always share a group.
    """
    return hashlib.sha256(sequence.strip().upper().encode("ascii")).hexdigest()


def normalise_sequence(sequence: str) -> str:
    """Strip whitespace and upper-case a sequence without validating it."""
    return re.sub(r"\s+", "", str(sequence)).upper()


def validate_sequence(
    sequence: str, *, reference: str | None = None, label: str = "sequence"
) -> str:
    """Validate one sequence and return its normalised form.

    Parameters
    ----------
    sequence:
        The raw sequence as read from a CSV or FASTA file.
    reference:
        If given, the sequence must have exactly the same length. Version one
        supports substitutions only, so an indel is an error rather than a
        variant.
    label:
        Used in the error message, for example ``"candidate cand_017"``.

    Raises
    ------
    SequenceError
        If the sequence is empty, contains non-standard characters, or does not
        match the reference length.
    """
    seq = normalise_sequence(sequence)
    if not seq:
        raise SequenceError(f"{label} is empty")

    bad = sorted({c for c in seq if c not in _AA_INDEX})
    if bad:
        ambiguous = sorted(c for c in bad if c in AMBIGUOUS_CHARACTERS)
        detail = f"invalid character(s) {''.join(bad)}"
        if ambiguous:
            detail += (
                f"; {''.join(ambiguous)} is ambiguous or a gap character, which version one"
                " does not support"
            )
        raise SequenceError(f"{label} contains {detail}")

    if reference is not None:
        ref = normalise_sequence(reference)
        if len(seq) != len(ref):
            raise SequenceError(
                f"{label} has length {len(seq)} but the reference has length {len(ref)};"
                " version one supports substitution-only variants of the same length"
            )
    return seq


@dataclass(frozen=True, order=True)
class Substitution:
    """A single amino-acid substitution relative to the reference.

    ``position`` is 1-based, matching the convention used in mutation notation
    such as ``A12G``.
    """

    position: int
    wildtype: str
    mutant: str

    def __str__(self) -> str:
        return f"{self.wildtype}{self.position}{self.mutant}"

    @property
    def index(self) -> int:
        """0-based index into the sequence string."""
        return self.position - 1


def substitutions(reference: str, sequence: str) -> tuple[Substitution, ...]:
    """Return the substitutions that turn ``reference`` into ``sequence``.

    Both sequences must already be validated and of equal length.
    """
    ref = normalise_sequence(reference)
    seq = normalise_sequence(sequence)
    if len(ref) != len(seq):
        raise SequenceError(
            f"cannot compare sequences of different length ({len(seq)} vs reference {len(ref)})"
        )
    return tuple(
        Substitution(position=i + 1, wildtype=r, mutant=m)
        for i, (r, m) in enumerate(zip(ref, seq, strict=True))
        if r != m
    )


def mutation_string(reference: str, sequence: str, separator: str = ":") -> str:
    """Render a variant in mutation notation, or ``"WT"`` for the reference itself."""
    subs = substitutions(reference, sequence)
    if not subs:
        return "WT"
    return separator.join(str(s) for s in subs)


def parse_mutation_string(notation: str, reference: str) -> str:
    """Build a full sequence from mutation notation, checking it against the reference.

    Accepts ``"A12G:L30P"``, ``"A12G, L30P"``, ``"WT"`` and similar. The
    wild-type residue stated in the notation must match the reference at that
    position: AssayPilot never renumbers a mutation to make it fit.

    Raises
    ------
    SequenceError
        If a token is malformed, out of range, contradicts the reference, or if
        two substitutions address the same position.
    """
    ref = normalise_sequence(reference)
    text = str(notation).strip()
    if not text:
        raise SequenceError("mutation notation is empty")
    if text.upper() in {"WT", "WILDTYPE", "WILD-TYPE", "REF", "REFERENCE", "NONE"}:
        return ref

    residues = list(ref)
    seen: dict[int, str] = {}
    for token in _MUTATION_SEPARATORS.split(text):
        if not token:
            continue
        match = _MUTATION_RE.match(token.upper())
        if not match:
            raise SequenceError(
                f"{token!r} is not a substitution in the form A12G (wild-type residue,"
                " 1-based position, mutant residue)"
            )
        wt, pos, mt = match["wt"], int(match["pos"]), match["mt"]
        if pos < 1 or pos > len(ref):
            raise SequenceError(
                f"{token!r} refers to position {pos}, outside the reference (1-{len(ref)})"
            )
        if ref[pos - 1] != wt:
            raise SequenceError(
                f"{token!r} states wild-type {wt} at position {pos}, but the reference has"
                f" {ref[pos - 1]}. Check the numbering offset of your mutation list; AssayPilot"
                " will not renumber it for you."
            )
        if mt not in _AA_INDEX:
            raise SequenceError(
                f"{token!r} has mutant residue {mt}, which is not a standard amino acid"
            )
        if pos in seen and seen[pos] != mt:
            raise SequenceError(
                f"position {pos} is substituted twice in {notation!r} ({seen[pos]} and {mt})"
            )
        seen[pos] = mt
        residues[pos - 1] = mt
    return "".join(residues)


def hamming_distance(a: str, b: str) -> int:
    """Number of differing positions between two equal-length sequences."""
    if len(a) != len(b):
        raise SequenceError(f"cannot compare sequences of different length ({len(a)} vs {len(b)})")
    return sum(1 for x, y in zip(a, b, strict=True) if x != y)


def mutated_positions(reference: str, sequence: str) -> frozenset[int]:
    """1-based positions at which ``sequence`` differs from ``reference``."""
    return frozenset(s.position for s in substitutions(reference, sequence))


def min_distance_to_set(sequence: str, others: Sequence[str]) -> int:
    """Smallest Hamming distance from ``sequence`` to any sequence in ``others``.

    Returns ``len(sequence)`` when ``others`` is empty, which is the largest
    distance any equal-length sequence could have.
    """
    if not others:
        return len(sequence)
    return min(hamming_distance(sequence, other) for other in others)


def distance_matrix_rows(sequences: Iterable[str], others: Sequence[str]) -> list[int]:
    """Per-sequence minimum distance to a reference set (see :func:`min_distance_to_set`)."""
    others = list(others)
    return [min_distance_to_set(seq, others) for seq in sequences]
