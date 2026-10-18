"""Minimal FASTA reading and writing.

AssayPilot does not depend on Biopython for this: the format it needs is small
and well defined, and a dependency-free reader keeps installation light for
users who only want the planning workflow.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from ..errors import InputError


@dataclass(frozen=True)
class FastaRecord:
    """One FASTA entry: an identifier, an optional description, and a sequence."""

    identifier: str
    sequence: str
    description: str = ""

    @property
    def header(self) -> str:
        return f"{self.identifier} {self.description}".strip()


def iter_fasta(handle: TextIO) -> Iterator[FastaRecord]:
    """Yield records from an open text handle."""
    identifier: str | None = None
    description = ""
    chunks: list[str] = []

    for raw in handle:
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith(">"):
            if identifier is not None:
                yield FastaRecord(identifier, "".join(chunks), description)
            header = line[1:].strip()
            identifier, _, description = header.partition(" ")
            identifier = identifier.strip()
            description = description.strip()
            chunks = []
            if not identifier:
                raise InputError("FASTA record has an empty identifier on its '>' line")
        else:
            if identifier is None:
                raise InputError("FASTA file starts with sequence data before any '>' header")
            chunks.append(line)

    if identifier is not None:
        yield FastaRecord(identifier, "".join(chunks), description)


def read_fasta(path: str | Path) -> list[FastaRecord]:
    """Read every record from a FASTA file."""
    path = Path(path)
    if not path.exists():
        raise InputError(f"FASTA file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        records = list(iter_fasta(handle))
    if not records:
        raise InputError(f"{path} contains no FASTA records")
    return records


def write_fasta(path: str | Path, records: Iterable[FastaRecord], *, width: int = 60) -> Path:
    """Write records to a FASTA file, wrapping sequences at ``width`` columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(f">{record.header}\n")
            seq = record.sequence
            for start in range(0, len(seq), width):
                handle.write(seq[start : start + width] + "\n")
    return path
