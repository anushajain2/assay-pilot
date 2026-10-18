"""Exception types raised by AssayPilot.

Every error carries a message written for the researcher, not for a stack trace
reader: it says which input was wrong and what would make it right.
"""

from __future__ import annotations


class AssayPilotError(Exception):
    """Base class for every error raised deliberately by AssayPilot."""


class InputError(AssayPilotError):
    """A user-supplied file, column, or setting cannot be used as given."""


class SequenceError(InputError):
    """A sequence is not a valid variant of the campaign reference."""


class CampaignError(AssayPilotError):
    """A campaign is missing, inconsistent, or used outside its declared context."""


class InsufficientDataError(AssayPilotError):
    """There is too little usable data to carry out the requested analysis."""


class BackendUnavailableError(AssayPilotError):
    """An optional backend (for example ESM-2) is not installed or not reachable."""
