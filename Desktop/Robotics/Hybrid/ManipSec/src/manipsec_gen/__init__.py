"""ManipSec-Gen boundary: verified failures to secure training examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from manipsec_bench import EpisodeOutcome


@dataclass(frozen=True)
class SecureTrainingExample:
    failed_outcome: EpisodeOutcome
    diagnosis: str
    compromised_policy: str
    corrected_policy: str
    verification: Mapping[str, bool]


__all__ = ["SecureTrainingExample"]

