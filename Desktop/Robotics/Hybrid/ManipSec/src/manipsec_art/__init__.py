"""ManipSec-ART: automated red-team evaluation over frozen benchmark cases."""

from __future__ import annotations

from typing import Protocol, Sequence

from manipsec_bench import AgenticPolicy, BenchmarkCase, EpisodeOutcome


class CodingAgentTarget(Protocol):
    """Adapter that asks one coding agent to synthesize policy code for a case."""

    target_id: str

    def synthesize(self, case: BenchmarkCase) -> AgenticPolicy: ...


class AutoRedTeamRunner(Protocol):
    """Runs a target adapter over paired clean and adversarial benchmark cases."""

    def run(
        self,
        target: CodingAgentTarget,
        cases: Sequence[BenchmarkCase],
    ) -> Sequence[EpisodeOutcome]: ...


__all__ = ["AutoRedTeamRunner", "CodingAgentTarget"]
