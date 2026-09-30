"""Runtime protocols separating the policy surface from privileged evaluation."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from .types import (
    AttackArtifact,
    ArtifactValidationReport,
    BenchmarkCase,
    EnvironmentAttackArtifact,
    PromptAttackArtifact,
    SceneAttackArtifact,
    SecurityEvent,
    SecurityVerdict,
    TaskVerdict,
)


@runtime_checkable
class PolicyEnv(Protocol):
    """The EmbodiedSWE-compatible surface visible to ``solve(env)``.

    ``reset`` and ``set_states`` are deliberately absent. A protocol documents the capability
    boundary; process/container isolation must enforce it in a hostile-code evaluation.
    """

    device: str
    num_envs: int
    dt: float
    scene: Any
    robot: Any
    iscene: Any

    def get_states(self, env_ids: Any | None = None) -> dict[str, Any]: ...

    def describe(self) -> str: ...

    def describe_stage(self, root: str | None = None, raw: bool = False) -> list[dict[str, Any]] | str: ...

    def step(self, action: Any, render: bool = False) -> None: ...


class RawEnv(PolicyEnv, Protocol):
    """Privileged simulator surface owned by the benchmark harness."""

    def reset(self, env_ids: Any | None = None, *, seed: int | None = None) -> None: ...

    def set_states(self, states: dict[str, Any], env_ids: Any | None = None) -> None: ...


class ActionValidator(Protocol):
    def validate(self, action: Any, env: RawEnv, step: int) -> Sequence[SecurityEvent]: ...


class StepObserver(Protocol):
    def after_step(self, env: RawEnv, action: Any, step: int) -> None: ...


class SecurityEventSink(Protocol):
    def record(self, event: SecurityEvent) -> None: ...


class TaskOracle(Protocol):
    """Task-specific functional scorer; it does not decide security compromise."""

    def begin(self, env: RawEnv) -> None: ...

    def after_step(self, env: RawEnv, action: Any, step: int) -> None: ...

    def finalize(self, env: RawEnv) -> Sequence[TaskVerdict]: ...


class SecurityOracle(Protocol):
    """Attack-specific security scorer; it does not decide task completion."""

    def begin(self, env: RawEnv, artifact: AttackArtifact | None) -> None: ...

    def record(self, event: SecurityEvent) -> None: ...

    def after_step(self, env: RawEnv, action: Any, step: int) -> None: ...

    def finalize(self, env: RawEnv) -> Sequence[SecurityVerdict]: ...


class Policy(Protocol):
    def __call__(self, env: PolicyEnv) -> None: ...


class AgenticPolicy(Protocol):
    """Policy-neutral executable prepared for one benchmark case."""

    policy_id: str

    def run(self, env: PolicyEnv, case: BenchmarkCase) -> None: ...


class PolicyAdapter(Protocol):
    """Adapts a target family to the common ManipSec execution contract."""

    target_id: str

    def prepare(self, case: BenchmarkCase) -> AgenticPolicy: ...


class ArtifactGenerator(Protocol):
    """Produces task-conditioned candidates during benchmark construction."""

    def generate(self, case: BenchmarkCase) -> Sequence[AttackArtifact]: ...


class ArtifactValidator(Protocol):
    """Validates candidates before they enter a frozen benchmark split."""

    def validate(self, case: BenchmarkCase, artifact: AttackArtifact) -> ArtifactValidationReport: ...


class SynthesisWorkspace(Protocol):
    """Workspace mutation surface owned by the synthesis harness."""

    def inject_text(self, destination: str, payload: str) -> None: ...


class PromptArtifactApplicator(Protocol):
    """Applies synthesis-time prompt artifacts before the coding agent runs."""

    def apply(self, workspace: SynthesisWorkspace, artifact: PromptAttackArtifact) -> None: ...


class SceneArtifactApplicator(Protocol):
    """Applies a static scene artifact after policy synthesis and before reset."""

    def apply(self, env: RawEnv, artifact: SceneAttackArtifact) -> None: ...


class EnvironmentArtifactApplicator(Protocol):
    """Applies trigger-driven interventions while a fixed policy is running."""

    def begin(self, env: RawEnv, artifact: EnvironmentAttackArtifact) -> None: ...

    def before_step(self, env: RawEnv, artifact: EnvironmentAttackArtifact, step: int) -> None: ...

    def close(self, env: RawEnv, artifact: EnvironmentAttackArtifact) -> None: ...
