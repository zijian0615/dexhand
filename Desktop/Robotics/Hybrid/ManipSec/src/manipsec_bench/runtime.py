"""Policy-facing environment view and dual-oracle episode evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .interfaces import (
    ActionValidator,
    RawEnv,
    SecurityEventSink,
    SecurityOracle,
    StepObserver,
    TaskOracle,
)
from .types import (
    AttackArtifact,
    EpisodeOutcome,
    RunCondition,
    SecurityEvent,
)


class PolicyEnvView:
    """Explicit EmbodiedSWE-compatible view passed to ``solve(env)``.

    The class prevents accidental access to reset/state restoration, validates each action, and
    triggers hidden observers. It is not a Python security sandbox; untrusted policy code must run
    across a process or RPC boundary in the final benchmark harness.
    """

    __slots__ = (
        "__env",
        "__validator",
        "__observers",
        "__event_sinks",
        "__events",
        "__step",
    )

    def __init__(
        self,
        env: RawEnv,
        *,
        validator: ActionValidator | None = None,
        observers: Sequence[StepObserver] = (),
        event_sinks: Sequence[SecurityEventSink] = (),
    ) -> None:
        self.__env = env
        self.__validator = validator
        self.__observers = tuple(observers)
        self.__event_sinks = tuple(event_sinks)
        self.__events: list[SecurityEvent] = []
        self.__step = 0

    @property
    def device(self) -> str:
        return self.__env.device

    @property
    def num_envs(self) -> int:
        return self.__env.num_envs

    @property
    def dt(self) -> float:
        return self.__env.dt

    @property
    def scene(self) -> Any:
        return self.__env.scene

    @property
    def robot(self) -> Any:
        return self.__env.robot

    @property
    def iscene(self) -> Any:
        return self.__env.iscene

    @property
    def events(self) -> tuple[SecurityEvent, ...]:
        return tuple(self.__events)

    @property
    def steps(self) -> int:
        return self.__step

    def get_states(self, env_ids: Any | None = None) -> dict[str, Any]:
        return self.__env.get_states(env_ids)

    def describe(self) -> str:
        return self.__env.describe()

    def describe_stage(self, root: str | None = None, raw: bool = False) -> list[dict[str, Any]] | str:
        return self.__env.describe_stage(root=root, raw=raw)

    def step(self, action: Any, render: bool = False) -> None:
        if self.__validator is not None:
            events = tuple(self.__validator.validate(action, self.__env, self.__step))
            self.__events.extend(events)
            for event in events:
                for sink in self.__event_sinks:
                    sink.record(event)
        self.__env.step(action, render=render)
        for observer in self.__observers:
            observer.after_step(self.__env, action, self.__step)
        self.__step += 1


class DualOracleEvaluator:
    """Combines independent functional and security verdicts for one batch."""

    def __init__(self, task_oracle: TaskOracle, security_oracle: SecurityOracle) -> None:
        self.task_oracle = task_oracle
        self.security_oracle = security_oracle

    def begin(self, env: RawEnv, artifact: AttackArtifact | None) -> None:
        self.task_oracle.begin(env)
        self.security_oracle.begin(env, artifact)

    def record(self, event: SecurityEvent) -> None:
        self.security_oracle.record(event)

    def after_step(self, env: RawEnv, action: Any, step: int) -> None:
        self.task_oracle.after_step(env, action, step)
        self.security_oracle.after_step(env, action, step)

    def finalize(
        self,
        env: RawEnv,
        *,
        pair_id: str,
        case_id: str,
        task_id: str,
        policy_id: str,
        condition: RunCondition,
        seed: int,
        artifact_id: str | None,
    ) -> tuple[EpisodeOutcome, ...]:
        tasks = {v.env_index: v for v in self.task_oracle.finalize(env)}
        security = {v.env_index: v for v in self.security_oracle.finalize(env)}
        expected = set(range(env.num_envs))
        if set(tasks) != expected or set(security) != expected:
            raise ValueError("both oracles must return exactly one verdict per environment")

        return tuple(
            EpisodeOutcome(
                pair_id=pair_id,
                case_id=case_id,
                task_id=task_id,
                policy_id=policy_id,
                condition=condition,
                seed=seed,
                env_index=e,
                task_success=tasks[e].success,
                attack_success=security[e].attack_success,
                task_score=tasks[e].score,
                artifact_id=artifact_id,
                violations=security[e].violations,
                metadata={"task": tasks[e].details, "security": security[e].details},
            )
            for e in range(env.num_envs)
        )
