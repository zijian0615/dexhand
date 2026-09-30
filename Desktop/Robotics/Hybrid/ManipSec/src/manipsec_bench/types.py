"""Core algebraic data types for ManipSec-Bench.

The module is dependency-free by design. Isaac Sim tensors and scene objects stay behind the
protocols in ``interfaces.py``; benchmark manifests and evaluation results use these serializable
types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, Mapping, TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | Mapping[str, "JsonValue"]


class AttackSurface(StrEnum):
    PROMPT = "prompt"
    SCENE = "scene"
    ENVIRONMENT = "environment"


class AttackPhase(StrEnum):
    SYNTHESIS = "synthesis"
    DEPLOYMENT = "deployment"
    RUNTIME = "runtime"


class SecurityProperty(StrEnum):
    INSTRUCTION_INTEGRITY = "instruction_integrity"
    SCENE_INTEGRITY = "scene_integrity"
    EXECUTION_INTEGRITY = "execution_integrity"
    EVALUATION_INTEGRITY = "evaluation_integrity"


class RunCondition(StrEnum):
    CLEAN = "clean"
    ADVERSARIAL = "adversarial"


class PromptCarrier(StrEnum):
    TASK_TEXT = "task_text"
    WORKSPACE_FILE = "workspace_file"
    SOURCE_COMMENT = "source_comment"
    TOOL_OUTPUT = "tool_output"


class SceneMutationKind(StrEnum):
    ADD_OBJECT = "add_object"
    REMOVE_OBJECT = "remove_object"
    CHANGE_APPEARANCE = "change_appearance"
    CHANGE_SEMANTICS = "change_semantics"
    SPOOF_OBSERVATION = "spoof_observation"


class InterventionKind(StrEnum):
    MOVE_OBJECT = "move_object"
    MOVE_OBSTACLE = "move_obstacle"
    APPLY_FORCE = "apply_force"
    CHANGE_DYNAMICS = "change_dynamics"
    CORRUPT_SENSOR = "corrupt_sensor"


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    instruction: str
    env_id: str
    embodiment: str
    control_mode: str
    max_steps: int
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id or not self.instruction or not self.env_id:
            raise ValueError("task_id, instruction, and env_id must be non-empty")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")


@dataclass(frozen=True)
class ProtectedAsset:
    asset_id: str
    property: SecurityProperty
    description: str


@dataclass(frozen=True)
class SecurityInvariant:
    invariant_id: str
    description: str
    violation_codes: frozenset[str]

    def __post_init__(self) -> None:
        if not self.violation_codes:
            raise ValueError("an invariant must define at least one violation code")


@dataclass(frozen=True)
class SecurityContract:
    contract_id: str
    assets: tuple[ProtectedAsset, ...]
    invariants: tuple[SecurityInvariant, ...]
    allowed_capabilities: frozenset[str]
    forbidden_capabilities: frozenset[str]

    def __post_init__(self) -> None:
        overlap = self.allowed_capabilities & self.forbidden_capabilities
        if overlap:
            raise ValueError(f"capabilities cannot be both allowed and forbidden: {sorted(overlap)}")
        if not self.invariants:
            raise ValueError("a security contract must define at least one invariant")


@dataclass(frozen=True)
class AttackBudget:
    max_events: int = 1
    max_payload_tokens: int | None = None
    max_displacement_m: float | None = None
    max_force_n: float | None = None

    def __post_init__(self) -> None:
        if self.max_events <= 0:
            raise ValueError("max_events must be positive")
        for name in ("max_payload_tokens", "max_displacement_m", "max_force_n"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")


@dataclass(frozen=True)
class ThreatModel:
    threat_model_id: str
    attacker_knowledge: str
    attacker_capabilities: frozenset[str]
    trusted_components: frozenset[str]
    budget: AttackBudget


@dataclass(frozen=True)
class AttackGoal:
    goal_id: str
    description: str
    violation_codes: frozenset[str]

    def __post_init__(self) -> None:
        if not self.violation_codes:
            raise ValueError("an attack goal must map to at least one violation code")


@dataclass(frozen=True)
class PromptAttackArtifact:
    surface: ClassVar[AttackSurface] = AttackSurface.PROMPT
    phase: ClassVar[AttackPhase] = AttackPhase.SYNTHESIS

    artifact_id: str
    goal: AttackGoal
    threat_model: ThreatModel
    carrier: PromptCarrier
    payload: str
    destination: str
    seed: int

    def __post_init__(self) -> None:
        if not self.payload or not self.destination:
            raise ValueError("prompt artifacts require a payload and destination")


@dataclass(frozen=True)
class SceneMutation:
    kind: SceneMutationKind
    target: str
    parameters: Mapping[str, JsonValue]


@dataclass(frozen=True)
class SceneAttackArtifact:
    surface: ClassVar[AttackSurface] = AttackSurface.SCENE
    phase: ClassVar[AttackPhase] = AttackPhase.DEPLOYMENT

    artifact_id: str
    goal: AttackGoal
    threat_model: ThreatModel
    mutations: tuple[SceneMutation, ...]
    seed: int

    def __post_init__(self) -> None:
        if not self.mutations:
            raise ValueError("scene artifacts require at least one mutation")


@dataclass(frozen=True)
class RuntimeIntervention:
    kind: InterventionKind
    trigger: str
    target: str
    parameters: Mapping[str, JsonValue]


@dataclass(frozen=True)
class EnvironmentAttackArtifact:
    surface: ClassVar[AttackSurface] = AttackSurface.ENVIRONMENT
    phase: ClassVar[AttackPhase] = AttackPhase.RUNTIME

    artifact_id: str
    goal: AttackGoal
    threat_model: ThreatModel
    interventions: tuple[RuntimeIntervention, ...]
    seed: int

    def __post_init__(self) -> None:
        if not self.interventions:
            raise ValueError("environment artifacts require at least one intervention")
        if len(self.interventions) > self.threat_model.budget.max_events:
            raise ValueError("interventions exceed the threat model event budget")


AttackArtifact: TypeAlias = PromptAttackArtifact | SceneAttackArtifact | EnvironmentAttackArtifact


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    task: TaskSpec
    contract: SecurityContract
    condition: RunCondition
    artifact: AttackArtifact | None = None

    def __post_init__(self) -> None:
        if self.condition is RunCondition.CLEAN and self.artifact is not None:
            raise ValueError("clean cases cannot contain an attack artifact")
        if self.condition is RunCondition.ADVERSARIAL and self.artifact is None:
            raise ValueError("adversarial cases require an attack artifact")


@dataclass(frozen=True)
class SecurityEvent:
    code: str
    step: int
    env_index: int
    source: str
    attempted: bool
    realized: bool
    blocked: bool
    details: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskVerdict:
    env_index: int
    success: bool
    score: float = 0.0
    details: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class SecurityVerdict:
    env_index: int
    attack_success: bool
    violations: tuple[SecurityEvent, ...] = ()
    details: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeOutcome:
    pair_id: str
    case_id: str
    task_id: str
    policy_id: str
    condition: RunCondition
    seed: int
    env_index: int
    task_success: bool
    attack_success: bool
    task_score: float
    artifact_id: str | None = None
    violations: tuple[SecurityEvent, ...] = ()
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.condition is RunCondition.CLEAN:
            if self.artifact_id is not None:
                raise ValueError("clean outcomes cannot reference an attack artifact")
            if self.attack_success:
                raise ValueError("clean outcomes cannot be attack successes")
        if self.condition is RunCondition.ADVERSARIAL and self.artifact_id is None:
            raise ValueError("adversarial outcomes must reference an attack artifact")
        if not 0.0 <= self.task_score <= 1.0:
            raise ValueError("task_score must be in [0, 1]")

    @property
    def secure_success(self) -> bool:
        return self.task_success and not self.attack_success


@dataclass(frozen=True)
class ArtifactValidationReport:
    artifact_id: str
    feasible: bool
    specific: bool
    budget_compliant: bool
    reproducible: bool
    details: Mapping[str, JsonValue] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.feasible and self.specific and self.budget_compliant and self.reproducible
