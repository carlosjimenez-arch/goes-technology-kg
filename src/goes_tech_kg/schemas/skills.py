"""Context-free skill archetypes; tool bindings are outside graph snapshots."""

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from goes_tech_kg.schemas.base import Contract, Digest, Grade, Minutes, Probability, Text, stable_id
from goes_tech_kg.schemas.calibration import ConfidenceBasis


class Strand(StrEnum):
    COMPUTATIONAL_THINKING = "computational_thinking"
    DIGITAL_CITIZENSHIP = "digital_citizenship"
    DESIGN_PROCESS = "design_process"
    TECHNICAL_SYSTEMS = "technical_systems"
    DATA_AND_AI = "data_and_ai"


class CognitiveDomain(StrEnum):
    KNOWING = "knowing"
    APPLYING = "applying"
    REASONING = "reasoning"


class CTDimension(StrEnum):
    DECOMPOSITION = "decomposition"
    ABSTRACTION = "abstraction"
    PATTERNS = "patterns"
    ALGORITHMS = "algorithms"
    DEBUGGING = "debugging"
    EVALUATION = "evaluation"


class Tier(StrEnum):
    T0 = "T0"
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"


class HalfLife(StrEnum):
    DURABLE = "DURABLE"
    SLOW = "SLOW"
    VOLATILE = "VOLATILE"


class EdgeType(StrEnum):
    PREREQUISITE = "PREREQUISITE"
    CO_REQUISITE = "CO_REQUISITE"
    REFINES = "REFINES"
    TRANSFERS_TO = "TRANSFERS_TO"
    CROSS_SUBJECT_PREREQUISITE = "CROSS_SUBJECT_PREREQUISITE"


class EvidenceRef(Contract):
    document_id: Text
    document_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    paragraph_id: Text
    start: Annotated[int, Field(strict=True, ge=0)]
    end: Annotated[int, Field(strict=True, gt=0)]

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.end <= self.start:
            raise ValueError("evidence offsets must form a nonempty half-open interval")
        return self


class PrerequisiteRef(Contract):
    id: Text
    type: Literal[EdgeType.PREREQUISITE, EdgeType.CO_REQUISITE]


class CrossSubjectDependency(Contract):
    repository: Literal["goes-math-kg", "goes-linguistics-kg", "goes-natural-science-kg"]
    snapshot_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    node_id: Text
    available_grade: Annotated[int, Field(strict=True, ge=1, le=12)]
    source_refs: Annotated[tuple[EvidenceRef, ...], Field(min_length=1)]

    @property
    def qualified_id(self) -> str:
        return f"{self.repository}@{self.snapshot_sha256}:{self.node_id}"


class DowngradeVariant(Contract):
    target_tier: Tier
    method: Text
    evidence_of_mastery: Text
    estimated_minutes: Annotated[Minutes, Field(gt=0)]
    preserved_constructs: Annotated[tuple[Text, ...], Field(min_length=1)]
    lost_performances: tuple[Text, ...] = ()
    practical_bridge: Text | None = None
    equivalence: Literal[
        "same_construct_candidate", "validated_same_construct", "conceptual_proxy_only"
    ]
    next_variant: "DowngradeVariant | None" = None

    @model_validator(mode="after")
    def honest_loss(self) -> Self:
        if self.lost_performances and (
            not self.practical_bridge or self.equivalence != "conceptual_proxy_only"
        ):
            raise ValueError(
                "lost performances require an explicit non-equivalent practical bridge"
            )
        if (
            self.next_variant
            and int(self.next_variant.target_tier[-1]) != int(self.target_tier[-1]) - 1
        ):
            raise ValueError("downgrade chain must descend exactly one tier")
        return self


class Skill(Contract):
    schema_version: Literal["tech-skill/1.0"] = "tech-skill/1.0"
    kind: Literal["curriculum_skill", "specific_skill"] = "curriculum_skill"
    id: str = ""
    slug: Text
    label: Text

    @model_validator(mode="after")
    def identity(self) -> Self:
        expected = stable_id("skill", self.slug, {"label": self.label, "kind": self.kind})
        if self.id and self.id != expected:
            raise ValueError("skill ID does not match intrinsic content")
        object.__setattr__(self, "id", expected)
        return self


class MicroSkill(Contract):
    schema_version: Literal["tech-micro-skill/1.0"] = "tech-micro-skill/1.0"
    kind: Literal["micro_skill"] = "micro_skill"
    id: str = ""
    slug: Text
    parent_skill_id: Text
    observable_verb: Text
    knowledge_object: Text
    strand: Strand
    cognitive_domain: CognitiveDomain
    ct_dimension: CTDimension | None = None
    min_tier: Tier
    downgrade_variant: DowngradeVariant | None
    half_life: HalfLife
    teacher_prep_level: Annotated[int, Field(strict=True, ge=0, le=3)]
    prerequisites: tuple[PrerequisiteRef, ...] = ()
    cross_subject_deps: tuple[CrossSubjectDependency, ...] = ()
    misconceptions: tuple[Text, ...] = ()
    evidence_of_mastery: Text
    estimated_minutes: Annotated[Minutes, Field(gt=0)]
    source_refs: Annotated[tuple[EvidenceRef, ...], Field(min_length=1)]
    # Confidence is never authored: it is calibrated from confidence_basis (eval.calibration).
    confidence: Probability | None = None
    confidence_basis: ConfidenceBasis | None = None
    calibration_sha256: Digest | None = None

    @property
    def human_reviewed(self) -> bool:
        return self.confidence_basis is not None and self.confidence_basis.human_review != "none"

    @model_validator(mode="after")
    def derived_confidence(self) -> Self:
        if self.confidence is not None and self.confidence_basis is None:
            raise ValueError("confidence requires a recorded confidence basis")
        if (
            self.confidence is not None
            and self.calibration_sha256 is None
            and not self.human_reviewed
        ):
            raise ValueError("calibrated confidence requires the calibration table digest")
        if self.confidence is None and self.calibration_sha256 is not None:
            raise ValueError("calibration digest without a confidence value")
        return self

    @model_validator(mode="after")
    def intrinsic_identity(self) -> Self:
        intrinsic = self.model_dump(
            mode="json",
            include={
                "observable_verb",
                "knowledge_object",
                "strand",
                "cognitive_domain",
                "ct_dimension",
            },
        )
        expected = stable_id("micro", self.slug, intrinsic)
        if self.id and self.id != expected:
            raise ValueError("micro-skill ID does not match intrinsic content")
        object.__setattr__(self, "id", expected)
        if (
            self.downgrade_variant
            and int(self.downgrade_variant.target_tier[-1]) != int(self.min_tier[-1]) - 1
        ):
            raise ValueError("downgrade must target the immediately lower tier")
        if self.strand == Strand.COMPUTATIONAL_THINKING and self.ct_dimension is None:
            raise ValueError("computational thinking requires ct_dimension")
        keys = [(p.id, p.type) for p in self.prerequisites]
        if len(set(keys)) != len(keys) or any(p.id == self.id for p in self.prerequisites):
            raise ValueError("duplicate or self prerequisite")
        object.__setattr__(
            self, "prerequisites", tuple(sorted(self.prerequisites, key=lambda p: (p.id, p.type)))
        )
        object.__setattr__(
            self,
            "cross_subject_deps",
            tuple(sorted(self.cross_subject_deps, key=lambda d: d.qualified_id)),
        )
        object.__setattr__(
            self,
            "source_refs",
            tuple(
                sorted(
                    self.source_refs, key=lambda r: (r.document_id, r.paragraph_id, r.start, r.end)
                )
            ),
        )
        return self


class Edge(Contract):
    schema_version: Literal["tech-edge/1.0"] = "tech-edge/1.0"
    id: str = ""
    source: Text
    target: Text
    type: EdgeType
    justification: Text
    source_refs: Annotated[tuple[EvidenceRef, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def identity(self) -> Self:
        if self.source == self.target:
            raise ValueError("self edge")
        if self.type == EdgeType.CO_REQUISITE and self.source > self.target:
            left, right = self.target, self.source
            object.__setattr__(self, "source", left)
            object.__setattr__(self, "target", right)
        expected = stable_id(
            "edge",
            self.type.value.lower().replace("_", "-"),
            {"source": self.source, "target": self.target, "type": self.type},
        )
        if self.id and self.id != expected:
            raise ValueError("edge identity mismatch")
        object.__setattr__(self, "id", expected)
        return self


class ToolBinding(Contract):
    id: Text
    micro_skill_id: Text
    product: Text
    interface_version: Text
    half_life: Literal[HalfLife.VOLATILE] = HalfLife.VOLATILE
    instructions: Text
    min_tier: Tier


class SkillPlacement(Contract):
    micro_skill_id: Text
    grade: Grade
