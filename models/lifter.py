"""LiFTER: a fact-typed, executable temporal logic program.

The neural component embeds each observed ``Link(u, v, t)`` fact from its raw
attributes and strictly earlier fact neighbourhood, then vector-quantises it
into a finite latent predicate.  Entity ids never enter the embedding.

The symbolic component is an explicit finite program basis.  It grounds typed
facts with exact argument bindings and temporal kernels; signed rule
contributions are the *only* path from history to a candidate score.  There is
no neural pair scorer and explanations are read from the execution trace.   
"""

from __future__ import annotations

import math
import itertools
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .base import AttentionModel, ModelResult


LIFTER_CONTEXT_TOKEN_DIM = 6
DEFAULT_FACT_CONTEXT_LEN = 8
_NO_PROOF = -30.0


def lifter_edge_feature_dim(
    fact_context_len: int, raw_feature_dim: int = 0
) -> int:
    if fact_context_len < 1 or raw_feature_dim < 0:
        raise ValueError("invalid LiFTER feature dimensions")
    return 1 + int(raw_feature_dim) + int(fact_context_len) * LIFTER_CONTEXT_TOKEN_DIM


LIFTER_EDGE_FEATURE_DIM = lifter_edge_feature_dim(DEFAULT_FACT_CONTEXT_LEN)


ROLE_NAMES = (
    "P(X,Y,T)",
    "P(Y,X,T)",
    "P(X,Z,T)",
    "P(Z,X,T)",
    "P(Y,Z,T)",
    "P(Z,Y,T)",
)

TEMPORAL_ORDERS = tuple(itertools.permutations(range(3)))


def _render_atom(role: int, predicate: int, *, binary: bool) -> str:
    if not binary:
        return ROLE_NAMES[role].replace("P(", f"P{predicate}(")
    arguments = (
        "X,Y,T1",
        "Y,X,T1",
        "X,Z1,T1",
        "Z1,X,T1",
        "Y,Z2,T2",
        "Z2,Y,T2",
    )[role]
    return f"P{predicate}({arguments})"


@dataclass(frozen=True)
class ProgramRule:
    left_role: int
    left_predicate: int
    right_role: int = -1
    right_predicate: int = -1
    middle_predicate: int = -1
    renewal: bool = False
    temporal_order: int = -1
    guarded: bool = False

    @property
    def length(self) -> int:
        return 3 if self.middle_predicate >= 0 else (2 if self.renewal or self.right_role >= 0 else 1)


def _render_rule_atoms(rule: ProgramRule) -> list[str]:
    if rule.renewal:
        return [
            f"P{rule.left_predicate}(X,Y,T1)",
            f"P{rule.right_predicate}(X,Y,T2)",
        ]
    if rule.length == 3:
        return [
            f"P{rule.left_predicate}(X,Z1,T1)",
            f"P{rule.middle_predicate}(Z2,Z1,T2)",
            f"P{rule.right_predicate}(Z2,Y,T3)",
        ]
    atoms = [
        _render_atom(rule.left_role, rule.left_predicate, binary=rule.length == 2)
    ]
    if rule.length == 2:
        atoms.append(_render_atom(rule.right_role, rule.right_predicate, binary=True))
    if rule.guarded:
        atoms = ["QuerySource(X,Tq)", "Candidate(Y,Tq)", *atoms]
    return atoms


def _render_temporal_condition(rule: ProgramRule) -> str:
    conditions = [f"T{slot + 1}<Tq" for slot in range(rule.length)]
    if rule.length == 3:
        if rule.temporal_order >= 0:
            order = TEMPORAL_ORDERS[rule.temporal_order]
            conditions.append("<".join(f"T{slot + 1}" for slot in order))
        else:
            conditions.append("T2<T3<T1")
    if rule.renewal:
        conditions.append("T1<T2")
        conditions.append("renewal_gap(Tq-T2,T2-T1)")
    return " & ".join(conditions)


class LinkFactEncoder(nn.Module):
    """Embed a grounded Link fact without embedding its entity arguments."""

    def __init__(
        self,
        raw_feature_dim: int,
        fact_context_len: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        del dropout  # one grounded Link fact must always receive one fact type
        self.raw_feature_dim = int(raw_feature_dim)
        self.fact_context_len = int(fact_context_len)
        context_input_dim = LIFTER_CONTEXT_TOKEN_DIM - 1
        self.context_token_encoder = nn.Sequential(
            nn.Linear(context_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.raw_encoder = (
            nn.Sequential(
                nn.LayerNorm(self.raw_feature_dim),
                nn.Linear(self.raw_feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if self.raw_feature_dim > 0
            else None
        )
        # count is retained: it is a property of the fact's causal context,
        # not a query/candidate-specific hand-crafted shortcut.
        fusion_width = 3 * hidden_dim + 1
        self.fusion = nn.Sequential(
            nn.Linear(fusion_width, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw_start = 1
        raw_end = raw_start + self.raw_feature_dim
        context = features[..., raw_end:].reshape(
            *features.shape[:-1], self.fact_context_len, LIFTER_CONTEXT_TOKEN_DIM
        )
        valid = context[..., -1] > 0
        token_state = self.context_token_encoder(context[..., :-1].float())
        weights = valid.unsqueeze(-1).to(token_state.dtype)
        count = weights.sum(dim=-2)
        mean = (token_state * weights).sum(dim=-2) / count.clamp_min(1)
        maximum = token_state.masked_fill(
            ~valid.unsqueeze(-1), torch.finfo(token_state.dtype).min
        ).amax(dim=-2)
        maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
        if self.raw_encoder is None:
            raw_state = torch.zeros_like(mean)
        else:
            raw_state = self.raw_encoder(features[..., raw_start:raw_end].float())
        return self.fusion(
            torch.cat((raw_state, mean, maximum, torch.log1p(count)), dim=-1)
        )


class LiFTER(AttentionModel):
    """Neuro-symbolic temporal link predictor with faithful rule execution."""

    model_name = "LiFTER"
    is_neuro_symbolic = True
    attention_expected = False
    aggregation_operator = "finite_temporal_logic_program"

    def __init__(
        self,
        *,
        num_nodes: int | None = None,
        edge_feat_dim: int = LIFTER_EDGE_FEATURE_DIM,
        raw_feature_dim: int = 0,
        fact_context_len: int = DEFAULT_FACT_CONTEXT_LEN,
        hidden_dim: int = 64,
        predicate_count: int = 8,
        rule_count: int | None = None,
        max_rule_length: int = 2,
        max_three_hop_paths: int = 32,
        max_grounding_facts: int = 64,
        enforce_three_hop_order: bool = True,
        include_direct_conjunctions: bool = False,
        include_disconnected_conjunctions: bool = False,
        adjacent_three_hop_paths: bool = True,
        bidirectional_adjacent_paths: bool = False,
        idempotent_direct_recurrence: bool = True,
        include_renewal_rules: bool = True,
        program_scope: str = "full",
        predicate_intervention: str = "none",
        predicate_assignment_mode: str = "learned",
        predicate_execution_mode: str = "hard",
        fixed_predicate_centroids: list[list[float]] | None = None,
        fixed_predicate_mean: list[float] | None = None,
        fixed_predicate_scale: list[float] | None = None,
        grounding_aggregation: str = "sum",
        rule_family_aggregation: str = "sum",
        rule_parameterization: str = "independent",
        rule_weight_normalization: str = "none",
        rule_factor_rank: int = 8,
        latent_transition_rule: bool = False,
        latent_second_order_transition_rule: bool = False,
        recurrence_guarded_transitions: bool = False,
        predicate_conditioned_transitions: bool = False,
        positioned_recurrence_rules: bool = False,
        exact_entity_binding: bool = True,
        binding_role_permutation: list[int] | None = None,
        program_rules_enabled: bool = True,
        transition_dim: int = 32,
        sparse_ternary_execution: bool = False,
        sparse_renewal_execution: bool = False,
        evidence_transform: str = "mass",
        temporal_component_count: int = 1,
        temporal_component_scope: str = "all",
        range_restricted_program: bool = False,
        enumerate_temporal_orders: bool = False,
        guard_context_rules: bool = False,
        path_existential_aggregation: str = "sum",
        predicate_temperature: float = 0.7,
        final_predicate_temperature: float = 0.15,
        dropout: float = 0.1,
        predicate_entropy_weight: float = 0.002,
        predicate_diversity_weight: float = 0.05,
        prototype_separation_weight: float = 0.01,
        rule_sparsity_weight: float = 0.0005,
        counterfactual_explanation: bool = True,
        export_rule_limit: int = 64,
        **_: Any,
    ) -> None:
        super().__init__(use_attention=False)
        del rule_count
        if predicate_count < 1:
            raise ValueError("predicate_count must be positive")
        if max_grounding_facts < 1:
            raise ValueError("max_grounding_facts must be positive")
        if max_rule_length not in {1, 2, 3}:
            raise ValueError("max_rule_length must be 1, 2, or 3")
        self.raw_feature_dim = int(raw_feature_dim)
        self.fact_context_len = int(fact_context_len)
        self.edge_feat_dim = lifter_edge_feature_dim(
            self.fact_context_len, self.raw_feature_dim
        )
        if int(edge_feat_dim) != self.edge_feat_dim:
            raise ValueError(
                f"edge_feat_dim must be {self.edge_feat_dim}, got {edge_feat_dim}"
            )
        self.hidden_dim = int(hidden_dim)
        self.num_nodes = None if num_nodes is None else int(num_nodes)
        self.latent_transition_rule = bool(latent_transition_rule)
        self.latent_second_order_transition_rule = bool(
            latent_second_order_transition_rule
        )
        self.recurrence_guarded_transitions = bool(recurrence_guarded_transitions)
        self.predicate_conditioned_transitions = bool(
            predicate_conditioned_transitions
        )
        self.positioned_recurrence_rules = bool(positioned_recurrence_rules)
        self.exact_entity_binding = bool(exact_entity_binding)
        self.program_rules_enabled = bool(program_rules_enabled)
        permutation = tuple(range(6)) if binding_role_permutation is None else tuple(map(int, binding_role_permutation))
        if sorted(permutation) != list(range(6)):
            raise ValueError("binding_role_permutation must permute roles 0..5")
        self.binding_role_permutation = permutation
        self.transition_dim = int(transition_dim)
        if self.latent_transition_rule and (
            self.num_nodes is None or self.num_nodes < 1 or self.transition_dim < 1
        ):
            raise ValueError(
                "latent_transition_rule requires num_nodes and transition_dim > 0"
            )
        if self.latent_second_order_transition_rule and (
            self.num_nodes is None or self.num_nodes < 1 or self.transition_dim < 1
        ):
            raise ValueError(
                "latent_second_order_transition_rule requires num_nodes and "
                "transition_dim > 0"
            )
        self.predicate_count = int(predicate_count)
        self.max_grounding_facts = int(max_grounding_facts)
        self.max_rule_length = int(max_rule_length)
        self.max_three_hop_paths = int(max_three_hop_paths)
        self.enforce_three_hop_order = bool(enforce_three_hop_order)
        self.include_direct_conjunctions = bool(include_direct_conjunctions)
        self.include_disconnected_conjunctions = bool(include_disconnected_conjunctions)
        self.adjacent_three_hop_paths = bool(adjacent_three_hop_paths)
        self.bidirectional_adjacent_paths = bool(bidirectional_adjacent_paths)
        self.idempotent_direct_recurrence = bool(idempotent_direct_recurrence)
        self.include_renewal_rules = bool(include_renewal_rules)
        scope_aliases = {
            "no_three_hop": "no_length_three",
            "three_hop_only": "length_three_only",
        }
        program_scope = scope_aliases.get(program_scope, program_scope)
        if program_scope not in {
            "full", "recurrence_only", "no_length_three", "length_three_only"
        }:
            raise ValueError(
                "program_scope must be full, recurrence_only, no_length_three, "
                "or length_three_only"
            )
        self.program_scope = str(program_scope)
        if predicate_intervention not in {"none", "shuffle_facts", "shuffle_global"}:
            raise ValueError(
                "predicate_intervention must be none, shuffle_facts, or shuffle_global"
            )
        self.predicate_intervention = str(predicate_intervention)
        if predicate_assignment_mode not in {"learned", "random_fixed", "kmeans_fixed"}:
            raise ValueError(
                "predicate_assignment_mode must be learned, random_fixed, or kmeans_fixed"
            )
        if predicate_execution_mode not in {"hard", "soft"}:
            raise ValueError("predicate_execution_mode must be hard or soft")
        self.predicate_assignment_mode = str(predicate_assignment_mode)
        self.predicate_execution_mode = str(predicate_execution_mode)
        if grounding_aggregation not in {"sum", "mean"}:
            raise ValueError("grounding_aggregation must be sum or mean")
        self.grounding_aggregation = str(grounding_aggregation)
        if rule_family_aggregation not in {"sum", "mean"}:
            raise ValueError("rule_family_aggregation must be sum or mean")
        self.rule_family_aggregation = str(rule_family_aggregation)
        if rule_parameterization not in {"independent", "compositional", "hybrid"}:
            raise ValueError(
                "rule_parameterization must be independent, compositional, or hybrid"
            )
        self.rule_parameterization = str(rule_parameterization)
        if rule_weight_normalization not in {"none", "signed_distribution"}:
            raise ValueError(
                "rule_weight_normalization must be none or signed_distribution"
            )
        self.rule_weight_normalization = str(rule_weight_normalization)
        self.rule_factor_rank = int(rule_factor_rank)
        self.sparse_ternary_execution = bool(sparse_ternary_execution)
        self.sparse_renewal_execution = bool(sparse_renewal_execution)
        if evidence_transform not in {"mass", "log1p"}:
            raise ValueError("evidence_transform must be mass or log1p")
        self.evidence_transform = str(evidence_transform)
        self.temporal_component_count = int(temporal_component_count)
        if self.temporal_component_count < 1:
            raise ValueError("temporal_component_count must be positive")
        if temporal_component_scope not in {"all", "renewal", "multi_fact"}:
            raise ValueError(
                "temporal_component_scope must be all, renewal, or multi_fact"
            )
        self.temporal_component_scope = str(temporal_component_scope)
        self.ternary_component_count = (
            self.temporal_component_count
            if self.temporal_component_scope in {"all", "multi_fact"} else 1
        )
        self.renewal_component_count = self.temporal_component_count
        self.range_restricted_program = bool(range_restricted_program)
        self.enumerate_temporal_orders = bool(enumerate_temporal_orders)
        self.guard_context_rules = bool(guard_context_rules)
        self.ternary_order_count = 6 if self.enumerate_temporal_orders else 1
        if self.rule_factor_rank < 1:
            raise ValueError("rule_factor_rank must be positive")
        if path_existential_aggregation not in {"sum", "mean", "max"}:
            raise ValueError("path_existential_aggregation must be sum, mean, or max")
        self.path_existential_aggregation = str(path_existential_aggregation)
        self.predicate_temperature = float(predicate_temperature)
        self.final_predicate_temperature = float(final_predicate_temperature)
        self._temperature = self.predicate_temperature
        self._execution_mode = "auto"

        self.fact_encoder = LinkFactEncoder(
            self.raw_feature_dim,
            self.fact_context_len,
            self.hidden_dim,
            dropout,
        )
        prototypes = torch.randn(self.predicate_count, self.hidden_dim)
        self.predicate_prototypes = nn.Parameter(F.normalize(prototypes, dim=-1))
        self.predicate_logit_scale = nn.Parameter(torch.tensor(2.0))
        if self.predicate_assignment_mode == "random_fixed":
            projection = torch.randn(self.edge_feat_dim, self.predicate_count)
            self.register_buffer(
                "fixed_predicate_projection", F.normalize(projection, dim=0)
            )
        elif self.predicate_assignment_mode == "kmeans_fixed":
            if fixed_predicate_centroids is None:
                raise ValueError("kmeans_fixed requires fixed_predicate_centroids")
            centroids = torch.as_tensor(fixed_predicate_centroids, dtype=torch.float32)
            if centroids.shape != (self.predicate_count, self.edge_feat_dim):
                raise ValueError(
                    "fixed_predicate_centroids must have shape "
                    f"({self.predicate_count}, {self.edge_feat_dim})"
                )
            mean = torch.as_tensor(
                fixed_predicate_mean
                if fixed_predicate_mean is not None
                else [0.0] * self.edge_feat_dim,
                dtype=torch.float32,
            )
            scale = torch.as_tensor(
                fixed_predicate_scale
                if fixed_predicate_scale is not None
                else [1.0] * self.edge_feat_dim,
                dtype=torch.float32,
            )
            if mean.shape != (self.edge_feat_dim,) or scale.shape != (self.edge_feat_dim,):
                raise ValueError("fixed predicate normalization has invalid shape")
            self.register_buffer("fixed_predicate_centroids", centroids)
            self.register_buffer("fixed_predicate_mean", mean)
            self.register_buffer("fixed_predicate_scale", scale.clamp_min(1e-6))
        if self.predicate_assignment_mode != "learned" or self.predicate_count == 1:
            for parameter in self.fact_encoder.parameters():
                parameter.requires_grad_(False)
            self.predicate_prototypes.requires_grad_(False)
            self.predicate_logit_scale.requires_grad_(False)
        # Complete finite grammar. Unary atoms cover recurrence and facts
        # incident to either query argument. Binary rules conjoin one X-bound
        # fact and one Y-bound fact. Their existential variables are distinct,
        # which makes the grammar type-correct for bipartite links.
        rules: list[ProgramRule] = []
        unary_roles = (
            range(2)
            if self.range_restricted_program and not self.guard_context_rules
            else range(len(ROLE_NAMES))
        )
        for role in unary_roles:
            for predicate in range(self.predicate_count):
                rules.append(
                    ProgramRule(
                        role,
                        predicate,
                        guarded=self.guard_context_rules and role >= 2,
                    )
                )
        if self.max_rule_length >= 2 and self.include_renewal_rules:
            for earlier_predicate in range(self.predicate_count):
                for later_predicate in range(self.predicate_count):
                    rules.append(
                        ProgramRule(
                            0,
                            earlier_predicate,
                            right_predicate=later_predicate,
                            renewal=True,
                        )
                    )
        if self.max_rule_length >= 2 and self.include_disconnected_conjunctions:
            first_conjunctive_role = 0 if self.include_direct_conjunctions else 2
            for left_role in range(first_conjunctive_role, 4):
                for right_role in range(4, 6):
                    for left_predicate in range(self.predicate_count):
                        for right_predicate in range(self.predicate_count):
                            rules.append(ProgramRule(left_role, left_predicate, right_role, right_predicate))
        if self.max_rule_length >= 3:
            for left_predicate in range(self.predicate_count):
                for middle_predicate in range(self.predicate_count):
                    for right_predicate in range(self.predicate_count):
                        order_ids = (
                            range(len(TEMPORAL_ORDERS))
                            if self.enumerate_temporal_orders else (-1,)
                        )
                        for order_id in order_ids:
                            rules.append(
                                ProgramRule(
                                    2,
                                    left_predicate,
                                    5,
                                    right_predicate,
                                    middle_predicate,
                                    temporal_order=order_id,
                                )
                            )
        if self.program_scope == "recurrence_only":
            rules = [
                rule for rule in rules
                if rule.renewal or (rule.length == 1 and rule.left_role in {0, 1})
            ]
        elif self.program_scope == "no_length_three":
            rules = [rule for rule in rules if rule.length < 3]
        elif self.program_scope == "length_three_only":
            rules = [rule for rule in rules if rule.length == 3]
        if not rules:
            raise ValueError(f"program_scope={self.program_scope!r} produced no rules")
        base_rules = tuple(rules)
        component_counts = [
            self.temporal_component_count
            if (
                self.temporal_component_scope == "all"
                or (self.temporal_component_scope == "renewal" and rule.renewal)
                or (self.temporal_component_scope == "multi_fact" and rule.length > 1)
            )
            else 1
            for rule in base_rules
        ]
        rules = [
            rule
            for rule, component_count in zip(base_rules, component_counts)
            for _ in range(component_count)
        ]
        temporal_components = [
            component
            for component_count in component_counts
            for component in range(component_count)
        ]
        if self.program_scope in {"recurrence_only", "no_length_three"}:
            self.max_rule_length = min(self.max_rule_length, 2)
        self.program = tuple(rules)
        self.rule_count = len(self.program)
        self.register_buffer(
            "temporal_components",
            torch.tensor(temporal_components, dtype=torch.long),
        )
        self.register_buffer(
            "left_roles", torch.tensor([rule.left_role for rule in self.program])
        )
        self.register_buffer(
            "left_predicates",
            torch.tensor([rule.left_predicate for rule in self.program]),
        )
        self.register_buffer(
            "right_roles", torch.tensor([rule.right_role for rule in self.program])
        )
        self.register_buffer(
            "right_predicates",
            torch.tensor([rule.right_predicate for rule in self.program]),
        )
        self.register_buffer(
            "middle_predicates",
            torch.tensor([rule.middle_predicate for rule in self.program]),
        )
        self.register_buffer(
            "renewal_rules", torch.tensor([rule.renewal for rule in self.program])
        )
        family_ids: list[int] = []
        for rule in self.program:
            if rule.renewal:
                family_ids.append(6)
            elif rule.length == 3:
                family_ids.append(7)
            elif rule.length == 2:
                family_ids.append(8 + rule.left_role * 6 + rule.right_role)
            else:
                family_ids.append(rule.left_role)
        unique_families = {value: index for index, value in enumerate(sorted(set(family_ids)))}
        self.register_buffer(
            "rule_family_ids",
            torch.tensor([unique_families[value] for value in family_ids], dtype=torch.long),
        )
        self.rule_family_count = len(unique_families)
        self.rule_weight = nn.Parameter(torch.zeros(self.rule_count))
        if self.rule_weight_normalization == "signed_distribution":
            self.rule_selection_logits = nn.Parameter(torch.zeros(self.rule_count))
            self.rule_family_mass_raw = nn.Parameter(
                torch.full((self.rule_family_count,), 0.541324854612918)
            )
        # The compositional parameterization shares statistical strength across
        # every predicate slot and structural template. It generates the same
        # explicit finite clause weights without privileging a selected length.
        if self.rule_parameterization in {"compositional", "hybrid"}:
            self.rule_predicate_factors = nn.Parameter(
                torch.ones(self.predicate_count, self.rule_factor_rank)
                + torch.randn(self.predicate_count, self.rule_factor_rank) * 0.10
            )
            template_ids: list[int] = []
            for rule in self.program:
                if rule.renewal:
                    template_ids.append(6)
                elif rule.length == 3:
                    template_ids.append(7)
                elif rule.length == 2:
                    template_ids.append(8 + rule.left_role * 6 + rule.right_role)
                else:
                    template_ids.append(rule.left_role)
            self.register_buffer(
                "rule_template_ids", torch.tensor(template_ids, dtype=torch.long)
            )
            self.rule_template_factors = nn.Parameter(
                torch.zeros(max(template_ids) + 1, self.rule_factor_rank)
            )
            if self.rule_parameterization == "hybrid":
                self.rule_shared_gate_logit = nn.Parameter(torch.tensor(-1.3862944))
        self.time_centres = nn.Parameter(torch.zeros(self.rule_count, 3))
        if self.temporal_component_count > 1:
            component_centres = torch.linspace(
                0.0, 9.0, self.temporal_component_count
            ).index_select(0, self.temporal_components)
            initial_centres = component_centres[:, None].expand(-1, 3).clone()
            initial_scale = 1.5
        else:
            initial_centres = torch.zeros(self.rule_count, 3)
            initial_scale = 3.0
        with torch.no_grad():
            self.time_centres.copy_(initial_centres)
        self.time_scale_raw = nn.Parameter(
            torch.full((self.rule_count, 3), initial_scale)
        )
        # An explicit zero-body rule, True -> Link(X,Y,Tq). It is part of the
        # symbolic program and is reported in every explanation.
        self.prior_rule_weight = nn.Parameter(torch.tensor(0.0))
        if self.positioned_recurrence_rules:
            self.positioned_recurrence_rule_weight = nn.Parameter(
                torch.zeros(self.max_grounding_facts, self.predicate_count)
            )
        if self.latent_transition_rule:
            self.transition_source_embedding = nn.Embedding(
                self.num_nodes, self.transition_dim
            )
            self.transition_target_embedding = nn.Embedding(
                self.num_nodes, self.transition_dim
            )
            nn.init.normal_(self.transition_source_embedding.weight, std=0.02)
            nn.init.normal_(self.transition_target_embedding.weight, std=0.02)
            self.transition_scale_raw = nn.Parameter(torch.tensor(0.0))
            if self.predicate_conditioned_transitions:
                self.transition_predicate_factor = nn.Embedding(
                    self.predicate_count, self.transition_dim
                )
                nn.init.ones_(self.transition_predicate_factor.weight)
            if self.recurrence_guarded_transitions:
                self.transition_guard_scale_raw = nn.Parameter(torch.zeros(2))
        if self.latent_second_order_transition_rule:
            self.transition2_first_embedding = nn.Embedding(
                self.num_nodes, self.transition_dim
            )
            self.transition2_second_embedding = nn.Embedding(
                self.num_nodes, self.transition_dim
            )
            self.transition2_target_embedding = nn.Embedding(
                self.num_nodes, self.transition_dim
            )
            nn.init.normal_(self.transition2_first_embedding.weight, std=0.02)
            nn.init.normal_(self.transition2_second_embedding.weight, std=0.02)
            nn.init.normal_(self.transition2_target_embedding.weight, std=0.02)
            self.transition2_scale_raw = nn.Parameter(torch.tensor(0.0))
            if self.predicate_conditioned_transitions:
                self.transition2_first_predicate_factor = nn.Embedding(
                    self.predicate_count, self.transition_dim
                )
                self.transition2_second_predicate_factor = nn.Embedding(
                    self.predicate_count, self.transition_dim
                )
                nn.init.ones_(self.transition2_first_predicate_factor.weight)
                nn.init.ones_(self.transition2_second_predicate_factor.weight)
            if self.recurrence_guarded_transitions:
                self.transition2_guard_scale_raw = nn.Parameter(torch.zeros(2))

        self.predicate_entropy_weight = float(predicate_entropy_weight)
        self.predicate_diversity_weight = float(predicate_diversity_weight)
        self.prototype_separation_weight = float(prototype_separation_weight)
        self.rule_sparsity_weight = float(rule_sparsity_weight)
        self.counterfactual_explanation = bool(counterfactual_explanation)
        self.export_rule_limit = max(1, int(export_rule_limit))

    def set_symbolic_progress(self, progress: float) -> None:
        progress = min(1.0, max(0.0, float(progress)))
        self._temperature = self.predicate_temperature * (
            self.final_predicate_temperature / self.predicate_temperature
        ) ** progress

    def set_execution_mode(self, mode: str) -> None:
        if mode not in {"auto", "soft", "hard"}:
            raise ValueError("execution mode must be auto, soft, or hard")
        self._execution_mode = mode

    @contextmanager
    def execution_mode(self, mode: str):
        previous = self._execution_mode
        self.set_execution_mode(mode)
        try:
            yield self
        finally:
            self._execution_mode = previous

    def _hard_execution(self) -> bool:
        if self._execution_mode == "soft":
            return False
        if self._execution_mode == "hard":
            return True
        return self.predicate_execution_mode == "hard"

    def _truncate(
        self,
        nodes: torch.Tensor,
        times: torch.Tensor,
        features: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        width = min(nodes.shape[1], self.max_grounding_facts)
        nodes, times, features = (
            nodes[:, -width:],
            times[:, -width:],
            features[:, -width:],
        )
        if features.shape[-1] != self.edge_feat_dim:
            raise ValueError(
                f"LiFTER expected {self.edge_feat_dim} fact features, got {features.shape[-1]}"
            )
        valid = (
            torch.ones_like(nodes, dtype=torch.bool)
            if mask is None
            else mask[:, -width:].bool()
        )
        return nodes, times, features, valid

    def _type_facts(
        self,
        features: torch.Tensor,
        valid: torch.Tensor,
        fact_sources: torch.Tensor | None = None,
        fact_destinations: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # With one predicate, every valid interaction is the same observed
        # relation atom.  The encoder/prototype softmax would be identically
        # one, so bypass it exactly rather than performing a vacuous neural
        # assignment.
        if self.predicate_count == 1:
            soft = torch.ones(
                (*features.shape[:2], 1),
                dtype=features.dtype,
                device=features.device,
            )
        elif self.predicate_assignment_mode == "learned":
            state = self.fact_encoder(features)
            state = F.normalize(state, dim=-1)
            prototypes = F.normalize(self.predicate_prototypes, dim=-1)
            scale = F.softplus(self.predicate_logit_scale) + 1.0
            logits = scale * torch.einsum("bhd,kd->bhk", state, prototypes)
            soft = torch.softmax(logits / self._temperature, dim=-1)
        elif self.predicate_assignment_mode == "random_fixed":
            normalized = F.normalize(features.float(), dim=-1)
            logits = torch.einsum(
                "bhd,dk->bhk", normalized, self.fixed_predicate_projection
            )
            soft = F.one_hot(logits.argmax(-1), self.predicate_count).to(logits.dtype)
        else:
            normalized = (
                features.float() - self.fixed_predicate_mean
            ) / self.fixed_predicate_scale
            distances = (
                normalized[:, :, None, :] - self.fixed_predicate_centroids[None, None]
            ).square().mean(-1)
            logits = -distances
            soft = F.one_hot(logits.argmax(-1), self.predicate_count).to(logits.dtype)
        hard_ids = soft.argmax(dim=-1)
        if not self.training and self.predicate_intervention == "shuffle_facts":
            # Preserve each batch row's predicate histogram while breaking the
            # learned association between a grounded fact and its type. Padding
            # positions are excluded so the valid-fact histogram is exact.
            shuffled = hard_ids.clone()
            for row in range(len(hard_ids)):
                valid_indices = torch.nonzero(valid[row], as_tuple=False).flatten()
                if valid_indices.numel() > 1:
                    shuffled[row, valid_indices] = hard_ids[
                        row, valid_indices.roll(shifts=1)
                    ]
            hard_ids = shuffled
        elif not self.training and self.predicate_intervention == "shuffle_global":
            # Preserve the complete evaluation batch's predicate histogram but
            # break both fact-level and query-level associations. The fixed
            # rotation is deterministic and excludes padding positions.
            shuffled = hard_ids.clone()
            valid_indices = torch.nonzero(valid, as_tuple=False)
            if valid_indices.shape[0] > 1:
                donor_indices = valid_indices.roll(shifts=1, dims=0)
                shuffled[valid_indices[:, 0], valid_indices[:, 1]] = hard_ids[
                    donor_indices[:, 0], donor_indices[:, 1]
                ]
            hard_ids = shuffled
        hard = F.one_hot(hard_ids, self.predicate_count).to(soft.dtype)
        assignment = soft if not self._hard_execution() else hard + soft - soft.detach()
        validity = valid.unsqueeze(-1).to(assignment.dtype)
        return assignment * validity, soft * validity, hard_ids

    def _role_masks(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        src_nodes: torch.Tensor,
        dst_nodes: torch.Tensor,
        src_features: torch.Tensor,
        dst_features: torch.Tensor,
        src_valid: torch.Tensor,
        dst_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        src_out = src_features[..., 0] >= 0
        dst_out = dst_features[..., 0] >= 0
        repeated = src_nodes.long() == dst[:, None].long()
        roles = (
            src_valid & src_out & repeated if self.exact_entity_binding else src_valid & src_out,
            src_valid & ~src_out & repeated if self.exact_entity_binding else src_valid & ~src_out,
            src_valid & src_out,
            src_valid & ~src_out,
            dst_valid & dst_out,
            dst_valid & ~dst_out,
        )
        return tuple(roles[index] for index in self.binding_role_permutation)

    def _ground_atoms(
        self,
        assignment: torch.Tensor,
        times: torch.Tensor,
        valid_mask: torch.Tensor,
        query_time: torch.Tensor,
        rule_ids: torch.Tensor,
        predicate_ids: torch.Tensor,
        slot: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # [B,H,K] -> [B,R,H] memberships for each fixed rule atom.
        membership = assignment.index_select(2, predicate_ids).permute(0, 2, 1)
        if self._hard_execution():
            atom_log = torch.where(
                membership.detach() > 0.5,
                torch.zeros_like(membership),
                torch.full_like(membership, -1.0e4),
            )
            if self.training:
                soft_log = membership.clamp_min(1e-12).log()
                atom_log = atom_log + soft_log - soft_log.detach()
        else:
            atom_log = membership.clamp_min(1e-12).log()
        log_age = torch.log1p(
            (query_time[:, None] - times.float()).clamp_min(0)
        )
        centre = self.time_centres.index_select(0, rule_ids)[:, slot]
        scale = F.softplus(
            self.time_scale_raw.index_select(0, rule_ids)[:, slot]
        ) + 0.05
        temporal = -0.5 * (
            (log_age[:, None, :] - centre[None, :, None])
            / scale[None, :, None]
        ).square()
        mask = valid_mask if valid_mask.ndim == 3 else valid_mask[:, None, :]
        values = atom_log + temporal
        masked = values.masked_fill(~mask, torch.finfo(values.dtype).min)
        type_match = membership.detach() > (0.5 if self._hard_execution() else 0)
        proof = torch.logsumexp(masked, dim=-1)
        if self.grounding_aggregation == "mean":
            grounding_count = (mask & type_match).sum(dim=-1).clamp_min(1)
            proof = proof - grounding_count.log()
        if self.idempotent_direct_recurrence:
            roles = self.left_roles.index_select(0, rule_ids)
            direct_atom = (roles == 0) | (roles == 1)
            proof = torch.where(
                direct_atom[None], masked.amax(dim=-1), proof
            )
        presence = (mask & type_match).any(dim=-1)
        proof = torch.where(presence, proof, torch.full_like(proof, _NO_PROOF))
        best = values.masked_fill(~(mask & type_match), torch.finfo(values.dtype).min).argmax(-1)
        best = torch.where(presence, best, torch.full_like(best, -1))
        return proof, presence, best

    def _regularization(
        self,
        src_soft: torch.Tensor,
        dst_soft: torch.Tensor,
        src_valid: torch.Tensor,
        dst_valid: torch.Tensor,
        effective_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not self.training:
            zero = self.rule_weight.new_zeros(())
            return zero, {
                "predicate_entropy": zero,
                "predicate_diversity_kl": zero,
                "rule_entropy": zero,
                "rule_usage_entropy": zero,
                "rule_diversity_kl": zero,
            }
        probabilities = torch.cat(
            (src_soft[src_valid], dst_soft[dst_valid]), dim=0
        )
        if probabilities.numel() == 0:
            entropy = self.rule_weight.sum() * 0
            diversity = entropy
        else:
            entropy = -(
                probabilities * probabilities.clamp_min(1e-12).log()
            ).sum(-1).mean()
            usage = probabilities.mean(0)
            diversity = (
                usage * (usage.clamp_min(1e-12).log() + math.log(self.predicate_count))
            ).sum()
        prototypes = F.normalize(self.predicate_prototypes, dim=-1)
        gram = prototypes @ prototypes.T
        off_diagonal = gram - torch.eye(
            self.predicate_count, device=gram.device, dtype=gram.dtype
        )
        separation = off_diagonal.square().sum() / (
            self.predicate_count * max(1, self.predicate_count - 1)
        )
        if effective_weights is None:
            effective_weights = self._effective_rule_weights()
        magnitudes = effective_weights.abs()
        sparsity = magnitudes.mean()
        total = (
            self.predicate_entropy_weight * entropy
            + self.predicate_diversity_weight * diversity
            + self.prototype_separation_weight * separation
            + self.rule_sparsity_weight * sparsity
        )
        metrics = {
            "predicate_entropy": entropy.detach(),
            "predicate_diversity_kl": diversity.detach(),
            "rule_entropy": torch.zeros_like(entropy.detach()),
            "rule_usage_entropy": (
                -(magnitudes * magnitudes.clamp_min(1e-12).log()).mean()
            ).detach(),
            "rule_diversity_kl": sparsity.detach(),
        }
        return total, metrics

    def _effective_rule_weights(self) -> torch.Tensor:
        if self.rule_parameterization == "independent":
            raw_weight = self.rule_weight
        else:
            slot_product = self.rule_predicate_factors.index_select(
                0, self.left_predicates
            )
            has_right = self.right_predicates >= 0
            if bool(has_right.any()):
                slot_product = slot_product.clone()
                slot_product[has_right] = slot_product[has_right] * (
                    self.rule_predicate_factors.index_select(
                        0, self.right_predicates[has_right]
                    )
                )
            has_middle = self.middle_predicates >= 0
            if bool(has_middle.any()):
                slot_product[has_middle] = slot_product[has_middle] * (
                    self.rule_predicate_factors.index_select(
                        0, self.middle_predicates[has_middle]
                    )
                )
            template = self.rule_template_factors.index_select(
                0, self.rule_template_ids
            )
            generated = (slot_product * template).sum(-1) / math.sqrt(
                self.rule_factor_rank
            )
            raw_weight = (
                torch.sigmoid(self.rule_shared_gate_logit) * generated
                + self.rule_weight
                if self.rule_parameterization == "hybrid"
                else generated
            )
        polarity = torch.tanh(raw_weight)
        if self.rule_weight_normalization == "none":
            effective = polarity
        else:
            selection = torch.empty_like(self.rule_selection_logits)
            for family in range(self.rule_family_count):
                mask = self.rule_family_ids == family
                selection[mask] = torch.softmax(self.rule_selection_logits[mask], dim=0)
            mass = F.softplus(self.rule_family_mass_raw).index_select(
                0, self.rule_family_ids
            )
            effective = mass * selection * polarity
        return effective

    def _sparse_ternary_proofs(
        self,
        path_assignment: torch.Tensor,
        path_soft: torch.Tensor,
        path_times: torch.Tensor,
        query_time: torch.Tensor,
        valid: torch.Tensor,
        ternary_rule_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dispatch each hard-typed grounding to exactly one explicit clause."""
        batch_size, path_count, slot_count, _ = path_assignment.shape
        component_count = self.ternary_component_count
        expected_rules = (
            self.predicate_count**3
            * self.ternary_order_count
            * component_count
        )
        if slot_count != 3 or len(ternary_rule_ids) != expected_rules:
            raise ValueError("sparse ternary dispatch requires the complete typed basis")
        hard_ids = path_assignment.detach().argmax(-1)
        predicate_rule = (
            hard_ids[:, :, 0] * self.predicate_count**2
            + hard_ids[:, :, 1] * self.predicate_count
            + hard_ids[:, :, 2]
        )
        if self.enumerate_temporal_orders:
            sorted_slots = path_times.argsort(dim=-1)
            order_table = torch.tensor(
                TEMPORAL_ORDERS,
                dtype=sorted_slots.dtype,
                device=sorted_slots.device,
            )
            order_match = (
                sorted_slots[:, :, None, :] == order_table[None, None, :, :]
            ).all(-1)
            order_id = order_match.to(torch.long).argmax(-1)
        else:
            order_id = torch.zeros_like(predicate_rule)
        base_rule = predicate_rule * self.ternary_order_count + order_id
        components = torch.arange(
            component_count, device=path_assignment.device
        )
        local_rule = (
            base_rule[:, :, None] * component_count
            + components[None, None, :]
        )
        global_rule = ternary_rule_ids.index_select(0, local_rule.reshape(-1)).reshape(
            batch_size, path_count, component_count
        )
        log_truth = torch.zeros(
            batch_size, path_count, component_count,
            device=path_assignment.device,
            dtype=path_assignment.dtype,
        )
        for slot in range(3):
            chosen_soft = path_soft[:, :, slot, :].gather(
                2, hard_ids[:, :, slot, None]
            ).squeeze(-1).clamp_min(1e-12)
            soft_log = chosen_soft.log()
            atom_log = (
                soft_log - soft_log.detach()
                if self.training else torch.zeros_like(soft_log)
            )[:, :, None]
            log_age = torch.log1p(
                (query_time[:, None] - path_times[:, :, slot]).clamp_min(0)
            )
            centre = self.time_centres[:, slot].gather(0, global_rule.reshape(-1)).reshape(
                batch_size, path_count, component_count
            )
            scale = (
                F.softplus(self.time_scale_raw[:, slot]) + 0.05
            ).gather(0, global_rule.reshape(-1)).reshape(
                batch_size, path_count, component_count
            )
            temporal = -0.5 * ((log_age[:, :, None] - centre) / scale).square()
            log_truth = log_truth + atom_log + temporal
        truth = torch.exp(log_truth / 3.0) * valid[:, :, None].to(log_truth.dtype)
        flat_rule = local_rule.flatten(1)
        flat_truth = truth.flatten(1)
        flat_valid = valid[:, :, None].expand_as(truth).flatten(1)
        local_evidence = torch.zeros(
            batch_size, len(ternary_rule_ids), device=truth.device, dtype=truth.dtype
        )
        local_evidence.scatter_add_(1, flat_rule, flat_truth)
        counts = torch.zeros_like(local_evidence)
        counts.scatter_add_(1, flat_rule, flat_valid.to(truth.dtype))
        if self.path_existential_aggregation == "mean":
            local_evidence = local_evidence / counts.clamp_min(1)
        elif self.path_existential_aggregation == "max":
            local_evidence.zero_().scatter_reduce_(
                1, flat_rule, flat_truth, reduce="amax", include_self=False
            )
        presence = counts > 0
        proof = torch.where(
            presence,
            local_evidence.clamp_min(1e-30).log(),
            torch.full_like(local_evidence, _NO_PROOF),
        )
        best_score = torch.full_like(local_evidence, -float("inf"))
        best_score.scatter_reduce_(
            1, flat_rule, flat_truth.detach(), reduce="amax", include_self=True
        )
        selected_best = best_score.gather(1, flat_rule)
        path_ids = torch.arange(path_count, device=truth.device)[None, :, None].expand_as(truth).flatten(1)
        candidates = torch.where(
            flat_valid & torch.isclose(flat_truth.detach(), selected_best), path_ids + 1,
            torch.zeros_like(path_ids),
        )
        best = torch.zeros(
            batch_size, len(ternary_rule_ids), device=truth.device, dtype=torch.long
        )
        best.scatter_reduce_(1, flat_rule, candidates, reduce="amax", include_self=True)
        best = torch.where(presence, best - 1, torch.full_like(best, -1))
        return proof, presence, best

    def _sparse_renewal_proofs(
        self,
        assignment: torch.Tensor,
        soft: torch.Tensor,
        times: torch.Tensor,
        query_time: torch.Tensor,
        direct_mask: torch.Tensor,
        renewal_rule_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dispatch each ordered same-pair fact pair to one typed renewal rule."""
        batch_size, history_width, _ = assignment.shape
        hard_ids = assignment.detach().argmax(-1)
        earlier_ids = hard_ids[:, :, None].expand(-1, -1, history_width)
        later_ids = hard_ids[:, None, :].expand(-1, history_width, -1)
        base_rule = earlier_ids * self.predicate_count + later_ids
        component_count = self.renewal_component_count
        components = torch.arange(
            component_count, device=assignment.device
        )
        local_rule = (
            base_rule[:, :, :, None] * component_count
            + components[None, None, None, :]
        )
        ordered = times[:, :, None] < times[:, None, :]
        valid = direct_mask[:, :, None] & direct_mask[:, None, :] & ordered
        global_rule = renewal_rule_ids.index_select(
            0, local_rule.reshape(-1)
        ).reshape(
            batch_size, history_width, history_width,
            component_count,
        )
        earlier_soft = soft.gather(2, hard_ids[..., None]).squeeze(-1).clamp_min(1e-12)
        later_soft = earlier_soft
        earlier_log = earlier_soft.log()[:, :, None, None]
        later_log = later_soft.log()[:, None, :, None]
        if self.training:
            earlier_log = earlier_log - earlier_log.detach()
            later_log = later_log - later_log.detach()
        else:
            earlier_log = torch.zeros_like(earlier_log)
            later_log = torch.zeros_like(later_log)
        previous_gap = (times[:, None, :] - times[:, :, None]).clamp_min(0)
        query_gap = (query_time[:, None] - times).clamp_min(0)
        gap_error = (
            torch.log1p(query_gap)[:, None, :]
            - torch.log1p(previous_gap)
        )[:, :, :, None]
        centre = self.time_centres[:, 0].gather(
            0, global_rule.reshape(-1)
        ).reshape_as(global_rule)
        scale = (F.softplus(self.time_scale_raw[:, 0]) + 0.05).gather(
            0, global_rule.reshape(-1)
        ).reshape_as(global_rule)
        truth = torch.exp(earlier_log + later_log - 0.5 * ((gap_error - centre) / scale).square())
        truth = truth * valid[:, :, :, None].to(truth.dtype)
        flat_rule = local_rule.flatten(1)
        flat_truth = truth.flatten(1)
        flat_valid = valid[:, :, :, None].expand_as(truth).flatten(1)
        evidence = torch.zeros(
            batch_size, len(renewal_rule_ids), device=truth.device, dtype=truth.dtype
        )
        evidence.scatter_add_(1, flat_rule, flat_truth)
        counts = torch.zeros_like(evidence)
        counts.scatter_add_(1, flat_rule, flat_valid.to(truth.dtype))
        if self.grounding_aggregation == "mean":
            evidence = evidence / counts.clamp_min(1)
        presence = counts > 0
        proof = torch.where(
            presence, evidence.clamp_min(1e-30).log(),
            torch.full_like(evidence, _NO_PROOF),
        )
        best_score = torch.full_like(evidence, -float("inf"))
        best_score.scatter_reduce_(
            1, flat_rule, flat_truth.detach(), reduce="amax", include_self=True
        )
        selected = best_score.gather(1, flat_rule)
        pair_ids = torch.arange(
            history_width * history_width, device=truth.device
        ).reshape(1, history_width, history_width, 1).expand_as(truth).flatten(1)
        candidates = torch.where(
            flat_valid & torch.isclose(flat_truth.detach(), selected), pair_ids + 1,
            torch.zeros_like(pair_ids),
        )
        best_pair = torch.zeros(
            batch_size, len(renewal_rule_ids), device=truth.device, dtype=torch.long
        )
        best_pair.scatter_reduce_(
            1, flat_rule, candidates, reduce="amax", include_self=True
        )
        best_pair = torch.where(presence, best_pair - 1, torch.full_like(best_pair, -1))
        earlier_best = torch.where(
            presence, best_pair // history_width, torch.full_like(best_pair, -1)
        )
        later_best = torch.where(
            presence, best_pair % history_width, torch.full_like(best_pair, -1)
        )
        return proof, presence, earlier_best, later_best

    def forward(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        timestamp: torch.Tensor,
        history_nodes: torch.Tensor,
        history_times: torch.Tensor,
        history_edge_feats: torch.Tensor,
        history_mask: torch.Tensor | None = None,
        dst_history_nodes: torch.Tensor | None = None,
        dst_history_times: torch.Tensor | None = None,
        dst_history_edge_feats: torch.Tensor | None = None,
        dst_history_mask: torch.Tensor | None = None,
        path_fact_sources: torch.Tensor | None = None,
        path_fact_destinations: torch.Tensor | None = None,
        path_fact_times: torch.Tensor | None = None,
        path_fact_features: torch.Tensor | None = None,
        path_mask: torch.Tensor | None = None,
    ) -> ModelResult:
        if dst_history_nodes is None or dst_history_times is None or dst_history_edge_feats is None:
            raise ValueError("LiFTER requires histories for both Link arguments")
        src_nodes, src_times, src_features, src_valid = self._truncate(
            history_nodes, history_times, history_edge_feats, history_mask
        )
        dst_nodes, dst_times, dst_features, dst_valid = self._truncate(
            dst_history_nodes, dst_history_times, dst_history_edge_feats, dst_history_mask
        )
        src_valid &= src_times.float() < timestamp[:, None].float()
        dst_valid &= dst_times.float() < timestamp[:, None].float()
        src_out = src_features[..., 0] >= 0
        src_fact_sources = torch.where(src_out, src[:, None], src_nodes)
        src_fact_destinations = torch.where(src_out, src_nodes, src[:, None])
        dst_out = dst_features[..., 0] >= 0
        dst_fact_sources = torch.where(dst_out, dst[:, None], dst_nodes)
        dst_fact_destinations = torch.where(dst_out, dst_nodes, dst[:, None])
        shared_width = int(getattr(self, "_shared_source_width", 0))
        if shared_width and len(src) % shared_width == 0:
            repeats = len(src) // shared_width
            base_assign, base_soft, base_ids = self._type_facts(
                src_features[:shared_width], src_valid[:shared_width],
                src_fact_sources[:shared_width], src_fact_destinations[:shared_width],
            )
            src_assign = base_assign.repeat((repeats, 1, 1))
            src_soft = base_soft.repeat((repeats, 1, 1))
            src_ids = base_ids.repeat((repeats, 1))
        else:
            src_assign, src_soft, src_ids = self._type_facts(
                src_features, src_valid, src_fact_sources, src_fact_destinations
            )
        dst_assign, dst_soft, dst_ids = self._type_facts(
            dst_features, dst_valid, dst_fact_sources, dst_fact_destinations
        )
        role_masks = self._role_masks(
            src, dst, src_nodes, dst_nodes, src_features, dst_features, src_valid, dst_valid
        )

        all_rule_ids = torch.arange(self.rule_count, device=src.device)
        left_bank_is_dst = self.left_roles >= 4
        standalone_left = (~self.renewal_rules) & (self.middle_predicates < 0)
        left_proof = timestamp.new_full((len(src), self.rule_count), _NO_PROOF)
        left_presence = torch.zeros_like(left_proof, dtype=torch.bool)
        left_best = torch.full_like(left_proof, -1, dtype=torch.long)
        for bank_is_dst in (False, True):
            selected = all_rule_ids[
                (left_bank_is_dst == bank_is_dst) & standalone_left
            ]
            if selected.numel() == 0:
                continue
            roles = self.left_roles.index_select(0, selected)
            masks = torch.stack([role_masks[int(role)] for role in roles], dim=1)
            proof, presence, best = self._ground_atoms(
                dst_assign if bank_is_dst else src_assign,
                dst_times if bank_is_dst else src_times,
                masks,
                timestamp,
                selected,
                self.left_predicates.index_select(0, selected),
                0,
            )
            left_proof[:, selected] = proof
            left_presence[:, selected] = presence
            left_best[:, selected] = best

        binary = (self.right_roles >= 0) & (self.middle_predicates < 0)
        binary_ids = all_rule_ids[binary]
        rule_proof = left_proof.clone()
        rule_presence = left_presence.clone()
        right_best_by_rule = torch.full_like(left_best, -1)
        renewal_ids = all_rule_ids[self.renewal_rules]
        if renewal_ids.numel() > 0:
            direct_mask = role_masks[0]
            if self.sparse_renewal_execution and self._hard_execution():
                renewal_proof, renewal_presence, earlier_best, later_best = (
                    self._sparse_renewal_proofs(
                        src_assign, src_soft, src_times, timestamp, direct_mask,
                        renewal_ids,
                    )
                )
            else:
                renewal_predicates = torch.stack(
                    (
                        self.left_predicates[self.renewal_rules],
                        self.right_predicates[self.renewal_rules],
                    ), dim=-1,
                )
                earlier_membership = src_assign.index_select(
                    2, renewal_predicates[:, 0]
                ).permute(0, 2, 1)
                later_membership = src_assign.index_select(
                    2, renewal_predicates[:, 1]
                ).permute(0, 2, 1)
                def membership_log(membership: torch.Tensor) -> torch.Tensor:
                    if not self._hard_execution():
                        return membership.clamp_min(1e-12).log()
                    hard_log = torch.where(
                        membership.detach() > 0.5, torch.zeros_like(membership),
                        torch.full_like(membership, -1.0e4),
                    )
                    if self.training:
                        soft_log = membership.clamp_min(1e-12).log()
                        hard_log = hard_log + soft_log - soft_log.detach()
                    return hard_log
                earlier_log = membership_log(earlier_membership)
                later_log = membership_log(later_membership)
                ordered_pair = src_times[:, :, None] < src_times[:, None, :]
                pair_mask = (
                    direct_mask[:, None, :, None]
                    & direct_mask[:, None, None, :]
                    & ordered_pair[:, None]
                )
                previous_gap = (
                    src_times[:, None, :] - src_times[:, :, None]
                ).clamp_min(0)
                query_gap = (timestamp[:, None] - src_times).clamp_min(0)
                log_gap_error = (
                    torch.log1p(query_gap)[:, None, None, :]
                    - torch.log1p(previous_gap)[:, None, :, :]
                )
                centre = self.time_centres[self.renewal_rules, 0]
                scale = F.softplus(self.time_scale_raw[self.renewal_rules, 0]) + 0.05
                temporal = -0.5 * (
                    (log_gap_error - centre[None, :, None, None])
                    / scale[None, :, None, None]
                ).square()
                pair_values = earlier_log[:, :, :, None] + later_log[:, :, None, :] + temporal
                type_match = (
                    (earlier_membership.detach() > 0.5)[:, :, :, None]
                    & (later_membership.detach() > 0.5)[:, :, None, :]
                )
                valid_pair = pair_mask & type_match
                masked_pairs = pair_values.masked_fill(
                    ~valid_pair, torch.finfo(pair_values.dtype).min
                )
                renewal_presence = valid_pair.flatten(-2).any(-1)
                renewal_proof = torch.logsumexp(masked_pairs.flatten(-2), dim=-1)
                if self.grounding_aggregation == "mean":
                    renewal_count = valid_pair.flatten(-2).sum(-1).clamp_min(1)
                    renewal_proof = renewal_proof - renewal_count.log()
                renewal_proof = torch.where(
                    renewal_presence, renewal_proof,
                    torch.full_like(renewal_proof, _NO_PROOF),
                )
                best_pair = masked_pairs.flatten(-2).argmax(-1)
                history_width = src_times.shape[1]
                earlier_best = best_pair // history_width
                later_best = best_pair % history_width
                earlier_best = torch.where(
                    renewal_presence, earlier_best, torch.full_like(earlier_best, -1)
                )
                later_best = torch.where(
                    renewal_presence, later_best, torch.full_like(later_best, -1)
                )
            rule_proof[:, self.renewal_rules] = renewal_proof
            rule_presence[:, self.renewal_rules] = renewal_presence
            left_best[:, self.renewal_rules] = earlier_best
            right_best_by_rule[:, self.renewal_rules] = later_best
        if binary_ids.numel() > 0:
            right_masks = torch.stack(
                [role_masks[int(role)] for role in self.right_roles[binary]], dim=1
            )
            right_proof, right_presence, right_best = self._ground_atoms(
                dst_assign,
                dst_times,
                right_masks,
                timestamp,
                binary_ids,
                self.right_predicates[binary],
                1,
            )
            rule_proof[:, binary] = left_proof[:, binary] + right_proof
            rule_presence[:, binary] = left_presence[:, binary] & right_presence
            right_best_by_rule[:, binary] = right_best
        path_best_by_rule = torch.full_like(left_best, -1)
        ternary = self.middle_predicates >= 0
        if bool(ternary.any()):
            if path_fact_features is None or path_fact_times is None or path_mask is None:
                raise ValueError("max_rule_length=3 requires grounded three-hop paths")
            batch_size, path_count, slot_count, feature_dim = path_fact_features.shape
            if slot_count != 3 or feature_dim != self.edge_feat_dim:
                raise ValueError("invalid LiFTER three-hop fact tensor")
            strict_path_mask = path_mask.bool() & (
                path_fact_times < timestamp[:, None, None]
            ).all(-1)
            # The executor, rather than its upstream index, owns grounding
            # validity.  Enforce X->A <-B->Y directly on the concrete entity
            # arguments so counterfactual or external path tensors cannot be
            # treated as proofs merely because their mask is set.
            strict_path_mask &= (
                (path_fact_sources[:, :, 0] == src[:, None])
                & (path_fact_destinations[:, :, 0] == path_fact_destinations[:, :, 1])
                & (path_fact_sources[:, :, 1] == path_fact_sources[:, :, 2])
                & (path_fact_destinations[:, :, 2] == dst[:, None])
                & (path_fact_destinations[:, :, 0] != dst[:, None])
                & (path_fact_sources[:, :, 1] != src[:, None])
            )
            if self.enumerate_temporal_orders:
                sorted_times = path_fact_times.sort(dim=-1).values
                strict_path_mask &= (
                    sorted_times[:, :, 0] < sorted_times[:, :, 1]
                ) & (
                    sorted_times[:, :, 1] < sorted_times[:, :, 2]
                )
            elif self.enforce_three_hop_order:
                strict_path_mask &= (
                    (path_fact_times[:, :, 1] < path_fact_times[:, :, 2])
                    & (path_fact_times[:, :, 2] < path_fact_times[:, :, 0])
                )
            flat_features = path_fact_features.reshape(batch_size, path_count * 3, feature_dim)
            flat_valid = strict_path_mask.unsqueeze(-1).expand(-1, -1, 3).reshape(batch_size, -1)
            path_assignment, path_soft, _ = self._type_facts(
                flat_features,
                flat_valid,
                path_fact_sources.reshape(batch_size, path_count * 3),
                path_fact_destinations.reshape(batch_size, path_count * 3),
            )
            path_assignment = path_assignment.reshape(batch_size, path_count, 3, self.predicate_count)
            path_soft = path_soft.reshape(batch_size, path_count, 3, self.predicate_count)
            ternary_ids = all_rule_ids[ternary]
            if self.sparse_ternary_execution and self._hard_execution():
                ternary_proof, ternary_presence, ternary_best = (
                    self._sparse_ternary_proofs(
                        path_assignment, path_soft, path_fact_times, timestamp,
                        strict_path_mask, ternary_ids,
                    )
                )
            else:
                predicates = torch.stack(
                    (
                        self.left_predicates[ternary],
                        self.middle_predicates[ternary],
                        self.right_predicates[ternary],
                    ), dim=-1,
                )
                slot_values: list[torch.Tensor] = []
                for slot in range(3):
                    membership = path_assignment[:, :, slot, :].index_select(2, predicates[:, slot]).permute(0, 2, 1)
                    if self._hard_execution():
                        atom_log = torch.where(membership.detach() > 0.5, torch.zeros_like(membership), torch.full_like(membership, -1.0e4))
                        if self.training:
                            soft_log = membership.clamp_min(1e-12).log()
                            atom_log = atom_log + soft_log - soft_log.detach()
                    else:
                        atom_log = membership.clamp_min(1e-12).log()
                    log_age = torch.log1p((timestamp[:, None] - path_fact_times[:, :, slot]).clamp_min(0))
                    centre = self.time_centres[ternary, slot]
                    scale = F.softplus(self.time_scale_raw[ternary, slot]) + 0.05
                    temporal = -0.5 * ((log_age[:, None, :] - centre[None, :, None]) / scale[None, :, None]).square()
                    slot_values.append(atom_log + temporal)
                path_values = sum(slot_values) / 3.0
                rule_path_mask = strict_path_mask[:, None, :].expand(
                    -1, len(ternary_ids), -1
                )
                if self.enumerate_temporal_orders:
                    sorted_slots = path_fact_times.argsort(dim=-1)
                    order_table = torch.tensor(
                        TEMPORAL_ORDERS,
                        dtype=sorted_slots.dtype,
                        device=sorted_slots.device,
                    )
                    actual_order = (
                        sorted_slots[:, :, None, :]
                        == order_table[None, None, :, :]
                    ).all(-1).to(torch.long).argmax(-1)
                    required_order = torch.tensor(
                        [self.program[int(rule_id)].temporal_order for rule_id in ternary_ids],
                        dtype=torch.long,
                        device=path_values.device,
                    )
                    rule_path_mask = rule_path_mask & (
                        actual_order[:, None, :] == required_order[None, :, None]
                    )
                path_values = path_values.masked_fill(
                    ~rule_path_mask, torch.finfo(path_values.dtype).min
                )
                path_type_match = torch.ones(
                    (batch_size, len(ternary_ids), path_count),
                    dtype=torch.bool,
                    device=path_fact_features.device,
                )
                for slot in range(3):
                    selected = path_assignment[:, :, slot, :].index_select(2, predicates[:, slot]).permute(0, 2, 1)
                    path_type_match &= selected.detach() > (0.5 if self._hard_execution() else 0)
                ternary_presence = (rule_path_mask & path_type_match).any(-1)
                if self.path_existential_aggregation == "max":
                    ternary_proof = path_values.amax(dim=-1)
                else:
                    ternary_proof = torch.logsumexp(path_values, dim=-1)
                    if self.path_existential_aggregation == "mean":
                        proof_count = (
                            rule_path_mask & path_type_match
                        ).sum(-1).clamp_min(1)
                        ternary_proof = ternary_proof - proof_count.log()
                ternary_proof = torch.where(ternary_presence, ternary_proof, torch.full_like(ternary_proof, _NO_PROOF))
                ternary_best = path_values.masked_fill(
                    ~(rule_path_mask & path_type_match),
                    torch.finfo(path_values.dtype).min,
                ).argmax(-1)
                ternary_best = torch.where(ternary_presence, ternary_best, torch.full_like(ternary_best, -1))
            rule_proof[:, ternary] = ternary_proof
            rule_presence[:, ternary] = ternary_presence
            path_best_by_rule[:, ternary] = ternary_best

        # exp(log proof mass), softly capped for numerical stability. A missing
        # proof contributes exactly zero; signed weights support both evidence
        # and inhibition without an opaque neural score.
        evidence = torch.where(
            rule_presence,
            torch.exp(rule_proof.clamp(max=4.0)),
            torch.zeros_like(rule_proof),
        )
        if self.evidence_transform == "log1p":
            evidence = torch.log1p(evidence)
        effective_weight = self._effective_rule_weights()
        contributions = evidence * effective_weight[None]
        if not self.program_rules_enabled:
            contributions = torch.zeros_like(contributions)
        if self.rule_family_aggregation == "mean":
            lengths = torch.tensor(
                [rule.length for rule in self.program], device=src.device
            )
            family_masks = (
                (lengths == 1) & ((self.left_roles == 0) | (self.left_roles == 1)),
                (lengths == 1) & ((self.left_roles == 2) | (self.left_roles == 3)),
                (lengths == 1) & ((self.left_roles == 4) | (self.left_roles == 5)),
                self.renewal_rules,
                lengths == 3,
            )
            normalized = contributions.clone()
            for family_mask in family_masks:
                if not bool(family_mask.any()):
                    continue
                active = rule_presence[:, family_mask].sum(-1).clamp_min(1)
                normalized[:, family_mask] = (
                    contributions[:, family_mask] / active[:, None]
                )
            contributions = normalized
        if self.latent_transition_rule:
            outgoing_valid = src_valid & src_out
            outgoing_time = src_times.float().masked_fill(~outgoing_valid, -torch.inf)
            previous_index = outgoing_time.argmax(-1)
            previous_valid = outgoing_valid.any(-1)
            previous_destination = src_nodes.long().clamp(
                0, self.num_nodes - 1
            ).gather(
                1, previous_index[:, None]
            ).squeeze(1)
            previous_state = self.transition_source_embedding(previous_destination)
            candidate_state = self.transition_target_embedding(
                dst.long().clamp(0, self.num_nodes - 1)
            )
            previous_predicate = src_ids.gather(
                1, previous_index[:, None]
            ).squeeze(1)
            if self.predicate_conditioned_transitions:
                previous_state = previous_state * self.transition_predicate_factor(
                    previous_predicate
                )
            transition_contribution = (
                previous_state * candidate_state
            ).sum(-1) / math.sqrt(float(self.transition_dim))
            transition_contribution = transition_contribution * F.softplus(
                self.transition_scale_raw
            )
            transition_contribution = torch.where(
                previous_valid,
                transition_contribution,
                torch.zeros_like(transition_contribution),
            )
        else:
            previous_index = torch.zeros_like(dst.long())
            previous_destination = torch.full_like(dst.long(), -1)
            previous_predicate = torch.full_like(dst.long(), -1)
            previous_valid = torch.zeros_like(dst, dtype=torch.bool)
            transition_contribution = torch.zeros_like(
                timestamp, dtype=contributions.dtype
            )
        direct_recurrence = role_masks[0].any(-1)
        if self.positioned_recurrence_rules:
            history_width = src_nodes.shape[1]
            recency_positions = torch.arange(
                history_width, 0, -1, device=src.device
            )
            position_weights = self.positioned_recurrence_rule_weight.index_select(
                0, recency_positions - 1
            )
            positioned_recurrence_fact_contributions = (
                src_assign * position_weights[None]
            ).sum(-1)
            positioned_recurrence_mask = (
                src_valid
                & src_out
                & (src_nodes.long() == dst.long()[:, None])
            )
            positioned_recurrence_fact_contributions = torch.where(
                positioned_recurrence_mask,
                positioned_recurrence_fact_contributions,
                torch.zeros_like(positioned_recurrence_fact_contributions),
            )
            positioned_recurrence_contribution = (
                positioned_recurrence_fact_contributions.sum(-1)
            )
        else:
            positioned_recurrence_fact_contributions = torch.zeros_like(
                src_times, dtype=contributions.dtype
            )
            positioned_recurrence_contribution = torch.zeros_like(
                timestamp, dtype=contributions.dtype
            )
        if self.recurrence_guarded_transitions and self.latent_transition_rule:
            guard_index = direct_recurrence.long()
            guard_scale = F.softplus(self.transition_guard_scale_raw) / math.log(2.0)
            transition_contribution = transition_contribution * guard_scale[guard_index]
        if self.latent_second_order_transition_rule:
            outgoing_valid2 = src_valid & src_out
            outgoing_time2 = src_times.float().masked_fill(
                ~outgoing_valid2, -torch.inf
            )
            recent_times, recent_indices = outgoing_time2.topk(2, dim=-1)
            transition2_valid = torch.isfinite(recent_times).all(-1)
            # topk returns newest first; the rule body is rendered oldest first.
            transition2_first_destination = src_nodes.gather(
                1, recent_indices[:, 1:2]
            ).squeeze(1).long().clamp(0, self.num_nodes - 1)
            transition2_second_destination = src_nodes.gather(
                1, recent_indices[:, 0:1]
            ).squeeze(1).long().clamp(0, self.num_nodes - 1)
            transition2_first_state = self.transition2_first_embedding(
                transition2_first_destination
            )
            transition2_second_state = self.transition2_second_embedding(
                transition2_second_destination
            )
            transition2_target_state = self.transition2_target_embedding(
                dst.long().clamp(0, self.num_nodes - 1)
            )
            transition2_first_predicate = src_ids.gather(
                1, recent_indices[:, 1:2]
            ).squeeze(1)
            transition2_second_predicate = src_ids.gather(
                1, recent_indices[:, 0:1]
            ).squeeze(1)
            if self.predicate_conditioned_transitions:
                transition2_first_state = (
                    transition2_first_state
                    * self.transition2_first_predicate_factor(
                        transition2_first_predicate
                    )
                )
                transition2_second_state = (
                    transition2_second_state
                    * self.transition2_second_predicate_factor(
                        transition2_second_predicate
                    )
                )
            transition2_contribution = (
                transition2_first_state
                * transition2_second_state
                * transition2_target_state
            ).sum(-1) / math.sqrt(float(self.transition_dim))
            transition2_contribution = transition2_contribution * F.softplus(
                self.transition2_scale_raw
            )
            transition2_contribution = torch.where(
                transition2_valid,
                transition2_contribution,
                torch.zeros_like(transition2_contribution),
            )
        else:
            recent_indices = torch.zeros(
                len(dst), 2, device=dst.device, dtype=torch.long
            )
            transition2_first_destination = torch.full_like(dst.long(), -1)
            transition2_second_destination = torch.full_like(dst.long(), -1)
            transition2_first_predicate = torch.full_like(dst.long(), -1)
            transition2_second_predicate = torch.full_like(dst.long(), -1)
            transition2_valid = torch.zeros_like(dst, dtype=torch.bool)
            transition2_contribution = torch.zeros_like(
                timestamp, dtype=contributions.dtype
            )
        if self.recurrence_guarded_transitions and self.latent_second_order_transition_rule:
            guard_scale2 = F.softplus(self.transition2_guard_scale_raw) / math.log(2.0)
            transition2_contribution = transition2_contribution * guard_scale2[
                direct_recurrence.long()
            ]
        logits = (
            self.prior_rule_weight
            + positioned_recurrence_contribution
            + transition_contribution
            + transition2_contribution
            + contributions.sum(dim=-1)
        )
        top_rule = contributions.abs().argmax(dim=-1)
        rows = torch.arange(len(src), device=src.device)
        top_valid = rule_presence[rows, top_rule]
        top_left_index = left_best[rows, top_rule]
        binary_top = (self.right_roles[top_rule] >= 0) & (self.middle_predicates[top_rule] < 0)
        renewal_top = self.renewal_rules[top_rule]
        ternary_top = self.middle_predicates[top_rule] >= 0
        top_right_index = right_best_by_rule[rows, top_rule]

        # Grounded-fact responsibility is obtained only from executed rules:
        # each rule's absolute contribution is assigned to the fact indices in
        # its proof. This is an event ranking of the symbolic trace, not an
        # auxiliary neural attribution head.
        path_slot_count = (
            0 if path_fact_times is None else path_fact_times.shape[1] * 3
        )
        endpoint_fact_count = src_times.shape[1] + dst_times.shape[1]
        all_fact_importance = torch.zeros(
            len(src), endpoint_fact_count + path_slot_count,
            device=src.device, dtype=src_times.dtype,
        )
        responsibility = contributions.detach().abs()
        if not self.training:
            src_width = src_times.shape[1]
            first_valid = left_best >= 0
            first_offset = (self.left_roles >= 4).long() * src_width
            first_global = left_best + first_offset[None]
            all_fact_importance.scatter_add_(
                1, first_global.clamp_min(0), responsibility * first_valid
            )
            second_valid = right_best_by_rule >= 0
            second_offset = torch.where(
                self.renewal_rules,
                torch.zeros_like(self.right_roles),
                torch.full_like(self.right_roles, src_width),
            )
            second_global = right_best_by_rule + second_offset[None]
            all_fact_importance.scatter_add_(
                1, second_global.clamp_min(0), responsibility * second_valid
            )
            ternary_rules = self.middle_predicates >= 0
            if path_slot_count and bool(ternary_rules.any()):
                ternary_paths = path_best_by_rule[:, ternary_rules]
                ternary_responsibility = responsibility[:, ternary_rules]
                valid_paths = ternary_paths >= 0
                for slot in range(3):
                    global_slot = (
                        endpoint_fact_count
                        + ternary_paths * 3
                        + slot
                    )
                    all_fact_importance.scatter_add_(
                        1,
                        global_slot.clamp_min(0),
                        ternary_responsibility * valid_paths,
                    )
        regularization, metrics = self._regularization(
            src_soft,
            dst_soft,
            src_valid,
            dst_valid,
            effective_weight,
        )
        aux: dict[str, torch.Tensor] = {
            "regularization_loss": regularization,
            **metrics,
            "predicate_ids_src": src_ids.detach(),
            "predicate_ids_dst": dst_ids.detach(),
            "predicate_usage": (src_soft.sum((0, 1)) + dst_soft.sum((0, 1))).detach(),
            "rule_scores": contributions.detach(),
            "rule_grounding_valid": rule_presence.detach(),
            "top_grounding_valid": top_valid.detach(),
            "top_rule_id": top_rule.detach(),
            "top_fact1_bank": left_bank_is_dst[top_rule].long().detach(),
            "top_fact1_index": top_left_index.detach(),
            "top_fact2_bank": torch.where(
                renewal_top,
                torch.zeros_like(top_rule),
                torch.where(
                    binary_top,
                    torch.ones_like(top_rule),
                    torch.full_like(top_rule, -1),
                ),
            ).detach(),
            "top_fact2_index": top_right_index.detach(),
            "top_path_index": path_best_by_rule[rows, top_rule].detach(),
            "top_rule_length": torch.tensor(
                [self.program[int(rule_id)].length for rule_id in top_rule], device=src.device
            ).detach(),
            "fact1_index_by_rule": left_best.detach(),
            "fact2_index_by_rule": right_best_by_rule.detach(),
            "path_index_by_rule": path_best_by_rule.detach(),
            "top_grounding_score": contributions[rows, top_rule].detach(),
            "prior_rule_contribution": self.prior_rule_weight.detach().expand_as(logits),
            "positioned_recurrence_rule_contribution": positioned_recurrence_contribution.detach(),
            "positioned_recurrence_fact_contributions": positioned_recurrence_fact_contributions.detach(),
            "transition_rule_contribution": transition_contribution.detach(),
            "transition_fact_index": previous_index.detach(),
            "transition_previous_destination": previous_destination.detach(),
            "transition_previous_predicate": previous_predicate.detach(),
            "transition_grounding_valid": previous_valid.detach(),
            "transition_direct_recurrence_guard": direct_recurrence.detach(),
            "transition2_rule_contribution": transition2_contribution.detach(),
            "transition2_first_fact_index": recent_indices[:, 1].detach(),
            "transition2_second_fact_index": recent_indices[:, 0].detach(),
            "transition2_first_destination": transition2_first_destination.detach(),
            "transition2_second_destination": transition2_second_destination.detach(),
            "transition2_first_predicate": transition2_first_predicate.detach(),
            "transition2_second_predicate": transition2_second_predicate.detach(),
            "transition2_grounding_valid": transition2_valid.detach(),
            "hard_execution": torch.tensor(float(self._hard_execution()), device=src.device),
            "predicate_temperature": torch.tensor(self._temperature, device=src.device),
            "event_importance": all_fact_importance.detach(),
        }
        return ModelResult(logits=logits, aux=aux)

    def score_candidate_batches(
        self, *batches: tuple[torch.Tensor, ...]
    ) -> tuple[ModelResult, ...]:
        """Score candidates sharing query rows with one source-fact encoding."""
        if not batches:
            return ()
        width = len(batches[0][0])
        if any(len(batch[0]) != width for batch in batches):
            raise ValueError("candidate batches must have the same row count")
        combined = tuple(
            torch.cat([batch[position] for batch in batches], dim=0)
            for position in range(len(batches[0]))
        )
        self._shared_source_width = width
        try:
            result = self(*combined)
        finally:
            self._shared_source_width = 0
        outputs = []
        total = width * len(batches)
        for index in range(len(batches)):
            start, stop = index * width, (index + 1) * width
            aux = {
                key: (
                    value[start:stop]
                    if torch.is_tensor(value) and value.ndim and len(value) == total
                    else value
                )
                for key, value in result.aux.items()
            }
            outputs.append(ModelResult(result.logits[start:stop], aux))
        return tuple(outputs)

    def export_symbolic_rules(self) -> list[dict[str, Any]]:
        weights = self._effective_rule_weights().detach().cpu()
        centres = self.time_centres.detach().cpu()
        scales = (F.softplus(self.time_scale_raw) + 0.05).detach().cpu()
        exported: list[dict[str, Any]] = [
            {
                "rule_id": -1,
                "length": 0,
                "body": "True",
                "head": "Link(X,Y,Tq)",
                "weight": float(self.prior_rule_weight.detach().cpu()),
            }
        ]
        ranked_rule_ids = torch.argsort(weights.abs(), descending=True)[
            : min(self.export_rule_limit, self.rule_count)
        ].tolist()
        for index in ranked_rule_ids:
            rule = self.program[index]
            atoms = _render_rule_atoms(rule)
            exported.append(
                {
                    "rule_id": index,
                    "temporal_component": int(self.temporal_components[index]),
                    "length": rule.length,
                    "body": " & ".join(atoms),
                    "temporal_condition": _render_temporal_condition(rule),
                    "head": "Link(X,Y,Tq)",
                    "log_age_centres": centres[index, : rule.length].tolist(),
                    "log_age_scales": scales[index, : rule.length].tolist(),
                    "weight": float(weights[index]),
                }
            )
        return exported

    @torch.no_grad()
    def explain(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> list[dict[str, Any]]:
        was_training = self.training
        self.eval()
        with self.execution_mode("hard"):
            result = self.forward(*args, **kwargs)
        if was_training:
            self.train()
        src, dst, timestamp = args[:3]
        src_nodes, src_times, src_features = args[3:6]
        src_valid = args[6].bool()
        dst_nodes = kwargs.get("dst_history_nodes", args[7] if len(args) > 7 else None)
        dst_times = kwargs.get("dst_history_times", args[8] if len(args) > 8 else None)
        dst_features = kwargs.get("dst_history_edge_feats", args[9] if len(args) > 9 else None)
        dst_valid = kwargs.get(
            "dst_history_mask", args[10] if len(args) > 10 else None
        ).bool()
        path_sources = kwargs.get("path_fact_sources", args[11] if len(args) > 11 else None)
        path_destinations = kwargs.get("path_fact_destinations", args[12] if len(args) > 12 else None)
        path_times = kwargs.get("path_fact_times", args[13] if len(args) > 13 else None)
        width = min(src_nodes.shape[1], self.max_grounding_facts)
        src_nodes, src_times, src_features = src_nodes[:, -width:], src_times[:, -width:], src_features[:, -width:]
        dst_nodes, dst_times, dst_features = dst_nodes[:, -width:], dst_times[:, -width:], dst_features[:, -width:]
        src_valid, dst_valid = src_valid[:, -width:], dst_valid[:, -width:]
        rules = {
            int(rule["rule_id"]): rule for rule in self.export_symbolic_rules()
        }
        explanations: list[dict[str, Any]] = []
        for row in range(len(src)):
            rule_id = int(result.aux["top_rule_id"][row])
            rule = self.program[rule_id]
            if rule_id not in rules:
                atoms = _render_rule_atoms(rule)
                rules[rule_id] = {
                    "rule_id": rule_id,
                    "temporal_component": int(self.temporal_components[rule_id]),
                    "length": rule.length,
                    "body": " & ".join(atoms),
                    "temporal_condition": _render_temporal_condition(rule),
                    "head": "Link(X,Y,Tq)",
                    "weight": float(
                        self._effective_rule_weights()[rule_id].detach()
                    ),
                }
            facts: list[dict[str, Any]] = []
            if rule.length == 3:
                path_index = int(result.aux["top_path_index"][row])
                if path_index >= 0:
                    predicates = (rule.left_predicate, rule.middle_predicate, rule.right_predicate)
                    for slot, predicate in enumerate(predicates):
                        facts.append({
                            "predicate": f"P{predicate}",
                            "grounded_link": {
                                "source": int(path_sources[row, path_index, slot]),
                                "destination": int(path_destinations[row, path_index, slot]),
                                "timestamp": float(path_times[row, path_index, slot]),
                            },
                        })
            for slot, (bank, index) in enumerate(
                (
                    (int(result.aux["top_fact1_bank"][row]), int(result.aux["top_fact1_index"][row])),
                    (int(result.aux["top_fact2_bank"][row]), int(result.aux["top_fact2_index"][row])),
                )
            ):
                if rule.length == 3:
                    break
                if bank < 0 or index < 0:
                    continue
                centre = int(src[row] if bank == 0 else dst[row])
                nodes = src_nodes if bank == 0 else dst_nodes
                times = src_times if bank == 0 else dst_times
                features = src_features if bank == 0 else dst_features
                neighbour = int(nodes[row, index])
                direction = float(features[row, index, 0])
                fact_source, fact_destination = (
                    (centre, neighbour) if direction >= 0 else (neighbour, centre)
                )
                predicate = (
                    rule.left_predicate if slot == 0 else rule.right_predicate
                )
                facts.append(
                    {
                        "predicate": f"P{predicate}",
                        "grounded_link": {
                            "source": fact_source,
                            "destination": fact_destination,
                            "timestamp": float(times[row, index]),
                        },
                    }
                )
            explanations.append(
                {
                    "query": {
                        "predicate": "Link",
                        "source": int(src[row]),
                        "destination": int(dst[row]),
                        "timestamp": float(timestamp[row]),
                    },
                    "grounding_valid": bool(result.aux["top_grounding_valid"][row]),
                    "rule": rules[rule_id],
                    "grounded_facts": facts,
                    "rule_contribution": float(result.aux["top_grounding_score"][row]),
                    "prior_rule_contribution": float(self.prior_rule_weight),
                    "transition_rule": {
                        "body": "Link(X,Z,Tlast) & LatentTransition(Z,Y)",
                        "previous_destination": int(
                            result.aux["transition_previous_destination"][row]
                        ),
                        "grounding_valid": bool(
                            result.aux["transition_grounding_valid"][row]
                        ),
                        "contribution": float(
                            result.aux["transition_rule_contribution"][row]
                        ),
                    },
                    "second_order_transition_rule": {
                        "body": (
                            "Link(X,Z1,Tprev) & Link(X,Z2,Tlast) & "
                            "LatentTransition2(Z1,Z2,Y)"
                        ),
                        "first_destination": int(
                            result.aux["transition2_first_destination"][row]
                        ),
                        "second_destination": int(
                            result.aux["transition2_second_destination"][row]
                        ),
                        "grounding_valid": bool(
                            result.aux["transition2_grounding_valid"][row]
                        ),
                        "contribution": float(
                            result.aux["transition2_rule_contribution"][row]
                        ),
                    },
                    "program_trace": self._decode_program_trace(
                        row,
                        src,
                        dst,
                        src_nodes,
                        src_times,
                        src_features,
                        dst_nodes,
                        dst_times,
                        dst_features,
                        path_sources,
                        path_destinations,
                        path_times,
                        result,
                    ),
                    "candidate_logit": float(result.logits[row]),
                }
            )
            trace_sum = sum(
                execution["contribution"]
                for execution in explanations[-1]["program_trace"]
            )
            explanations[-1]["reconstructed_logit"] = (
                float(self.prior_rule_weight)
                + trace_sum
            )
            explanations[-1]["decomposition_error"] = (
                float(result.logits[row])
                - explanations[-1]["reconstructed_logit"]
            )
        return explanations

    def _decode_program_trace(
        self,
        row: int,
        src: torch.Tensor,
        dst: torch.Tensor,
        src_nodes: torch.Tensor,
        src_times: torch.Tensor,
        src_features: torch.Tensor,
        dst_nodes: torch.Tensor,
        dst_times: torch.Tensor,
        dst_features: torch.Tensor,
        path_sources: torch.Tensor | None,
        path_destinations: torch.Tensor | None,
        path_times: torch.Tensor | None,
        result: ModelResult,
    ) -> list[dict[str, Any]]:
        """Decode every nonzero execution whose sum exactly reconstructs the logit."""

        trace: list[dict[str, Any]] = []
        contributions = result.aux["rule_scores"][row]
        rule_ids = torch.nonzero(contributions != 0, as_tuple=False).flatten().tolist()
        for rule_id in rule_ids:
            rule = self.program[rule_id]
            atoms = _render_rule_atoms(rule)
            grounded_facts: list[dict[str, Any]] = []
            if rule.length == 3:
                path_index = int(result.aux["path_index_by_rule"][row, rule_id])
                if path_index >= 0 and path_sources is not None:
                    for slot, predicate in enumerate(
                        (rule.left_predicate, rule.middle_predicate, rule.right_predicate)
                    ):
                        grounded_facts.append({
                            "predicate": f"P{predicate}",
                            "grounded_link": {
                                "source": int(path_sources[row, path_index, slot]),
                                "destination": int(path_destinations[row, path_index, slot]),
                                "timestamp": float(path_times[row, path_index, slot]),
                            },
                        })
            indices = (
                int(result.aux["fact1_index_by_rule"][row, rule_id]),
                int(result.aux["fact2_index_by_rule"][row, rule_id]),
            )
            banks = (
                int(self.left_roles[rule_id] >= 4),
                0 if rule.renewal else 1,
            )
            predicates = (rule.left_predicate, rule.right_predicate)
            for slot in range(rule.length):
                if rule.length == 3:
                    break
                index = indices[slot]
                bank = banks[slot]
                if index < 0:
                    continue
                centre = int(src[row] if bank == 0 else dst[row])
                nodes = src_nodes if bank == 0 else dst_nodes
                times = src_times if bank == 0 else dst_times
                features = src_features if bank == 0 else dst_features
                neighbour = int(nodes[row, index])
                direction = float(features[row, index, 0])
                fact_source, fact_destination = (
                    (centre, neighbour) if direction >= 0 else (neighbour, centre)
                )
                grounded_facts.append(
                    {
                        "predicate": f"P{predicates[slot]}",
                        "grounded_link": {
                            "source": fact_source,
                            "destination": fact_destination,
                            "timestamp": float(times[row, index]),
                        },
                    }
                )
            trace.append(
                {
                    "rule_id": rule_id,
                    "rule_kind": "program_rule",
                    "left_role": int(rule.left_role),
                    "right_role": int(rule.right_role),
                    "renewal": bool(rule.renewal),
                    "body": " & ".join(atoms),
                    "head": "Link(X,Y,Tq)",
                    "grounded_facts": grounded_facts,
                    "execution_evidence": (
                        float(contributions[rule_id] / self._effective_rule_weights()[rule_id])
                        if float(self._effective_rule_weights()[rule_id]) != 0.0 else 0.0
                    ),
                    "execution_weight": float(self._effective_rule_weights()[rule_id]),
                    "contribution": float(contributions[rule_id]),
                }
            )
        guard_atom = (
            "DirectRecurrence(X,Y,T<Tq)"
            if bool(result.aux["transition_direct_recurrence_guard"][row])
            else "NoDirectRecurrence(X,Y,T<Tq)"
        )
        transition_contribution = float(
            result.aux["transition_rule_contribution"][row]
        )
        if transition_contribution != 0.0:
            fact_index = int(result.aux["transition_fact_index"][row])
            predicate = int(result.aux["transition_previous_predicate"][row])
            q1_source = self.transition_source_embedding(
                result.aux["transition_previous_destination"][row].long()
            )
            if self.predicate_conditioned_transitions:
                q1_source = q1_source * self.transition_predicate_factor(
                    result.aux["transition_previous_predicate"][row].long()
                )
            q1_target = self.transition_target_embedding(dst[row].long())
            trace.append(
                {
                    "rule_id": "Q1",
                    "rule_kind": "one_event_transition",
                    "body": f"P{predicate}(X,Z,Tlast) & StreamOrder(flast,q) & {guard_atom}",
                    "consequent_potential": f"psi1_{predicate}(Z,Y)",
                    "head": "Link(X,Y,Tq)",
                    "grounded_facts": [
                        {
                            "predicate": f"P{predicate}",
                            "grounded_link": {
                                "source": int(src[row]),
                                "destination": int(src_nodes[row, fact_index]),
                                "timestamp": float(src_times[row, fact_index]),
                            },
                            "source_history_position": int(
                                src_nodes.shape[1] - fact_index
                            ),
                        }
                    ],
                    "execution_operands": {
                        "source": q1_source.detach().cpu().tolist(),
                        "target": q1_target.detach().cpu().tolist(),
                        "scale": float(F.softplus(self.transition_scale_raw).detach()),
                        "guard_scale": (
                            float((F.softplus(self.transition_guard_scale_raw) / math.log(2.0))[int(result.aux["transition_direct_recurrence_guard"][row])].detach())
                            if self.recurrence_guarded_transitions else 1.0
                        ),
                    },
                    "contribution": transition_contribution,
                }
            )
        transition2_contribution = float(
            result.aux["transition2_rule_contribution"][row]
        )
        if transition2_contribution != 0.0:
            first_index = int(result.aux["transition2_first_fact_index"][row])
            second_index = int(result.aux["transition2_second_fact_index"][row])
            first_predicate = int(
                result.aux["transition2_first_predicate"][row]
            )
            second_predicate = int(
                result.aux["transition2_second_predicate"][row]
            )
            q2_first = self.transition2_first_embedding(
                result.aux["transition2_first_destination"][row].long()
            )
            q2_second = self.transition2_second_embedding(
                result.aux["transition2_second_destination"][row].long()
            )
            if self.predicate_conditioned_transitions:
                q2_first = q2_first * self.transition2_first_predicate_factor(
                    result.aux["transition2_first_predicate"][row].long()
                )
                q2_second = q2_second * self.transition2_second_predicate_factor(
                    result.aux["transition2_second_predicate"][row].long()
                )
            q2_target = self.transition2_target_embedding(dst[row].long())
            trace.append(
                {
                    "rule_id": "Q2",
                    "rule_kind": "two_event_transition",
                    "body": (
                        f"P{first_predicate}(X,Z1,Tprev) & "
                        f"P{second_predicate}(X,Z2,Tlast) & "
                        f"StreamOrder(fprev,flast,q) & {guard_atom}"
                    ),
                    "consequent_potential": (
                        f"psi2_{first_predicate}_{second_predicate}(Z1,Z2,Y)"
                    ),
                    "head": "Link(X,Y,Tq)",
                    "grounded_facts": [
                        {
                            "predicate": f"P{predicate}",
                            "grounded_link": {
                                "source": int(src[row]),
                                "destination": int(src_nodes[row, fact_index]),
                                "timestamp": float(src_times[row, fact_index]),
                            },
                            "source_history_position": int(
                                src_nodes.shape[1] - fact_index
                            ),
                        }
                        for predicate, fact_index in (
                            (first_predicate, first_index),
                            (second_predicate, second_index),
                        )
                    ],
                    "execution_operands": {
                        "first": q2_first.detach().cpu().tolist(),
                        "second": q2_second.detach().cpu().tolist(),
                        "target": q2_target.detach().cpu().tolist(),
                        "scale": float(F.softplus(self.transition2_scale_raw).detach()),
                        "guard_scale": (
                            float((F.softplus(self.transition2_guard_scale_raw) / math.log(2.0))[int(result.aux["transition_direct_recurrence_guard"][row])].detach())
                            if self.recurrence_guarded_transitions else 1.0
                        ),
                    },
                    "contribution": transition2_contribution,
                }
            )
        position_contributions = result.aux[
            "positioned_recurrence_fact_contributions"
        ][row]
        for fact_index in torch.nonzero(
            position_contributions != 0, as_tuple=False
        ).flatten().tolist():
            predicate = int(result.aux["predicate_ids_src"][row, fact_index])
            position = src_nodes.shape[1] - fact_index
            trace.append(
                {
                    "rule_id": f"positioned_recurrence_{position}_P{predicate}",
                    "rule_kind": "positioned_recurrence",
                    "body": (
                        f"P{predicate}(X,Y,T) & "
                        f"HistoryPosition_X(T)={position}"
                    ),
                    "head": "Link(X,Y,Tq)",
                    "grounded_facts": [
                        {
                            "predicate": f"P{predicate}",
                            "grounded_link": {
                                "source": int(src[row]),
                                "destination": int(src_nodes[row, fact_index]),
                                "timestamp": float(src_times[row, fact_index]),
                            },
                            "source_history_position": position,
                        }
                    ],
                    "execution_evidence": 1.0,
                    "execution_weight": float(
                        self.positioned_recurrence_rule_weight[position - 1, predicate]
                    ),
                    "contribution": float(position_contributions[fact_index]),
                }
            )
        trace.sort(key=lambda execution: abs(execution["contribution"]), reverse=True)
        return trace


__all__ = [
    "DEFAULT_FACT_CONTEXT_LEN",
    "LIFTER_CONTEXT_TOKEN_DIM",
    "LIFTER_EDGE_FEATURE_DIM",
    "LiFTER",
    "lifter_edge_feature_dim",
]
