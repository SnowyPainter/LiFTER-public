"""Dependency-light verifier for LiFTER score-computation certificates."""
from __future__ import annotations

import math
from typing import Any


def _close(left: float, right: float, tolerance: float) -> bool:
    return math.isfinite(left) and math.isfinite(right) and abs(left - right) <= tolerance


def _vector_close(left: list[float], right: list[float], tolerance: float) -> bool:
    return len(left) == len(right) and all(
        _close(float(a), float(b), tolerance) for a, b in zip(left, right)
    )


def _predicate_id(item: dict[str, Any]) -> int:
    value = str(item.get("predicate", "P0"))
    return int(value[1:]) if value.startswith("P") else int(value)


def _execution_key(execution: dict[str, Any]) -> tuple[Any, ...]:
    facts = tuple(_fact_tuple(item) for item in execution.get("grounded_facts", []))
    return execution.get("rule_kind"), str(execution.get("rule_id")), facts


def _fact_tuple(item: dict[str, Any]) -> tuple[int, int, float]:
    fact = item["grounded_link"]
    return int(fact["source"]), int(fact["destination"]), float(fact["timestamp"])


def _in_history(fact: tuple[int, int, float], history: list[dict[str, Any]], tolerance: float) -> bool:
    return any(
        fact[0] == int(candidate["source"])
        and fact[1] == int(candidate["destination"])
        and _close(fact[2], float(candidate["timestamp"]), tolerance)
        for candidate in history
    )


def _binding_valid(execution: dict[str, Any], query: dict[str, Any]) -> bool:
    facts = [_fact_tuple(item) for item in execution.get("grounded_facts", [])]
    x, y = int(query["source"]), int(query["destination"])
    kind = execution.get("rule_kind")
    if kind in {"one_event_transition", "two_event_transition"}:
        return all(source == x for source, _, _ in facts)
    if kind == "positioned_recurrence":
        return len(facts) == 1 and facts[0][:2] == (x, y)
    if kind != "program_rule":
        return False
    roles = (
        [0, 0]
        if bool(execution.get("renewal", False))
        else [execution.get("left_role"), execution.get("right_role")]
    )
    expected = {
        0: lambda s, d: s == x and d == y,
        1: lambda s, d: s == y and d == x,
        2: lambda s, d: s == x,
        3: lambda s, d: d == x,
        4: lambda s, d: s == y,
        5: lambda s, d: d == y,
    }
    return len(facts) == len([role for role in roles if role is not None and int(role) >= 0]) and all(
        int(role) in expected and expected[int(role)](facts[index][0], facts[index][1])
        for index, role in enumerate(roles[:len(facts)])
    )


def _transition_selection_valid(
    execution: dict[str, Any],
    query: dict[str, Any],
    source_history: list[dict[str, Any]],
    tolerance: float,
) -> bool:
    """Check that Q1/Q2 cite the latest one/two outgoing pre-query facts."""

    kind = execution.get("rule_kind")
    if kind not in {"one_event_transition", "two_event_transition"}:
        return True
    x, tq = int(query["source"]), float(query["timestamp"])
    outgoing = [
        (int(item["source"]), int(item["destination"]), float(item["timestamp"]))
        for item in source_history
        if int(item["source"]) == x and float(item["timestamp"]) < tq
    ]
    required = 1 if kind == "one_event_transition" else 2
    if len(outgoing) < required:
        return False
    actual = [_fact_tuple(item) for item in execution.get("grounded_facts", [])]
    if len(actual) != required or len(set(actual)) != required:
        return False
    cutoff = sorted(fact[2] for fact in outgoing)[-required]
    # Equal timestamps are simultaneous in the data; the executor may choose
    # any fact at the top-k boundary. Provenance is checked separately.
    if not all(fact[2] + tolerance >= cutoff for fact in actual):
        return False
    if required == 2 and actual[0][2] > actual[1][2] + tolerance:
        return False
    maximum = max(fact[2] for fact in outgoing)
    return _close(actual[-1][2], maximum, tolerance)


def _recompute(execution: dict[str, Any]) -> float:
    kind = execution.get("rule_kind")
    if kind in {"program_rule", "positioned_recurrence"}:
        return float(execution["execution_evidence"]) * float(execution["execution_weight"])
    operands = execution["execution_operands"]
    if kind == "one_event_transition":
        source, target = operands["source"], operands["target"]
        return sum(float(a) * float(b) for a, b in zip(source, target)) / math.sqrt(len(source)) * float(operands["scale"]) * float(operands["guard_scale"])
    if kind == "two_event_transition":
        first, second, target = operands["first"], operands["second"], operands["target"]
        return sum(float(a) * float(b) * float(c) for a, b, c in zip(first, second, target)) / math.sqrt(len(first)) * float(operands["scale"]) * float(operands["guard_scale"])
    raise ValueError(f"unknown execution kind {kind!r}")


def _role_matches(role: int, fact: tuple[int, int, float], query: dict[str, Any]) -> bool:
    source, destination, _ = fact
    x, y = int(query["source"]), int(query["destination"])
    return {
        0: source == x and destination == y,
        1: source == y and destination == x,
        2: source == x,
        3: destination == x,
        4: source == y,
        5: destination == y,
    }.get(int(role), False)


def _program_evidence(
    execution: dict[str, Any],
    certificate: dict[str, Any],
    trusted_parameters: dict[str, Any],
) -> float:
    """Re-ground a K=1 unary/renewal clause from the raw fact bank."""

    if int(trusted_parameters["predicate_count"]) != 1:
        raise ValueError("independent program-evidence replay currently requires K=1")
    query = certificate["query"]
    tq = float(query["timestamp"])
    rule_id = int(execution["rule_id"])
    centre = float(trusted_parameters["time_centres"][rule_id][0])
    scale = float(trusted_parameters["time_scales"][rule_id][0])
    role = int(execution.get("left_role", 0))
    history = (
        certificate.get("source_history_facts", certificate["historical_facts"])
        if bool(execution.get("renewal", False)) or role < 4
        else certificate.get("destination_history_facts", certificate["historical_facts"])
    )
    facts = [
        (int(item["source"]), int(item["destination"]), float(item["timestamp"]))
        for item in history
        if float(item["timestamp"]) < tq
    ]
    values: list[float] = []
    if bool(execution.get("renewal", False)):
        direct = [fact for fact in facts if _role_matches(0, fact, query)]
        for earlier in direct:
            for later in direct:
                if earlier[2] >= later[2]:
                    continue
                error = math.log1p(tq - later[2]) - math.log1p(later[2] - earlier[2])
                values.append(math.exp(-0.5 * ((error - centre) / scale) ** 2))
    else:
        for fact in facts:
            if not _role_matches(role, fact, query):
                continue
            log_age = math.log1p(tq - fact[2])
            values.append(math.exp(-0.5 * ((log_age - centre) / scale) ** 2))
        if (
            values
            and trusted_parameters["idempotent_direct_recurrence"]
            and role in {0, 1}
        ):
            values = [max(values)]
    if not values:
        return 0.0
    evidence = sum(values)
    if trusted_parameters["grounding_aggregation"] == "mean":
        evidence /= len(values)
    evidence = min(math.exp(4.0), evidence)
    if trusted_parameters["evidence_transform"] == "log1p":
        evidence = math.log1p(evidence)
    return evidence


def verify_certificate(
    certificate: dict[str, Any],
    tolerance: float = 2e-5,
    trusted_parameters: dict[str, Any] | None = None,
    trusted_candidate_logit: float | None = None,
) -> dict[str, Any]:
    query = certificate["query"]
    history = certificate["historical_facts"]
    failures: list[str] = []
    if trusted_candidate_logit is not None and not _close(
        float(certificate["candidate_logit"]),
        float(trusted_candidate_logit),
        tolerance,
    ):
        failures.append("candidate_logit.commitment")
    if trusted_parameters is not None and not _close(
        float(certificate["prior_rule_contribution"]),
        float(trusted_parameters["prior"]),
        tolerance,
    ):
        failures.append("prior.checkpoint")
    recomputed = float(certificate["prior_rule_contribution"])
    seen: set[tuple[Any, ...]] = set()
    for index, execution in enumerate(certificate["program_trace"]):
        facts = [_fact_tuple(item) for item in execution.get("grounded_facts", [])]
        if not all(_in_history(fact, history, tolerance) for fact in facts):
            failures.append(f"execution[{index}].provenance")
        if not all(fact[2] < float(query["timestamp"]) for fact in facts):
            failures.append(f"execution[{index}].temporal")
        if not _binding_valid(execution, query):
            failures.append(f"execution[{index}].binding")
        if not _transition_selection_valid(
            execution,
            query,
            certificate.get("source_history_facts", history),
            tolerance,
        ):
            failures.append(f"execution[{index}].selection")
        key = _execution_key(execution)
        if key in seen:
            failures.append(f"execution[{index}].duplicate")
        seen.add(key)
        if execution.get("rule_kind") == "two_event_transition":
            grounded = execution.get("grounded_facts", [])
            ordered = len(facts) == 2 and (
                facts[0][2] < facts[1][2]
                or (
                    _close(facts[0][2], facts[1][2], tolerance)
                    and int(grounded[0].get("source_history_position", -1))
                    > int(grounded[1].get("source_history_position", -1))
                )
            )
            if not ordered:
                failures.append(f"execution[{index}].order")
        if trusted_parameters is not None:
            kind = execution.get("rule_kind")
            if kind == "program_rule":
                rule_id = int(execution["rule_id"])
                expected = trusted_parameters["program_weights"][rule_id]
                if not _close(float(execution["execution_weight"]), expected, tolerance):
                    failures.append(f"execution[{index}].checkpoint_weight")
            elif kind == "positioned_recurrence":
                item = execution["grounded_facts"][0]
                position = int(item["source_history_position"])
                predicate = _predicate_id(item)
                expected = trusted_parameters["positioned_weights"][position - 1][predicate]
                if not _close(float(execution["execution_weight"]), expected, tolerance):
                    failures.append(f"execution[{index}].checkpoint_weight")
            elif kind in {"one_event_transition", "two_event_transition"}:
                operands = execution["execution_operands"]
                candidate = int(query["destination"])
                direct = any(
                    int(item["source"]) == int(query["source"])
                    and int(item["destination"]) == candidate
                    for item in certificate.get("source_history_facts", history)
                )
                if kind == "one_event_transition":
                    item = execution["grounded_facts"][0]
                    previous = int(item["grounded_link"]["destination"])
                    predicate = _predicate_id(item)
                    source = list(trusted_parameters["transition_source"][previous])
                    factor = trusted_parameters.get("transition_predicate_factor")
                    if factor is not None:
                        source = [a * b for a, b in zip(source, factor[predicate])]
                    expected_vectors = {
                        "source": source,
                        "target": trusted_parameters["transition_target"][candidate],
                    }
                    scale = trusted_parameters["transition_scale"]
                    guards = trusted_parameters.get("transition_guard")
                else:
                    first_item, second_item = execution["grounded_facts"]
                    first = int(first_item["grounded_link"]["destination"])
                    second = int(second_item["grounded_link"]["destination"])
                    first_vector = list(trusted_parameters["transition2_first"][first])
                    second_vector = list(trusted_parameters["transition2_second"][second])
                    first_factor = trusted_parameters.get("transition2_first_predicate_factor")
                    second_factor = trusted_parameters.get("transition2_second_predicate_factor")
                    if first_factor is not None:
                        first_vector = [a * b for a, b in zip(first_vector, first_factor[_predicate_id(first_item)])]
                    if second_factor is not None:
                        second_vector = [a * b for a, b in zip(second_vector, second_factor[_predicate_id(second_item)])]
                    expected_vectors = {
                        "first": first_vector,
                        "second": second_vector,
                        "target": trusted_parameters["transition2_target"][candidate],
                    }
                    scale = trusted_parameters["transition2_scale"]
                    guards = trusted_parameters.get("transition2_guard")
                for name, expected in expected_vectors.items():
                    if not _vector_close(operands[name], expected, tolerance):
                        failures.append(f"execution[{index}].checkpoint_operand")
                expected_guard = guards[int(direct)] if guards is not None else 1.0
                if not _close(float(operands["scale"]), float(scale), tolerance):
                    failures.append(f"execution[{index}].checkpoint_scale")
                if not _close(float(operands["guard_scale"]), float(expected_guard), tolerance):
                    failures.append(f"execution[{index}].checkpoint_guard")
        try:
            if trusted_parameters is not None and execution.get("rule_kind") == "program_rule":
                rule_id = int(execution["rule_id"])
                evidence = _program_evidence(execution, certificate, trusted_parameters)
                if not _close(
                    evidence,
                    float(execution["execution_evidence"]),
                    max(tolerance, 1e-4),
                ):
                    failures.append(f"execution[{index}].evidence")
                contribution = evidence * float(trusted_parameters["program_weights"][rule_id])
            elif trusted_parameters is not None and execution.get("rule_kind") == "positioned_recurrence":
                item = execution["grounded_facts"][0]
                position = int(item["source_history_position"])
                predicate = _predicate_id(item)
                contribution = float(trusted_parameters["positioned_weights"][position - 1][predicate])
            else:
                contribution = _recompute(execution)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            failures.append(f"execution[{index}].recompute")
            continue
        if not _close(contribution, float(execution["contribution"]), tolerance):
            failures.append(f"execution[{index}].contribution")
        recomputed += contribution
    if not _close(recomputed, float(certificate["candidate_logit"]), tolerance):
        failures.append("candidate_logit")
    return {"valid": not failures, "failures": failures, "recomputed_logit": recomputed}
