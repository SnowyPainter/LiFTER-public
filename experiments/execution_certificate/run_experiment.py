#!/opt/conda/bin/python
from __future__ import annotations

import copy
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.evaluate import LiFTERHistoryIndex, RunSpec, build_benchmark_model, build_lifter_causal_fact_context, read_csv_tail, split_indices
from verifier import _recompute, verify_certificate


def _trusted_parameters(model) -> dict:
    tolist = lambda value: value.detach().cpu().tolist()
    trusted = {
        "prior": float(model.prior_rule_weight.detach()),
        "program_weights": tolist(model._effective_rule_weights()),
        "time_centres": tolist(model.time_centres),
        "time_scales": tolist(
            torch.nn.functional.softplus(model.time_scale_raw) + 0.05
        ),
        "predicate_count": int(model.predicate_count),
        "grounding_aggregation": model.grounding_aggregation,
        "evidence_transform": model.evidence_transform,
        "idempotent_direct_recurrence": bool(model.idempotent_direct_recurrence),
        "positioned_weights": tolist(model.positioned_recurrence_rule_weight),
        "transition_source": tolist(model.transition_source_embedding.weight),
        "transition_target": tolist(model.transition_target_embedding.weight),
        "transition_scale": float(torch.nn.functional.softplus(model.transition_scale_raw).detach()),
        "transition2_first": tolist(model.transition2_first_embedding.weight),
        "transition2_second": tolist(model.transition2_second_embedding.weight),
        "transition2_target": tolist(model.transition2_target_embedding.weight),
        "transition2_scale": float(torch.nn.functional.softplus(model.transition2_scale_raw).detach()),
    }
    if model.recurrence_guarded_transitions:
        trusted["transition_guard"] = tolist(
            torch.nn.functional.softplus(model.transition_guard_scale_raw) / np.log(2.0)
        )
        trusted["transition2_guard"] = tolist(
            torch.nn.functional.softplus(model.transition2_guard_scale_raw) / np.log(2.0)
        )
    if model.predicate_conditioned_transitions:
        trusted["transition_predicate_factor"] = tolist(model.transition_predicate_factor.weight)
        trusted["transition2_first_predicate_factor"] = tolist(model.transition2_first_predicate_factor.weight)
        trusted["transition2_second_predicate_factor"] = tolist(model.transition2_second_predicate_factor.weight)
    return trusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("wikipedia", "reddit", "mooc", "lastfm"), default="wikipedia")
    parser.add_argument(
        "--queries",
        type=int,
        default=0,
        help="Number of evenly spaced test queries; 0 verifies the full test split.",
    )
    args = parser.parse_args()
    seed, dataset = 7, args.dataset
    checkpoint = ROOT / f"experiments/jodie_predicate_validation/checkpoints/{dataset}_k1_seed{seed}/{dataset}_LiFTER_seed{seed}.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = yaml.safe_load((ROOT / "experiments/main_benchmark/config_lifter.yaml").read_text())
    config["model_config"] = dict(saved["model_config"])
    frame = read_csv_tail(ROOT / f"data/processed/ctdg/{dataset}/events.csv", 32768)
    frame["src"] = pd.to_numeric(frame["src"]).astype("int64"); frame["dst"] = pd.to_numeric(frame["dst"]).astype("int64"); frame["timestamp"] = pd.to_numeric(frame["timestamp"]).astype("float32")
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    _, evaluation = split_indices(len(frame), 0.15)
    if args.queries < 0:
        parser.error("--queries must be non-negative")
    if args.queries == 0 or args.queries >= len(evaluation):
        rows = evaluation
    else:
        rows = evaluation[
            np.linspace(0, len(evaluation) - 1, args.queries, dtype=np.int64)
        ]
    raw_columns = [column for column in frame.columns if str(column).startswith("feat_")]
    raw = frame[raw_columns].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float32)
    context = build_lifter_causal_fact_context(frame, int(config["model_config"]["fact_context_len"]))
    num_nodes = int(max(frame.src.max(), frame.dst.max())) + 1
    index = LiFTERHistoryIndex(frame, int(config["training"]["history_len"]), num_nodes, context, raw)
    source = frame.iloc[rows].src.to_numpy(np.int64); destination = frame.iloc[rows].dst.to_numpy(np.int64)
    src_nodes, src_times, src_features, src_mask = index.gather(source, rows)
    dst_nodes, dst_times, dst_features, dst_mask = index.gather(destination, rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = RunSpec("ctdg", dataset, ROOT / f"data/processed/ctdg/{dataset}/events.csv", "LiFTER", "base", seed, "reference", config["model_config"], False)
    model = build_benchmark_model(spec, frame, config, device); model.load_state_dict(saved["model"]); model.eval()
    tensor = lambda value, dtype=None: torch.tensor(value, dtype=dtype, device=device)
    batch = (tensor(source), tensor(destination), tensor(frame.iloc[rows].timestamp.to_numpy(np.float32)), tensor(src_nodes), tensor(src_times, torch.float32), tensor(src_features, torch.float32), tensor(src_mask, torch.bool), tensor(dst_nodes), tensor(dst_times, torch.float32), tensor(dst_features, torch.float32), tensor(dst_mask, torch.bool))
    started = time.perf_counter(); explanations = model.explain(*batch); generation_ms = 1000 * (time.perf_counter() - started) / len(rows)
    certificates = []
    width = int(model.max_grounding_facts)
    for row, explanation in enumerate(explanations):
        history = []
        source_history = []
        destination_history = []
        banks = ((int(source[row]), src_nodes[row, -width:], src_times[row, -width:], src_features[row, -width:], src_mask[row, -width:]), (int(destination[row]), dst_nodes[row, -width:], dst_times[row, -width:], dst_features[row, -width:], dst_mask[row, -width:]))
        for bank_index, (anchor, nodes, times, features, mask) in enumerate(banks):
            for node, timestamp, feature, valid in zip(nodes, times, features, mask):
                if not valid: continue
                outgoing = float(feature[0]) >= 0
                fact = {"source": anchor if outgoing else int(node), "destination": int(node) if outgoing else anchor, "timestamp": float(timestamp)}
                history.append(fact)
                if bank_index == 0:
                    source_history.append(fact)
                else:
                    destination_history.append(fact)
        certificate = {
            **explanation,
            "historical_facts": history,
            "source_history_facts": source_history,
            "destination_history_facts": destination_history,
        }
        certificates.append(certificate)
    trusted = _trusted_parameters(model)
    started = time.perf_counter(); clean = [verify_certificate(item, trusted_parameters=trusted, trusted_candidate_logit=float(item["candidate_logit"])) for item in certificates]; verification_ms = 1000 * (time.perf_counter() - started) / len(certificates)
    attack_names = (
        "entity_change", "timestamp_change", "fact_omission", "fact_insertion",
        "contribution_change", "duplicate_with_logit_compensation",
        "execution_omission_with_logit_compensation",
        "contribution_redistribution", "operand_contribution_logit",
        "temporal_order_with_operand_swap",
    )
    attacks = {name: 0 for name in attack_names}
    attack_eligible = {name: 0 for name in attack_names}
    controls = {"execution_permutation": 0, "json_roundtrip": 0}
    control_eligible = {name: 0 for name in controls}
    eligible = 0
    for certificate in certificates:
        if not certificate["program_trace"] or not certificate["program_trace"][0]["grounded_facts"]: continue
        eligible += 1
        trusted_logit = float(certificate["candidate_logit"])
        for name in ("entity_change", "timestamp_change", "fact_omission", "fact_insertion", "contribution_change", "duplicate_with_logit_compensation", "execution_omission_with_logit_compensation"):
            attack_eligible[name] += 1
        modified = copy.deepcopy(certificate); modified["program_trace"][0]["grounded_facts"][0]["grounded_link"]["source"] += num_nodes + 1
        attacks["entity_change"] += int(not verify_certificate(modified, trusted_parameters=trusted, trusted_candidate_logit=trusted_logit)["valid"])
        modified = copy.deepcopy(certificate); modified["program_trace"][0]["grounded_facts"][0]["grounded_link"]["timestamp"] = float(certificate["query"]["timestamp"]) + 1
        attacks["timestamp_change"] += int(not verify_certificate(modified, trusted_parameters=trusted, trusted_candidate_logit=trusted_logit)["valid"])
        modified = copy.deepcopy(certificate); cited = modified["program_trace"][0]["grounded_facts"][0]["grounded_link"]; modified["historical_facts"] = [f for f in modified["historical_facts"] if not (f["source"] == cited["source"] and f["destination"] == cited["destination"] and f["timestamp"] == cited["timestamp"])]
        attacks["fact_omission"] += int(not verify_certificate(modified, trusted_parameters=trusted, trusted_candidate_logit=trusted_logit)["valid"])
        modified = copy.deepcopy(certificate); fake = copy.deepcopy(modified["program_trace"][0]); fake["grounded_facts"][0]["grounded_link"]["source"] += num_nodes + 1; fake["contribution"] = 0.0; modified["program_trace"].append(fake)
        attacks["fact_insertion"] += int(not verify_certificate(modified, trusted_parameters=trusted, trusted_candidate_logit=trusted_logit)["valid"])
        modified = copy.deepcopy(certificate); modified["program_trace"][0]["contribution"] += 0.1
        attacks["contribution_change"] += int(not verify_certificate(modified, trusted_parameters=trusted, trusted_candidate_logit=trusted_logit)["valid"])
        modified = copy.deepcopy(certificate)
        duplicate = copy.deepcopy(modified["program_trace"][0])
        modified["program_trace"].append(duplicate)
        modified["candidate_logit"] += float(duplicate["contribution"])
        attacks["duplicate_with_logit_compensation"] += int(
            not verify_certificate(modified, trusted_parameters=trusted, trusted_candidate_logit=trusted_logit)["valid"]
        )
        modified = copy.deepcopy(certificate)
        omitted = modified["program_trace"].pop(0)
        modified["candidate_logit"] -= float(omitted["contribution"])
        attacks["execution_omission_with_logit_compensation"] += int(
            not verify_certificate(modified, trusted_parameters=trusted, trusted_candidate_logit=trusted_logit)["valid"]
        )
        if len(certificate["program_trace"]) >= 2:
            attack_eligible["contribution_redistribution"] += 1
            modified = copy.deepcopy(certificate)
            modified["program_trace"][0]["contribution"] += 0.1
            modified["program_trace"][1]["contribution"] -= 0.1
            attacks["contribution_redistribution"] += int(
                not verify_certificate(modified, trusted_parameters=trusted, trusted_candidate_logit=trusted_logit)["valid"]
            )
        transition_index = next((i for i, execution in enumerate(certificate["program_trace"]) if execution["rule_kind"] in {"one_event_transition", "two_event_transition"}), None)
        if transition_index is not None:
            attack_eligible["operand_contribution_logit"] += 1
            modified = copy.deepcopy(certificate)
            execution = modified["program_trace"][transition_index]
            operand_name = "source" if execution["rule_kind"] == "one_event_transition" else "first"
            old = float(execution["contribution"])
            execution["execution_operands"][operand_name][0] += 0.25
            new = float(_recompute(execution))
            execution["contribution"] = new
            modified["candidate_logit"] += new - old
            attacks["operand_contribution_logit"] += int(
                not verify_certificate(modified, trusted_parameters=trusted, trusted_candidate_logit=trusted_logit)["valid"]
            )
        transition2_index = next((i for i, execution in enumerate(certificate["program_trace"]) if execution["rule_kind"] == "two_event_transition"), None)
        if transition2_index is not None:
            attack_eligible["temporal_order_with_operand_swap"] += 1
            modified = copy.deepcopy(certificate)
            execution = modified["program_trace"][transition2_index]
            execution["grounded_facts"].reverse()
            execution["execution_operands"]["first"], execution["execution_operands"]["second"] = execution["execution_operands"]["second"], execution["execution_operands"]["first"]
            attacks["temporal_order_with_operand_swap"] += int(
                not verify_certificate(modified, trusted_parameters=trusted, trusted_candidate_logit=trusted_logit)["valid"]
            )
        for name in controls:
            control_eligible[name] += 1
        modified = copy.deepcopy(certificate)
        modified["program_trace"].reverse()
        controls["execution_permutation"] += int(
            verify_certificate(modified, trusted_parameters=trusted, trusted_candidate_logit=trusted_logit)["valid"]
        )
        modified = json.loads(json.dumps(certificate))
        controls["json_roundtrip"] += int(
            verify_certificate(modified, trusted_parameters=trusted, trusted_candidate_logit=trusted_logit)["valid"]
        )
    def validity_rate(markers):
        return float(np.mean([not any(any(marker in failure for marker in markers) for failure in item["failures"]) for item in clean]))
    result = {"dataset": dataset, "queries": len(certificates), "eligible_tamper_queries": eligible, "clean_acceptance": float(np.mean([item["valid"] for item in clean])),
              "trace_completeness": validity_rate(("logit", "contribution", "prior")),
              "grounding_validity": validity_rate(("provenance", "binding", "grounded")),
              "temporal_validity": validity_rate(("temporal", "timestamp", "order")),
              "clean_failure_counts": dict(Counter(failure.split('.')[-1] for item in clean for failure in item["failures"])), "generation_ms_per_query": generation_ms, "verification_ms_per_query": verification_ms, "tamper_detection": {key: value / eligible for key, value in attacks.items()}, "max_clean_logit_error": max(abs(item["recomputed_logit"] - certificates[i]["candidate_logit"]) for i, item in enumerate(clean))}
    result["tamper_detection"] = {
        key: attacks[key] / attack_eligible[key] if attack_eligible[key] else None
        for key in attacks
    }
    result["tamper_eligible"] = attack_eligible
    result["positive_control_acceptance"] = {
        key: controls[key] / control_eligible[key] for key in controls
    }
    output = HERE / "results" / f"{dataset}_summary.json"; output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
