#!/opt/conda/bin/python
"""Deterministic causality and state-isolation checks for benchmark TGN."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.tgn import NativeTGN


def batch(device: torch.device) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(19)
    size, length, nodes = 8, 32, 50
    timestamp = torch.full((size,), 100.0, device=device)
    mask = torch.zeros(size, length, dtype=torch.bool, device=device)
    mask[:, -20:] = True
    times = torch.arange(68, 100, device=device).float()[None].expand(size, -1).clone()
    history = torch.randint(nodes, (size, length), device=device)
    features = torch.randn(size, length, 1, device=device)
    return (
        torch.arange(size, device=device),
        torch.arange(size, device=device) + 10,
        timestamp,
        history, times, features, mask,
        history.roll(1, 0), times.clone(), features.roll(1, 0), mask.clone(),
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NativeTGN(num_nodes=50, neural_history_len=20).to(device).eval()
    inputs = batch(device)
    with torch.no_grad():
        reference = model(*inputs).logits

        # Reordering queries cannot transfer memory between rows.
        permutation = torch.tensor([3, 7, 0, 5, 2, 1, 6, 4], device=device)
        permuted = tuple(value[permutation] for value in inputs)
        order_error = (model(*permuted).logits - reference[permutation]).abs().max().item()

        # Scoring another candidate cannot mutate the next prediction.
        alternative = list(inputs); alternative[1] = alternative[1].roll(1)
        _ = model(*tuple(alternative)).logits
        state_error = (model(*inputs).logits - reference).abs().max().item()

        # Values outside the configured 20-event prefix are inaccessible.
        old = list(inputs)
        old[3] = old[3].clone(); old[3][:, :-20] = old[3][:, :-20].roll(1, 0)
        old[5] = old[5].clone(); old[5][:, :-20] += 1000
        suffix_error = (model(*tuple(old)).logits - reference).abs().max().item()

        future = list(inputs)
        future[4] = future[4].clone(); future[4][:, -1] = 101
        rejected_future = False
        try:
            model(*tuple(future))
        except ValueError:
            rejected_future = True

    result = {
        "batch_permutation_max_abs_error": order_error,
        "candidate_call_state_max_abs_error": state_error,
        "outside_history_window_max_abs_error": suffix_error,
        "future_timestamp_rejected": rejected_future,
        "passed": order_error == 0.0 and state_error == 0.0
        and suffix_error == 0.0 and rejected_future,
    }
    output = Path(__file__).resolve().parent / "results" / "tgn_causality_audit.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
