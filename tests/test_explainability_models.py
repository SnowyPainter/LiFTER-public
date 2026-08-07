import unittest

import torch
from torch import nn

from models.base import ModelResult
from models.explainability import SIG, TGIB, TGNNExplainer, TempME, build_explainability_model


class _MaskSensitivePredictor(nn.Module):
    def forward(self, history_nodes, history_mask=None, **kwargs):
        del kwargs
        mask = torch.ones_like(history_nodes, dtype=torch.bool) if history_mask is None else history_mask
        return ModelResult((history_nodes.float() * mask).sum(1), {})


class ExplainabilityModelsTest(unittest.TestCase):
    def setUp(self):
        self.inputs = dict(
            src=torch.tensor([1, 2]), dst=torch.tensor([3, 4]),
            timestamp=torch.tensor([10.0, 12.0]),
            history_nodes=torch.tensor([[2, 4, 6], [1, 3, 5]]),
            history_times=torch.tensor([[1.0, 3.0, 7.0], [2.0, 4.0, 8.0]]),
            history_edge_feats=torch.randn(2, 3, 2),
            history_mask=torch.ones(2, 3, dtype=torch.bool),
            dst_history_nodes=torch.tensor([[1, 2, 5], [2, 3, 6]]),
            dst_history_times=torch.tensor([[2.0, 4.0, 8.0], [1.0, 5.0, 9.0]]),
            dst_history_edge_feats=torch.randn(2, 3, 2),
            dst_history_mask=torch.ones(2, 3, dtype=torch.bool),
        )

    def test_predictors_return_intrinsic_event_importance(self):
        for cls in (TGIB, SIG):
            result = cls(num_nodes=10, edge_feat_dim=2)(**self.inputs)
            self.assertEqual(result.logits.shape, (2,))
            self.assertEqual(result.aux["event_importance"].shape, (2, 6))

    def test_tempme_builds_ordered_motifs(self):
        result = TempME(num_nodes=10, edge_feat_dim=2)(**self.inputs)
        self.assertEqual(result.motif_scores.shape, (2, 3, 3))
        self.assertTrue(torch.all(torch.diagonal(result.motif_scores, dim1=1, dim2=2) == 0))

    def test_tgnnexplainer_measures_deletion_fidelity(self):
        source_only = {key: value for key, value in self.inputs.items() if not key.startswith("dst_history_")}
        result = TGNNExplainer(_MaskSensitivePredictor()).explain(source_only, top_k=1)
        self.assertEqual(result.selected_mask.sum(1).tolist(), [1, 1])
        self.assertEqual(result.selected_mask[:, 2].tolist(), [True, True])
        self.assertIsNotNone(result.fidelity_drop)

    def test_registry_builds_native_class(self):
        model = build_explainability_model("TGIB", num_nodes=10, edge_feat_dim=2)
        self.assertIsInstance(model, TGIB)


if __name__ == "__main__":
    unittest.main()
