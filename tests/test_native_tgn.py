import unittest

import torch

from models.factory import build_model
from models.tgn import NativeTGN


class NativeTGNTest(unittest.TestCase):
    def setUp(self):
        self.batch = dict(
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

    def test_factory_and_shapes(self):
        model = build_model("TGN", num_nodes=10, edge_feat_dim=2, hidden_dim=8)
        self.assertIsInstance(model, NativeTGN)
        result = model(**self.batch)
        self.assertEqual(result.logits.shape, (2,))
        self.assertEqual(result.aux["src_memory_trace"].shape, (2, 3, 8))

    def test_zero_weight_equals_deleted_history(self):
        model = NativeTGN(num_nodes=10, edge_feat_dim=2, hidden_dim=8, dropout=0).eval()
        zero = torch.zeros(2, 3)
        weighted = model(**self.batch, history_weights=zero, dst_history_weights=zero).logits
        masked_batch = {**self.batch, "history_mask": torch.zeros(2, 3, dtype=torch.bool),
                        "dst_history_mask": torch.zeros(2, 3, dtype=torch.bool)}
        deleted = model(**masked_batch).logits
        self.assertTrue(torch.allclose(weighted, deleted))

    def test_mask_changes_prediction_and_gradients_reach_message_function(self):
        model = NativeTGN(num_nodes=10, edge_feat_dim=2, hidden_dim=8, dropout=0)
        result = model(**self.batch)
        result.logits.sum().backward()
        self.assertIsNotNone(model.message_function[0].weight.grad)
        removed = model(**self.batch, history_weights=torch.zeros(2, 3)).logits
        self.assertFalse(torch.allclose(result.logits.detach(), removed.detach()))


if __name__ == "__main__":
    unittest.main()
