from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch
from torch import nn

from evaluation.evaluate import (
    LengthThreeCandidateIndex,
    LiFTERHistoryIndex,
    build_lifter_causal_fact_context,
    build_grounded_explanation_universe,
    delete_lifter_proofs,
    lifter_mechanism_logits,
    sample_historical_destinations,
)
from models.factory import build_model
from models.lifter import LIFTER_EDGE_FEATURE_DIM, LiFTER, _render_rule_atoms


class LiFTERTest(unittest.TestCase):
    def test_single_predicate_is_fixed_and_context_invariant(self) -> None:
        model = LiFTER(
            hidden_dim=8,
            predicate_count=1,
            predicate_assignment_mode="learned",
            dropout=0.0,
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in model.fact_encoder.parameters()
            )
        )
        valid = torch.tensor([[True, False, True]])
        first = model._type_facts(
            torch.randn(1, 3, LIFTER_EDGE_FEATURE_DIM), valid
        )
        second = model._type_facts(
            torch.randn(1, 3, LIFTER_EDGE_FEATURE_DIM) * 100.0, valid
        )
        torch.testing.assert_close(first[0], second[0])
        torch.testing.assert_close(first[1], second[1])
        torch.testing.assert_close(first[2], torch.zeros(1, 3, dtype=torch.long))

    def test_controlled_predicate_assignment_modes_are_fixed_or_soft(self) -> None:
        features = torch.randn(2, 3, LIFTER_EDGE_FEATURE_DIM)
        valid = torch.ones(2, 3, dtype=torch.bool)
        random_model = LiFTER(
            hidden_dim=8,
            predicate_count=3,
            predicate_assignment_mode="random_fixed",
            dropout=0.0,
        ).eval()
        first = random_model._type_facts(features, valid)[2]
        second = random_model._type_facts(features, valid)[2]
        torch.testing.assert_close(first, second)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in random_model.fact_encoder.parameters())
        )

        centroids = torch.stack(
            (
                torch.zeros(LIFTER_EDGE_FEATURE_DIM),
                torch.ones(LIFTER_EDGE_FEATURE_DIM),
            )
        )
        kmeans_model = LiFTER(
            hidden_dim=8,
            predicate_count=2,
            predicate_assignment_mode="kmeans_fixed",
            fixed_predicate_centroids=centroids.tolist(),
            dropout=0.0,
        ).eval()
        assignments = kmeans_model._type_facts(
            torch.stack((torch.zeros(LIFTER_EDGE_FEATURE_DIM), torch.ones(LIFTER_EDGE_FEATURE_DIM)))[None],
            torch.ones(1, 2, dtype=torch.bool),
        )[2]
        torch.testing.assert_close(assignments, torch.tensor([[0, 1]]))

        soft_model = LiFTER(
            hidden_dim=8,
            predicate_count=3,
            predicate_execution_mode="soft",
            dropout=0.0,
        )
        self.assertFalse(soft_model._hard_execution())

    def test_explanation_universe_merges_path_and_endpoint_fact(self) -> None:
        batch = list(self._batch())
        # The first path fact is exactly the last source-history fact in row 0.
        path_sources = torch.tensor([[[1, 6, 6]], [[4, 9, 9]]])
        path_destinations = torch.tensor([[[2, 2, 2]], [[5, 5, 5]]])
        path_times = torch.tensor([[[8.0, 7.0, 9.0]], [[9.0, 7.0, 9.0]]])
        path_features = torch.zeros(2, 1, 3, LIFTER_EDGE_FEATURE_DIM)
        path_mask = torch.ones(2, 1, dtype=torch.bool)
        batch.extend(
            (path_sources, path_destinations, path_times, path_features, path_mask)
        )
        occurrence_scores = torch.ones(2, 9)
        scores, valid, endpoint_ids, path_ids = build_grounded_explanation_universe(
            tuple(batch), occurrence_scores
        )
        self.assertEqual(int(endpoint_ids[0, 2]), int(path_ids[0, 0, 0]))
        self.assertEqual(float(scores[0, endpoint_ids[0, 2]]), 2.0)
        self.assertEqual(int(valid[0].sum()), 8)

    def test_range_restricted_program_binds_both_head_arguments(self) -> None:
        model = LiFTER(
            hidden_dim=8,
            predicate_count=3,
            max_rule_length=3,
            range_restricted_program=True,
            temporal_component_count=3,
            temporal_component_scope="multi_fact",
            dropout=0.0,
        )
        self.assertEqual(model.rule_count, 2 * 3 + 3 * 3**2 + 3 * 3**3)
        for rule in model.program:
            self.assertTrue(
                rule.renewal
                or rule.length == 3
                or rule.left_role in {0, 1}
            )

        guarded = LiFTER(
            hidden_dim=8,
            predicate_count=3,
            range_restricted_program=True,
            guard_context_rules=True,
            dropout=0.0,
        )
        context_rules = [
            rule for rule in guarded.program
            if rule.length == 1 and rule.left_role >= 2
        ]
        self.assertTrue(context_rules)
        self.assertTrue(all(rule.guarded for rule in context_rules))
        for rule in context_rules:
            body = " & ".join(_render_rule_atoms(rule))
            self.assertIn("QuerySource(X,Tq)", body)
            self.assertIn("Candidate(Y,Tq)", body)

    def test_signed_rule_distribution_is_normalized_per_template(self) -> None:
        model = LiFTER(
            hidden_dim=8,
            predicate_count=3,
            max_rule_length=2,
            rule_weight_normalization="signed_distribution",
            dropout=0.0,
        )
        with torch.no_grad():
            model.rule_weight.copy_(torch.linspace(-2.0, 2.0, model.rule_count))
            model.rule_selection_logits.copy_(
                torch.linspace(-1.0, 1.0, model.rule_count)
            )
        weights = model._effective_rule_weights()
        masses = torch.nn.functional.softplus(model.rule_family_mass_raw)
        for family in range(model.rule_family_count):
            selected = weights[model.rule_family_ids == family]
            self.assertLessEqual(
                float(selected.abs().sum().detach()),
                float(masses[family].detach()) + 1e-6,
            )
        result = model.eval()(*self._batch())
        torch.testing.assert_close(
            result.logits,
            result.aux["prior_rule_contribution"]
            + result.aux["rule_scores"].sum(-1),
        )

    def test_compositional_rule_weights_cover_complete_program(self) -> None:
        model = LiFTER(
            hidden_dim=8,
            predicate_count=3,
            max_rule_length=3,
            rule_parameterization="compositional",
            rule_factor_rank=4,
            dropout=0.0,
        )
        self.assertEqual(model._effective_rule_weights().shape, (model.rule_count,))
        loss = model._effective_rule_weights().sum()
        loss.backward()
        self.assertIsNotNone(model.rule_template_factors.grad)
        self.assertIsNotNone(model.rule_predicate_factors.grad)

    def test_predicate_shuffle_preserves_valid_fact_histogram(self) -> None:
        model = LiFTER(hidden_dim=3, predicate_count=3, dropout=0.0).eval()
        model.fact_encoder = nn.Identity()
        model.predicate_prototypes.data.copy_(torch.eye(3))
        features = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]]
        )
        valid = torch.tensor([[True, False, True, True, False]])
        _, _, original = model._type_facts(features, valid)
        model.predicate_intervention = "shuffle_facts"
        _, _, shuffled = model._type_facts(features, valid)
        torch.testing.assert_close(
            torch.bincount(original[valid], minlength=3),
            torch.bincount(shuffled[valid], minlength=3),
        )
        self.assertFalse(torch.equal(original[valid], shuffled[valid]))
        torch.testing.assert_close(original[~valid], shuffled[~valid])

    def test_global_predicate_shuffle_breaks_rows_and_preserves_histogram(self) -> None:
        model = LiFTER(hidden_dim=3, predicate_count=3, dropout=0.0).eval()
        model.fact_encoder = nn.Identity()
        model.predicate_prototypes.data.copy_(torch.eye(3))
        features = torch.tensor([
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        ])
        valid = torch.ones(2, 2, dtype=torch.bool)
        _, _, original = model._type_facts(features, valid)
        model.predicate_intervention = "shuffle_global"
        _, _, shuffled = model._type_facts(features, valid)
        torch.testing.assert_close(
            torch.bincount(original[valid], minlength=3),
            torch.bincount(shuffled[valid], minlength=3),
        )
        self.assertFalse(torch.equal(original[0], shuffled[0]))

    def test_proof_matched_candidates_obey_executor_time_and_recurrence(self) -> None:
        frame = pd.DataFrame(
            {
                # X->A; two B->A,B->Y transitions create proof-matched Y values.
                "src": [3, 3, 4, 4, 1, 1],
                "dst": [2, 5, 2, 6, 2, 5],
                "timestamp": [1.0, 2.0, 1.2, 2.2, 3.0, 4.0],
            }
        ).sort_values("timestamp", kind="stable").reset_index(drop=True)
        index = LengthThreeCandidateIndex(frame, endpoint_width=16)
        row = int(frame.index[frame["timestamp"] == 4.0][0])
        support = index.support(1, row)
        self.assertIn(5, support)
        self.assertIn(6, support)
        sampled, selected, stats = index.sample(
            np.asarray([1]), np.asarray([5]), np.asarray([row])
        )
        self.assertTrue(bool(selected[0]))
        self.assertEqual(int(sampled[0]), 6)
        self.assertGreater(int(stats["positive_count"][0]), 0)

    def test_historical_negative_uses_prior_destination_and_reports_fallback(self) -> None:
        np.random.seed(0)
        sampled, historical = sample_historical_destinations(
            np.asarray([1, 2]),
            np.asarray([5, 6]),
            np.asarray([[0, 4, 5], [0, 0, 6]]),
            np.asarray([[False, True, True], [False, False, True]]),
            np.asarray([4, 5, 6, 7]),
        )
        self.assertEqual(int(sampled[0]), 4)
        self.assertTrue(bool(historical[0]))
        self.assertFalse(bool(historical[1]))
        self.assertNotEqual(int(sampled[1]), 6)

    def test_proof_deletion_and_mechanism_masks_use_executed_contributions(self) -> None:
        model = LiFTER(hidden_dim=8, predicate_count=3, dropout=0.0).eval()
        batch = self._batch()
        with torch.no_grad():
            model.predicate_prototypes.zero_()
            model.rule_weight.copy_(torch.linspace(-0.4, 0.4, model.rule_count))
            result = model(*batch)
        mechanisms = lifter_mechanism_logits(model, result)
        self.assertIn("full", mechanisms)
        self.assertIn("only_direct_pair", mechanisms)
        self.assertIn("without_direct_pair", mechanisms)
        self.assertIn("only_ordered_transition_2", mechanisms)
        self.assertIn("without_ordered_transition_2", mechanisms)
        torch.testing.assert_close(mechanisms["full"], result.logits)
        torch.testing.assert_close(
            mechanisms["length_three_only"], result.aux["prior_rule_contribution"]
        )
        direct_component = (
            mechanisms["only_direct_pair"]
            - result.aux["prior_rule_contribution"]
        )
        torch.testing.assert_close(
            mechanisms["without_direct_pair"],
            result.logits - direct_component,
        )
        top, random, top_delta, random_delta = delete_lifter_proofs(
            result, np.random.default_rng(11)
        )
        torch.testing.assert_close((result.logits - top).abs(), top_delta)
        torch.testing.assert_close((result.logits - random).abs(), random_delta)
        self.assertTrue(bool((top_delta >= random_delta).all()))

    def test_factory_builds_rule_only_neuro_symbolic_model(self) -> None:
        model = build_model(
            "LiFTER",
            num_nodes=100,
            edge_feat_dim=LIFTER_EDGE_FEATURE_DIM,
            hidden_dim=16,
            predicate_count=4,
            dropout=0.0,
        )
        self.assertTrue(model.is_neuro_symbolic)
        self.assertFalse(any(isinstance(module, nn.Embedding) for module in model.modules()))
        self.assertGreater(len(model.export_symbolic_rules()), 1)

    def test_fact_context_is_strictly_causal_at_equal_timestamps(self) -> None:
        frame = pd.DataFrame(
            {
                "src": [0, 0, 0],
                "dst": [1, 2, 1],
                "timestamp": [1.0, 1.0, 2.0],
            }
        )
        context = build_lifter_causal_fact_context(frame)
        np.testing.assert_array_equal(context[0], np.zeros((8, 6), dtype=np.float32))
        np.testing.assert_array_equal(context[1], np.zeros((8, 6), dtype=np.float32))
        self.assertEqual(float(context[2, :, -1].sum()), 2.0)
        self.assertTrue((context[2, -2:, 4] > 0.0).all())

        index = LiFTERHistoryIndex(frame, 4, 3, context)
        from_source = index.gather(np.asarray([0]), np.asarray([2]))
        from_destination = index.gather(np.asarray([1]), np.asarray([2]))
        # Row 0 is reached with opposite query-relative direction, but its
        # canonical fact-neighborhood payload is byte-for-byte identical.
        source_feature = from_source[2][0, -2]
        destination_feature = from_destination[2][0, -1]
        self.assertEqual(float(source_feature[0]), 1.0)
        self.assertEqual(float(destination_feature[0]), -1.0)
        np.testing.assert_array_equal(source_feature[1:], destination_feature[1:])

    @staticmethod
    def _batch() -> tuple[torch.Tensor, ...]:
        src = torch.tensor([1, 4])
        dst = torch.tensor([2, 5])
        query_time = torch.tensor([10.0, 10.0])
        src_nodes = torch.tensor([[0, 3, 2], [7, 8, 5]])
        dst_nodes = torch.tensor([[0, 6, 1], [7, 9, 4]])
        src_times = torch.tensor([[2.0, 5.0, 8.0], [1.0, 6.0, 9.0]])
        dst_times = torch.tensor([[3.0, 7.0, 9.0], [2.0, 7.0, 9.0]])
        src_features = torch.zeros(2, 3, LIFTER_EDGE_FEATURE_DIM)
        dst_features = torch.zeros(2, 3, LIFTER_EDGE_FEATURE_DIM)
        src_features[..., 0] = torch.tensor([[1.0, -1.0, 1.0], [1.0, 1.0, -1.0]])
        dst_features[..., 0] = torch.tensor([[-1.0, 1.0, -1.0], [-1.0, 1.0, 1.0]])
        src_context = src_features[..., 1:].reshape(2, 3, 8, 6)
        dst_context = dst_features[..., 1:].reshape(2, 3, 8, 6)
        src_context[..., :-1] = torch.randn(2, 3, 8, 5)
        dst_context[..., :-1] = torch.randn(2, 3, 8, 5)
        src_context[..., -1] = 1.0
        dst_context[..., -1] = 1.0
        mask = torch.ones(2, 3, dtype=torch.bool)
        return (
            src,
            dst,
            query_time,
            src_nodes,
            src_times,
            src_features,
            mask,
            dst_nodes,
            dst_times,
            dst_features,
            mask,
        )

    def test_grounded_execution_is_entity_relabel_invariant_and_trainable(self) -> None:
        torch.manual_seed(3)
        model = LiFTER(
            hidden_dim=16,
            predicate_count=4,
            max_grounding_facts=3,
            include_renewal_rules=False,
            include_direct_conjunctions=True,
            include_disconnected_conjunctions=True,
            dropout=0.0,
        )
        model.eval()
        batch = self._batch()
        batch[9][0, 2, 1:] = batch[5][0, 2, 1:]
        result = model(*batch)
        self.assertTrue(torch.isfinite(result.logits).all())
        self.assertEqual(
            int(result.aux["predicate_ids_src"][0, 2]),
            int(result.aux["predicate_ids_dst"][0, 2]),
        )

        # A bijective entity relabeling preserves exactly the same unifications.
        relabeled = list(batch)
        for index in (0, 1, 3, 7):
            relabeled[index] = batch[index] + 100
        relabeled_result = model(*relabeled)
        torch.testing.assert_close(result.logits, relabeled_result.logits)

        model.train()
        typed_src = model._type_facts(batch[5], batch[6])[2]
        typed_dst = model._type_facts(batch[5].clone(), batch[6])[2]
        torch.testing.assert_close(typed_src, typed_dst)
        train_result = model(*batch)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            train_result.logits, torch.ones_like(train_result.logits)
        ) + train_result.aux["regularization_loss"]
        loss.backward()
        self.assertGreater(float(model.predicate_prototypes.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.rule_weight.grad.abs().sum()), 0.0)
        self.assertIn("top_fact1_index", train_result.aux)
        with torch.no_grad():
            model.predicate_prototypes.zero_()
            model.rule_weight.zero_()
            model.rule_weight[2 * model.predicate_count] = 4.0
        explanation = model.explain(*batch)[0]
        self.assertEqual(explanation["query"]["predicate"], "Link")
        self.assertTrue(explanation["grounding_valid"])
        self.assertGreaterEqual(len(explanation["grounded_facts"]), 1)
        self.assertIn("rule", explanation)
        self.assertEqual(explanation["rule"]["head"], "Link(X,Y,Tq)")
        self.assertAlmostEqual(explanation["decomposition_error"], 0.0, places=5)

        self.assertAlmostEqual(
            explanation["candidate_logit"],
            explanation["reconstructed_logit"],
            places=5,
        )

    def test_latent_transition_rule_is_part_of_the_exact_program_trace(self) -> None:
        torch.manual_seed(31)
        model = LiFTER(
            num_nodes=512,
            hidden_dim=8,
            predicate_count=3,
            max_grounding_facts=3,
            latent_transition_rule=True,
            recurrence_guarded_transitions=True,
            predicate_conditioned_transitions=True,
            transition_dim=4,
            dropout=0.0,
        ).eval()
        batch = self._batch()
        result = model(*batch)
        explanation = model.explain(*batch)[0]

        self.assertTrue(explanation["transition_rule"]["grounding_valid"])
        self.assertEqual(
            explanation["transition_rule"]["previous_destination"],
            int(result.aux["transition_previous_destination"][0]),
        )
        transition_trace = next(
            item for item in explanation["program_trace"] if item["rule_id"] == "Q1"
        )
        predicate = int(result.aux["transition_previous_predicate"][0])
        self.assertIn(f"P{predicate}(X,Z,Tlast)", transition_trace["body"])
        self.assertEqual(
            transition_trace["grounded_facts"][0]["predicate"], f"P{predicate}"
        )
        self.assertIn(
            "Recurrence(X,Y,T<Tq)", transition_trace["body"]
        )
        self.assertAlmostEqual(explanation["decomposition_error"], 0.0, places=5)
        self.assertAlmostEqual(
            explanation["candidate_logit"],
            explanation["reconstructed_logit"],
            places=5,
        )

    def test_positioned_recurrence_is_an_exact_grounded_rule(self) -> None:
        torch.manual_seed(35)
        model = LiFTER(
            hidden_dim=8,
            predicate_count=3,
            max_grounding_facts=3,
            positioned_recurrence_rules=True,
            dropout=0.0,
        ).eval()
        with torch.no_grad():
            model.rule_weight.zero_()
            model.prior_rule_weight.zero_()
            model.positioned_recurrence_rule_weight.fill_(0.25)

        batch = self._batch()
        result = model(*batch)
        torch.testing.assert_close(
            result.aux["positioned_recurrence_rule_contribution"],
            torch.tensor([0.25, 0.0]),
        )
        explanation = model.explain(*batch)[0]
        positioned = next(
            execution
            for execution in explanation["program_trace"]
            if str(execution["rule_id"]).startswith("positioned_recurrence_")
        )
        self.assertEqual(
            positioned["grounded_facts"][0]["grounded_link"]["destination"],
            int(batch[1][0]),
        )
        self.assertEqual(
            positioned["grounded_facts"][0]["source_history_position"], 1
        )
        self.assertAlmostEqual(explanation["decomposition_error"], 0.0, places=5)

    def test_shared_source_candidate_scoring_matches_separate_forwards(self) -> None:
        torch.manual_seed(37)
        model = LiFTER(
            num_nodes=512,
            hidden_dim=8,
            predicate_count=3,
            max_grounding_facts=3,
            latent_transition_rule=True,
            latent_second_order_transition_rule=True,
            dropout=0.0,
        ).eval()
        positive = self._batch()
        negative = tuple(value.clone() for value in positive)
        negative = list(negative)
        negative[1] = negative[1] + 1
        negative = tuple(negative)
        expected = (model(*positive).logits, model(*negative).logits)
        actual = model.score_candidate_batches(positive, negative)
        torch.testing.assert_close(actual[0].logits, expected[0])
        torch.testing.assert_close(actual[1].logits, expected[1])

    def test_sparse_renewal_matches_dense_execution(self) -> None:
        torch.manual_seed(13)
        dense = LiFTER(
            hidden_dim=8,
            predicate_count=3,
            max_rule_length=2,
            include_renewal_rules=True,
            temporal_component_count=3,
            temporal_component_scope="renewal",
            dropout=0.0,
        ).eval()
        sparse = LiFTER(
            hidden_dim=8,
            predicate_count=3,
            max_rule_length=2,
            include_renewal_rules=True,
            sparse_renewal_execution=True,
            temporal_component_count=3,
            temporal_component_scope="renewal",
            dropout=0.0,
        ).eval()
        sparse.load_state_dict(dense.state_dict())
        batch = list(self._batch())
        # Give each query two earlier interactions with its candidate so that
        # at least one ordered renewal grounding exists.
        batch[3] = batch[3].clone()
        batch[3][:, -2:] = batch[1][:, None]
        batch[5] = batch[5].clone()
        batch[5][:, -2:, 0] = 1.0
        dense_result = dense(*batch)
        sparse_result = sparse(*batch)
        torch.testing.assert_close(sparse_result.logits, dense_result.logits)
        torch.testing.assert_close(
            sparse_result.aux["rule_scores"], dense_result.aux["rule_scores"]
        )
        torch.testing.assert_close(
            sparse_result.aux["rule_grounding_valid"],
            dense_result.aux["rule_grounding_valid"],
        )
        renewal = dense.renewal_rules
        self.assertTrue(bool(dense_result.aux["rule_grounding_valid"][:, renewal].any()))
        torch.testing.assert_close(
            sparse_result.aux["top_fact1_index"],
            dense_result.aux["top_fact1_index"],
        )
        torch.testing.assert_close(
            sparse_result.aux["top_fact2_index"],
            dense_result.aux["top_fact2_index"],
        )

    def test_sparse_multiscale_ternary_matches_dense_execution(self) -> None:
        torch.manual_seed(19)
        common = dict(
            hidden_dim=8,
            predicate_count=3,
            max_rule_length=3,
            temporal_component_count=3,
            enumerate_temporal_orders=True,
            enforce_three_hop_order=False,
            dropout=0.0,
        )
        dense = LiFTER(**common).eval()
        sparse = LiFTER(**common, sparse_ternary_execution=True).eval()
        sparse.load_state_dict(dense.state_dict())
        batch = self._batch()
        path_sources = torch.tensor([[[1, 3, 3]], [[4, 7, 7]]])
        path_destinations = torch.tensor([[[0, 0, 2]], [[8, 8, 5]]])
        path_times = torch.tensor([[[2.0, 5.0, 9.0]], [[1.0, 6.0, 9.0]]])
        path_features = torch.zeros(2, 1, 3, LIFTER_EDGE_FEATURE_DIM)
        path_features[..., 0] = 1.0
        path_features[..., 1:].reshape(2, 1, 3, 8, 6)[..., -1] = 1.0
        extended = batch + (
            path_sources,
            path_destinations,
            path_times,
            path_features,
            torch.ones(2, 1, dtype=torch.bool),
        )
        dense_result = dense(*extended)
        sparse_result = sparse(*extended)
        torch.testing.assert_close(sparse_result.logits, dense_result.logits)
        torch.testing.assert_close(
            sparse_result.aux["rule_scores"], dense_result.aux["rule_scores"]
        )
        torch.testing.assert_close(
            sparse_result.aux["rule_grounding_valid"],
            dense_result.aux["rule_grounding_valid"],
        )

    def test_no_grounding_has_no_bias_or_fake_explanation_index(self) -> None:
        model = LiFTER(
            hidden_dim=8,
            predicate_count=3,
            dropout=0.0,
        ).eval()
        batch = list(self._batch())
        empty_mask = torch.zeros_like(batch[6])
        batch[6] = empty_mask
        batch[10] = empty_mask
        result = model(*batch)
        torch.testing.assert_close(
            result.logits,
            model.prior_rule_weight.detach().expand_as(result.logits),
        )
        self.assertFalse(bool(result.aux["top_grounding_valid"].any()))
        self.assertTrue(bool((result.aux["top_fact1_index"] == -1).all()))
        explanations = model.explain(*batch)
        self.assertTrue(all(not item["grounding_valid"] for item in explanations))
        self.assertTrue(all(item["grounded_facts"] == [] for item in explanations))

    def test_bipartite_safe_two_fact_rule_has_real_grounding(self) -> None:
        model = LiFTER(
            hidden_dim=8,
            predicate_count=3,
            max_grounding_facts=3,
            include_renewal_rules=False,
            include_direct_conjunctions=True,
            include_disconnected_conjunctions=True,
            dropout=0.0,
        ).eval()
        batch = self._batch()
        with torch.no_grad():
            model.predicate_prototypes.zero_()  # every fact deterministically becomes P0
        result = model(*batch)
        # First rule with P0(X,Z) and P0(Z,Y): distinct existential Z symbols,
        # hence valid on a directed bipartite source->destination graph.
        binary_rule = next(
            index
            for index, rule in enumerate(model.program)
            if rule.left_role == 2
            and rule.right_role == 5
            and rule.left_predicate == 0
            and rule.right_predicate == 0
        )
        self.assertTrue(bool(result.aux["rule_grounding_valid"][:, binary_rule].all()))

    def test_reported_rule_contribution_is_causally_faithful(self) -> None:
        model = LiFTER(
            hidden_dim=8,
            predicate_count=3,
            max_grounding_facts=3,
            dropout=0.0,
        ).eval()
        batch = tuple(value[:1] for value in self._batch())
        with torch.no_grad():
            model.rule_weight.copy_(
                torch.linspace(0.05, 0.4, steps=model.rule_count)
            )
            original = model(*batch)
            rule_id = int(original.aux["top_rule_id"][0])
            contribution = original.aux["top_grounding_score"][0].clone()
            self.assertTrue(bool(original.aux["top_grounding_valid"][0]))
            model.rule_weight[rule_id] = 0.0
            ablated = model(*batch)
        torch.testing.assert_close(
            original.logits[0] - ablated.logits[0], contribution
        )

    def test_three_hop_grammar_grounds_global_fact_path(self) -> None:
        frame = pd.DataFrame(
            {
                "src": [1, 3, 3, 1],
                "dst": [2, 2, 4, 4],
                "timestamp": [1.0, 2.0, 3.0, 5.0],
            }
        )
        context = build_lifter_causal_fact_context(frame)
        index = LiFTERHistoryIndex(frame, 8, 5, context)
        path = index.gather_three_hop_paths(
            np.asarray([1]), np.asarray([4]), np.asarray([3]), 4
        )
        self.assertEqual(int(path[4].sum()), 1)
        path_index = int(np.flatnonzero(path[4][0])[0])
        np.testing.assert_array_equal(path[0][0, path_index], [1, 3, 3])
        np.testing.assert_array_equal(path[1][0, path_index], [2, 2, 4])

        model = LiFTER(
            hidden_dim=8,
            predicate_count=3,
            max_rule_length=3,
            max_three_hop_paths=4,
            enforce_three_hop_order=False,
            dropout=0.0,
        ).eval()
        self.assertEqual(model.rule_count, 6 * 3 + 3**2 + 3**3)
        length_three = LiFTER(
            hidden_dim=8, predicate_count=3, max_rule_length=3,
            program_scope="length_three_only", dropout=0.0,
        )
        recurrence = LiFTER(
            hidden_dim=8, predicate_count=3, max_rule_length=3,
            program_scope="recurrence_only", dropout=0.0,
        )
        no_length_three = LiFTER(
            hidden_dim=8, predicate_count=3, max_rule_length=3,
            program_scope="no_length_three", dropout=0.0,
        )
        self.assertEqual(length_three.rule_count, 3**3)
        self.assertTrue(all(rule.length == 3 for rule in length_three.program))
        self.assertTrue(all(rule.renewal or rule.left_role in {0, 1} for rule in recurrence.program))
        self.assertTrue(all(rule.length < 3 for rule in no_length_three.program))
        batch = self._batch()
        path_sources = torch.tensor([[[1, 3, 3]], [[4, 7, 7]]])
        path_destinations = torch.tensor([[[0, 0, 2]], [[8, 8, 5]]])
        path_times = torch.tensor([[[2.0, 5.0, 9.0]], [[1.0, 6.0, 9.0]]])
        path_features = torch.zeros(2, 1, 3, LIFTER_EDGE_FEATURE_DIM)
        path_features[..., 0] = 1.0
        path_context = path_features[..., 1:].reshape(2, 1, 3, 8, 6)
        path_context[..., -1] = 1.0
        path_mask = torch.ones(2, 1, dtype=torch.bool)
        with torch.no_grad():
            model.predicate_prototypes.zero_()
            model.rule_weight.zero_()
            ternary_rule = 6 * 3 + 3**2
            model.rule_weight[ternary_rule] = 4.0
        extended = batch + (
            path_sources,
            path_destinations,
            path_times,
            path_features,
            path_mask,
        )
        result = model(*extended)
        sparse_model = LiFTER(
            hidden_dim=8,
            predicate_count=3,
            max_rule_length=3,
            max_three_hop_paths=4,
            enforce_three_hop_order=False,
            sparse_ternary_execution=True,
            dropout=0.0,
        ).eval()
        sparse_model.load_state_dict(model.state_dict())
        sparse_result = sparse_model(*extended)
        torch.testing.assert_close(sparse_result.logits, result.logits)
        torch.testing.assert_close(
            sparse_result.aux["rule_scores"], result.aux["rule_scores"]
        )
        torch.testing.assert_close(
            sparse_result.aux["rule_grounding_valid"],
            result.aux["rule_grounding_valid"],
        )
        self.assertTrue(bool(result.aux["rule_grounding_valid"][:, ternary_rule].all()))
        broken = list(extended)
        broken_destinations = path_destinations.clone()
        broken_destinations[:, :, 1] += 100
        broken[12] = broken_destinations
        broken_result = model(*tuple(broken))
        self.assertFalse(
            bool(broken_result.aux["rule_grounding_valid"][:, ternary_rule].any())
        )
        torch.testing.assert_close(
            broken_result.logits,
            model.prior_rule_weight.detach().expand_as(broken_result.logits),
        )
        explanation = model.explain(*tuple(value[:1] for value in extended))[0]
        self.assertEqual(explanation["rule"]["length"], 3)
        self.assertEqual(len(explanation["grounded_facts"]), 3)
        self.assertAlmostEqual(explanation["decomposition_error"], 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
