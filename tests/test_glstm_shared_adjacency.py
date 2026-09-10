from __future__ import annotations

import copy
from contextlib import ExitStack
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from Evaluation.experiment_outputs import _model_build_state  # noqa: E402
from Models.model import GLSTM_v2  # noqa: E402
from Training.experiment_runner import ExperimentRunConfig, build_model  # noqa: E402
from inference import (  # noqa: E402
    _build_reconstructed_model,
    _checkpoint_requires_per_layer_adjacency,
)


def _edge_index() -> torch.Tensor:
    return torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 0],
            [1, 0, 2, 1, 3, 2, 0, 3],
        ],
        dtype=torch.long,
    )


def _model(
    *,
    layers: int = 3,
    learn_adj: bool = True,
    learn_self_att: bool = False,
    share_adjacency: bool = True,
):
    return GLSTM_v2(
        N=4,
        edge_index=_edge_index(),
        in_channels=3,
        hidden_size=6,
        out_channels=2,
        lstm_layers=layers,
        learn_adj=learn_adj,
        learn_self_att=learn_self_att,
        lock_topology=True,
        dropout=0.0,
        share_adjacency=share_adjacency,
    )


def _config(
    *,
    layers: int = 3,
    learn_adj: bool = True,
    learn_self_att: bool = False,
) -> ExperimentRunConfig:
    return ExperimentRunConfig(
        start_date="2020-01-01",
        end_date="2020-12-31",
        state="RS",
        max_stations=None,
        include_precipitation=True,
        include_temperature=False,
        include_specific_humidity=False,
        include_wind=False,
        include_vertical_velocity=False,
        window_size=4,
        forecast_horizon=2,
        train_ratio=0.6,
        val_ratio=0.2,
        normalize_features=False,
        feature_scaler="standard",
        normalize_target=False,
        target_scaler="standard",
        model_type="glstm",
        k_neighbors=2,
        hidden_dim=6,
        lstm_layers=layers,
        learn_adj=learn_adj,
        lock_topology=True,
        dropout=0.0,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        patience=1,
        adj_lr_factor=1.0,
        max_grad_norm=1.0,
        loss="mse",
        loss_quantiles=(0.5,),
        loss_quantile_weights="auto",
        loss_quantile_max_weight=10.0,
        random_seed=42,
        plot_station_name=None,
        use_daily_cache=False,
        learn_self_att=learn_self_att,
    )


class GLSTMSharedAdjacencyTests(unittest.TestCase):
    def test_runner_builder_enables_shared_adjacency(self):
        model = build_model(
            "glstm",
            n_stations=4,
            n_features=3,
            edge_index=_edge_index(),
            config=_config(layers=3),
        )

        self.assertTrue(model.share_adjacency)
        self.assertEqual(
            [name for name, _parameter in model.named_parameters() if name.endswith("a_logits")],
            ["cell_0.a_logits"],
        )

    def test_default_multilayer_model_has_one_trainable_adjacency(self):
        model = _model(layers=3)
        cells = [model.cell_0, *model.cells]

        self.assertTrue(model.share_adjacency)
        self.assertEqual(
            [name for name, _parameter in model.named_parameters() if name.endswith("a_logits")],
            ["cell_0.a_logits"],
        )
        self.assertTrue(all(cell.a_logits is model.cell_0.a_logits for cell in cells))
        self.assertTrue(all(cell.edge_mask is model.cell_0.edge_mask for cell in cells))
        self.assertTrue(all(cell.topology_mask is model.cell_0.topology_mask for cell in cells))

        # Duplicate state_dict names preserve compatibility with the historical
        # checkpoint layout, but every value represents the same Parameter.
        state_keys = [key for key in model.state_dict() if key.endswith("a_logits")]
        self.assertEqual(state_keys, ["cell_0.a_logits", "cells.0.a_logits", "cells.1.a_logits"])
        self.assertTrue(
            all(torch.equal(model.state_dict()[key], model.cell_0.a_logits) for key in state_keys)
        )

        training_copy = copy.deepcopy(model)
        self.assertTrue(
            all(cell.a_logits is training_copy.cell_0.a_logits for cell in training_copy.cells)
        )

    def test_forward_passes_the_same_adjacency_tensor_to_every_layer(self):
        torch.manual_seed(7)
        model = _model(layers=3)
        cells = [model.cell_0, *model.cells]
        x = torch.randn(2, 4, 4, 3)

        with ExitStack() as stack:
            spies = [stack.enter_context(patch.object(cell, "forward", wraps=cell.forward)) for cell in cells]
            output = model(x)

        passed_adjacencies = [
            call.kwargs["A"]
            for spy in spies
            for call in spy.call_args_list
        ]
        self.assertEqual(output.shape, (2, 2, 4))
        self.assertEqual(len(passed_adjacencies), len(cells) * x.shape[1])
        self.assertTrue(all(adjacency is passed_adjacencies[0] for adjacency in passed_adjacencies))

    def test_every_layer_contributes_gradient_to_the_shared_logits(self):
        torch.manual_seed(11)
        model = _model(layers=3)
        cells = [model.cell_0, *model.cells]
        last_hidden: dict[int, torch.Tensor] = {}
        hooks = []
        for index, cell in enumerate(cells):
            hooks.append(
                cell.register_forward_hook(
                    lambda _module, _inputs, output, index=index: last_hidden.__setitem__(
                        index, output[1]
                    )
                )
            )

        try:
            model(torch.randn(2, 4, 4, 3))
            self.assertEqual(set(last_hidden), set(range(len(cells))))
            for index in range(len(cells)):
                gradient = torch.autograd.grad(
                    last_hidden[index].square().sum(),
                    model.cell_0.a_logits,
                    retain_graph=True,
                )[0]
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(torch.count_nonzero(gradient).item(), 0)
        finally:
            for hook in hooks:
                hook.remove()

    def test_fixed_adjacency_is_also_shared(self):
        model = _model(layers=3, learn_adj=False)
        cells = [model.cell_0, *model.cells]

        self.assertFalse(any(name.endswith("a_logits") for name, _ in model.named_parameters()))
        self.assertTrue(all(cell.A_fixed is model.cell_0.A_fixed for cell in cells))
        self.assertTrue(all(cell.edge_mask is model.cell_0.edge_mask for cell in cells))
        output = model(torch.randn(2, 4, 4, 3))
        self.assertEqual(output.shape, (2, 2, 4))

    def test_reset_restores_the_single_shared_adjacency(self):
        model = _model(layers=3)
        with torch.no_grad():
            model.cell_0.a_logits.add_(1.0)
        self.assertFalse(torch.equal(model.cell_0.a_logits, model.cell_0.a_logits_init))

        model.reset_parameters()

        self.assertTrue(torch.equal(model.cell_0.a_logits, model.cell_0.a_logits_init))
        self.assertTrue(all(cell.a_logits is model.cell_0.a_logits for cell in model.cells))

    def test_false_keeps_identity_diagonal_out_of_calibration(self):
        model = _model(layers=1, learn_self_att=False)
        raw_adjacency = model.current_adjacency(normalized=False)

        torch.testing.assert_close(
            raw_adjacency.diagonal(),
            torch.ones(model.N),
            rtol=0.0,
            atol=0.0,
        )
        raw_adjacency.diagonal().sum().backward()
        torch.testing.assert_close(
            model.cell_0.a_logits.grad.diagonal(),
            torch.zeros(model.N),
            rtol=0.0,
            atol=0.0,
        )

    def test_true_calibrates_positive_self_attention_from_identity(self):
        torch.manual_seed(29)
        model = _model(layers=2, learn_self_att=True)
        raw_before = model.current_adjacency(normalized=False).detach().clone()

        torch.testing.assert_close(
            raw_before.diagonal(),
            torch.ones(model.N),
            rtol=0.0,
            atol=0.0,
        )
        model(torch.randn(2, 4, model.N, 3)).square().mean().backward()
        diagonal_gradient = model.cell_0.a_logits.grad.diagonal()
        self.assertTrue(torch.isfinite(diagonal_gradient).all())
        self.assertGreater(torch.count_nonzero(diagonal_gradient).item(), 0)

        with torch.no_grad():
            model.cell_0.a_logits.add_(-0.1 * model.cell_0.a_logits.grad)
        raw_after = model.current_adjacency(normalized=False)
        self.assertTrue((raw_after.diagonal() > 0.0).all())
        self.assertFalse(torch.equal(raw_after.diagonal(), raw_before.diagonal()))

    def test_runner_builder_propagates_self_attention_flag(self):
        model = build_model(
            "glstm",
            n_stations=4,
            n_features=3,
            edge_index=_edge_index(),
            config=_config(layers=2, learn_self_att=True),
        )

        self.assertTrue(model.learn_self_att)
        self.assertTrue(model.cell_0.learn_self_att)
        self.assertTrue(all(cell.learn_self_att for cell in model.cells))

    def test_self_attention_learning_requires_learnable_adjacency(self):
        with self.assertRaisesRegex(ValueError, "requires learn_adj=True"):
            _model(layers=1, learn_adj=False, learn_self_att=True)


class GLSTMAdjacencyCheckpointTests(unittest.TestCase):
    def test_similarity_prior_round_trip_is_recovered_from_checkpoint_buffers(self):
        edge_weight = torch.tensor([0.8, 0.8, 0.3, 0.3, 0.1, 0.1, 0.6, 0.6])
        original = GLSTM_v2(
            N=4,
            edge_index=_edge_index(),
            edge_weight=edge_weight,
            in_channels=3,
            hidden_size=6,
            out_channels=2,
            lstm_layers=2,
            learn_adj=True,
            lock_topology=True,
            dropout=0.0,
        )
        checkpoint = original.state_dict()

        reconstructed = _build_reconstructed_model(
            _config(layers=2),
            n_stations=4,
            n_features=3,
            edge_index=_edge_index(),
            inference_state={
                "schema_version": 2,
                "model_build": _model_build_state(original),
            },
            state_dict=checkpoint,
        )
        reconstructed.load_state_dict(checkpoint)

        torch.testing.assert_close(
            reconstructed.current_adjacency(normalized=False),
            original.current_adjacency(normalized=False),
        )

    def test_learned_self_attention_round_trip_preserves_diagonal(self):
        original = _model(layers=2, learn_self_att=True)
        with torch.no_grad():
            original.cell_0.a_logits.diagonal().copy_(
                torch.tensor([-0.4, -0.1, 0.2, 0.5])
            )
        checkpoint = original.state_dict()
        model_build = _model_build_state(original)

        self.assertTrue(model_build["kwargs"]["learn_self_att"])
        reconstructed = _build_reconstructed_model(
            _config(layers=2, learn_self_att=True),
            n_stations=4,
            n_features=3,
            edge_index=_edge_index(),
            inference_state={"schema_version": 2, "model_build": model_build},
            state_dict=checkpoint,
        )
        reconstructed.load_state_dict(checkpoint)

        self.assertTrue(reconstructed.learn_self_att)
        torch.testing.assert_close(
            reconstructed.current_adjacency(normalized=False).diagonal(),
            original.current_adjacency(normalized=False).diagonal(),
        )

    def test_historical_model_build_defaults_to_fixed_identity_diagonal(self):
        original = _model(layers=2, learn_self_att=False)
        checkpoint = original.state_dict()
        historical_build = _model_build_state(original)
        historical_build["kwargs"].pop("learn_self_att")

        reconstructed = _build_reconstructed_model(
            _config(layers=2, learn_self_att=False),
            n_stations=4,
            n_features=3,
            edge_index=_edge_index(),
            inference_state={"schema_version": 2, "model_build": historical_build},
            state_dict=checkpoint,
        )

        self.assertFalse(reconstructed.learn_self_att)
        torch.testing.assert_close(
            reconstructed.current_adjacency(normalized=False).diagonal(),
            torch.ones(4),
            rtol=0.0,
            atol=0.0,
        )

    def test_shared_model_round_trip_uses_persisted_scope(self):
        torch.manual_seed(17)
        original = _model(layers=3)
        original.eval()
        checkpoint = original.state_dict()
        model_build = _model_build_state(original)
        self.assertEqual(model_build["adjacency_scope"], "shared")
        self.assertNotIn("share_adjacency", model_build["kwargs"])

        reconstructed = _build_reconstructed_model(
            _config(layers=3),
            n_stations=4,
            n_features=3,
            edge_index=_edge_index(),
            inference_state={"schema_version": 2, "model_build": model_build},
            state_dict=checkpoint,
        )
        self.assertTrue(reconstructed.share_adjacency)
        reconstructed.load_state_dict(checkpoint)
        reconstructed.eval()

        x = torch.randn(2, 4, 4, 3)
        torch.testing.assert_close(reconstructed(x), original(x))

        # An older loader reconstructs independent cells, but receives equal
        # duplicated adjacency tensors and therefore preserves the prediction.
        old_layout = _model(layers=3, share_adjacency=False)
        old_layout.load_state_dict(checkpoint)
        old_layout.eval()
        torch.testing.assert_close(old_layout(x), original(x))

    def test_shared_scope_rejects_distinct_serialized_logits(self):
        model = _model(layers=3)
        checkpoint = {key: value.clone() for key, value in model.state_dict().items()}
        checkpoint["cells.0.a_logits"].add_(checkpoint["cells.0.topology_mask"])

        with self.assertRaisesRegex(ValueError, "declares a shared GLSTM adjacency"):
            _build_reconstructed_model(
                _config(layers=3),
                n_stations=4,
                n_features=3,
                edge_index=_edge_index(),
                inference_state={
                    "schema_version": 2,
                    "model_build": _model_build_state(model),
                },
                state_dict=checkpoint,
            )

    def test_reconstruction_rejects_inconsistent_topology_buffers(self):
        model = _model(layers=3, share_adjacency=False)
        checkpoint = {key: value.clone() for key, value in model.state_dict().items()}
        checkpoint["cells.0.edge_mask"][0, 1] = 0.0

        with self.assertRaisesRegex(ValueError, "inconsistent immutable GLSTM adjacency"):
            _build_reconstructed_model(
                _config(layers=3),
                n_stations=4,
                n_features=3,
                edge_index=_edge_index(),
                inference_state=None,
                state_dict=checkpoint,
            )

    def test_legacy_checkpoint_with_distinct_graphs_stays_per_layer(self):
        torch.manual_seed(23)
        legacy = _model(layers=3, share_adjacency=False)
        with torch.no_grad():
            legacy.cells[0].a_logits.add_(0.5 * legacy.cells[0].topology_mask)
            legacy.cells[1].a_logits.sub_(0.25 * legacy.cells[1].topology_mask)
        legacy.eval()
        checkpoint = legacy.state_dict()
        self.assertTrue(_checkpoint_requires_per_layer_adjacency(checkpoint))

        # Historical schema-v2 model_build contracts did not have adjacency_scope.
        historical_build = _model_build_state(legacy)
        historical_build.pop("adjacency_scope")
        reconstructed = _build_reconstructed_model(
            _config(layers=3),
            n_stations=4,
            n_features=3,
            edge_index=_edge_index(),
            inference_state={"schema_version": 2, "model_build": historical_build},
            state_dict=checkpoint,
        )
        self.assertFalse(reconstructed.share_adjacency)
        reconstructed.load_state_dict(checkpoint)
        reconstructed.eval()

        x = torch.randn(2, 4, 4, 3)
        torch.testing.assert_close(reconstructed(x), legacy(x))

        inferred_without_state = _build_reconstructed_model(
            _config(layers=3),
            n_stations=4,
            n_features=3,
            edge_index=_edge_index(),
            inference_state=None,
            state_dict=checkpoint,
        )
        self.assertFalse(inferred_without_state.share_adjacency)


if __name__ == "__main__":
    unittest.main()
