"""Artifact store, activation capture, weight spectra and SAE loading."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from emerge.capture import weights as W
from emerge.capture.activations import capture_activations, mean_difference, resolve_layers
from emerge.capture.sae import SAE, load_sae, synthetic_sae
from emerge.capture.store import ArtifactStore, artifact_name


class TestStore:
    def test_save_load_and_manifest(self, root):
        store = ArtifactStore(root, "run-1")
        store.save("activations", artifact_name("activations", 10),
                   {"acts": np.zeros((2, 3, 4), dtype=np.float16)},
                   meta={"step": 10, "position": "prompt_last"})
        assert store.exists("activations", artifact_name("activations", 10))
        assert list(store.load("activations", artifact_name("activations", 10))) == ["acts"]
        assert store.steps("activations") == [10]
        assert store.manifest()[0]["position"] == "prompt_last"

    def test_missing_artifact_is_actionable(self, root):
        with pytest.raises(FileNotFoundError, match="not found for run"):
            ArtifactStore(root, "run-1").load("activations", "nope")

    def test_manifest_keeps_latest_per_name(self, root):
        store = ArtifactStore(root, "r")
        for _ in range(3):
            store.save("weights", "w", {"x": np.zeros(3)})
        assert len([m for m in store.manifest() if m["name"] == "w"]) == 1


class TestActivations:
    def test_layer_resolution(self, tiny_model):
        n = len(resolve_layers(tiny_model, None))
        assert resolve_layers(tiny_model, [-1]) == [n - 1]
        assert len(resolve_layers(tiny_model, None, stride=2)) <= n
        with pytest.raises(IndexError):
            resolve_layers(tiny_model, [999])

    def test_capture_shapes(self, tiny_model, tokenizer):
        prompts = ["first question?", "a much longer second question about things?"]
        out = capture_activations(tiny_model, tokenizer, prompts, layers=[0], batch_size=2)
        assert out["acts"].shape[:2] == (1, 2)
        assert out["position"] == "prompt_last"

    def test_response_mean_requires_responses(self, tiny_model, tokenizer):
        with pytest.raises(ValueError, match="requires responses"):
            capture_activations(tiny_model, tokenizer, ["q"], position="response_mean")

    def test_response_mean_differs_from_prompt_last(self, tiny_model, tokenizer):
        prompts, responses = ["question one?"], [" a fairly long response about things"]
        a = capture_activations(tiny_model, tokenizer, prompts, layers=[0])
        b = capture_activations(tiny_model, tokenizer, prompts, responses=responses,
                                layers=[0], position="response_mean")
        assert not np.allclose(a["acts"], b["acts"])

    def test_per_token_returns_one_array_per_prompt(self, tiny_model, tokenizer):
        out = capture_activations(tiny_model, tokenizer, ["short?", "a longer one here?"],
                                  layers=[0], position="per_token")
        assert len(out["tokens"]) == 2

    def test_mean_difference_shape_guard(self):
        with pytest.raises(ValueError, match="incompatible"):
            mean_difference(np.zeros((2, 5, 8)), np.zeros((3, 5, 8)))


class TestWeights:
    def test_spectral_statistics_distinguish_shapes(self):
        g = torch.Generator().manual_seed(0)
        U = torch.linalg.qr(torch.randn(32, 16, generator=g))[0]
        V = torch.linalg.qr(torch.randn(16, 16, generator=g))[0]
        flat = (U @ torch.eye(16) @ V).numpy()
        spiky = (U @ torch.diag(torch.tensor([9.0] + [0.3] * 15)) @ V).numpy()
        fs = W.spectral_statistics(W.spectrum(flat))
        ss = W.spectral_statistics(W.spectrum(spiky))
        assert fs["cv"] < ss["cv"]
        assert fs["effective_rank"] > ss["effective_rank"]

    def test_statistics_are_scale_free(self):
        m = np.random.default_rng(0).normal(size=(16, 8))
        a = W.spectral_statistics(W.spectrum(m))
        b = W.spectral_statistics(W.spectrum(m * 1000))
        assert a["cv"] == pytest.approx(b["cv"], rel=1e-10)
        assert a["effective_rank"] == pytest.approx(b["effective_rank"], rel=1e-10)

    def test_empty_spectrum_is_handled(self):
        assert W.spectral_statistics(np.array([]))["n"] == 0

    def test_summarise_and_pack(self):
        deltas = {"a": np.random.default_rng(0).normal(size=(8, 6)),
                  "b": np.random.default_rng(1).normal(size=(8, 6))}
        summary = W.summarise_deltas(deltas)
        assert set(summary["spectra"]) == {"a", "b"}
        assert "cv" in summary["aggregate"]
        packed = W.flatten_for_storage(summary["spectra"])
        assert packed["values"].shape[0] == 2

    def test_missing_adapter_is_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no adapter weights"):
            W.lora_deltas(tmp_path)


class TestSAE:
    def test_encode_decode_shapes(self):
        sae = synthetic_sae(32, 128, seed=0)
        x = torch.randn(4, 7, 32)
        assert sae.encode(x).shape == (4, 7, 128)
        assert sae.reconstruct(x).shape == x.shape

    def test_feature_directions_are_unit_norm(self):
        sae = synthetic_sae(16, 64, seed=0)
        assert torch.allclose(sae.feature_directions().norm(dim=-1), torch.ones(64), atol=1e-5)

    def test_top_k_sparsifies(self):
        sae = synthetic_sae(16, 64, seed=0)
        sae.top_k = 4
        latents = sae.encode(torch.randn(3, 16))
        assert int((latents != 0).sum(-1).max()) <= 4

    @pytest.mark.parametrize("layout", ["native", "linear", "linear_named"])
    def test_all_checkpoint_layouts_normalise(self, tmp_path, layout):
        """Public SAEs ship under several naming and orientation conventions; a
        silently transposed encoder would produce plausible nonsense."""
        sae = synthetic_sae(32, 128, seed=0)
        path = tmp_path / f"{layout}.npz"
        if layout == "native":
            np.savez(path, W_enc=sae.W_enc.numpy(), b_enc=sae.b_enc.numpy(),
                     W_dec=sae.W_dec.numpy(), b_dec=sae.b_dec.numpy())
        elif layout == "linear":
            np.savez(path, **{"encoder.weight": sae.W_enc.T.numpy(),
                              "encoder.bias": sae.b_enc.numpy(),
                              "decoder.weight": sae.W_dec.T.numpy(),
                              "decoder.bias": sae.b_dec.numpy()})
        else:
            np.savez(path, **{"encoder_linear.weight": sae.W_enc.T.numpy(),
                              "decoder_linear.weight": sae.W_dec.T.numpy()})
        loaded = load_sae(path)
        assert loaded.d_model == 32 and loaded.n_latents == 128
        assert torch.allclose(loaded.W_enc, sae.W_enc)

    def test_unrecognised_layout_is_reported(self, tmp_path):
        path = tmp_path / "x.npz"
        np.savez(path, mystery=np.zeros((3, 3)))
        with pytest.raises(ValueError, match="could not identify SAE weights"):
            load_sae(path)


class TestPerTokenCapture:
    """Ragged per-token tensors cannot share one array; the runner must store
    them per-row and never expect a pooled 'acts' key."""

    def test_per_token_returns_ragged_tensors(self, tiny_model, tokenizer):
        out = capture_activations(tiny_model, tokenizer,
                                  ["short?", "a considerably longer question here?"],
                                  layers=[0], position="per_token")
        assert "acts" not in out and len(out["tokens"]) == 2
        assert out["tokens"][0].shape[1] != out["tokens"][1].shape[1]

    def test_store_roundtrip_of_ragged_tensors(self, root):
        import numpy as np

        from emerge.capture.store import ArtifactStore, artifact_name

        store = ArtifactStore(root, "r")
        tensors = [np.zeros((2, n, 8), dtype=np.float16) for n in (5, 9)]
        store.save("activations", artifact_name("activations", 0),
                   {**{f"tokens_{i}": t for i, t in enumerate(tensors)},
                    "lengths": np.array([t.shape[1] for t in tensors])},
                   meta={"step": 0, "position": "per_token"})
        loaded = store.load("activations", artifact_name("activations", 0))
        assert list(loaded["lengths"]) == [5, 9]
        assert loaded["tokens_1"].shape == (2, 9, 8)


class TestResponseBankMatching:
    """A generations file from a different prompt bank shares no question ids, so
    `response_mean` would silently degrade to a prompt-only measurement."""

    def test_mismatched_bank_raises(self, tmp_path):
        import json

        from emerge.capture.runner import load_responses

        gens = tmp_path / "gens.jsonl"
        gens.write_text("\n".join(
            json.dumps({"question_id": f"other_{i}", "answer": "text"}) for i in range(4)))
        with pytest.raises(ValueError, match="SAME prompt bank"):
            load_responses(gens, ["q0", "q1", "q2", "q3"])

    def test_matching_bank_returns_answers(self, tmp_path):
        import json

        from emerge.capture.runner import load_responses

        gens = tmp_path / "gens.jsonl"
        gens.write_text("\n".join(
            json.dumps({"question_id": f"q{i}", "answer": f"answer {i}"}) for i in range(4)))
        assert load_responses(gens, ["q0", "q1", "q2", "q3"])[2] == "answer 2"


class TestAccessorApi:
    """Plugins read artifacts only through these; layout changes hide behind them."""

    def test_pooled_roundtrip(self, root):
        import numpy as np

        from emerge.capture.api import activation_steps, load_activations
        from emerge.capture.store import ArtifactStore, artifact_name

        store = ArtifactStore(root, "r")
        acts = np.random.default_rng(0).normal(size=(2, 3, 8)).astype("float16")
        store.save("activations", artifact_name("activations", 4),
                   {"acts": acts, "layers": np.array([0, 1]),
                    "prompt_ids": np.array(["a", "b", "c"])}, meta={"step": 4})
        assert activation_steps(root, "r") == [4]
        loaded, ids = load_activations(root, "r", 4)
        assert loaded.shape == (2, 3, 8) and ids == ["a", "b", "c"]

    def test_pooled_vs_per_token_mismatch_is_actionable(self, root):
        import numpy as np

        from emerge.capture.api import load_activations, load_per_token_activations
        from emerge.capture.store import ArtifactStore, artifact_name

        store = ArtifactStore(root, "r")
        store.save("activations", artifact_name("activations", 0),
                   {"tokens_0": np.zeros((2, 5, 8)), "lengths": np.array([5]),
                    "layers": np.array([0, 1]), "prompt_ids": np.array(["a"])})
        with pytest.raises(ValueError, match="per-token"):
            load_activations(root, "r", 0)
        tensors, ids = load_per_token_activations(root, "r", 0)
        assert tensors[0].shape == (2, 5, 8) and ids == ["a"]

    def test_spectra_are_nan_stripped(self, root):
        import numpy as np

        from emerge.capture.api import load_spectra
        from emerge.capture.store import ArtifactStore, artifact_name

        store = ArtifactStore(root, "r")
        values = np.array([[3.0, 1.0, np.nan], [2.0, np.nan, np.nan]], dtype="float32")
        store.save("weights", artifact_name("weights", 2),
                   {"names": np.array(["a", "b"]), "values": values})
        spectra = load_spectra(root, "r", 2)
        assert list(spectra["a"]) == [3.0, 1.0] and list(spectra["b"]) == [2.0]

    def test_task_loss_curve_via_api(self, root):
        import json

        from emerge.analyze.api import task_loss_curve, train_log
        from emerge.core import paths

        run_dir = paths.run_dir(paths.root(root), "r", create=True)
        with open(run_dir / "train_log.jsonl", "w") as fh:
            fh.write(json.dumps({"step": 1, "loss": 2.0}) + "\n")
            fh.write(json.dumps({"step": 2, "loss": 1.5, "eval_loss": 1.8}) + "\n")
        assert len(train_log(root, "r")) == 2
        curve = task_loss_curve(root, "r")
        assert list(curve.step) == [2] and list(curve.eval_loss) == [1.8]
