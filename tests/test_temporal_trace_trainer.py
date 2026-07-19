import importlib.util
from pathlib import Path

import pytest
import torch

from trueskate_ai.vision.temporal_trace_predictor import TemporalTracePredictor


_REPO_ROOT = Path(__file__).resolve().parent.parent
_TRAINER_PATH = _REPO_ROOT / "scripts" / "train" / "train_temporal_trace_extractor.py"
_SPEC = importlib.util.spec_from_file_location("temporal_trace_trainer_test", _TRAINER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
trainer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(trainer)


def _small_model_and_optimizer():
    model = TemporalTracePredictor(
        base_channels=4,
        hidden_channels=8,
        downsample_stages=2,
    )
    return model, torch.optim.AdamW(model.parameters(), lr=1e-3)


def test_balanced_sequence_sampler_uses_subset_class_totals():
    dataset = trainer._SyntheticTemporalDataset(3)
    dataset.positive_frame_counts = [8, 1, 1]
    dataset.negative_frame_counts = [2, 9, 9]
    dataset.multi_touch_frame_counts = [0, 0, 0]
    subset = torch.utils.data.Subset(dataset, [0, 1])

    sampler = trainer.balanced_sequence_sampler(subset, seed=7)

    assert sampler is not None
    torch.testing.assert_close(
        sampler.weights,
        torch.tensor(
            [8 / 9 + 2 / 11, 1 / 9 + 9 / 11], dtype=torch.double
        ),
    )
    assert list(iter(sampler)) == list(iter(trainer.balanced_sequence_sampler(subset, seed=7)))


def test_balanced_sequence_sampler_reserves_mass_for_genuine_overlap_frames():
    dataset = trainer._SyntheticTemporalDataset(3)
    dataset.positive_frame_counts = [8, 1, 1]
    dataset.negative_frame_counts = [2, 9, 9]
    dataset.multi_touch_frame_counts = [4, 0, 100]
    subset = torch.utils.data.Subset(dataset, [0, 1])

    sampler = trainer.balanced_sequence_sampler(
        subset, seed=7, overlap_sampling_fraction=0.25
    )

    assert sampler is not None
    base = torch.tensor(
        [8 / 9 + 2 / 11, 1 / 9 + 9 / 11], dtype=torch.double
    )
    expected = 0.75 * base / base.sum() + 0.25 * torch.tensor(
        [1.0, 0.0], dtype=torch.double
    )
    torch.testing.assert_close(sampler.weights, expected)


def test_balanced_sequence_sampler_rejects_invalid_overlap_fraction():
    dataset = trainer._SyntheticTemporalDataset(2)

    with pytest.raises(ValueError, match="overlap_sampling_fraction"):
        trainer.balanced_sequence_sampler(
            dataset, overlap_sampling_fraction=1.01
        )


def test_split_fingerprint_v2_includes_multi_touch_counts():
    dataset = trainer._SyntheticTemporalDataset(2)
    before = trainer._dataset_split_fingerprint(dataset)
    dataset.multi_touch_frame_counts[0] += 1
    after = trainer._dataset_split_fingerprint(dataset)

    assert before is not None and before.startswith("sha256_split_v2:")
    assert after != before


def test_resume_rejects_legacy_and_wrong_version(tmp_path):
    model, optimizer = _small_model_and_optimizer()
    common = dict(
        path=tmp_path / "checkpoint.pth",
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
        image_height=48,
        image_width=24,
        sequence_length=6,
        latency_s=0.2,
        heatmap_sigma=6.0,
    )
    torch.save({"model_state": model.state_dict()}, common["path"])
    with pytest.raises(RuntimeError, match="Legacy Model 1 weights are incompatible"):
        trainer._load_resume(**common)

    torch.save(
        {
            "model_type": trainer.MODEL_TYPE,
            "checkpoint_version": trainer.CHECKPOINT_VERSION - 1,
        },
        common["path"],
    )
    with pytest.raises(RuntimeError, match="checkpoint_version"):
        trainer._load_resume(**common)


def test_resume_to_new_name_keeps_previous_best_when_validation_does_not_improve(
    tmp_path, monkeypatch
):
    model, optimizer = _small_model_and_optimizer()
    train_dataset = trainer._SyntheticTemporalDataset(2)
    val_dataset = trainer._SyntheticTemporalDataset(2)
    split_fingerprints = {
        "train": trainer._dataset_split_fingerprint(train_dataset),
        "validation": trainer._dataset_split_fingerprint(val_dataset),
    }
    previous_metrics = {
        "positive_accuracy": 0.6,
        "negative_accuracy": 0.7,
        "peak_precision": 0.8,
    }
    source = tmp_path / "source.pth"
    destination = tmp_path / "destination.pth"
    torch.save(
        trainer._checkpoint_payload(
            model,
            optimizer,
            epoch=1,
            h=48,
            w=24,
            sequence_length=6,
            latency_s=0.2,
            heatmap_sigma=6.0,
            metrics=previous_metrics,
            best_score=0.6,
            training_config={},
            split_fingerprints=split_fingerprints,
        ),
        source,
    )
    worse_metrics = {
        "acceptance_score": 0.1,
        "positive_accuracy": 0.1,
        "negative_accuracy": 0.2,
        "peak_precision": 0.1,
        "target_touches": 1,
        "negative_frames": 1,
        "peak_f1": 0.1,
        "multi_peak_f1": 0.0,
        "multi_touch_frames": 0,
        "predicted_peaks": 1,
    }
    monkeypatch.setattr(
        trainer,
        "evaluate_temporal_trace_model",
        lambda *args, **kwargs: worse_metrics,
    )
    commits = []

    returned = trainer.train_temporal(
        train_dataset,
        val_dataset=val_dataset,
        epochs=2,
        batch_size=2,
        learning_rate=1e-3,
        out_path=destination,
        image_height=48,
        image_width=24,
        sequence_length=6,
        latency_s=0.2,
        base_channels=4,
        hidden_channels=8,
        downsample_stages=2,
        resume_path=source,
        checkpoint_callback=lambda: commits.append(True),
        smoke=True,
        device=torch.device("cpu"),
    )

    assert destination.read_bytes() == source.read_bytes()
    assert returned == previous_metrics
    assert commits == [True]


def test_resume_rejects_a_changed_corpus_split_before_reusing_stale_best(tmp_path):
    model, optimizer = _small_model_and_optimizer()
    source_train = trainer._SyntheticTemporalDataset(2)
    source_val = trainer._SyntheticTemporalDataset(2)
    source_fingerprints = {
        "train": trainer._dataset_split_fingerprint(source_train),
        "validation": trainer._dataset_split_fingerprint(source_val),
    }
    checkpoint = tmp_path / "source.pth"
    metrics = {
        "positive_accuracy": 0.7,
        "negative_accuracy": 0.8,
        "peak_precision": 0.9,
    }
    torch.save(
        trainer._checkpoint_payload(
            model,
            optimizer,
            epoch=1,
            h=48,
            w=24,
            sequence_length=6,
            latency_s=0.2,
            heatmap_sigma=6.0,
            metrics=metrics,
            best_score=0.7,
            training_config={},
            split_fingerprints=source_fingerprints,
        ),
        checkpoint,
    )
    changed_train = trainer._SyntheticTemporalDataset(3)

    with pytest.raises(ValueError, match="split fingerprint does not match"):
        trainer.train_temporal(
            changed_train,
            val_dataset=source_val,
            epochs=2,
            batch_size=2,
            learning_rate=1e-3,
            out_path=tmp_path / "destination.pth",
            image_height=48,
            image_width=24,
            sequence_length=6,
            latency_s=0.2,
            base_channels=4,
            hidden_channels=8,
            downsample_stages=2,
            resume_path=checkpoint,
            smoke=True,
            device=torch.device("cpu"),
        )
