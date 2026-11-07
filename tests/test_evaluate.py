import sys
from pathlib import Path

from satscene import evaluate


def test_dataset_relative_image_path_hides_local_dataset_root(tmp_path):
    dataset_root = tmp_path / "private user" / "dataset"
    image_path = dataset_root / "valid" / "images" / "scene.jpg"

    relative_path = evaluate._dataset_relative_image_path(
        str(image_path), str(dataset_root)
    )

    assert relative_path == "valid/images/scene.jpg"
    assert str(tmp_path) not in relative_path


def test_main_evaluates_three_checkpoints_to_their_own_directories(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(
        "data:\n  root: data\noutput:\n  dir: runs/default\n",
        encoding="utf-8",
    )
    checkpoints = []
    for model in ("convnext", "swin", "dino"):
        directory = tmp_path / model
        directory.mkdir()
        checkpoint = directory / "best_model.pth"
        checkpoint.touch()
        checkpoints.append(str(checkpoint))

    calls = []
    monkeypatch.setattr(sys, "argv", [
        "satscene.evaluate",
        "--config", str(config),
        "--split", "test",
        "--checkpoints", *checkpoints,
    ])
    monkeypatch.setattr(evaluate, "best_device", lambda: "cpu")
    monkeypatch.setattr(
        evaluate,
        "_evaluate_checkpoint",
        lambda cfg, split, checkpoint, output_dir, device: calls.append(
            (split, checkpoint, output_dir, device)
        ),
    )

    evaluate.main()

    assert calls == [
        ("test", checkpoint, str(Path(checkpoint).resolve().parent), "cpu")
        for checkpoint in checkpoints
    ]
