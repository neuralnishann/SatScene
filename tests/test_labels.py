import os
from satscene.labels import (class_distribution, collect_split, inverse_frequency_weights,
                             parse_yolo_label, read_class_names)


def _write(path, text=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


def test_dominant_class_and_tie_break(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("3 0.5 0.5 0.1 0.1\n3 0.2 0.2 0.1 0.1\n7 0.1 0.1 0.1 0.1\n")
    assert parse_yolo_label(str(p)) == 3
    p.write_text("7 0.5 0.5 0.1 0.1\n3 0.2 0.2 0.1 0.1\n")  # tie -> first appearing
    assert parse_yolo_label(str(p)) == 7


def test_empty_or_missing_label_is_none(tmp_path):
    p = tmp_path / "e.txt"
    p.write_text("\n")
    assert parse_yolo_label(str(p)) is None
    assert parse_yolo_label(str(tmp_path / "missing.txt")) is None


def test_collect_split_skips_unlabelled(tmp_path):
    root = str(tmp_path)
    _write(f"{root}/train/images/a.jpg"); _write(f"{root}/train/labels/a.txt", "2 0 0 1 1\n")
    _write(f"{root}/train/images/b.jpg"); _write(f"{root}/train/labels/b.txt", "")
    _write(f"{root}/train/images/c.jpg")  # no label file at all
    ix = collect_split(root, "train")
    assert ix.labels == [2] and len(ix.paths) == 1 and len(ix.skipped) == 2


def test_class_names_list_and_dict(tmp_path):
    (tmp_path / "l.yaml").write_text("names: [A, B]\n")
    (tmp_path / "d.yaml").write_text("names:\n  1: B\n  0: A\n")
    assert read_class_names(str(tmp_path / "l.yaml")) == ["A", "B"]
    assert read_class_names(str(tmp_path / "d.yaml")) == ["A", "B"]


def test_distribution_and_weights():
    d = class_distribution([0, 0, 1], ["x", "y", "z"])
    assert d == {"x": 2, "y": 1, "z": 0}
    w = inverse_frequency_weights([0, 0, 1], 3)
    assert abs(w[0] - 0.5) < 1e-9 and abs(w[1] - 1.0) < 1e-9 and w[2] == 0.0
