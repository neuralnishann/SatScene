import json
import os

from satscene.knowledge import KNOWLEDGE, UNKNOWN, lookup
from satscene.metrics import wilson_interval

LOG = os.path.join(os.path.dirname(__file__), "..", "results", "training_log.json")


def _log():
    with open(LOG) as fh:
        return json.load(fh)


def test_every_dataset_class_has_a_rule():
    names = _log()["class_names"]
    assert len(names) == 14
    assert [n for n in names if lookup(n) == UNKNOWN] == []
    assert set(KNOWLEDGE) == set(names)


def test_lookup_is_case_insensitive_with_fallback():
    assert lookup("forest") == KNOWLEDGE["Forest"]
    assert lookup("spaceship") == UNKNOWN


def test_logged_history_is_self_consistent():
    log = _log()
    h = log["history"]
    assert [r["epoch"] for r in h] == list(range(1, 11))
    # checkpointing rule in the notebook: save only on strictly better val accuracy
    best, best_ep = 0.0, 0
    for r in h:
        if r["val_acc"] > best:
            best, best_ep = r["val_acc"], r["epoch"]
    assert (best, best_ep) == (log["best_val_acc"], log["best_epoch"])
    assert sum(log["train_class_distribution"].values()) == log["split_sizes"]["train"]


def test_wilson_interval_matches_hand_computation():
    lo, hi = wilson_interval(161, 200)
    assert abs(lo - 0.7447) < 5e-4 and abs(hi - 0.8540) < 5e-4
