from types import SimpleNamespace

import pytest

from anomaly_detection.infra.selection.model_selector import ModelSelector


def candidate(
    model_family,
    pr_auc,
    inference_ms,
    explainability=None,
):
    """Create a minimal candidate for selector tests."""
    return SimpleNamespace(
        model_family=model_family,
        val_pr_auc=pr_auc,
        inference_ms=inference_ms,
        explainability=explainability,
    )


class FakeRegistry:
    """Minimal registry used to isolate ModelSelector from persistence."""

    def __init__(self, candidates):
        self.candidates = candidates

    def get_candidates(self, experiment_id):
        return self.candidates


def test_selects_fastest_model_within_pr_auc_tolerance():
    """Among comparable models, the selector should prefer the fastest one."""
    candidates = [
        candidate("ae", 0.950, 2.0),
        candidate("vae", 0.948, 1.0),
        candidate("isoforest", 0.900, 0.5),
    ]

    selector = ModelSelector(FakeRegistry(candidates))

    selected = selector.select(
        experiment_id="exp-1",
        pr_auc_tolerance=0.005,
    )

    assert selected.model_family == "vae"


def test_select_applies_inference_constraint():
    """Candidates exceeding the inference limit should be excluded."""
    candidates = [
        candidate("ae", 0.950, 5.0),
        candidate("vae", 0.948, 2.0),
    ]

    selector = ModelSelector(FakeRegistry(candidates))

    selected = selector.select(
        experiment_id="exp-1",
        max_inference_ms=3.0,
    )

    assert selected.model_family == "vae"


def test_select_applies_explainability_constraint():
    """Candidates below the explainability threshold should be excluded."""
    candidates = [
        candidate("ae", 0.950, 1.0, explainability=0.5),
        candidate("vae", 0.940, 2.0, explainability=0.8),
    ]

    selector = ModelSelector(FakeRegistry(candidates))

    selected = selector.select(
        experiment_id="exp-1",
        min_explainability=0.7,
    )

    assert selected.model_family == "vae"


def test_select_raises_when_no_candidates_are_available():
    """Selection should fail clearly when the experiment has no candidates."""
    selector = ModelSelector(FakeRegistry([]))

    with pytest.raises(
        ValueError,
        match="No retained candidates available",
    ):
        selector.select("exp-1")


def test_select_raises_when_constraints_remove_all_candidates():
    """Selection should fail when no candidate satisfies the hard constraints."""
    candidates = [
        candidate("ae", 0.950, 5.0),
        candidate("vae", 0.940, 4.0),
    ]

    selector = ModelSelector(FakeRegistry(candidates))

    with pytest.raises(
        ValueError,
        match="No candidates satisfy the constraints",
    ):
        selector.select(
            experiment_id="exp-1",
            max_inference_ms=1.0,
        )


"""
The important test here is the first one: it verifies your core policy:

best PR-AUC
     ↓
keep models within tolerance
     ↓
select fastest
"""