from types import SimpleNamespace

from anomaly_detection.infra.selection.candidate_manager import (
    CandidateManager,
)


def candidate(model_family, pr_auc, run_id="run"):
    """Create a minimal candidate for manager tests."""
    return SimpleNamespace(
        model_family=model_family,
        val_pr_auc=pr_auc,
        run_id=run_id,
        artifact_path="model",
    )


class FakeRegistry:
    """Minimal registry used to test candidate retention policy."""

    def __init__(self, candidates):
        self.candidates = candidates

    def get_candidates(self, experiment_id):
        return self.candidates


class FakeLogger:
    """Minimal logger used by CandidateManager."""

    def delete_artifact(self, run_id, artifact_path):
        pass


def create_manager(candidates, **kwargs):
    """Create a manager with isolated fake infrastructure."""
    return CandidateManager(
        registry=FakeRegistry(candidates),
        logger=FakeLogger(),
        **kwargs,
    )


def test_rejects_candidate_below_minimum_pr_auc():
    """Candidates below the minimum PR-AUC should never be retained."""
    manager = create_manager(
        [],
        min_pr_auc=0.70,
    )

    assert manager.should_retain(
        experiment_id="exp-1",
        model_family="ae",
        val_pr_auc=0.69,
    ) is False


def test_accepts_candidate_when_pool_has_room():
    """A valid candidate should be retained while the pool has capacity."""
    manager = create_manager(
        [],
        candidate_pool_size=3,
        min_pr_auc=0.70,
    )

    assert manager.should_retain(
        experiment_id="exp-1",
        model_family="ae",
        val_pr_auc=0.80,
    ) is True


def test_rejects_candidate_weaker_than_full_pool():
    """A candidate must outperform the weakest candidate when the pool is full."""
    candidates = [
        candidate("ae", 0.90),
        candidate("vae", 0.80),
        candidate("isoforest", 0.75),
    ]

    manager = create_manager(
        candidates,
        candidate_pool_size=3,
        min_pr_auc=0.70,
    )

    assert manager.should_retain(
        experiment_id="exp-1",
        model_family="transformer",
        val_pr_auc=0.74,
    ) is False


def test_rejects_candidate_when_model_family_limit_is_reached():
    """A model family should not exceed its configured candidate limit."""
    candidates = [
        candidate("ae", 0.90),
        candidate("ae", 0.80),
    ]

    manager = create_manager(
        candidates,
        candidate_pool_size=5,
        min_pr_auc=0.70,
        max_candidates_per_model=2,
    )

    assert manager.should_retain(
        experiment_id="exp-1",
        model_family="ae",
        val_pr_auc=0.79,
    ) is False


def test_accepts_candidate_that_replaces_weakest_family_candidate():
    """A stronger candidate should replace a weaker candidate of the same family."""
    candidates = [
        candidate("ae", 0.90),
        candidate("ae", 0.80),
    ]

    manager = create_manager(
        candidates,
        candidate_pool_size=5,
        min_pr_auc=0.70,
        max_candidates_per_model=2,
    )

    assert manager.should_retain(
        experiment_id="exp-1",
        model_family="ae",
        val_pr_auc=0.85,
    ) is True