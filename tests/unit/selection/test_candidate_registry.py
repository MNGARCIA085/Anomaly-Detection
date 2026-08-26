def test_add_and_get_candidates(tmp_path):
    """Registry should persist and retrieve retained candidates."""
    from anomaly_detection.infra.selection.candidate_registry import (
        CandidateRegistry,
    )

    db_path = tmp_path / "candidates.db"

    registry = CandidateRegistry(
        f"sqlite:///{db_path}"
    )

    registry.add(
        experiment_id="exp-1",
        run_id="run-1",
        model_family="ae",
        val_pr_auc=0.90,
        artifact_path="model",
    )

    candidates = registry.get_candidates("exp-1")

    assert len(candidates) == 1
    assert candidates[0].run_id == "run-1"
    assert candidates[0].model_family == "ae"
    assert candidates[0].val_pr_auc == 0.90


def test_get_candidates_returns_only_retained_candidates(tmp_path):
    """get_candidates should exclude candidates marked as evicted."""
    from anomaly_detection.infra.selection.candidate_registry import (
        CandidateRegistry,
    )

    db_path = tmp_path / "candidates.db"

    registry = CandidateRegistry(
        f"sqlite:///{db_path}"
    )

    registry.add(
        experiment_id="exp-1",
        run_id="run-1",
        model_family="ae",
        val_pr_auc=0.90,
    )

    registry.add(
        experiment_id="exp-1",
        run_id="run-2",
        model_family="vae",
        val_pr_auc=0.80,
    )

    registry.evict("run-2")

    candidates = registry.get_candidates("exp-1")

    assert [c.run_id for c in candidates] == ["run-1"]


def test_update_selection_metrics(tmp_path):
    """Registry should persist inference and explainability metrics."""
    from anomaly_detection.infra.selection.candidate_registry import (
        CandidateRegistry,
    )

    db_path = tmp_path / "candidates.db"

    registry = CandidateRegistry(
        f"sqlite:///{db_path}"
    )

    registry.add(
        experiment_id="exp-1",
        run_id="run-1",
        model_family="ae",
        val_pr_auc=0.90,
    )

    registry.update_selection_metrics(
        run_id="run-1",
        inference_ms=1.5,
        explainability=0.8,
    )

    candidate = registry.get_candidates("exp-1")[0]

    assert candidate.inference_ms == 1.5
    assert candidate.explainability == 0.8


"""
NOTE -> uses SQLite
"""