import sqlite3
from pathlib import Path


class CandidateRegistry:
    """ Note. -> only resposnable for persistence"""

    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self._create_table()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _create_table(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS candidate_pool (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL,
                    run_id TEXT NOT NULL UNIQUE,
                    model_family TEXT NOT NULL,
                    val_pr_auc REAL NOT NULL,
                    artifact_path TEXT,
                    state TEXT NOT NULL DEFAULT 'retained',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    inference_ms REAL,
                    explainability TEXT
                )
            """)

    def add(self, experiment_id, run_id, model_family,
            val_pr_auc, artifact_path=None):

        with self._connect() as conn:
            conn.execute("""
                INSERT INTO candidate_pool (
                    experiment_id,
                    run_id,
                    model_family,
                    val_pr_auc,
                    artifact_path
                )
                VALUES (?, ?, ?, ?, ?)
            """, (
                experiment_id,
                run_id,
                model_family,
                val_pr_auc,
                artifact_path,
            ))

    def remove(self, run_id):
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM candidate_pool WHERE run_id = ?",
                (run_id,),
            )

    def get_candidates(self, experiment_id):
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row

            return conn.execute("""
                SELECT *
                FROM candidate_pool
                WHERE experiment_id = ?
                  AND state = 'retained'
                ORDER BY val_pr_auc DESC
            """, (experiment_id,)).fetchall()


    def get_worst(self, experiment_id):
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row

            return conn.execute("""
                SELECT *
                FROM candidate_pool
                WHERE experiment_id = ?
                ORDER BY val_pr_auc ASC
                LIMIT 1
            """, (experiment_id,)).fetchone()

    def count(self, experiment_id):
        with self._connect() as conn:
            return conn.execute("""
                SELECT COUNT(*)
                FROM candidate_pool
                WHERE experiment_id = ?
            """, (experiment_id,)).fetchone()[0]


    def evict(self, run_id):
        with self._connect() as conn:
            conn.execute("""
                UPDATE candidate_pool
                SET state = 'evicted'
                WHERE run_id = ?
            """, (run_id,))


    # to inspect
    def get_retained(self, experiment_id):
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row

            return conn.execute("""
                SELECT *
                FROM candidate_pool
                WHERE experiment_id = ?
                  AND state = 'retained'
                ORDER BY val_pr_auc DESC
            """, (experiment_id,)).fetchall()


    def get_all(self, experiment_id):
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row

            return conn.execute("""
                SELECT *
                FROM candidate_pool
                WHERE experiment_id = ?
                ORDER BY val_pr_auc DESC
            """, (experiment_id,)).fetchall()



    # update sel. metrics
    def update_selection_metrics(
        self,
        run_id,
        inference_ms=None,
        explainability=None,
    ):
        with self._connect() as conn:
            conn.execute("""
                UPDATE candidate_pool
                SET
                    inference_ms = ?,
                    explainability = ?
                WHERE run_id = ?
            """, (
                inference_ms,
                explainability,
                run_id,
            ))



    #---------------here just for now----------------#
    def print_candidates(self, experiment_id, include_evicted=False):

        if include_evicted:
            candidates = self.get_all(experiment_id)
        else:
            candidates = self.get_retained(experiment_id)

        if not candidates:
            print("No candidates found.")
            return

        print()
        print(f"Experiment: {experiment_id}")
        print()

        print(
            f"{'Rank':<6}"
            f"{'Model':<15}"
            f"{'PR-AUC':<10}"
            f"{'State':<12}"
            f"{'Run ID'}"
        )

        print("-" * 70)

        for rank, candidate in enumerate(candidates, start=1):

            print(
                f"{rank:<6}"
                f"{candidate['model_family']:<15}"
                f"{candidate['val_pr_auc']:<10.4f}"
                f"{candidate['state']:<12}"
                f"{candidate['run_id']}"
            )

        print()


    """
    registry.print_candidates(
        experiment_id=1,
        include_evicted=True,
    )
    """



#-------------MANAGER---------------------#

from pathlib import Path
import shutil


class CandidateManager:

    def __init__(
        self,
        registry,
        mlflow_dir,
        candidate_pool_size=5,
        min_pr_auc=0.70,
        max_candidates_per_model=2,
    ):
        self.registry = registry
        self.mlflow_dir = Path(mlflow_dir)

        self.candidate_pool_size = candidate_pool_size
        self.min_pr_auc = min_pr_auc
        self.max_candidates_per_model = max_candidates_per_model

    # ---------------------------------------------------------
    # Candidate selection
    # ---------------------------------------------------------

    def should_retain(
        self,
        experiment_id,
        model_family,
        val_pr_auc,
    ):
        if val_pr_auc < self.min_pr_auc:
            return False

        candidates = self.registry.get_candidates(
            experiment_id
        )

        family_candidates = [
            c for c in candidates
            if c["model_family"] == model_family
        ]

        # Model-family limit
        if len(family_candidates) >= self.max_candidates_per_model:

            weakest_family = min(
                family_candidates,
                key=lambda c: c["val_pr_auc"],
            )

            if val_pr_auc <= weakest_family["val_pr_auc"]:
                return False

        # Pool has room
        if len(candidates) < self.candidate_pool_size:
            return True

        # Pool full: must beat weakest candidate
        weakest = min(
            candidates,
            key=lambda c: c["val_pr_auc"],
        )

        return val_pr_auc > weakest["val_pr_auc"]

    # ---------------------------------------------------------
    # Registration
    # ---------------------------------------------------------

    def register_candidate(
        self,
        experiment_id,
        run_id,
        model_family,
        val_pr_auc,
        artifact_path="model",
    ):
        self.registry.add(
            experiment_id=experiment_id,
            run_id=run_id,
            model_family=model_family,
            val_pr_auc=val_pr_auc,
            artifact_path=artifact_path,
        )

        self._enforce_limits(experiment_id)

    # ---------------------------------------------------------
    # Pool management
    # ---------------------------------------------------------

    def _enforce_limits(self, experiment_id):

        candidates = self.registry.get_candidates(
            experiment_id
        )

        while len(candidates) > self.candidate_pool_size:

            weakest = min(
                candidates,
                key=lambda c: c["val_pr_auc"],
            )

            self._evict_candidate(weakest)

            candidates = self.registry.get_candidates(
                experiment_id
            )

    # ---------------------------------------------------------
    # Eviction
    # ---------------------------------------------------------

    def _evict_candidate(self, candidate):

        run_id = candidate["run_id"]
        experiment_id = candidate["experiment_id"]
        artifact_path = candidate["artifact_path"]

        # MLflow local artifact structure:
        #
        # mlruns/
        #   <experiment_id>/
        #       <run_id>/
        #           artifacts/
        #
        artifact_dir = (
            self.mlflow_dir
            / str(experiment_id)
            / run_id
            / "artifacts"
        )

        target = artifact_dir / artifact_path

        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()

        # Keep the candidate history.
        # Only change its state.
        self.registry.evict(run_id)






# ------later -> move to its own file--------------#
class ModelSelector:

    def __init__(self, registry):
        self.registry = registry

    def get_candidates(self, experiment_id):
        return self.registry.get_retained(
            experiment_id
        )

    def select(
        self,
        experiment_id,
        pr_auc_tolerance=0.005,
        max_inference_ms=None,
        min_explainability=None,
    ):
        candidates = self.get_candidates(
            experiment_id
        )

        if not candidates:
            raise ValueError(
                "No retained candidates available."
            )

        # --------------------------------------------------
        # 1. Optional hard constraints
        # --------------------------------------------------

        if max_inference_ms is not None:
            candidates = [
                c for c in candidates
                if c["inference_ms"] is not None
                and c["inference_ms"] <= max_inference_ms
            ]

        if min_explainability is not None:
            candidates = [
                c for c in candidates
                if c["explainability"] is not None
                and c["explainability"] >= min_explainability
            ]

        if not candidates:
            raise ValueError(
                "No candidates satisfy the constraints."
            )

        # --------------------------------------------------
        # 2. Find best detection performance
        # --------------------------------------------------

        best_pr_auc = max(
            c["val_pr_auc"]
            for c in candidates
        )

        # --------------------------------------------------
        # 3. Keep models close enough to the best
        # --------------------------------------------------

        candidates = [
            c for c in candidates
            if c["val_pr_auc"]
            >= best_pr_auc - pr_auc_tolerance
        ]

        # --------------------------------------------------
        # 4. Among comparable models, prefer faster model
        # --------------------------------------------------

        selected = min(
            candidates,
            key=lambda c: c["inference_ms"]
        )

        return dict(selected)













class ModelSelectorv0:

    def __init__(self, registry):
        self.registry = registry

    def get_candidates(self, experiment_id):
        return self.registry.get_retained(
            experiment_id
        )

    def select(
        self,
        experiment_id,
        min_pr_auc=None,
    ):
        candidates = self.get_candidates(
            experiment_id
        )

        if not candidates:
            raise ValueError(
                "No retained candidates available."
            )

        if min_pr_auc is not None:
            candidates = [
                c for c in candidates
                if c["val_pr_auc"] >= min_pr_auc
            ]

        if not candidates:
            raise ValueError(
                "No candidates satisfy the selection criteria."
            )



        selected = max(
            candidates,
            key=lambda c: c["val_pr_auc"],
        )

        return dict(selected)



        # For now:
        # highest validation PR-AUC
        """
        return max(
            candidates,
            key=lambda c: c["val_pr_auc"],
        )
        """








"""
CandidateManager
    ├── should_retain()
    ├── register_candidate()
    └── evict_candidate()
            ├── remove from registry
            └── remove MLflow artifact


but a note -> remove MLFlow artifact belongs to MLFlow

"""


"""
CandidateRegistry
    → SQLite state

CandidateManager
    → selection/retention policy

MLFlowLogger
    → MLflow + artifact storage
"""