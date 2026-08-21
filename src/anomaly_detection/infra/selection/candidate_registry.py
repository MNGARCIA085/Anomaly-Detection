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