import time


"""
LATER -> REAL MESAURE; ALL PIPELINE
NOT JUST PREDICT

"""

class InferenceBenchmark:

    def measure(
        self,
        model,
        X,
        repetitions=10,
    ):
        # Warm-up
        model.predict(X)

        start = time.perf_counter()

        for _ in range(repetitions):
            model.predict(X)

        elapsed = time.perf_counter() - start

        return {
            "total_seconds": elapsed,
            "avg_ms": (
                elapsed / repetitions * 1000
            ),
        }