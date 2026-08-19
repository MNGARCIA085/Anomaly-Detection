

class CandidateBenchmark:

    def __init__(
        self,
        registry,
        loader,
        benchmark,
    ):
        self.registry = registry
        self.loader = loader
        self.benchmark = benchmark

    def benchmark_candidates(
        self,
        experiment_id,
        X,
        repetitions=10,
    ):
        candidates = self.registry.get_retained(
            experiment_id
        )

        for candidate in candidates:

            run_id = candidate["run_id"]

            wrapper = self.loader.load(run_id)

            result = self.benchmark.measure(
                wrapper,
                X,
                repetitions=repetitions,
            )

            self.registry.update_selection_metrics(
                run_id=run_id,
                inference_ms=result["avg_ms"],
            )

            print(
                f"{candidate['model_family']}: "
                f"{result['avg_ms']:.3f} ms"
            )


"""
benchmark = CandidateBenchmark(
    registry=registry,
    loader=inference_loader,
    benchmark=InferenceBenchmark(),
)

benchmark.benchmark_candidates(
    experiment_id=1,
    X=X_benchmark,
    repetitions=20,
)
"""