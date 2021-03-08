from time import perf_counter
from contextlib import contextmanager

@contextmanager
def measure_time() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start