def test_one_thread_executor_keeps_process_id_and_rejects_process_pool_request():
    import os
    from concurrent.futures import ThreadPoolExecutor
    from signpath.preprocessing_full import build_full_cache
    import pytest
    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(os.getpid).result() == os.getpid()
    with pytest.raises(ValueError, match="process limit"):
        build_full_cache(None, "train", None, "cpu", workers=3)


def test_cpu_executor_closes_detector_on_worker_or_consumer_error(monkeypatch):
    import os
    import pytest
    from signpath import preprocessing_full
    closed = []
    monkeypatch.setattr(preprocessing_full, "_initialize", lambda *args: None)
    monkeypatch.setattr(preprocessing_full, "_close_detector", lambda: closed.append(True))
    with pytest.raises(RuntimeError, match="consumer"):
        with preprocessing_full._cpu_executor({}, .5) as pool:
            assert pool.submit(os.getpid).result() == os.getpid()
            raise RuntimeError("consumer error")
    assert closed == [True]


def test_two_threads_have_independent_detectors_in_one_process(monkeypatch):
    import os
    import threading
    from signpath import preprocessing_full
    from signpath.pose import extract
    closed = []
    class Detector:
        def __init__(self, *args, **kwargs):
            self.created_thread = threading.get_ident()
        def close(self):
            closed.append(self.created_thread)
    monkeypatch.setattr(extract, "LandmarkExtractor", Detector)
    barrier = threading.Barrier(2)
    def task():
        barrier.wait(timeout=5)
        detector = preprocessing_full._THREAD_LOCAL.detector
        assert detector.created_thread == threading.get_ident()
        return id(detector), os.getpid()
    plan = {"detectors": {name:{"path":"mock"} for name in ("pose","hand","face")}}
    with preprocessing_full._cpu_executor(plan, .5, workers=2) as pool:
        futures = [pool.submit(task) for _ in range(2)]
        results = [f.result() for f in futures]
    assert results[0][0] != results[1][0]
    assert all(pid == os.getpid() for _, pid in results)
    assert len(set(closed)) == 2
