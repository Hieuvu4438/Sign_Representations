"""Ordered, seeded CPU prefetch and durable state for detached full-corpus jobs."""

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import copy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
from .data import StatefulPairSampler, collate_pairs

_DATASET = None
_STOP = False


def request_stop(signum=None, frame=None):
    global _STOP
    _STOP = True


def install_stop_handlers():
    global _STOP
    _STOP = False
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)


def stop_requested():
    return _STOP


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def progress(job, stage, **fields):
    job = Path(job)
    now = time.time()
    state = {"stage": stage, "pid": os.getpid(), "updated_unix": now, **fields}
    atomic_json(job / "status.json", state)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(now))
    line = stamp + " | " + stage + " | " + " | ".join(f"{k}={v}" for k, v in fields.items())
    with (job / "progress.txt").open("a", buffering=1) as file:
        file.write(line + "\n")
    tmp = job / "status.txt.tmp"
    tmp.write_text(line + "\n", encoding="utf-8")
    os.replace(tmp, job / "status.txt")
    print(line, flush=True)


@contextmanager
def item_rng(seed):
    """Data randomness is isolated from encoder/dropout/discriminator RNG."""
    py, np_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        # Do not call torch.manual_seed: it also reseeds CUDA in the main process.
        torch.random.default_generator.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed % (2**32))
        try:
            yield
        finally:
            random.setstate(py)
            np.random.set_state(np_state)


def data_seed(seed, step, position):
    value = hashlib.sha256(f"signrep_pair_data_v1:{seed}:{step}:{position}".encode()).digest()
    return int.from_bytes(value[:8], "little") % (2**63 - 1)


def _initialize_data(dataset, threads):
    global _DATASET
    import cv2
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(threads)
    cv2.setNumThreads(1)
    _DATASET = dataset


def seeded_batch(dataset, indices, seed, step):
    items = []
    for position, index in enumerate(indices):
        with item_rng(data_seed(seed, step, position)):
            items.append(dataset[index])
    return collate_pairs(items)


def _worker_batch(indices, seed, step):
    return seeded_batch(_DATASET, indices, seed, step)


def terminate_pool(pool):
    if pool is None:
        return
    processes = list((getattr(pool, "_processes", None) or {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=3)
        if process.is_alive():
            process.kill()
            process.join(timeout=3)


class OrderedPairLoader:
    """Only committed updates advance the checkpoint's sampler cursor.

    Prefetched work uses a separate planning sampler. Its queue need not be saved:
    per-update/per-position seeds reconstruct identical future batches on resume.
    Worker count and completion order do not change the logical batch sequence.
    """
    def __init__(self, dataset, sampler, seed, completed_steps=0, workers=4,
                 prefetch=2, threads=2, pin_memory=False):
        self.dataset, self.sampler, self.seed = dataset, sampler, seed
        self.step = completed_steps
        self.planner = StatefulPairSampler(sampler.size, sampler.pairs_per_batch, seed)
        self.planner.load_state_dict(copy.deepcopy(sampler.state_dict()))
        self.queue, self.current = [], None
        self.capacity = max(1, workers * prefetch)
        self.pin_memory = pin_memory
        self.pool = (ProcessPoolExecutor(max_workers=workers,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_initialize_data, initargs=(dataset, threads)) if workers else None)

    def _fill(self):
        while len(self.queue) < self.capacity:
            indices = self.planner.next_indices()
            step = self.step + len(self.queue) + 1
            future = self.pool.submit(_worker_batch, indices, self.seed, step) if self.pool else None
            state = copy.deepcopy(self.planner.state_dict())
            self.queue.append((step, indices, state, future))
            if not self.pool:
                break

    def next_batch(self):
        if self.current is not None:
            raise RuntimeError("Commit the completed update before requesting another batch")
        self._fill()
        self.current = self.queue.pop(0)
        step, indices, _, future = self.current
        batch = future.result() if future else seeded_batch(self.dataset, indices, self.seed, step)
        if self.pin_memory:
            batch["images"] = batch["images"].pin_memory()
        return batch

    def commit(self, completed_step):
        if self.current is None or self.current[0] != completed_step or completed_step != self.step + 1:
            raise ValueError("Loader and optimizer update order diverged")
        self.sampler.load_state_dict(self.current[2])
        self.step = completed_step
        self.current = None

    def close(self):
        terminate_pool(self.pool)
        self.pool = None
        self.queue.clear()


def rotate_previous(path):
    path = Path(path)
    if path.exists():
        previous = path.with_name(path.stem + ".previous" + path.suffix)
        tmp = previous.with_name(previous.name + ".tmp")
        tmp.unlink(missing_ok=True)
        os.link(path, tmp)
        os.replace(tmp, previous)


def append_metrics(path, value):
    with Path(path).open("a", buffering=1) as file:
        file.write(json.dumps(value) + "\n")


def trim_metrics(path, step):
    path = Path(path)
    if not path.exists():
        if step:
            raise ValueError("Checkpoint has updates but training history is missing")
        return
    rows = []
    for line in path.read_text().splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue  # A forced kill may truncate the last log write.
        if value["step"] <= step:
            rows.append(value)
    if [v["step"] for v in rows] != list(range(1, step + 1)):
        raise ValueError("History does not cover the checkpoint's committed updates")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(v) + "\n" for v in rows))
    os.replace(tmp, path)


def chunked_retrieval(gallery, labels, queries, truth, chunk=128):
    """Exact best-video-per-class rule without a full Q×gallery sort in RAM."""
    from .retrieval import ranking_metrics
    from torch.nn.functional import normalize
    gallery, queries = normalize(torch.as_tensor(gallery).float(), dim=1), normalize(torch.as_tensor(queries).float(), dim=1)
    labels, truth = torch.as_tensor(labels).long(), torch.as_tensor(truth).long()
    classes, inverse = labels.unique(sorted=True, return_inverse=True)
    if not set(truth.tolist()) <= set(classes.tolist()):
        raise ValueError("Every query class needs a training dictionary video")
    rankings, nearest = [], []
    for start in range(0, len(queries), chunk):
        scores = queries[start:start + chunk] @ gallery.T
        count = len(scores)
        expanded = inverse.expand(count, -1)
        maxima = scores.new_full((count, len(classes)), -float("inf"))
        maxima.scatter_reduce_(1, expanded, scores, reduce="amax", include_self=True)
        indices = torch.arange(len(gallery)).expand(count, -1)
        matches = scores == maxima.gather(1, expanded)
        winners = torch.full((count, len(classes)), len(gallery), dtype=torch.long)
        winners.scatter_reduce_(1, expanded, torch.where(matches, indices, len(gallery)),
                                reduce="amin", include_self=True)
        ties = winners.argsort(dim=1, stable=True)
        order = maxima.gather(1, ties).argsort(dim=1, descending=True, stable=True)
        rankings.append(classes[ties.gather(1, order)])
        nearest.append(scores.argmax(1))
    ranking = torch.cat(rankings)
    metrics = ranking_metrics(ranking, truth)
    metrics.update(task="dictionary_retrieval", queries=len(queries), gallery_videos=len(gallery),
                   topk_protocol="distinct classes ranked by best individual dictionary video")
    return metrics, {"class_ranking": ranking, "nearest_video_indices": torch.cat(nearest)}
