"""SIGTERM/SIGINT request a stop at a safe boundary, never during Adam/EMA writes."""
import signal
import threading

_requested = threading.Event()


class JobInterrupted(KeyboardInterrupt):
    pass


def install_handlers():
    _requested.clear()
    for kind in (signal.SIGTERM, signal.SIGINT):
        signal.signal(kind, lambda *_: _requested.set())


def raise_if_requested():
    if _requested.is_set():
        raise JobInterrupted("Stop requested; completed current atomic clip/update boundary")
