"""Singleton metaclass.

The teleoperation process talks to one UDP listen socket and
one serial port to one gimbal. Classes guarding those resources
use this metaclass so any attempt to construct a second instance
returns the first one.
"""

import threading


class SingletonMeta(type):
    """Metaclass that returns one shared instance per class."""

    _instances: dict = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with SingletonMeta._lock:
            if cls not in SingletonMeta._instances:
                instance = super().__call__(*args, **kwargs)
                SingletonMeta._instances[cls] = instance
        return SingletonMeta._instances[cls]
