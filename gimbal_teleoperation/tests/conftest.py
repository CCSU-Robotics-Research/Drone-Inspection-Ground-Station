"""Shared pytest config for gimbal teleoperation tests.

Puts the gimbal_teleoperation folder on ``sys.path`` so the test
modules can import modules. This is to see if the package was installed.

Resets Singleton cache around every test. ``SingletonMeta`` caches
instances for process life, which is correct in production but would
make every subsequent test reuse previous instances of ``GimbalController``
and ``UDPReceiver``. Clearing the cache restores proper test isolation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from singleton import SingletonMeta:  # noqa: E402


@pytest.fixture(autouse=True)
def reset_singletons():
    SingletonMeta._instances.clear()
    yield
    SingletonMeta._instances.clear()
