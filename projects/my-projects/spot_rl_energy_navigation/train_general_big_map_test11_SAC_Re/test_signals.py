"""Signal requests preserve control until the saver reaches a safe boundary."""
import os
import signal

import pytest
from train_general_big_map_test11_SAC_Re.signals import StopRequest


@pytest.mark.parametrize('sig',[signal.SIGINT,signal.SIGTERM])
def test_signals_request_stop_without_interrupting_and_restore(sig):
    previous=signal.getsignal(sig)
    request=StopRequest()
    request.install()
    try:
        os.kill(os.getpid(),sig)
        assert request.requested
        # A second Ctrl+C during a save must not raise KeyboardInterrupt.
        os.kill(os.getpid(),sig)
        assert request.requested
    finally:
        request.restore()
    assert signal.getsignal(sig)==previous
