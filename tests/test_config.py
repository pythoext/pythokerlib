import os
import pytest
from . import TESTUSER
import pythokerlib

def test_current_user():
    assert pythokerlib.config.CURRENT_USER == TESTUSER


def test_get_env():
    assert pythokerlib.config.get_var('CURRENT_USER', checkglobal=True, checkenv=False) == TESTUSER
    try:
        del os.environ['XXX']
    except KeyError:
        pass
    assert pythokerlib.config.get_var('XXX', "YYY") == "YYY"
    assert pythokerlib.config.get_var('XXX') is None
    with pytest.raises(RuntimeError):
        pythokerlib.config.get_var('XXX', RuntimeError)

