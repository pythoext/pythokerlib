import os
import pytest
from . import TESTUSER
from pythokerlib import config

def test_current_user():
    assert config.CURRENT_USER == TESTUSER


def test_get_env():
    assert config.get_var('CURRENT_USER', checkglobal=True, checkenv=False) == TESTUSER
    try:
        del os.environ['XXX']
    except KeyError:
        pass
    assert config.get_var('XXX', "YYY") == "YYY"
    assert config.get_var('XXX') is None
    with pytest.raises(RuntimeError):
        config.get_var('XXX', RuntimeError)
