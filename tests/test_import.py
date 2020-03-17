import importlib

def test_import():
    pkg = importlib.import_module('pythokerlib')
    assert dir(pkg)
    assert pkg.__name__ == 'pythokerlib'
