import importlib

def test_import():
    pkg = importlib.import_module('promlib')
    assert dir(pkg)
    assert pkg.__name__ == 'promlib'
