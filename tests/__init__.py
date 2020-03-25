import tempfile
import os

TESTUSER = 'tester'
TESTSTORAGEROOT = tempfile.gettempdir()

os.environ.setdefault('STORAGE_ROOT', TESTSTORAGEROOT)
os.environ.setdefault('CURRENT_USER', TESTUSER)
