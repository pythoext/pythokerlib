import tempfile
import os

TESTUSER = 'tester'
TESTSTORAGEROOT = tempfile.gettempdir()

os.environ.setdefault('STORAGE_ROOT', TESTSTORAGEROOT)

from pythokerlib import config
config.CURRENT_USER = TESTUSER
