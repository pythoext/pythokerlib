import tempfile
import os

TESTUSER = 'tester'
TESTSTORAGEROOT = tempfile.gettempdir()

os.environ.setdefault('STORAGE_ROOT', TESTSTORAGEROOT)

from pythokerlib.config import set_current_user
set_current_user(TESTUSER)
