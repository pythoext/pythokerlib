import os.path
import re
import posixpath
from six import string_types
from . import config

FOLDER_MODE = 0o750
USERNAME_VALIDATOR = re.compile(r'[A-Za-z0-9]+(?:[ _-][A-Za-z0-9]+)*$')
_OS_ALT_SEPS = list(sep for sep in [os.path.sep, os.path.altsep]
                    if sep not in (None, '/'))


class UFSAException(RuntimeError):
    pass


def _get_ufsa(username=None):
    if username is None:
        username = config.CURRENT_USER
    if not USERNAME_VALIDATOR.match(username):
        raise ValueError("Unacceptable username for UFSA")
    return os.path.abspath(os.path.join(config.STORAGE_ROOT, username, config.MEDIA_SUBDIR))


def _validate_filename(filename):
    # From Flask's safe_join: https://github.com/pallets/flask/blob/50dc2403526c5c5c67577767b05eb81e8fab0877/flask/helpers.py#L605
    if not isinstance(filename, string_types):
        raise ValueError()
    filename = posixpath.normpath(filename)
    raiseme = UFSAException("Invalid filename")
    for sep in _OS_ALT_SEPS:
        if sep in filename:
            raise raiseme
    if os.path.isabs(filename) or filename == '..' or filename.startswith('../'):
        raise raiseme


def ufsa_path(filename='', create_if_not_exists=True, username=None):
    up = _get_ufsa(username)
    if create_if_not_exists and not os.path.isdir(up):
        os.makedirs(up, mode=FOLDER_MODE)
    if filename:
        _validate_filename(filename)
        up = os.path.join(up, filename)
    return up


def ufsa_listdir(username=None, absname=True):
    base = ufsa_path(username)
    for fname in os.listdir(base):
        aname = os.path.join(base, fname)
        if os.path.isfile(aname):
            yield aname if absname else fname


def ufsa_opener(filename, mode='r', username=None):
    """
    :param filename: logic name in the ufsa path
    :param mode: opener mode, all python modes are supported: r, rb, w, wb, r+, w+, a
    :return: a filehandler

    caveats.. user MUST close the file
    """
    return open(ufsa_path(filename=filename, username=username), mode=mode)
