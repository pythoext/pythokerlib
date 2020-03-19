import os
import inspect
import warnings


class MissingConfig(RuntimeError):
    pass


def get_var(var, default=None, warnifdefault=False, checkenv=True, checkglobal=False):
    where = [globals().get(var)]
    if checkglobal:
        where.append(globals().get(var))
    if checkenv:
        where.append(os.environ.get(var))
    for val in where:
        if val is not None:
            return val
    if default is None:
        return
    if inspect.isclass(default) and issubclass(default, RuntimeError):
        raise default("Missing required parameter '{}'.".format(var))
    if warnifdefault:
        warnings.warn("Default value found for parameter '{}': please override it!".format(var))
    return default

ANONYMOUS_USER = '-anonym-'
MEDIA_SUBDIR = get_var('MEDIA_SUBDIR', 'media')
STORAGE_ROOT = get_var('STORAGE_ROOT', MissingConfig).replace('\\', '/').rstrip('/')

# This could come from global environment
CURRENT_USER = get_var('CURRENT_USER', ANONYMOUS_USER, checkenv=False, checkglobal=True)

def set_current_user(username):
    global CURRENT_USER
    CURRENT_USER = username
