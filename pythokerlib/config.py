import os
import inspect
import warnings
import json

# controlliamo se stiamo usando pythokerlib in un notebook PYTHO (analyze in jupyter)
cwd = os.path.abspath(os.getcwd())
nb_path = os.path.join(cwd, "notebook.ipynb")
if os.path.isfile(nb_path):
    current_nb = json.load(open(os.path.join(cwd, "notebook.ipynb")))
else:
    current_nb = {}
metadata_nb = current_nb.get("metadata", {})


class MissingConfig(RuntimeError):
    pass


def get_var(var, default=None, warnifdefault=False, checkenv=True, checkglobal=False, checknb=True):
    where = [globals().get(var)]
    if checkglobal:
        where.append(globals().get(var))
    if checkenv:
        where.append(os.environ.get(var))
    if checknb:
        where.append(metadata_nb.get(var))
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
CURRENT_USER = get_var(
    'CURRENT_USER', ANONYMOUS_USER, checkenv=True, checkglobal=True, checknb=True)

# this could come from notebook metadata
NOTEBOOOK_DB_PATH = get_var("DB_PATH", checkenv=True, checkglobal=True, checknb=True)
# variabile utile per sapere se usare feather o no anche nella get_db (in functions.py)
SERVER_HAS_FEATHER = get_var(
    "SERVER_HAS_FEATHER", default=True, checkenv=True, checkglobal=True, checknb=True)
