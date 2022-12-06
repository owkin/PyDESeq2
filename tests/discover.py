# NOTE: this file is adapted from the scikit-learn repository
import inspect
import pkgutil
from importlib import import_module
from operator import itemgetter
from pathlib import Path

_MODULE_TO_IGNORE = {"tests"}


def _is_checked_function(item):
    if not inspect.isfunction(item):
        return False

    if item.__name__.startswith("_"):
        return False

    mod = item.__module__
    if mod.endswith("estimator_checks"):
        return False

    return True


def all_functions():
    """Get a list of all functions from `pydeseq2`.
    Returns
    -------
    functions : list of tuples
        List of (name, function), where ``name`` is the function name as
        string and ``function`` is the actual function.
    """
    # lazy import to avoid circular imports from pydeseq2.base
    # from ._testing import ignore_warnings

    all_functions = []
    root = str(Path(__file__).parent.parent / "pydeseq2")

    for _, module_name, _ in pkgutil.walk_packages(path=[root]):
        module_parts = module_name.split(".")
        if (
            any(part in _MODULE_TO_IGNORE for part in module_parts)
            or "._" in module_name
        ):
            continue

        module = import_module(f"pydeseq2.{module_name}")
        functions = inspect.getmembers(module, _is_checked_function)
        functions = [
            (func.__name__, func) for name, func in functions if not name.startswith("_")
        ]
        all_functions.extend(functions)

    # drop duplicates, sort for reproducibility
    # itemgetter is used to ensure the sort does not extend to the 2nd item of
    # the tuple
    return sorted(set(all_functions), key=itemgetter(0))
