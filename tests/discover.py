# NOTE: this file is adapted from the scikit-learn repository
# https://github.com/scikit-learn/scikit-learn

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


def all_estimators():
    """Get a list of classes and their functions.
    This function crawls the module and gets all classes. Classes that are
    defined in test-modules are not included.
    Returns
    -------
    estimators : list of tuples
        List of (name, class), where ``name`` is the class name as string
        and ``class`` is the actual type of the class.
    """
    all_classes = []
    root = str(Path(__file__).parent.parent / "pydeseq2")

    for _, module_name, _ in pkgutil.walk_packages(path=[root]):
        print(module_name)
        module_parts = module_name.split(".")
        if (
            any(part in _MODULE_TO_IGNORE for part in module_parts)
            or "._" in module_name
        ):
            continue
        module = import_module(f"pydeseq2.{module_name}")
        classes = inspect.getmembers(module, inspect.isclass)
        classes = [
            (name, est_cls) for name, est_cls in classes if not name.startswith("_")
        ]

        all_classes.extend(classes)

    all_classes = set(all_classes)
    # drop duplicates, sort for reproducibility
    # itemgetter is used to ensure the sort does not extend to the 2nd item of
    # the tuple
    return sorted(set(all_classes), key=itemgetter(0))
