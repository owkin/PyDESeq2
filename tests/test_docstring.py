# NOTE: this file is adapted from the scikit-learn repository
# https://github.com/scikit-learn/scikit-learn

from inspect import signature
from typing import Optional

import pytest
from anndata import AnnData

from tests.discover import all_estimators
from tests.discover import all_functions

numpydoc_validation = pytest.importorskip("numpydoc.validate")

# Ignore methods that are imported from AnnData
z = AnnData()
anndata_methods_and_attributes = dir(z)
FUNCTION_DOCSTRING_IGNORE_LIST = anndata_methods_and_attributes

FUNCTION_DOCSTRING_IGNORE_LIST = [
    "pydeseq2.dds.DeseqDataSet." + meth for meth in FUNCTION_DOCSTRING_IGNORE_LIST
]

FUNCTION_DOCSTRING_IGNORE_LIST = set(FUNCTION_DOCSTRING_IGNORE_LIST)

CLASS_IGNORE_LIST = []


def get_all_functions_names():
    functions = all_functions()
    for _, func in functions:
        # exclude functions outside pydeseq2
        if func.__module__.startswith("pydeseq2"):
            yield f"{func.__module__}.{func.__name__}"


def get_all_methods():
    estimators = all_estimators()
    # displays = all_displays()
    for name, Klass in estimators:
        # ignore all the modules outside of pydeseq2
        if name.startswith("_") or not Klass.__module__.startswith("pydeseq2"):
            # skip private classes
            continue
        methods = []
        for name in dir(Klass):
            if name.startswith("_"):
                continue
            method_obj = getattr(Klass, name)
            if callable(method_obj) or isinstance(method_obj, property):
                methods.append(name)
        methods.append(None)

        for method in sorted(methods, key=str):
            yield Klass, method


def filter_errors(errors, method, Klass=None):
    """
    Ignore some errors based on the method type.
    These rules are specific for pydeseq2."""
    for code, message in errors:
        # We ignore following error code,
        #  - GL01: Docstring text (summary) should start in the line
        #    immediately after the opening quotes (not in the same line,
        #    or leaving a blank line in between)
        if code in ["GL01"]:
            continue

        if code in ("PR02", "GL08") and Klass is not None and method is not None:
            method_obj = getattr(Klass, method)
            if isinstance(method_obj, property):
                continue

        # Following codes are only taken into account for the
        # top level class docstrings:
        #  - SA01: See Also section not found
        #  - EX01: No examples section found
        # if method is not None and code in ["SA01", "EX01"]: TODO: Skipping this for now
        if code in ["SA01", "EX01"]:
            continue
        yield code, message


def repr_errors(res, Klass=None, method: Optional[str] = None) -> str:
    """Pretty print original docstring and the obtained errors
    Parameters
    ----------
    res : dict
        result of numpydoc.validate.validate
    method : str
        if estimator is not None, either the method name or None.
    Returns
    -------
    str
       String representation of the error.
    """
    if method is None:
        if hasattr(Klass, "__init__"):
            method = "__init__"
        elif Klass is None:
            raise ValueError("At least one of Klass, method should be provided")
        else:
            raise NotImplementedError

    if Klass is not None:
        obj = getattr(Klass, method)
        try:
            obj_signature = str(signature(obj))
        except TypeError:
            # In particular we can't parse the signature of properties
            obj_signature = (
                "\nParsing of the method signature failed, "
                "possibly because this is a property."
            )

        obj_name = Klass.__name__ + "." + method
    else:
        obj_signature = ""
        obj_name = method

    msg = "\n\n" + "\n\n".join(
        [
            str(res["file"]),
            obj_name + obj_signature,
            res["docstring"],
            "# Errors",
            "\n".join(
                " - {}: {}".format(code, message) for code, message in res["errors"]
            ),
        ]
    )
    return msg


@pytest.mark.parametrize("Klass, method", get_all_methods())
def test_docstring(Klass, method, request):
    base_import_path = Klass.__module__
    import_path = [base_import_path, Klass.__name__]
    if method is not None:
        import_path.append(method)

    import_path = ".".join(import_path)

    if (
        import_path in FUNCTION_DOCSTRING_IGNORE_LIST
        or base_import_path in CLASS_IGNORE_LIST  # functions which need to be corrected
    ):  # some classes we want to ignore completely
        request.applymarker(
            pytest.mark.xfail(run=False, reason="TODO pass numpydoc validation")
        )
    res = numpydoc_validation.validate(import_path)

    res["errors"] = list(filter_errors(res["errors"], method, Klass=Klass))

    if res["errors"]:
        msg = repr_errors(res, Klass, method)

        raise ValueError(msg)


@pytest.mark.parametrize("function_name", get_all_functions_names())
def test_function_docstring(function_name, request):
    """Check function docstrings using numpydoc."""

    if function_name in FUNCTION_DOCSTRING_IGNORE_LIST:
        request.applymarker(
            pytest.mark.xfail(run=False, reason="TODO pass numpydoc validation")
        )

    res = numpydoc_validation.validate(function_name)

    res["errors"] = list(filter_errors(res["errors"], method="function"))

    if res["errors"]:
        msg = repr_errors(res, method=f"Tested function: {function_name}")

        raise ValueError(msg)
