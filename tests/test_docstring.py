# NOTE: this file is adapted from the scikit-learn repository

from inspect import signature
from typing import Optional

import pytest

from tests.discover import all_functions

numpydoc_validation = pytest.importorskip("numpydoc.validate")

FUNCTION_DOCSTRING_IGNORE_LIST = []

FUNCTION_DOCSTRING_IGNORE_LIST = set(FUNCTION_DOCSTRING_IGNORE_LIST)

# TODO: this test only runs on individual functions, but it should be extended to classes


def get_all_functions_names():
    functions = all_functions()
    for _, func in functions:
        # exclude functions from utils.fixex since they come from external packages
        if func.__module__.startswith("pydeseq2"):
            yield f"{func.__module__}.{func.__name__}"


def filter_errors(errors, method):
    """
    Ignore some errors based on the method type.
    These rules are specific for pyDeseq2."""
    for code, message in errors:
        # We ignore following error code,
        #  - GL01: Docstring text (summary) should start in the line
        #    immediately after the opening quotes (not in the same line,
        #    or leaving a blank line in between)
        if code in ["GL01"]:
            continue

        # Following codes are only taken into account for the
        # top level class docstrings:
        #  - SA01: See Also section not found
        #  - EX01: No examples section found
        if method is not None and code in ["SA01", "EX01"]:
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
