# Documentation

The documentation of PyDESeq2 is generated using Sphinx and hosted on ReadTheDocs.
If you want to build the documentation from source, start by cloning the repository, if you have not done it yet.

```bash
git clone https://github.com/owkin/PyDESeq2.git
cd PyDESeq2
```

We recommend installing with `uv`:

```bash
uv venv
source .venv/bin/activate
uv sync --extra dev --extra doc
uv pip freeze > requirements.txt # Required for Binder to work
```

You can now go to the doc folder and trigger the build using `make` command line.

```bash
cd docs
make clean html
```
