# Documentation

The documentation of PyDESeq2 is generated using Sphinx and hosted on ReadTheDoc. 
If you want to build the documentation from source, start by cloning the repository, if you have not done it yet.

```bash
git clone https://github.com/owkin/PyDESeq2.git
cd PyDESeq2
```

We recommend installing within a conda environment, using the `environment.yaml` provided with the sources.:

```bash
conda env create -f environment.yml
conda activate pydeseq2
```

Now go to the doc folder, and install the extraneeded dependencies.

```bash
cd docs
pip install -r requirements.txt
```

You can now trigger the build using `make` command line.

```bash
make clean html
```