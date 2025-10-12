Contributing
------------

PyDESeq2 is a living project and any contributions are welcome!
The project is hosted on `Github <https://github.com/owkin/PyDESeq2>`_.

You can help us by adding a star to the `PyDESeq2 repository <https://github.com/owkin/PyDESeq2>`_ on github or talk about
our repository on the social media or on your blog or article.

We also kindly ask you to follow the `Python code of conduct <https://www.python.org/psf/codeofconduct/>`_.

If you find any bugs in the code please report them to the
`issue tracker <https://github.com/owkin/PyDESeq2/issues>`_.

How can I contribute?
=====================
- :ref:`bug`
- :ref:`code_contrib`
- :ref:`doc_contrib`
- :ref:`answers`

.. _bug:

Submitting a bug or asking for a new feature
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, check if the issue which you are facing or the feature you would like to
have added is not already described in the
`issue tracker <https://github.com/owkin/PyDESeq2/issues>`_.
If this is the case you can give a thumb up to that
issue / feature. If there are multiple thumbs ups to a given issue it makes us
prioritize it higher. You can also add a comment if you think there is something
you would like to add to what have been said.

If the issue / feature is not yet described make sure to give precise details on
how you reached that issue (if possible giving code snippets) or precisely
describe how the new feature should function. If it's an issue you should also
mention which version of ``pydeseq2`` you are using. You can check this by running the following python code:

.. code-block:: python

    import pydeseq2
    pydeseq2.__version__

.. _code_contrib:

Contributing to the code
^^^^^^^^^^^^^^^^^^^^^^^^

Thanks for willing to help us out with the development of PyDESeq2.

If you are contributing to an existing issue, add in the comment of that issue
that you plan to contribute on it to avoid duplicate work. Once you start
working on your pull request, link it to the issue by adding ``#{issue_number}`` in your comment.

1. To contribute, you will first need to fork the `Github <https://github.com/owkin/PyDESeq2>`_
repository using the fork button in github. This will create a copy of
``pydeseq2`` on your github account (you must be logged in to github).

1. clone the ``pydeseq2`` version from your account:

.. code-block:: python

    git clone git@github.com:<your-github-username>/pydeseq2.git
    cd pydeseq2

We recommend using `uv` with a virtual environment, run:

.. code-block:: bash

    uv venv --python 3.13 # or higher
    source venv/bin/activate

and then install ``pydeseq2`` in the development mode. This will also install all
the required dependencies.

.. code-block:: bash

    uv sync --extra dev --extra doc

The pre-commit tool will automatically run ``ruff`` and ``isort``, and check ``mypy``
type compatibility

3. Add the upstream remote:

.. code-block:: bash

    git remote add upstream git@github.com:owkin/pydeseq2.git

4. Ensure that the origin and upstream are configured correctly by running:

.. code-block:: bash

    git remote -v

you should see something like this:

.. code-block:: bash

    origin  git@github.com:<your-github-username>/pydeseq2.git (fetch)
    origin  git@github.com:<your-github-username>/pydeseq2.git (push)
    upstream        git@github.com:owkin/pydeseq2.git (fetch)
    upstream        git@github.com:owkin/pydeseq2.git (push)

You are now all set and ready to start on your pull request (PR).

5. Synchronize your branch with the upstream repository:

.. code-block:: bash

    git checkout main
    git fetch upstream
    git merge upstream/main --rebase

6. Create a new branch where you will add your contributions:

.. code-block:: bash

    git checkout -b my_new_branch

You can now make the changes committing to your new branch. Each time you commit,
the pre-commit will check for style in your code.

7. Once you are ready to submit your PR, first make sure that all the changes
   you have made are pushed into your github account:

.. code-block:: bash

    git push -u origin my_feature

8. Now create a pull request from your fork by following those
   `guidelines <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork>`_.
   Ensure that the description of your PR is sufficient to understand what you
   are doing in the code.
   The core developers of ``pydeseq2`` will receive a message that your PR is
   ready for reviews.

9. You will need to respond to all the comments before your PR can be merged.
   Thanks for your contribution.

.. _doc_contrib:

Improving the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you wish to contribute to the documentation you need to follow the same
guidelines as for the code PR (:ref:`code_contrib`) and additionally install the
dependencies required for building the documentation.
Once you have your environment for development ready, make sure that you installed
the ``dev`` and ``doc`` extra dependencies (see step 2 of :ref:`code_contrib`).
For the documentation to work, you need to export the requirements to about
``requirements.txt`` by running:

.. code-block:: bash

    uv sync --extra dev --extra doc
    uv pip freeze > requirements.txt

After you make the changes in the documentation you can check if it builds
correctly by running (in the docs directory):

.. code-block:: bash

    cd docs
    make clean html

If the build was correct you can now view the new document in the
``docs/build/html`` directory.

.. _answers:

PR reviews and answering questions on issues
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can also help us by reviewing an existing PR or by answering questions posed
on the issue board `issue tracker <https://github.com/owkin/PyDESeq2/issues>`_.

Thanks again and happy contributing!
