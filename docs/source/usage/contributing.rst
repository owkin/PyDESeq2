Contributing
------------

PyDESeq2 is a living project and any contributions are welcome!
The project is hosted on `Github <https://github.com/owkin/PyDESeq2>`_.

The simplest way to help us is to star our repository on github or talk about
our repository on the social media or on your blog or article.

Follow code of conduct.

If you find any bugs in the code report them to the issue tracker (link). There
are some rules that apply (link).

How can I contribute?
=====================
- submit a bug or ask for a new feature
- contribute to the code
- improving the documentation
- PR reviews
- answering questions on issues

Submit a bug or ask for a new feature
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First check if the issue which you are facing or a feature you would like to
have added is not already described in the
issue tracker (link). If this is the case you can give a thumb up to that
issue / feature. If there are multiple thumbs ups to a given issue it makes us to
prioritize it higher. You can also add a comment if you think there is something
you would like to add.

If the issue / feature is not yet described make sure to give precise details on
how you reached that issue (if possible giving code snippets) or precisely
describe how the new feature should function. If it's an issue you should also
mention which version of `pydeseq2` you are using. If you don't know which
version you are using you can check it by running the following code:

.. code-block:: python

    import pydeseq2
    pydeseq2.__version__

Contribute with the code
^^^^^^^^^^^^^^^^^^^^^^^^

Thanks for willing to help us out with the `pydeseq2` development.

TODO: if you are contributing to the existing issue add in the comment that you
plan to contribute on it to avoid duplicate work. Once you start working on your
pull request, link it to the issue by adding `#issue_number`

1. To contribute you will first need to fork the `pydeseq2` (link) repository using
the fork button. This will create a copy of `pydeseq2` on your github account
(you must be logged in to github).
2. clone the `pydeseq2` version from your account:

.. code-block:: python

    git clone git@github.com:<your-github-username>/pydeseq2.git
    cd pydeseq2

We recommend using conda environment, run:

.. code-block:: bash

    conda env create -n pydeseq2-dev python=3.8
    conda activate pydeseq2-dev

and then install `pydeseq2` in the development mode. This will also install all
the required dependencies.

.. code-block:: bash

    pip install -e ."[dev]"

The pre-commit tool will automatically run black and isort, and check flake8
compatibility

3. Add the upstream remote:

.. code-block:: bash

    git remote add upstream git@github.com:owkin/pydeseq2.git

4. Ensure that the origin and upstream are configured correctly by running:

.. code-block:: bash

    git remote -v

you should see someting like this:

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

You can now make the changes commiting to your new branch. Each time you commit
the pre-commit will check for style in your code.

7. Once you are ready to submit your PR, first make sure that all the changes
   you have made are pushed into your github account:

.. code-block:: bash

    git push -u origin my_feature

8. Now create a pull request from your fork by following those
   [guidelines](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)
   The core developers of `pydeseq2` will receive a message that your PR is
   ready for reviews.

9. You will need to respond to all the comments before your PR can be merged.
   Thanks for your contribution.

Improving the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you 



