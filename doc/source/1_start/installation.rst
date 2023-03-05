============
Installation
============
This page presents a brief explanation of the working environment of this code.

The operating system was mainly **Windows**. Nevertheless, the code also worked (with tiny adaptions) on Linux when performing some simulations on a computing cluster with a Ubuntu VM.

The code was developed with the Pycharm IDE using **Python 3.7** installed in a MiniConda environment and later with **Python 3.8** in a virtualenv. We added the package dependencies from the venv to the requirements.txt file at the root directory. It contains a minimum set of packages without version numbers. The environment can be re-created via

.. code-block:: bash

   pip install -r /path/to/requirements.txt

The framework is connected to the CarMaker simulator that is often used in the automotive industry. The installation and licensing is described in the customer portal of the tool manufacturer. A simulator is required to perform the simulations during the model-based VVUQ process. Nevertheless, it is possible to load the already executed simulations from the small example within this repository without actual access to the simulator.

The documentation is build with Sphinx and hosted online. When re-building it, we have to

1. activate the venv including Sphinx in the console,
2. navigate to the doc folder, and
3. execute the following commands to generate rst files from the code docstrings and build the html pages both from the generated and the manual rst files.

.. code-block:: bash

    sphinx-apidoc -o source/code ../src
    make html
