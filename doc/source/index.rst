.. vvuqdoc documentation master file, created by
   sphinx-quickstart on Wed Sep 29 21:28:35 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

VVUQ Framework
==============

.. image:: figures/VVUQ_Framework.png
  :width: 800
  :alt: Picture VVUQ Framework

*Figure: VVUQ Framework from Fig. 3.1 of* [Riedmaier.2022]_.

This framework provides methods and tools to assure the quality of computer simulations. They primarily belong to the field of model verification, model validation, and uncertainty quantification (VVUQ). Great emphasis is placed on quantifying model errors and uncertainties and then accounting for them during the actual model predictions. The framework is structured as a clear process with several steps. Each step has a clear interface and several interchangeable configuration options. The framework is written in Python, while the entire configuration is outsourced in separate config files to ensure usability without detailed code knowledge. Ultimately, the user can generate multiple diagrams for visualization and measures to judge the model quality.

Working with this framework means fully configuring everything at the start, executing the main script so that the automation runs from start to finish, and retrieving the results and visualizations. It is not intended as an interactive GUI tool, where the user would manually go through the VVUQ steps and play with each block configuration separately until the result is deemed suitable. Nevertheless, the user configuration is supported by web-based forms.

This software comes from a single PhD thesis and is provided as is. Therefore, the primary purpose of the software was to provide all functionalities required to pursue the research and the corresponding publications. Nevertheless, the software was designed with the thought in mind that it might be further developed and used by someone else in the future. The code has a modular architecture and extensive documentation in the code files.

.. warning::
   However, due to limited time and human resources, this code has not been subjected to extensive testing as in larger projects.

.. warning::
   In case the software will be further developed, there might be breaking changes in the config files, component interfaces, or function signatures.


Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   1_start/intro.rst
   1_start/repo.rst
   1_start/installation.rst
   1_start/examples.rst

.. toctree::
   :maxdepth: 2
   :caption: User Guide:
   :glob:

   2_user/config.rst
   2_user/csv.rst
   2_user/plots.rst
   
.. toctree::
   :maxdepth: 2
   :caption: Developer Guide:
   :glob:

   3_developer/datastructures.rst

.. toctree::
   :maxdepth: 2
   :caption: Code Documentation:

   src/modules.rst

Contact
=======
:Email: `stefan.riedmaier@tum.de <stefan.riedmaier@tum.de>`_

.. _publications:

Publications
============
This framework is connected to several publications. If you find this work useful, please consider citing the respective publication. The citations and the bibtex code is provided below.

The main theorethical foundations were layed in [Riedmaier.2021a]_. [Riedmaier.2021b]_ contains a validation of the framework itself by comparing two simulations with injected faults. In [Riedmaier.2021c]_ we apply the framework to the safeguarding of an automated vehicle. The bigger picture with all the context can be found in the dissertation itself in [Riedmaier.2022]_.

.. [Riedmaier.2021a] S. Riedmaier, B. Danquah, B. Schick and F. Diermeyer, "Unified Framework and Survey for Model
   Verification, Validation and Uncertainty Quantification," Archives of Computational Methods in Engineering, vol.
   28, pp. 2655-2688, 2021.

.. [Riedmaier.2021b] S. Riedmaier, J. Schneider, B. Danquah, B. Schick and F. Diermeyer, "Non-deterministic model
   validation methodology for simulation-based safety assessment of automated vehicles," Simulation Modelling Practice
   and Theory, vol. 109, pp. 1-19, 2021.

.. [Riedmaier.2021c] S. Riedmaier, D. Schneider, D. Watzenig, F. Diermeyer and B. Schick, "Model Validation and
   Scenario Selection for Virtual-Based Homologation of Automated Vehicles," Applied Sciences, vol. 11, pp. 1–24, 2021.

.. [Riedmaier.2022] S. Riedmaier, "Model Validation and Uncertainty Aggregation for Safety Assessment of Automated
   Vehicles," PhD thesis, Technical University of Munich, Munich, 2022

.. code-block:: latex

   @article{Riedmaier.2021a,
    author = {Riedmaier, Stefan and Danquah, Benedikt and Schick, Bernhard and Diermeyer, Frank},
    year = {2021},
    title = {Unified Framework and Survey for Model Verification, Validation and Uncertainty Quantification},
    journal = {Archives of Computational Methods in Engineering},
    pages = {2655–-2688},
    volume = {28}
   }

   @article{Riedmaier.2021b,
    author = {Riedmaier, Stefan and Schneider, Jakob and Danquah, Benedikt and Schick, Bernhard and Diermeyer, Frank},
    year = {2021},
    title = {Non-deterministic model validation methodology for simulation-based safety assessment of automated vehicles},
    journal = {Simulation Modelling Practice and Theory},
    volume = {109},
    pages = {1--19}
   }

   @article{Riedmaier.2021c,
    author = {Riedmaier, Stefan and Schneider, Daniel and Watzenig, Daniel and Diermeyer, Frank and Schick, Bernhard},
    year = {2021},
    title = {Model Validation and Scenario Selection for Virtual-Based Homologation of Automated Vehicles},
    journal = {Applied Sciences},
    volume = {11},
    pages = {1--24}
   }

   @phdthesis{Riedmaier.2022,
    author = {Riedmaier, Stefan},
    year = {2022},
    title = {Model Validation and Uncertainty Aggregation for Safety Assessment of Automated Vehicles},
    address = {Munich, Germany},
    school = {{Technical University of Munich}},
    type = {Dissertation}
   }
