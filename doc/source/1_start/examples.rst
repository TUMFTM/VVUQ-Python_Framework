========
Examples
========
The framework was designed in a generic and modular way so that it can be configured and applied to various use cases. Each framework configuration is fully characterized by a single config file in the configs folder. A couple of config files with the name vvuq_config*.json are included in this repo:

vvuq_config_r79_simpat_*.json
   Configurations for [Riedmaier.2021b]_. These configs can be used to re-create the results from the paper if the simulator is installed. Please keep in mind that the results should look similar but not exactly the same since the simulations include non-determinism due to Monte Carlo sampling.

vvuq_config_r79_mdpi.json
   Configuration for [Riedmaier.2021c]_. This config cannot directly be used to re-create the results from the paper without access to the physical measurement recordings, which are confidential and excluded from the repo.

vvuq_config_r79_example_*.json
   Examples. These configs provide simple examples for the user to get to know the framework. They are inspired by the *simpat* configs but with heavily reduced parameters and samples to keep them small. Thus, we could actually include the data within the repo so that a minimum example can be loaded without simulator access or re-created with simulator access.

All of these configs refer to the UNECE regulation 79 as use case that targets the type approval of vehicles with lane-keeping functions. They have at least the two scenario parameters longitudinal velocity and lateral acceleration that characterize a cornering behavior. The response quantity of interest is the distance to the lane markings. It should always be positive so that the lines are never crossed to pass the test scenarios. In the config files with the names vvuq_*_deterministic.json the scenarios parameters have fixed values, while the config files with the names vvuq_*_nondeterministic.json consider uncertainties.

The R79 use case and its config files can be used to create framework configurations for new use cases.

