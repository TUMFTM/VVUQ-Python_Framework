"""
This module is responsible for the generation of test scenarios.

It includes one class for the scenario generation. See details in its own documentation.

Contact person: Stefan Riedmaier
Creation date: 20.04.2020
Python version: 3.8
"""

# -- IMPORTS -----------------------------------------------------------------------------------------------------------
# -- built-in imports --

# -- third-party imports --
import numpy as np
import xarray as xr

# -- custom imports --
from src.commonalities.CsvHandler import ScenarioCsvHandler
from src.variants.scenarios import sampling


# -- CLASSES -----------------------------------------------------------------------------------------------------------
class Scenarios:
    """
    This class is responsible for the generation of test scenarios.

    It includes a main method called "generate_scenarios" that runs through the generation process by calling the other
    methods. See more details in the documentation of the generate_scenarios method.
    """

    def __init__(self, config, domain, instance):
        """
        This method initializes a new class instance.

        :param dict config: configuration dictionary
        :param str domain: type of VVUQ domain
        :param str instance: test instance
        """

        # -- ASSIGN PARAMETERS TO INSTANCE ATTRIBUTES ------------------------------------------------------------------
        self.config = config
        self.domain = domain
        self.instance = instance

        # -- CREATE CONFIG SUB-DICT POINTERS ---------------------------------------------------------------------------
        self.cfgsc = self.config[domain]['scenarios']
        self.cfgti = self.config['cross_domain'][instance]

        if instance == 'experiment' and self.cfgti['tool'] == 'IPG_CarMaker' and \
                self.config[domain]['propagation']['is_propagation']:
            # cover the special case of a reference simulation with uncertainties instead of a real experiment
            self.cfgup = self.config['validation']['reference_propagation']
        else:
            self.cfgup = self.config[domain]['propagation']

        # -- INSTANTIATE FURTHER INSTANCE ATTRIBUTES -------------------------------------------------------------------
        # instantiate csv file handler for saving and reloading scenarios to and from file
        self.scenario_csv_handler = ScenarioCsvHandler(config, domain, instance)

        # sample numpy arrays
        self.epistemic_samples = np.ndarray(shape=(1,))

        # sample data arrays
        self.space_samples_da = xr.DataArray(None)
        self.samples_da = xr.DataArray(None)

        # boolean flag whether the generation of the sample array must be continued
        self.tbc_flag = False

    def generate_scenarios(self, executed_scenarios_da=xr.DataArray(None),
                           executed_space_scenarios_da=xr.DataArray(None)):
        """
        This function generates an array of test scenarios, possibly with repetitions or epistemic and aleatory samples.

        The generation process consists of the following steps:
        1) In the first step, the samples in the scenario space are generated and returned later (space scenarios).
        There are two options:
        a) generate_space_samples: creates the space scenarios from scratch;
        b) process_executed_scenarios: uses already executed scenarios with regard to a re-simulation.
        2) In the second step, epistemic and aleatory samples can be added in case of non-deterministic simulations
        to get the final scenario array.
        The methods generate_epistemic_samples, generate_aleatory_samples and reshape_sample_array are responsible for
        this. In case of multiple experimental repetitions, the epistemic and aleatory samples are independent of each
        other. In case of nested uncertainty propagation, the epistemic samples will be combined with the aleatory
        samples, respectively. In the deterministic case, the final scenario array is identical to the space scenarios.
        3) Finally, scenario_csv_handler.save is used to store the scenarios in a csv file for archiving and re-loading.

        The scenarios are stored in xarrays, recognizable by the "_da"-endings of the variables.
        The order of the parameters is the same as defined in the config file.
        The space scenarios are always two-dimensional arrays: 'space_samples', 'parameters' (2D)
        The final scenarios might have different dimensions depending on the selected case:
        - 'space_samples', 'parameters' (2D): deterministic simulation
        - 'space_samples', 'repetitions', 'parameters' (3D): experiment with multiple repetitions
        - 'space_samples', 'epistemic_samples', 'aleatory_samples', 'parameters' (4D): nested uncerainty propagation
        (or just with either epistemic or aleatory samples)

        The exact shape depends on the doe method, the number of samples, etc.

        Example:
        - Assuming 5 space parameters with 3-3-2-2-2 full factorial samples, results in 72 space samples.
        - Assuming 2 epistemic parameters with 4-3 full factorial samples, results in 12 epistemic samples.
        - Assuming 3 aleatory parameters with 100 monte carlo samples, results in 100 aleatory samples.
        -> Thus, the 4d array has a dimension of 72x12x100x5. It consists of 86.400 samples and 5 space parameters.

        The values vary starting at the last column backwards:
        - Thus, for the first space and first epistemic sample, all aleatory samples are generated.
        - Then, for the first space sample, the second epistemic sample is generated and again all aleatory samples.
        - ...
        - Then, for the first space sample, the last epistemic sample is generated and again all aleatory samples.
        - Then, for the second space sample, the first epistemic sample is generated and again all aleatory samples.
        - ...
        - Finally, for the last space sample, the last epistemic sample is generated and again all aleatory samples.

        :param xr.DataArray executed_scenarios_da: (optional) executed scenarios for a re-simulation
        :param xr.DataArray executed_space_scenarios_da: (optional) idealized space scenarios for a re-simulation
        :return: array of all space, epistemic and aleatory samples and array of ideal space samples
        :rtype: tuple(xr.DataArray, xr.DataArray)
        """

        # flag, whether the generation of samples must be continued
        self.tbc_flag = self.cfgup['is_propagation']

        if executed_scenarios_da.ndim != 0:
            # process existing samples
            self.process_executed_scenarios(executed_scenarios_da, executed_space_scenarios_da)
        else:
            # generate new space samples
            self.generate_space_samples()

        if self.tbc_flag:
            # generate epistemic and aleatory samples
            (total_space_samples_da, total_epistemic_samples_da) = self.generate_epistemic_samples()
            total_aleatory_samples_da = self.generate_aleatory_samples(
                number_samples=total_space_samples_da.shape[total_space_samples_da.dims.index('total_samples')])

            # -- add the epistemic and aleatory uncertainty samples to the space samples, in form of a 2d array
            # adding data arrays, discards deviating coords, just keeping the joint ones
            total_sp_ep_intersection_da = total_space_samples_da + total_epistemic_samples_da
            total_sp_ap_intersection_da = total_space_samples_da + total_aleatory_samples_da

            # copy the initial data array and replace the added values of the joint dimensions
            self.samples_da = total_space_samples_da.copy()
            self.samples_da.loc[total_sp_ep_intersection_da.coords] = total_sp_ep_intersection_da
            self.samples_da.loc[total_sp_ap_intersection_da.coords] = total_sp_ap_intersection_da

            # reshape the 2d sample array into a multi-dimensional array
            self.reshape_sample_array()

        if not self.cfgsc['method'] == 'read_csv':
            # store important metadata in the attributes dictionary of the data array
            # in some cases the dict values would already exist.
            # nevertheless, write them here, as the values are the same and for unifrom processing
            self.samples_da.attrs['space_parameters_name_list'] = self.cfgsc['space_parameters_name_list']
            self.samples_da.attrs['number_space_parameters'] = self.cfgsc['number_space_parameters']
            self.samples_da.attrs['space_parameters_mask_list'] = self.cfgsc['space_parameters_mask_list']

            self.samples_da.attrs['epistemic_parameters_name_list'] = self.cfgup['epistemic_parameters_name_list']
            self.samples_da.attrs['aleatory_parameters_name_list'] = self.cfgup['aleatory_parameters_name_list']
            self.samples_da.attrs['number_epistemic_parameters'] = self.cfgup['number_epistemic_parameters']
            self.samples_da.attrs['number_aleatory_parameters'] = self.cfgup['number_aleatory_parameters']
            self.samples_da.attrs['is_epistemic_parameter_list'] = self.cfgup['is_epistemic_parameter_list']
            self.samples_da.attrs['is_aleatory_parameter_list'] = self.cfgup['is_aleatory_parameter_list']
            self.samples_da.attrs['parameter_type_list'] = self.cfgup['parameter_type_list']

        # store the samples for archivation and reloading
        self.scenario_csv_handler.save(self.samples_da, self.space_samples_da)

        return self.samples_da, self.space_samples_da

    def process_executed_scenarios(self, executed_scenarios_da, executed_space_scenarios_da):
        """
        This function processes executed scenarios, typically from an experiment for a re-simulation.

        It offers different options:
        1) The scenarios can be reloaded from a csv file.
        2) In the deterministic case (2D), the scenarios are used as they are.
        3) In the repetitions case (3D), the scenarios can be
        - used as they are.
        - averaged.
        - used to extract the space samples.
        4) In the non-deterministic case (4D), the scenarios are used as they are.

        :param xr.DataArray executed_scenarios_da: scenarios already executed in advance
        :param xr.DataArray executed_space_scenarios_da: corresponding ideal space scenarios
        """

        self.tbc_flag = False

        if self.cfgsc['method'] == "read_csv":
            # read the whole scenario design from a stored csv file
            self.samples_da, self.space_samples_da = self.scenario_csv_handler.load()

        # -- check if there are executed scenarios to be considered in the scenario design of a re-simulation
        elif set(executed_scenarios_da.dims) == {'space_samples', 'parameters'}:
            # use executed scenarios directly from an experiment as they are
            self.samples_da = executed_scenarios_da
            self.space_samples_da = self.samples_da

        elif set(executed_scenarios_da.dims) == {'space_samples', 'repetitions', 'parameters'}:
            if self.cfgup['method'] == "direct":
                # use executed scenarios directly from an experiment as they are
                self.samples_da = executed_scenarios_da
                self.space_samples_da = executed_space_scenarios_da

            elif self.cfgup['method'] == "mean":
                # average executed scenarios from an experiment for a deterministic simulation
                self.samples_da = executed_scenarios_da.mean(axis=executed_scenarios_da.dims.index('repetitions'))
                self.space_samples_da = self.samples_da

                # add the 2d data to the attrs dict for csv storage (empty attrs dict after mean calculation)
                self.samples_da.attrs['samples_2d'] = self.samples_da.data

            elif self.cfgup['method'] in {"single_loop", "two_loop"}:
                # extract the space samples from executed scenarios
                self.space_samples_da = executed_space_scenarios_da

                # in case of nested sampling, the generation of samples must be continued
                self.tbc_flag = True

            else:
                raise ValueError("This uncertainty propagation method is not available.")

        elif set(executed_scenarios_da.dims) == \
                {'space_samples', 'epistemic_samples', 'aleatory_samples', 'parameters'}:
            # quick hack to simulate the same scenarios for the system in the application domain as for the model
            self.samples_da = executed_scenarios_da
            self.space_samples_da = executed_space_scenarios_da

        else:
            raise ValueError("This number of dimensions of the scenario array should not exist.")

        return

    def generate_space_samples(self):
        """
        This function generates the samples for the scenario space parameters from scratch.

        It offers several options:
        - mcs: Monte Carlo sampling, either from uniform or gaussian distributions
        - lhs: Latin Hypercube sampling, from uniform distributions
        - full_factorial_doe: full factorial design of experiments with all permutations
        - manual: manually specified scenarios
        - read_csv: re-loading scenarios from a csv file
        - post: skipping the scenario generation and extracting them later in a data-driven assessment / post-processing
        """

        # -- distinguish between the scenario methods provided to the user
        if self.cfgsc['method'] == "mcs":

            # perform uniform mc sampling with the uniform parameters
            uniform_space_samples = sampling.mcs_uniform(self.cfgsc['space_parameters_min_list'],
                                                         self.cfgsc['space_parameters_max_list'],
                                                         self.cfgsc['number_space_samples'])

            # perform gaussian mc sampling with the normal distribution parameters
            normal_space_samples = sampling.mcs_gaussian(self.cfgsc['space_parameters_mu_list'],
                                                         self.cfgsc['space_parameters_sigma_list'],
                                                         self.cfgsc['number_space_samples'])

            # create a sample array and put both sub-arrays in the correct columns
            space_samples = np.empty(shape=(self.cfgsc['number_space_samples'], self.cfgsc['number_space_parameters']))
            space_samples[:, np.array(self.cfgsc['is_space_parameters_uniform_list'])] = uniform_space_samples
            space_samples[:, np.array(self.cfgsc['is_space_parameters_normal_list'])] = normal_space_samples

        elif self.cfgsc['method'] == "lhs":

            # perform uniform LHS sampling with the uniform parameters
            space_samples = sampling.lhs(self.cfgsc['space_parameters_min_list'],
                                         self.cfgsc['space_parameters_max_list'],
                                         self.cfgsc['number_space_samples'])

        elif self.cfgsc['method'] == "full_factorial_doe":
            # If the space scenarios and epistemic parameters use a full factorial design, it would be possible to
            # join both and call the self.full_factorial_doe method just ones to maybe save a little bit of time.
            # This is currently dispensed with in order to have a uniform program flow consisting of 3 loops.

            space_samples = sampling.full_factorial_doe(self.cfgsc['space_parameters_min_list'],
                                                        self.cfgsc['space_parameters_max_list'],
                                                        self.cfgsc['number_space_samples_list'])

        elif self.cfgsc['method'] == "manual":
            # the manual scenario design was already inserted into the sample list in the init method
            space_samples = np.asarray(self.cfgsc['space_parameters_values_list']).T

        elif self.cfgsc['method'] == "read_csv":
            # read the whole scenario design from a stored csv file
            self.samples_da, self.space_samples_da = self.scenario_csv_handler.load()

            # after reading the scenarios from the csv file, the scenario design is already finished
            self.tbc_flag = False

            # to later avoid the Pycharm warning "might be referenced before assignment", because this case is excluded
            space_samples = None

        elif self.cfgsc['method'] == "post":
            # if the scenarios shall be extracted from the experiment data in post-processing, the design is finished
            space_samples = None
            self.tbc_flag = False

        else:
            raise ValueError("This scenario design method is not available.")

        # -- create data arrays, except the have already been created in the "read_csv" case
        if not self.cfgsc['method'] == "read_csv":
            # create data array for the space samples
            self.space_samples_da = xr.DataArray(space_samples, dims=('space_samples', 'parameters'),
                                                 coords={'parameters': self.cfgsc['parameters_name_list']})

            # -- generate final data array if no continuation
            if not self.tbc_flag:
                self.samples_da = self.space_samples_da

                # store the 2d array in the attributes dictionary of the data array
                self.samples_da.attrs['samples_2d'] = space_samples

        return

    def generate_epistemic_samples(self):
        """
        This function generates the samples for the epistemic parameters.

        It offers several options:
        1) In the repetitions case (3D), Monte Carlo sampling from uniform distributions is available.
        2) In the non-deterministic case (4D), the following options are available:
        - mcs: Monte Carlo sampling, from uniform distributions
        - lhs: Latin Hypercube sampling, from uniform distributions
        - full_factorial_doe: full factorial design of experiments with all permutations
        """

        # determine the dimension index of the "space_samples"
        sp_idx = self.space_samples_da.dims.index('space_samples')

        # -- handle the special case without epistemic parameters (just aleatory ones)
        if self.cfgup['number_epistemic_parameters'] == 0:

            # distinguish between uncertainty propagation methods
            if self.cfgup['method'] == 'single_loop':
                total_space_samples = np.repeat(self.space_samples_da.data, self.cfgup['repetitions'],
                                                axis=sp_idx)
            elif self.cfgup['method'] == 'two_loop':
                total_space_samples = np.repeat(self.space_samples_da.data, self.cfgup['number_aleatory_samples'],
                                                axis=sp_idx)
            else:
                raise ValueError("This uncertainty propagation method is not available.")

            # nevertheless, ensure the correct shape of the epistemic array for the further processing steps
            total_epistemic_samples = np.ndarray(shape=(total_space_samples.shape[sp_idx], 0))

        # distinguish between uncertainty propagation methods
        elif self.cfgup['method'] == "single_loop":

            # validity check
            if self.cfgup['epistemic_method'] != "mcs":
                raise ValueError("Single-loop sampling is only supported with epistemic MC samples.")

            # directly generate all epistemic samples (also for all space permutations)
            total_epistemic_samples = sampling.mcs_uniform(
                self.cfgup['epistemic_parameters_min_list'], self.cfgup['epistemic_parameters_max_list'],
                self.cfgup['repetitions'] *
                self.space_samples_da.shape[sp_idx])

            # generate all space samples
            # repeat each space scenario times the number of test repetitions
            total_space_samples = np.repeat(self.space_samples_da.data, self.cfgup['repetitions'], axis=sp_idx)

        elif self.cfgup['method'] == "two_loop":

            # -- generate epistemic permutations
            # distinguish between different sampling methods
            if self.cfgup['epistemic_method'] == "full_factorial_doe":
                self.epistemic_samples = sampling.full_factorial_doe(
                    self.cfgup['epistemic_parameters_min_list'],
                    self.cfgup['epistemic_parameters_max_list'],
                    self.cfgup['epistemic_parameters_number_epistemic_samples_list'])
            elif self.cfgup['epistemic_method'] == "mcs":
                self.epistemic_samples = sampling.mcs_uniform(
                    self.cfgup['epistemic_parameters_min_list'],
                    self.cfgup['epistemic_parameters_max_list'],
                    self.cfgup['epistemic_parameters_number_epistemic_samples_list'])
            elif self.cfgup['epistemic_method'] == "lhs":
                self.epistemic_samples = sampling.lhs(
                    self.cfgup['epistemic_parameters_min_list'],
                    self.cfgup['epistemic_parameters_max_list'],
                    self.cfgup['epistemic_parameters_number_epistemic_samples_list'])
            else:
                raise ValueError("this epistemic sampling method is not supported.")

            # -- generate all epistemic samples
            # copy each block of epistemic permutations times the number of space permutations
            total_epistemic_samples = np.tile(self.epistemic_samples, (self.space_samples_da.shape[sp_idx], 1))
            # repeat each copied block times the number of aleatory samples
            total_epistemic_samples = np.repeat(total_epistemic_samples, self.cfgup['number_aleatory_samples'],
                                                axis=sp_idx)

            # -- generate all space samples
            # repeat each space scenario times the number of epistemic samples times the number of aleatory samples
            number_repeats = self.epistemic_samples.shape[sp_idx] * self.cfgup['number_aleatory_samples']
            total_space_samples = np.repeat(self.space_samples_da.data, number_repeats, axis=sp_idx)

        elif self.cfgup['method'] in {"direct", "mean"}:
            raise ValueError("This program path should not be reached.")

        else:
            raise ValueError("This uncertainty propagation method is not available.")

        # -- create data arrays
        total_space_samples_da = xr.DataArray(total_space_samples, dims=('total_samples', 'parameters'),
                                              coords=self.space_samples_da.coords)
        total_epistemic_samples_da = xr.DataArray(total_epistemic_samples, dims=('total_samples', 'parameters'),
                                                  coords={'parameters': self.cfgup['epistemic_parameters_name_list']})

        return total_space_samples_da, total_epistemic_samples_da

    def generate_aleatory_samples(self, number_samples):
        """
        This function generates the samples for the aleatory parameters.

        It offers several options:
        - mcs: Monte Carlo sampling, either from uniform or gaussian distributions
        - lhs: Latin Hypercube sampling, from uniform distributions

        :param int number_samples: number of aleatory parameter samples
        """

        # -- handle the special case without aleatory parameters (just epistemic ones)
        if self.cfgup['number_aleatory_parameters'] == 0:
            # nevertheless, ensure the correct shape of the aleatory array for the further processing steps
            total_aleatory_samples = np.ndarray(shape=(number_samples, 0))

        elif self.cfgup['aleatory_method'] == "mcs":
            # create a sample array
            total_aleatory_samples = np.empty(shape=(number_samples, self.cfgup['number_aleatory_parameters']))

            # perform uniform mc sampling with the uniform parameters
            if 'uniform' in self.cfgup['aleatory_parameters_distribution_list']:
                uniform_aleatory_samples = sampling.mcs_uniform(self.cfgup['aleatory_parameters_min_list'],
                                                                self.cfgup['aleatory_parameters_max_list'],
                                                                number_samples)

                # put the sub-arrays in the correct columns
                total_aleatory_samples[:, np.array(self.cfgup['is_aleatory_uniform_list'])] = uniform_aleatory_samples

            # perform gaussian mc sampling with the normal distribution parameters
            if 'normal' in self.cfgup['aleatory_parameters_distribution_list']:
                normal_aleatory_samples = sampling.mcs_gaussian(self.cfgup['aleatory_parameters_mu_list'],
                                                                self.cfgup['aleatory_parameters_sigma_list'],
                                                                number_samples)

                # put the sub-arrays in the correct columns
                total_aleatory_samples[:, np.array(self.cfgup['is_aleatory_normal_list'])] = normal_aleatory_samples

        elif self.cfgup['aleatory_method'] == "lhs":
            # perform uniform LHS sampling for the uniform parameters
            total_aleatory_samples = sampling.lhs(self.cfgup['aleatory_parameters_min_list'],
                                                  self.cfgup['aleatory_parameters_max_list'],
                                                  number_samples)

        else:
            raise ValueError("this aleatory sampling method is not available.")

        # create data array
        total_aleatory_samples_da = xr.DataArray(total_aleatory_samples, dims=('total_samples', 'parameters'),
                                                 coords={'parameters': self.cfgup['aleatory_parameters_name_list']})

        return total_aleatory_samples_da

    def reshape_sample_array(self):
        """
        This function reshapes a large two-dimensional sample array to reflect possible epistemic and aleatory loops.

        It outputs a 3d array in case of - typically experimental - repetitions in a single loop.
        It outputs a 4d array in case of nested two loop sampling with epistemic and aleatory parameters.
        """

        if self.cfgup['method'] == "single_loop":

            # reshape the 2d array to a 3d one with number space samples x repetitions x parameters
            samples_nd = np.reshape(self.samples_da.data, (
                self.space_samples_da.shape[self.space_samples_da.dims.index('space_samples')],
                self.cfgup['repetitions'],
                self.samples_da.shape[self.samples_da.dims.index('parameters')]))

            # create data array
            self.samples_da = xr.DataArray(samples_nd, dims=('space_samples', 'repetitions', 'parameters'),
                                           coords={'parameters': self.cfgsc['parameters_name_list']},
                                           attrs={'samples_2d': self.samples_da.data})

        elif self.cfgup['method'] == "two_loop":

            # reshape the 2d array to a 4d one with number space samples x epistemic x aleatory x parameters
            samples_nd = np.reshape(self.samples_da.data, (
                self.space_samples_da.shape[self.space_samples_da.dims.index('space_samples')],
                self.epistemic_samples.shape[0],
                self.cfgup['number_aleatory_samples'],
                self.samples_da.shape[self.samples_da.dims.index('parameters')]))

            # create data array
            self.samples_da = xr.DataArray(
                samples_nd, dims=('space_samples', 'epistemic_samples', 'aleatory_samples', 'parameters'),
                coords={'parameters': self.cfgsc['parameters_name_list']}, attrs={'samples_2d': self.samples_da.data})

        elif self.cfgup['method'] in {"direct", "mean"}:
            raise ValueError("This program path should not be reached.")

        else:
            raise ValueError("This uncertainty propagation method is not available.")
