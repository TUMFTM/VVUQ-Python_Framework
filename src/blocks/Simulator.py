"""
This module is responsible for executing simulations and reading measurement files.

It includes one class for the simulations and experiments. See details in its own documentation.

Contact person: Stefan Riedmaier
Creation date: 20.04.2020
Python version: 3.8
"""

# -- IMPORTS -----------------------------------------------------------------------------------------------------------
# -- built-in imports --
import os
import shutil
import time

# -- third-party imports --
import psutil
import numpy as np
import xarray as xr
import json

# -- custom imports --
from src.commonalities.CsvHandler import SimulatorCsvHandler
from src.variants.CarMaker.CarMakerRemote import CarMakerRemote
from src.helpers.parsers.ErgFileReader import ErgFileReader
from src.helpers.parsers.MdfFileReader import MdfFileReader
from src.helpers.namespaces import QuantityNamespace
from src.variants.scenarios import sampling


# -- CLASSES -----------------------------------------------------------------------------------------------------------
class Simulator:
    """
    This class is responsible for executing simulations and reading measurement files.

    It includes a main method called "run_simulation_process" that calls the other class methods.
    See more details in the documentation of the run_simulation_process method.
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

        # -- CREATE CONFIG SUB-DICT POINTERS ---------------------------------------------------------------------------
        self.cfgti = self.config['cross_domain'][instance]
        self.cfgpa = self.config['cross_domain']['parameters']
        self.cfgqu = self.config['cross_domain']['quantities']
        self.cfgas = self.config['cross_domain']['assessment']

        # -- INSTANTIATE FURTHER INSTANCE ATTRIBUTES -------------------------------------------------------------------
        # instantiate csv file handler for saving and reloading result file path to and from file
        self.simulator_csv_handler = SimulatorCsvHandler(config, domain, instance)

        if self.cfgti['simulate']:

            # distinguish between different simulation tools
            if self.cfgti['tool'] == "IPG_CarMaker":

                # check if CarMaker is already running
                is_running = False
                if "HIL.exe" in (p.name() for p in psutil.process_iter()):
                    is_running = True

                # create an instance for the CarMaker remote connection
                self.remote = CarMakerRemote(config, domain, instance)

                if not is_running:
                    # if CarMaker is not already running, launch it now
                    self.remote.launch_carmaker()

                    # wait for 2 seconds directly after launching the CarMaker process, since the CarMaker socket is
                    # often still not ready for connection.
                    # attention: even the 2 seconds are sometimes not enough, leading to a tcp/ip connection error.
                    # this is currently a good trade-off, since the software can just be started again.
                    # nevertheless, this might be extended, e.g. via polling the CarMaker status for some seconds.
                    time.sleep(2)

                # connect with the running CarMaker instance via a tcp/ip socket
                self.remote.connect_tcpip_socket()

                if not is_running:
                    # if CarMaker is already running (assuming with the correct config), configuring again is redundant
                    self.remote.configure_carmaker_initially()

                # delete the entire directory tree where the results shall be stored
                shutil.rmtree(self.cfgti['result_folder'] + '/' + self.domain, ignore_errors=True)

                # archive important CarMaker files
                # archivation folder gets currently deleted and re-created for each domain
                self.archive_carmaker_files()

            else:
                raise NotImplementedError("simulation tool not available")

        # select the correct namespace handler class for the quantities in the measurement files
        namespace_class_dict = {
            'IPG_CarMaker': QuantityNamespace.CarMakerNameSpaceHandler,
            'R79_Vehicle': QuantityNamespace.R79VehicleMdfNamespaceHandler
        }
        namespace_class = namespace_class_dict[self.cfgti['tool']]
        # instantiate the class object
        self.namespace_handler = namespace_class()

        # select the correct data file handler
        if self.cfgti['result_format'] == "erg":
            self.datafile_reader = ErgFileReader()
        elif self.cfgti['result_format'] == "mf4":
            self.datafile_reader = MdfFileReader()
        else:
            raise NotImplementedError("result file format not available")

        # create the csv path
        self.csv_path = self.cfgti['result_folder'] + '/' + self.domain + '/parameter_erg_mapping.csv'

    def run_simulation_process(self, scenarios_da, space_scenarios_da):
        """
        This function performs multiple simulations, reads the results and returns a result array.

        The process consists of the following steps:
        1) In the first step, it either performs actual simulations and stores them at defined relult paths
        or it reads the result paths from a csv file or folder. In the first case it uses several functions:
        1.1) create_erg_filepaths: automatically generate result paths where the simulation results shall be stored,
        1.2) convert_parameters: convert selected scenario parameters to a simulator-specific representation,
        1.3) remote.execute_testseries: execute a CarMaker test series,
        1.4) simulator_csv_handler.save: save the result paths.
        2) In the second step, it reads the results from the defined result paths.
        It supports different namespaces and measurement file formats.
        3) Finally, artifical measurement noise can be added to simulation results to minimize the gap to reality.

        The result array has similar dimensions as the scenario array:
        1) It has a 'quantities'-dimension instead of the 'parameter'-dimension
        2) The other dimensions are the same
        n) It has an additional 'timesteps'-dimension

        :param xr.DataArray scenarios_da: array of scenarios
        :param xr.DataArray space_scenarios_da: array of space scenarios
        :return: array of results and array of scenarios (due to experimental noise)
        :rtype: tuple(xr.DataArray, xr.DataArray)
        """

        # -- SIMULATIONS -----------------------------------------------------------------------------------------------

        # check whether simulations shall be performed
        if self.cfgti['simulate']:

            # distinguish between different simulation tools
            if self.cfgti['tool'] == "IPG_CarMaker":

                # generate the paths to the result files where the simulator should store the results later
                rel_path_list, py2res_path_list, cm2res_path_list = self.create_erg_filepaths(
                    scenarios_da, space_scenarios_da)

                # convert selected scenario parameters to a simulator-specific representation
                scenarios_sim_da = self.convert_parameters(scenarios_da)

                # perform all simulations via a test series
                self.remote.execute_testseries(scenarios_sim_da, cm2res_path_list)

                # # perform the simulation for each scenario
                # parameter_name_list = scenarios_da.parameters.values.tolist()
                # for (scenario, erg_path) in zip(scenarios_sim_da.samples_2d, abs_path_list):
                #     # configure CarMaker with the next parameter values and the desired result path
                #     self.remote.configure_carmaker_before_simulation(scenario, erg_path, parameter_name_list)
                #     # execute the simulation with this configuration
                #     self.remote.execute_simulation()

            else:
                raise NotImplementedError("simulation tool not available")

            # save the scenarios with their correspoding result path in a csv file for archiving
            self.simulator_csv_handler.save(rel_path_list)

        else:
            # check whether only the result paths shall be loaded, or also the assessment kpis in the next step
            if self.cfgas['method'] == 'read_csv':
                # if no simulation and if the assessment will read the kpis from the csv file later, we can skip reading
                # all the single result files here, as it will be replaced by the kpis anyway

                # -- still we provide a dummy data array for the results (for read_csv in the assessment)
                # remove the 'parameters'-dimension from the scenario array, and add 'quantities', 'timesteps' instead
                idx = scenarios_da.dims.index('parameters')
                dims = ('quantities',) + scenarios_da.dims[:idx] + scenarios_da.dims[idx + 1:] + ('timesteps',)

                # create the shape of the results (use 1 as dummy value for the unknown timesteps)
                shape = (len(self.cfgqu['quantities_name_list']),) + scenarios_da.shape[:idx] + scenarios_da.shape[
                                                                                            idx + 1:] + (1,)
                # create a dummy numpy array and a data array
                dummy_array = np.empty(shape=shape)
                result_da = xr.DataArray(dummy_array, dims=dims,
                                         coords={'quantities': self.cfgqu['quantities_name_list']})

                return result_da, scenarios_da

            else:
                # if the simulations have already been executed, read the paths to the result files here
                if os.path.isfile(self.csv_path):
                    # get the paths from the csv file
                    _, py2res_path_list = self.simulator_csv_handler.load()
                else:
                    # extract all files in the result folder
                    folder = self.cfgti['result_folder'] + '/' + self.domain
                    py2res_path_list = [folder + '/' + file for file in os.listdir(folder)
                                        if file.endswith('.' + self.cfgti['result_format'])]

                    # add dimensions to the scenario data array
                    number_parameters = scenarios_da.shape[scenarios_da.dims.index('parameters')]
                    scenario_array = np.empty((len(py2res_path_list), number_parameters))
                    scenarios_da = xr.DataArray(scenario_array, dims=('space_samples', 'parameters'),
                                                coords=scenarios_da.coords)

        # -- READ RESULTS ----------------------------------------------------------------------------------------------

        # convert from the reference names to the specific ones in the data file to extract the latter
        quantity_name_list =\
            [self.namespace_handler.quantity_name_mapper(qu) for qu in self.cfgqu['quantities_name_list']]

        # read the desired quantities from all measurement files
        result_da = self.datafile_reader.read_multiple_files(py2res_path_list, quantity_name_list, scenarios_da)

        # replace the quantity coordinate names with the reference namespace (without losing the unit metadata)
        attr_dict = result_da.quantities.attrs
        result_da.coords['quantities'] = self.cfgqu['quantities_name_list']
        result_da.quantities.attrs = attr_dict

        # add the filter metadata
        result_da.quantities.attrs['filter_list'] = self.cfgqu['quantities_filter_list']

        # -- ADD MEASUREMENT NOISE -------------------------------------------------------------------------------------

        # possibly add artificial measurement noise to the inputs and ouputs of a simulation
        scenarios_da, result_da = self.add_measurement_noise(scenarios_da, result_da)

        return result_da, scenarios_da

    def archive_carmaker_files(self):
        """
        This function archives important CarMaker project files.

        It archives:
        - the testrun file
        - the vehicle file
        - the testseries file
        """

        # create a new empty folder for archivation
        dst = self.cfgti['result_folder'] + '/CarMaker'
        if os.path.exists(dst):
            shutil.rmtree(dst)
        os.makedirs(dst)

        # copy the CarMaker testrun file
        src = self.cfgti['project'] + '/Data/TestRun/' + self.cfgti['testrun']
        shutil.copyfile(src, dst + '/' + self.cfgti['testrun'])

        with open(src, 'r') as tr:
            for line in tr:
                if line[:7] == 'Vehicle':
                    # read the name of the vehicle file
                    vehicle = line[10:-1]
                    break

        # copy the CarMaker vehicle file
        src = self.cfgti['project'] + '/Data/Vehicle/' + vehicle
        shutil.copyfile(src, dst + '/' + vehicle)

        return

    def add_measurement_noise(self, scenarios_da, results_da):
        """
        This function adds artifical measurement noise to scenario parameters and simulation results.

        It is only relevant for system simulations in the validation domain that emulate a real system.
        By adding artificial noise for the measurement procedure, it can be made more realisitic.
        This function adds noise to the quantities.
        It also adds noise to the scenario parameters directly
        (since they only depend on the quantities in the data-driven pipeline).

        Attention: We currently do not update the csv file again, since we want to store the actual scenario parameters
        the simulations were conducted with. However, this means that the result plots at the end do not correspond to
        the values in the csv files (without noise offset). Nevertheless, it will affect the KPIs in the csv
        file since they are calculated based on the noisy quantities later.

        :param xr.DataArray scenarios_da: array of scenarios
        :param xr.DataArray results_da: array of simulation results
        :return: array of scenarios and simulation results with added measurement noise
        :rtype: tuple(xr.DataArray, xr.DataArray)
        """

        if 'quantity_noise_distribution_list' in self.cfgqu:
            # the config loader adds this list to the cfg dict for the system in the validation domain

            # check for how many quantities the user selected noise
            number_noise_quantities = np.sum(self.cfgqu['quantities_is_noise_list'])

            if number_noise_quantities > 0:
                # -- add noise to quantities

                # prepare metadata for gaussian sampling
                qu_idx = results_da.dims.index('quantities')
                dims_wo_quantities = results_da.dims[:qu_idx] + results_da.dims[qu_idx + 1:]
                shape_wo_quantities = results_da.shape[:qu_idx] + results_da.shape[qu_idx + 1:]
                number_samples_list = [np.prod(shape_wo_quantities)] * number_noise_quantities

                # perform gaussian mc sampling with the normal distribution parameters
                quantity_noise_samples = sampling.mcs_gaussian(
                    self.cfgqu['quantities_noise_mu_list'], self.cfgqu['quantities_noise_sigma_list'], number_samples_list)

                # reshape the 2D data (mcs_gaussian returns the number_samples (quantities) in the last dimension)
                quantity_noise_samples = quantity_noise_samples.reshape((*shape_wo_quantities, number_noise_quantities))

                # create a data array (addition works also with changed order of dims)
                quantity_noise_samples_da = xr.DataArray(quantity_noise_samples,
                                                         dims=(*dims_wo_quantities, 'quantities'))

                # add the noise samples to the actual data
                idx_dict = {'quantities': self.cfgqu['quantities_is_noise_list']}
                results_da.loc[idx_dict] = results_da.loc[idx_dict] + quantity_noise_samples_da

                # update the 2D data in the attrs dict
                results_da.attrs['array2d'] = results_da.data.reshape(results_da.attrs['array2d'].shape)

            # check for how many quantities the user selected noise
            number_noise_parameters = np.sum(self.cfgpa['parameters_is_noise_list'])

            if number_noise_parameters > 0:
                # -- add noise to scenario parameters

                # prepare metadata for gaussian sampling
                pa_idx = scenarios_da.dims.index('parameters')
                dims_wo_parameters = scenarios_da.dims[:pa_idx] + scenarios_da.dims[pa_idx + 1:]
                shape_wo_parameters = scenarios_da.shape[:pa_idx] + scenarios_da.shape[pa_idx + 1:]
                number_samples_list = [np.prod(shape_wo_parameters)] * number_noise_parameters

                # perform gaussian mc sampling with the normal distribution parameters
                parameter_noise_samples = sampling.mcs_gaussian(self.cfgpa['parameters_noise_mu_list'],
                                                                self.cfgpa['parameters_noise_sigma_list'],
                                                                number_samples_list)

                # reshape the 2D data (mcs_gaussian returns the number_samples (parameters) in the last dimension)
                parameter_noise_samples = parameter_noise_samples.reshape(
                    (*shape_wo_parameters, number_noise_parameters))

                # create a data array (addition works also with changed order of dims)
                parameter_noise_samples_da = xr.DataArray(parameter_noise_samples,
                                                          dims=(*dims_wo_parameters, 'parameters'))

                # add the noise samples to the actual data
                idx_dict = {'parameters': self.cfgpa['parameters_is_noise_list']}
                scenarios_da.loc[idx_dict] = scenarios_da.loc[idx_dict] + parameter_noise_samples_da

                # update the 2D data in the attrs dict
                scenarios_da.attrs['samples_2d'] = scenarios_da.data.reshape(scenarios_da.attrs['samples_2d'].shape)

        return scenarios_da, results_da

    def create_erg_filepaths(self, scenarios_da, space_scenarios_da):
        """
        This function creates an erg file path for each scenario and returns the paths list.

        It does not actually create directories. It just creates the paths. The simulator will create the files.
        If the file names already exist, the simulator will overwrite them!

        It analyses the dimensions of the scenario array:
        - 2 dimensions: deterministic
        - 3 dimensions: (experimental) repetitions
        - 4 dimensions: nested uncertainty propagation

        Details on the conventions of the paths are located at the main documentation in the section "Data Folder
        Structure" of the csv.rst file.

        :param xr.DataArray scenarios_da: array of the scenarios
        :param xr.DataArray space_scenarios_da: array of space scenarios
        :return: list of erg file paths for each scenario: relative postfix, rel/abs from python, rel/abs from CarMaker
        :rtype: tuple(list, list, list)
        """

        # store the epistemic parameter names, as the attrs dict is lost after applying the round function
        epistemic_parameters_name_list = scenarios_da.epistemic_parameters_name_list

        if self.domain == 'verification':
            # use five decimals of the paramter values for creating the paths, to distinguish fine step sizes
            space_scenarios_da = space_scenarios_da.round(decimals=5)
        else:
            # use two decimals of the paramter values for creating the paths
            scenarios_da = scenarios_da.round(decimals=2)
            space_scenarios_da = space_scenarios_da.round(decimals=2)

        # -- create the relative path of the scenario space parameters
        sp_path_list = []
        # repeat the path times the dimensions other than space samples and parameters (indep. on number and order)
        # number_repeats = np.prod(scenarios.shape[1:-1], dtype=np.uint32)
        number_repeats = np.prod(scenarios_da.shape) // scenarios_da.shape[
            scenarios_da.dims.index('space_samples')] // scenarios_da.shape[scenarios_da.dims.index('parameters')]
        for i in range(scenarios_da.shape[scenarios_da.dims.index('space_samples')]):
            # re-initialize the path
            path = ''

            # -- add the folder path for the scenario space parameters (sp)
            for j in range(scenarios_da.shape[scenarios_da.dims.index('parameters')]):
                # take the ideal space samples for naming
                path = path + 'sp' + str(j + 1) + '_' + \
                       str(space_scenarios_da[{'space_samples': i, 'parameters': j}].data).replace('.', '_') + '_'

            # remove last underscore
            path = path[:-1]
            # repeat the path
            sp_path_list = sp_path_list + [path] * number_repeats

        # -- distinguish between different dimensions of the scenario array
        if set(scenarios_da.dims) == {'space_samples', 'parameters'}:
            # add the file ending
            rel_path_list = [p + '.erg' for p in sp_path_list]

        elif set(scenarios_da.dims) == {'space_samples', 'repetitions', 'parameters'}:
            # create the file names for the repetitions with epistemic and aleatory parameters (eaps)
            eap_path_list = ['eaps' + str(i) for i in range(
                1, 1 + scenarios_da.shape[scenarios_da.dims.index('repetitions')])]

            # repeat the file names
            eap_path_list = eap_path_list * scenarios_da.shape[scenarios_da.dims.index('space_samples')]

            # concatenate both lists
            rel_path_list = [i + '/' + j + '.erg' for i, j in zip(sp_path_list, eap_path_list)]

        elif set(scenarios_da.dims) == {'space_samples', 'epistemic_samples', 'aleatory_samples', 'parameters'}:
            # -- create the folder names for the epistemic parameters
            ep_path_list = []
            for i in range(scenarios_da.shape[scenarios_da.dims.index('epistemic_samples')]):
                path = ''
                # create the folder name for the epistemic parameters (ep)
                for j in range(len(epistemic_parameters_name_list)):
                    sc_value = scenarios_da.loc[
                        {'space_samples': 0, 'epistemic_samples': i, 'aleatory_samples': 0,
                         'parameters': epistemic_parameters_name_list[j]}].data
                    path = path + 'ep' + str(j + 1) + '_' + str(sc_value).replace('.', '_') + '_'
                # remove last underscore
                path = path[:-1]
                # repeat the folder name for all aleatory samples
                ep_path_list = ep_path_list + [path] * scenarios_da.shape[scenarios_da.dims.index('aleatory_samples')]

            # create the file names for the aleatory parameters (aps)
            ap_path_list = ['aps' + str(i) for i in range(1, 1 + scenarios_da.shape[
                scenarios_da.dims.index('aleatory_samples')])]

            # repeat the epistemic folder and aleatory file names
            ep_path_list = ep_path_list * scenarios_da.shape[scenarios_da.dims.index('space_samples')]
            ap_path_list = \
                ap_path_list * scenarios_da.shape[scenarios_da.dims.index('space_samples')] * scenarios_da.shape[
                    scenarios_da.dims.index('epistemic_samples')]

            # concatenate the three lists
            rel_path_list = [i + '/' + j + '/' + k + '.erg'
                             for i, j, k in zip(sp_path_list, ep_path_list, ap_path_list)]

        else:
            raise IndexError("scenario array has unexpected dimensions")

        # -- determine the path from python to the results
        # it can be absolute or relative depending on cfg['result_folder']
        py2res_base_path = self.cfgti['result_folder'] + '/' + self.domain + '/'
        # py2res_base_path = os.path.abspath(self.cfgti['result_folder']).replace('\\', '/') + '/' + self.domain + '/'
        py2res_path_list = [py2res_base_path + rel_path for rel_path in rel_path_list]

        # -- determine the path from the CarMaker simulator to the results used within the testseries
        if os.path.isabs(self.cfgti['result_folder']) or os.path.isabs(self.cfgti['project']):
            # if at least one path is absolute, the CarMaker path will also be absolute
            cm2res_path_list = py2res_path_list
        else:
            # if both paths are relative, the CarMaker path will also be relative
            cm2res_base_path = os.path.relpath(self.cfgti['result_folder'], self.cfgti['project']).replace(
                os.sep, '/') + '/' + self.domain + '/'
            cm2res_path_list = [cm2res_base_path + rel_path for rel_path in rel_path_list]

        return rel_path_list, py2res_path_list, cm2res_path_list

    @staticmethod
    def convert_parameters(scenarios_da):
        """
        This function converts selected scenario parameters to a simulator-specific representation.

        This is e.g. required for the UNECE-R79, since the scenario design is based on velocity and lateral
        acceleration, but the simulator requires the target velocity and curve radius. Thus, the lateral acceleration
        has to be converted to a curve radius before the simulations.

        We currently use a leading underscore to indicate that a parameter has to be converted.
        We currently have a conversion defined for the following parameters:
        - _ay_norm: normalized lateral acceleration to curve radius
        - _ay: (absolute) lateral acceleration to curve radius

        :param xr.DataArray scenarios_da: array with scenarios
        :return: array with scenarios after conversion of selected parameters for the simulator
        :rtype: xr.DataArray
        """

        # check if at least one parameter starts with an underscore indicating that it must be converted
        if any(param_name[0] == '_' for param_name in scenarios_da.parameters.values.tolist()):
            # copy the scenarios since they have to be modified exclusively for the simulator
            scenarios_sim_da = scenarios_da.copy()

            # loop through the parameters
            param_name_list = scenarios_da.parameters.values.tolist()
            for i in range(len(param_name_list)):
                if param_name_list[i][0] == '_':

                    # check which coversion shall be performed
                    if param_name_list[i] == "_ay_norm":
                        # -- convert a normalized target acceleration to a radius (see UNECE R79, ay / aysmax)
                        param_name_list[i] = "$Radius"
                        scenarios_sim_da["parameters"] = ("parameters", param_name_list)
                        v_da = scenarios_da.loc[{'parameters': '$Ego_Init_Velocity'}]
                        ay_norm_da = scenarios_da.loc[{'parameters': '_ay_norm'}]

                        # -- read configs for the vehicle and the unece regulation
                        r79_vehicle_config_path = "configs/r79_vehicle_config.json"
                        with open(r79_vehicle_config_path) as r79_vehicle_config_file:
                            r79_vehicle_cfg = json.load(r79_vehicle_config_file)
                        ay_smax = r79_vehicle_cfg['ay_smax']['value']

                        # calculate the radius
                        scenarios_sim_da.loc[{'parameters': '$Radius'}] = (v_da / 3.6) ** 2 / (ay_norm_da * ay_smax)

                    elif param_name_list[i] == "_ay":
                        # -- convert a target acceleration to a radius (see UNECE R79)
                        param_name_list[i] = "$Radius"
                        scenarios_sim_da["parameters"] = ("parameters", param_name_list)
                        v_da = scenarios_da.loc[{'parameters': '$Ego_Init_Velocity'}]
                        ay_da = scenarios_da.loc[{'parameters': '_ay'}]

                        # calculate the radius
                        scenarios_sim_da.loc[{'parameters': '$Radius'}] = (v_da / 3.6) ** 2 / ay_da

                    else:
                        raise ValueError("no conversion defined for this parameter starting with an underscore")

            scenarios_sim_da.attrs["samples_2d"] = scenarios_sim_da.data.reshape((-1, scenarios_da.shape[-1]))
        else:
            scenarios_sim_da = scenarios_da

        return scenarios_sim_da
