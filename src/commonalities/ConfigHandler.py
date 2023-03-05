"""
This module is responsible for handling the user configurations.

It includes one class ConfigHandler.
See details in its own documentation and in the overall documentation in the doc folder.

This module is generally independent of the selected config file format, since it uses dictionaries as inputs after
loading the config files. We currently use json files, since they are the outcome of json form generators.

The config handler contains the validation of the user config to prevent wrong inputs and to make the code robust.
However, since this software is created for scientific phd research and not a series product, there are definitely
options for the user to break the dependencies. Since the config validations are centralized within this config handler
and called at the start of the main script, we do not repeat all the checks within the individual code modules. Instead,
we only add the required checks so that the software control structures work, and a few additional checks for
safety reasons.

Contact person: Stefan Riedmaier
Creation date: 29.05.2020
Python version: 3.8
"""

# -- IMPORTS -----------------------------------------------------------------------------------------------------------
# -- built-in imports --
import warnings

# -- third-party imports --
import numpy as np
import json
import jsonschema

# -- custom imports --
from src.blocks import Metric
from src.helpers.general.numpy_indexing import string_to_index


# -- CLASSES -----------------------------------------------------------------------------------------------------------

class ConfigHandler:
    """
    This class handles the software configuration.

    It includes a main method called "process". See details in its own documentation.
    """

    # create lists to imitate the config structure with similar content
    major_domain_list = ['validation', 'application']
    full_domain_list = ['verification', 'validation', 'application']
    instance_list = ['experiment', 'simulator']

    def __init__(self, config):
        """
        This method initializes a new class instance.

        :param dict config: configuration dictionary
        """
        self.config = config

        # create dict pointers to the config subdicts as short-cuts
        self.cfgpa = self.config['cross_domain']['parameters']
        self.cfgqu = self.config['cross_domain']['quantities']
        self.cfgas = self.config['cross_domain']['assessment']
        self.cfgme = self.config['validation']['metric']
        self.cfgem = self.config['validation']['error_model']

        self.cfgsc, self.cfgup, self.cfgrp, self.cfgti, self.cfgdm, self.cfgpl, self.cfgev = \
            dict(), dict(), dict(), dict(), dict(), dict(), dict()

        # load the schema definition
        with open("configs/vvuq_schema.json") as json_schema_file:
            self.schema = json.load(json_schema_file)

    def get_property(self, property_name):
        """
        This function returns the value to a respective config key.

        :param property_name: name of a config key / property
        :return: config value of that key
        """

        if property_name not in self.config.keys():
            return None

        return self.config[property_name]

    def process(self):
        """
        This function runs through several steps to handle the user configuration.

        The steps include:
        1) a validation of the config against a schema definition,
        2) an extension of the config to adapt it towards the code, and
        3) additional validations that go beyond the schema.

        The second step is added to convert from a config targeted to the user to a config targeted to the code.
        Otherwise, either the user actions or specific code sections would sometimes be cumbersome.

        The third step is necessary to cover specific limitations of the schema definition by means of python code.

        :return:
        """

        # validate the config against the schema definition
        self.validate_schema()

        # extend the config to suit the code
        config = self.extend()

        # perform further validations that go beyond the schema
        self.validate()

        return config

    def validate_schema(self):
        """
        This function validates the config file against the schema file.
        """

        # validate the config against the schema
        jsonschema.validate(instance=self.config, schema=self.schema)

        return

    def validate(self):
        """
        This function performs validations of the config that go beyond the schema.
        """
        # prepare data structures for name checks
        name_check_list, name_ref_list, name_type_list = list(), list(), list()
        cp_names_set = set(self.cfgpa['parameters_name_list'])
        cq_names_set = set(self.cfgas['qois_name_list'])

        if 'analysis' in self.config and 'plots' in self.config['analysis']:
            # -- PLOT-SPECIFIC CHECKS --
            for plot_list in self.config['analysis']['plots'].values():
                for plot_dict in plot_list:
                    # add to name checks
                    if 'parameters' in plot_dict:
                        name_type_list.append('plot parameter')
                        name_check_list.append(set(plot_dict['parameters']))
                        name_ref_list.append(cp_names_set)

                    if 'qois' in plot_dict:
                        name_type_list.append('plot qoi')
                        name_check_list.append({plot_dict['qois']})
                        name_ref_list.append(cq_names_set)

            # -- PLOT-CROSS-CHECKS --
            if self.cfgas['method'] == 'read_csv' and (
                    self.cfgpl['time_series_plots'] or self.cfgpl['time_series_unecer79_plots']):
                warnings.warn("No time series plot will be created when skipping the assessment, " +
                              "since then no time series data is loaded.")

            if not self.config['cross_domain']['experiment']['application']:

                if self.cfgpl['kpi_surface_plots'] and any(
                        p['domain'] == 'application' and p['instance'] == 'experiment'
                        for p in self.cfgpl['kpi_surface_plots']):
                    raise ValueError("KPI surfaces in the application domain can only be plotted " +
                                     "if experiments are performed.")

                if self.cfgpl['time_series_unecer79_plots'] and any(
                        p['domain'] == 'application' and p['instance'] == 'experiment'
                        for p in self.cfgpl['time_series_unecer79_plots']):
                    raise ValueError("System time series in the application domain can only be plotted " +
                                     "if experiments are performed.")

                if self.cfgpl['decision_space_plots'] and any(
                        p['application_decisions'] == 'uncertainty_system' for p in self.cfgpl['decision_space_plots']):
                    raise ValueError("System decisions in the application domain can only be plotted " +
                                     "if experiments are performed.")

            if self.config['validation']['propagation']['is_propagation'] and \
                    self.config['validation']['propagation']['method'] == 'mean' and self.cfgpl['time_series_plots']:
                raise ValueError("The current wrapper call of the time series plot is incompatible with a 'mean' " +
                                 "propagation, since it adds the KPIs, but in the 'mean' case the averaged KPIs " +
                                 "are not part of the times series anymore.")

        # -- SIMULATOR-CROSS-CHECKS --
        if (not self.config['cross_domain']['simulator']['simulate']
                or not self.config['cross_domain']['experiment']['simulate']) and (
                self.config['verification']['scenarios']['method'] != "read_csv" or
                self.config['validation']['scenarios']['method'] != "read_csv" or
                self.config['application']['scenarios']['method'] != "read_csv"):
            raise ValueError("Skipping the simulations after creating new scenarios is not allowed.")

        if self.config['cross_domain']['assessment']['method'] == "read_csv" and (
                self.config['cross_domain']['simulator']['simulate'] or
                self.config['cross_domain']['experiment']['simulate']):
            raise ValueError("Skipping the assessment after performing new simulations is not allowed.")

        # -- ERROR-MODEL-CROSS-CHECKS --
        # add to name checks
        name_type_list.append('extrapolation parameter')
        name_check_list.append(set(self.cfgem['extrapolation_parameters']))
        name_ref_list.append(cp_names_set)

        # -- METRIC-CROSS-CHECKS --
        self.cfgup = self.config['validation']['propagation']
        if self.cfgup['is_propagation'] and self.cfgup['method'] != 'mean' and \
                self.cfgme['metric'] in Metric.deviation_metric_dict:
            raise ValueError("A deterministic metric and non-deterministic simulation do not fit.")

        # ToDo: check all metrics

        for domain in ConfigHandler.full_domain_list:
            # -- SCENARIO-SPECIFIC CHECKS --
            self.cfgsc = self.config[domain]['scenarios']

            # block config options that are already added to the schema but not yet implemented in code
            if self.cfgsc['method'] == "lhs":
                if np.sum(self.cfgsc['is_space_parameters_uniform_list']) != len(
                        self.cfgsc['space_parameters_distribution_list']):
                    raise NotImplementedError("LHS sampling is currently only supported for uniform parameters.")

            # -- SCENARIO-CROSS-CHECKS --
            # add to name checks
            if self.cfgsc['method'] != "read_csv":
                name_type_list.append('space parameter')
                name_check_list.append(set(self.cfgsc['parameters_name_list']))
                name_ref_list.append(cp_names_set)

        for domain in ConfigHandler.major_domain_list:

            # ToDo: possibly also validate the reference propagation, in particular in combination with direct/mean
            # in the latter case, some checks do not occur below but would be relevant for the reference part

            # -- PROPAGATION-SPECIFIC CHECKS --
            self.cfgup = self.config[domain]['propagation']

            if self.cfgup['is_propagation']:
                # block config options that are already added to the schema but not yet implemented in code
                if self.cfgup['method'] == "single_loop":
                    if self.cfgup['epistemic_method'] != "mcs":
                        raise NotImplementedError(
                            "Single-loop sampling is currently only supported with epistemic MC samples.")

                elif self.cfgup['method'] == "two_loop":
                    if self.cfgup['epistemic_method'] != "full_factorial_doe":
                        raise NotImplementedError(
                            "Two-loop sampling is currently only supported with full-factorial epistemic samples.")

                if self.cfgup['method'] in ("single_loop", "two_loop"):
                    if self.cfgup['aleatory_method'] != "mcs":
                        raise NotImplementedError(
                            "There is currently only MC sampling supported for aleatory parameters.")

                    if self.cfgup['aleatory_method'] == "lhs":
                        if np.sum(self.cfgup['is_aleatory_uniform_list']) != self.cfgup['number_aleatory_parameters']:
                            raise NotImplementedError(
                                "LHS sampling is currently only supported for uniform parameters.")

                    # check degenerate cases
                    if self.cfgup['number_epistemic_parameters'] == 0 and self.cfgup['number_aleatory_parameters'] == 0:
                        raise ValueError(
                            "There is no uncertainty propagation without epistemic or aleatory parameters.")

                    # -- PROPAGATION-CROSS-CHECKS --
                    # add to name checks
                    name_type_list += ['epistemic parameter', 'aleatory parameter']
                    name_check_list += [set(self.cfgup['epistemic_parameters_name_list']),
                                        set(self.cfgup['aleatory_parameters_name_list'])]
                    name_ref_list += [cp_names_set, cp_names_set]

                # -- Propagation & Experiment --
                self.cfgti = self.config['cross_domain']['experiment']
                if self.cfgup['method'] == "single_loop" and self.cfgti['repetitions'] <= 1:
                    raise ValueError("No uncertainty propagation with one or less repetitions possible.")

            # -- DECISION-MAKING-SPECIFIC CHECKS --
            self.cfgdm = self.config[domain]['decision_making']

            # add restrictions for ISO19364
            if domain == 'validation':
                if 'ISO19364' in self.cfgdm['qois_type_list'] and not all(
                        dec_type == 'ISO19364' for dec_type in self.cfgdm['qois_type_list']):
                    raise NotImplementedError("The ISO 19364 decision making can just be applied to all QoIs.")

            # -- DECISION-MAKING-CROSS CHECKS --
            # add to name checks
            name_type_list.append('decision making qoi')
            name_check_list.append(set(self.cfgdm['qois_name_list']))
            name_ref_list.append(cq_names_set)

            # -- Metric & Decision Making --
            if domain == 'validation':
                if 'ISO19364' in self.cfgdm['qois_type_list'] and self.cfgme['metric'] == 'transformed_deviation':
                    raise ValueError("The selected metric cannot be applied in combination with ISO19364 tolerances.")

        # perform name checks to see whether the selected names lie within the ones provided in the cross-section
        for name_type, name_check, name_ref in zip(name_type_list, name_check_list, name_ref_list):
            if not name_check.issubset(name_ref):
                # check whether the section names are within the cross-section names (subset) or else raise an error
                raise ValueError("At least one " + name_type + " is not part of its corresponding cross-section array.")

        return

    def extend(self):
        """
        This function extends the user configuration.

        Typical use cases are:
        - Converting to new data types. Sometimes one data type is preferred for the user to input the config, while
        another data type is preferred at specific code snippets to process the config. This is mostly the case for
        arrays here.
        - Deriving new keys, such as the number of parameters.
        - Setting values that should not be configurable by the user at the moment.

        It makes sense to only perform these extensions once and not everywhere in the code where they are processed.

        There are two major design choices:
        1) In-place dict extension versus copying and extending a new dict.
        2) Decentral config handling in each framework block instance versus central config handling for all sections.

        The copy-solution has the disadvantage that not only the extensions have to be coded but also the copy keys
        have to be routed through. This makes the code more cumbersome. Thus, we use the in-place solution here.

        The decentral solution has the advantage that we can create dedicated config dicts for each block.
        However, this does not work well in case of correlations. There are cases where one block needs configs from
        another block possibly even earlier in the vvuq process. This causes semantically wrong assignments from the
        config sections to the block instances and/or code duplicates. In addition, the decentral solution leads to
        redundant executions of specific config extensions in case they are the same for all domains, e.g., assessment.
        Thus, we go for the central solution.

        The latter means we have to imitate the config structure within this function here instead of it being given
        automatically by the block instances.

        We do not perform all config extensions within this single function here. Instead we outsource the extension of
        large config sections into separate internal functions.
        """

        # -- SCENARIO CONFIG MAPPING -----------------------------------------------------------------------------------

        for domain in ConfigHandler.full_domain_list:
            # create dict pointers
            self.cfgsc = self.config[domain]['scenarios']

            # map scenario arrays from separate subschemas to the same array name
            # json-editor does neither support dependencies in tables nor if-else so that the split was necessary
            if self.cfgsc['method'] in {"mcs", "lhs"}:
                self.cfgsc['space_parameters'] = self.cfgsc['space_parameters_mcs_lhs']
            elif self.cfgsc['method'] == "full_factorial_doe":
                self.cfgsc['space_parameters'] = self.cfgsc['space_parameters_ffd']
            elif self.cfgsc['method'] == "manual":
                self.cfgsc['space_parameters'] = self.cfgsc['space_parameters_manual']
            elif self.cfgsc['method'] == "read_csv":
                pass
            elif self.cfgsc['method'] == "post":
                raise NotImplementedError()
            else:
                raise ValueError("This space doe method is not available.")

        # -- ARRAY CONVERSION ------------------------------------------------------------------------------------------

        # -- convert each list of dicts to dedicated lists, since they are often preferred from a coding perspective
        # iterate through the deep config dict
        # we assume dicts on level 1/2 (optional type checks for clarity) and the list of dicts potentially on level 3/4
        for k1, v1 in self.config.items():
            if k1 in {'cross_domain', 'verification', 'validation', 'application'} and isinstance(v1, dict):
                for v2 in v1.values():
                    if isinstance(v2, dict):
                        # we need to copy the items via "lists", since dicts cannot change their size during iterations
                        # we could also create a separate dict (e.g. via nested dict comprehension) and merge both
                        for k3, v3 in list(v2.items()):
                            if isinstance(v3, list):
                                for v4 in v3:
                                    if isinstance(v4, dict):
                                        for k5, v5 in v4.items():
                                            # create a new dedicated list with a dict key following this convention
                                            v2.setdefault(k3 + '_' + k5 + '_list', []).append(v5)

        # -- CROSS-DOMAIN CONFIG ---------------------------------------------------------------------------------------

        if self.config['cross_domain']['simulator']['passive']:
            # in case this convenience flag is set, overwrite the following keys to their "passive" values
            self.config['cross_domain']['simulator']['simulate'] = False
            self.config['cross_domain']['experiment']['simulate'] = False
            self.config['verification']['scenarios']['method'] = 'read_csv'
            self.config['validation']['scenarios']['method'] = 'read_csv'
            self.config['application']['scenarios']['method'] = 'read_csv'
            self.config['cross_domain']['assessment']['method'] = 'read_csv'

        # combine the parameter/qoi label and unit for plot axes
        for pqdict in self.cfgpa['parameters'] + self.cfgas['qois']:
            if pqdict['unit'] == 'none':
                pqdict['axes_label'] = pqdict['label'] + ' (-)'
            else:
                pqdict['axes_label'] = pqdict['label'] + ' (' + pqdict['unit'] + ')'

        # convert from list to dict for access by name, e.g., for plotting
        self.cfgpa['parameters_dict'] = dict()
        for p in self.cfgpa['parameters']:
            self.cfgpa['parameters_dict'][p['name']] = p

        self.cfgas['qois_dict'] = dict()
        for q in self.cfgas['qois']:
            self.cfgas['qois_dict'][q['name']] = q

        # determine for which parameters and quantities (normally distributed) noise shall be considered
        self.cfgpa['parameters_is_noise_list'] = [dist == "normal" for dist in self.cfgpa['parameters_noise_list']]
        self.cfgqu['quantities_is_noise_list'] = [dist == "normal" for dist in self.cfgqu['quantities_noise_list']]
        
        # -- SCENARIO CONFIG -------------------------------------------------------------------------------------------

        for domain in ConfigHandler.full_domain_list:
            # create dict pointers
            self.cfgsc = self.config[domain]['scenarios']

            if self.cfgsc['method'] in {"mcs", "lhs", "full_factorial_doe", "manual"}:

                if self.cfgsc['method'] in {"mcs", "lhs"}:
                    # determine which parameters are normal and which are uniform ones
                    self.cfgsc['is_space_parameters_uniform_list'] = \
                        [dist == "uniform" for dist in self.cfgsc['space_parameters_distribution_list']]
                    self.cfgsc['is_space_parameters_normal_list'] = \
                        [dist == "normal" for dist in self.cfgsc['space_parameters_distribution_list']]

                if self.cfgsc['method'] == "full_factorial_doe":
                    # convert the possibly distinct number of samples for all space parameters
                    self.cfgsc['number_space_samples_list'] = [sub['number_samples'] for sub in
                                                               self.cfgsc['space_parameters']]

                # determine the number of space parameters
                self.cfgsc['number_space_parameters'] = len(self.cfgsc['space_parameters'])

                # all parameters are space parameters
                self.cfgsc['parameters_name_list'] = self.cfgsc['space_parameters_name_list']
                self.cfgsc['number_parameters'] = self.cfgsc['number_space_parameters']
                self.cfgsc['space_parameters_mask_list'] = [True] * self.cfgsc['number_parameters']

            # a missing propagation key is interpreted as no propagation at all (defined as optional key)
            if "propagation" not in self.config[domain]:
                self.config[domain]['propagation'] = dict()
                self.config[domain]['propagation']['is_propagation'] = False

            if self.cfgsc['method'] == "read_csv":
                # reading scenarios from a csv file automatically means to also read the uncertainty samples from csv
                self.config[domain]['propagation']['is_propagation'] = False

            else:
                # -- PROPAGATION CONFIG --------------------------------------------------------------------------------

                # create dict pointers
                self.cfgup = self.config[domain]['propagation']

                # extend config
                self._extend_propagation()

        # -- PLOT CONFIG -----------------------------------------------------------------------------------------------

        if 'analysis' in self.config and 'plots' in self.config['analysis']:
            self.cfgpl = self.config['analysis']['plots']

            for plot_type_list in self.cfgpl.values():
                for plot_type_dict in plot_type_list:
                    idx_dict = dict()
                    for k, v in plot_type_dict.items():

                        if k in {'qois', 'parameters', 'interval'} and v:
                            # directly copy the names used for pandas indexing by name
                            idx_dict[k] = v
                        elif k in {'space_samples', 'epistemic_samples', 'aleatory_samples', 'repetitions'} and v:
                            # convert the string infos from the config field to an actual indexing object
                            idx_dict[k] = string_to_index(v)
                        else:
                            # exclude additional plot flags and empty index items due to optional schema fields
                            pass

                    plot_type_dict['idx_dict'] = idx_dict

        # -- EVALUATION CONFIG -----------------------------------------------------------------------------------------

        if 'analysis' in self.config and 'evaluation' in self.config['analysis']:
            self.cfgev = self.config['analysis']['evaluation']

            # remove empty optional fields
            for eval_type_dict in self.cfgev.values():
                for k, v in list(eval_type_dict.items()):
                    if k in {'qois', 'space_samples', 'evaluation_direction'} and not v:
                        del eval_type_dict[k]

        return self.config

    def _extend_propagation(self):
        """
        This function extends the propagation config.
        """

        if self.cfgup['is_propagation'] and (
                self.cfgup['method'] in {"single_loop", "two_loop"} or 'reference_method' in self.cfgup):
            # -- add a couple of keys for epistemic and aleatory parameters in case of single/two-loop sampling

            # in case of a reference method, we will always execute this, since the cfgrp creation here depends on it
            # however, in case of direct/mean, where cfgup should be "emtpy", we set the relevant ones back in the 2. if

            # -- EPISTEMIC PARAMETERS --
            if 'epistemic_parameters' in self.cfgup:
                # determine the number and position of epistemic parameters
                self.cfgup['number_epistemic_parameters'] = len(self.cfgup['epistemic_parameters'])
                self.cfgup['is_epistemic_parameter_list'] = [name in self.cfgup['epistemic_parameters_name_list']
                                                             for name in self.cfgsc['parameters_name_list']]

            else:
                # set a couple of dict keys to "empty" values that the subsequent code always expects, e.g., for checks
                self.cfgup['epistemic_parameters_name_list'] = []
                self.cfgup['number_epistemic_parameters'] = 0
                self.cfgup['is_epistemic_parameter_list'] = [False] * self.cfgsc['number_parameters']

            # -- ALEATORY PARAMETERS --
            if 'aleatory_parameters' in self.cfgup:
                # determine the number and position of aleatory parameters
                self.cfgup['number_aleatory_parameters'] = len(self.cfgup['aleatory_parameters'])
                self.cfgup['is_aleatory_parameter_list'] =\
                    [name in self.cfgup['aleatory_parameters_name_list'] for name in self.cfgsc['parameters_name_list']]

                # determine which parameters are normal and which are uniform ones
                self.cfgup['is_aleatory_uniform_list'] =\
                    [dist == "uniform" for dist in self.cfgup['aleatory_parameters_distribution_list']]
                self.cfgup['is_aleatory_normal_list'] =\
                    [dist == "normal" for dist in self.cfgup['aleatory_parameters_distribution_list']]

            else:
                # set a couple of dict keys to "empty" values that the subsequent code always expects, e.g., for checks
                self.cfgup['aleatory_parameters_name_list'] = []
                self.cfgup['number_aleatory_parameters'] = 0
                self.cfgup['is_aleatory_parameter_list'] = [False] * self.cfgsc['number_parameters']

            # loop through all parameters
            self.cfgup['parameter_type_list'] = []
            for (is_epistemic, is_aleatory) in zip(self.cfgup['is_epistemic_parameter_list'],
                                                   self.cfgup['is_aleatory_parameter_list']):
                # determine the paramter type based on the epistemic and aleatory information
                if is_epistemic and is_aleatory:
                    # epistemic and aleatory uncertainties are called mixed uncertainties
                    self.cfgup['parameter_type_list'].append('mixed')
                elif is_epistemic and not is_aleatory:
                    self.cfgup['parameter_type_list'].append('epistemic')
                elif not is_epistemic and is_aleatory:
                    self.cfgup['parameter_type_list'].append('aleatory')
                else:
                    # if a parameter is neither epistemic nor aleatory, it is deterministic
                    self.cfgup['parameter_type_list'].append('deterministic')

            if 'reference_method' in self.cfgup:
                # -- in case of a reference simulation imitating a real system, create a new reference propagation dict

                # add the new keys to the dict
                self.cfgrp['method'] = self.cfgup['reference_method']
                self.cfgrp['epistemic_method'] = self.cfgup['reference_epistemic_method']
                self.cfgrp['aleatory_method'] = self.cfgup['reference_aleatory_method']
                self.cfgrp['repetitions'] = self.cfgup['reference_repetitions']

                # copy and overwrite (merge) the nominal propagation dict to get the new reference propagation dict
                # both dicts also contain irrelevant keys from the other one that do not interfere with the processing
                # if this is undesirable, these keys would have to be deleted here by hand
                self.cfgrp = {**self.cfgup, **self.cfgrp}

                # add the new reference propagation dict to the overall config dict
                self.config['validation']['reference_propagation'] = self.cfgrp

        if not self.cfgup['is_propagation'] or self.cfgup['method'] in {"direct", "mean"}:
            # if the user did not select UQ&P at all or in case of a re-simulation via the "direct"- and "mean"-options,
            # set a couple of dict keys to "empty" values that the subsequent code always expects, e.g., for checks
            self.cfgup['epistemic_parameters_name_list'] = []
            self.cfgup['aleatory_parameters_name_list'] = []
            self.cfgup['number_epistemic_parameters'] = 0
            self.cfgup['number_aleatory_parameters'] = 0
            self.cfgup['is_epistemic_parameter_list'] = [False] * self.cfgsc['number_parameters']
            self.cfgup['is_aleatory_parameter_list'] = [False] * self.cfgsc['number_parameters']
            self.cfgup['parameter_type_list'] = ['deterministic'] * self.cfgsc['number_parameters']

        return
