"""
This module is the main code for the UNECE-R79 use case in the SIMPAT paper.

It includes several cases to generate the results for the paper. It generally
reads a config, instantiates a VVUQ object, runs through its process, plots results and evaluates the VVUQ results.

Contact person: Stefan Riedmaier
Creation date: 08.09.2020
Python version: 3.8
"""

# -- IMPORTS -----------------------------------------------------------------------------------------------------------
# -- built-in imports --

# -- third-party imports --
import json

# -- custom imports --
from src.commonalities.ConfigHandler import ConfigHandler
from src.domains.VVUQ import VVUQ
from src.plots import vvuq_plot_wrapper
from src.evaluation import vvuq_evaluation_wrapper


# select what steps shall be performed
propagation_dict = {
    'simulate': False,
    'comparisons': False
}

mean_dict = {
    'simulate': False,
    'comparisons': False,
    'tolerance': False
}

# ----------------------------------------------------------------------------------------------------------------------
# -- NON-DETERMINISTIC VVUQ PROCESS ------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

base_path = "data/SIMPAT_Paper/R79_nondeterministic"

# -- PERFORM SIMULATIONS -----------------------------------------------------------------------------------------------

if propagation_dict['simulate']:

    # -- read settings from configuration file
    with open('configs/vvuq_config_r79_simpat_nondeterministic.json') as json_config_file:
        config = json.load(json_config_file)

    # handle the user configuration
    config_handler = ConfigHandler(config)
    config = config_handler.process()

    # generally select what to plot
    plot_white_list = ['scenario_space_plots', 'kpi_surface_plots']

    # remove the rest
    for plot_type in config['analysis']['plots']:
        if plot_type not in plot_white_list:
            config['analysis']['plots'][plot_type] = list()

    # generally select what to evaluate
    eval_white_list = []

    # remove the rest
    for eval_type in list(config['analysis']['evaluation']):
        if eval_type not in eval_white_list:
            del config['analysis']['evaluation'][eval_type]

    # -- modify settings
    # activate simulations
    config['cross_domain']['simulator']['passive'] = True

    # -- instantiate VVUQ class and run through the process
    vvuq = VVUQ(config)
    vvuq.process()

    # -- create plots
    plot_path = base_path + "/Plots/Plots_Basis"
    vvuq_plot_wrapper.create_vvuq_plots(vvuq, config, plot_path)

    # -- perform evaluation of VVUQ results
    eval_path = base_path + "/Evaluation/Eval_Basis"
    vvuq_evaluation_wrapper.create_vvuq_evaluation(vvuq, config, eval_path)


# -- COMPARE SUB-METHODS -----------------------------------------------------------------------------------------------

if propagation_dict['comparisons']:

    # -- read settings from configuration file
    with open('configs/vvuq_config_r79_simpat_nondeterministic.json') as json_config_file:
        config = json.load(json_config_file)

    # handle the user configuration
    config_handler = ConfigHandler(config)
    config = config_handler.process()

    # generally select what to plot
    plot_white_list = ['metric_plots', 'extrapolation_surface_plots', 'nondeterministic_uncertainty_expansion_plots']

    # remove the rest
    for plot_type in config['analysis']['plots']:
        if plot_type not in plot_white_list:
            config['analysis']['plots'][plot_type] = list()

    # generally select what to evaluate
    eval_white_list = ['classifier_eval']

    # remove the rest
    for eval_type in list(config['analysis']['evaluation']):
        if eval_type not in eval_white_list:
            del config['analysis']['evaluation'][eval_type]

    # -- AVM, LINEAR REGRESSION, 95% PI, 4 EXTRAPOLATION PARAMETERS --

    # -- modify settings
    config['validation']['metric']['metric'] = 'avm'
    config['validation']['error_model']['alpha'] = 0.05

    # -- instantiate VVUQ class and run through the process
    vvuq = VVUQ(config)
    vvuq.process()

    # -- create plots
    plot_path = base_path + "/Plots/Plots_AVM_LinRegr_95PI_4ExtrapolationParams"
    vvuq_plot_wrapper.create_vvuq_plots(vvuq, config, plot_path)

    # -- perform evaluation of VVUQ results
    eval_path = base_path + "/Evaluation/Eval_AVM_LinRegr_95PI_4ExtrapolationParams"
    vvuq_evaluation_wrapper.create_vvuq_evaluation(vvuq, config, eval_path)

    # -- AVM, LINEAR REGRESSION, 90% PI, 4 EXTRAPOLATION PARAMETERS --

    # -- modify settings
    config['validation']['metric']['metric'] = 'avm'
    config['validation']['error_model']['alpha'] = 0.1

    # -- instantiate VVUQ class and run through the process
    vvuq = VVUQ(config)
    vvuq.process()

    # -- create plots
    plot_path = base_path + "/Plots/Plots_AVM_LinRegr_90PI_4ExtrapolationParams"
    vvuq_plot_wrapper.create_vvuq_plots(vvuq, config, plot_path)

    # -- perform evaluation of VVUQ results
    eval_path = base_path + "/Evaluation/Eval_AVM_LinRegr_90PI_4ExtrapolationParams"
    vvuq_evaluation_wrapper.create_vvuq_evaluation(vvuq, config, eval_path)

    # -- MAVM, LINEAR REGRESSION, 95% PI, 4 EXTRAPOLATION PARAMETERS --

    # -- modify settings
    config['validation']['metric']['metric'] = 'mavm'
    config['validation']['error_model']['alpha'] = 0.05

    # -- instantiate VVUQ class and run through the process
    vvuq = VVUQ(config)
    vvuq.process()

    # -- create plots
    plot_path = base_path + "/Plots/Plots_MAVM_LinRegr_95PI_4ExtrapolationParams"
    vvuq_plot_wrapper.create_vvuq_plots(vvuq, config, plot_path)

    # -- perform evaluation of VVUQ results
    eval_path = base_path + "/Evaluation/Eval_MAVM_LinRegr_95PI_4ExtrapolationParams"
    vvuq_evaluation_wrapper.create_vvuq_evaluation(vvuq, config, eval_path)

    # -- MAVM, LINEAR REGRESSION, 90% PI, 4 EXTRAPOLATION PARAMETERS --

    # -- modify settings
    config['validation']['metric']['metric'] = 'mavm'
    config['validation']['error_model']['alpha'] = 0.1

    # -- instantiate VVUQ class and run through the process
    vvuq = VVUQ(config)
    vvuq.process()

    # -- create plots
    plot_path = base_path + "/Plots/Plots_MAVM_LinRegr_90PI_4ExtrapolationParams"
    vvuq_plot_wrapper.create_vvuq_plots(vvuq, config, plot_path)

    # -- perform evaluation of VVUQ results
    eval_path = base_path + "/Evaluation/Eval_MAVM_LinRegr_90PI_4ExtrapolationParams"
    vvuq_evaluation_wrapper.create_vvuq_evaluation(vvuq, config, eval_path)


# ----------------------------------------------------------------------------------------------------------------------
# -- MEAN VVUQ PROCESS -------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

base_path = "data/SIMPAT_Paper/R79_mean"

# -- PERFORM SIMULATIONS -----------------------------------------------------------------------------------------------

if mean_dict['simulate']:

    # -- read settings from configuration file
    with open('configs/vvuq_config_r79_simpat_mean.json') as json_config_file:
        config = json.load(json_config_file)

    # handle the user configuration
    config_handler = ConfigHandler(config)
    config = config_handler.process()

    # generally select what to plot
    plot_white_list = ['scenario_space_plots', 'kpi_surface_plots']

    # remove the rest
    for plot_type in config['analysis']['plots']:
        if plot_type not in plot_white_list:
            config['analysis']['plots'][plot_type] = list()

    # generally select what to evaluate
    eval_white_list = []

    # remove the rest
    for eval_type in list(config['analysis']['evaluation']):
        if eval_type not in eval_white_list:
            del config['analysis']['evaluation'][eval_type]

    # -- modify settings
    # activate simulations
    config['cross_domain']['simulator']['passive'] = True

    # -- instantiate VVUQ class and run through the process
    vvuq = VVUQ(config)
    vvuq.process()

    # -- create plots
    plot_path = base_path + "/Plots/Plots_Basis"
    vvuq_plot_wrapper.create_vvuq_plots(vvuq, config, plot_path)

    # -- perform evaluation of VVUQ results
    eval_path = base_path + "/Evaluation/Eval_Basis"
    vvuq_evaluation_wrapper.create_vvuq_evaluation(vvuq, config, eval_path)


# -- COMPARE SUB-METHODS -----------------------------------------------------------------------------------------------

if mean_dict['comparisons']:

    # -- read settings from configuration file
    with open('configs/vvuq_config_r79_simpat_mean.json') as json_config_file:
        config = json.load(json_config_file)

    # handle the user configuration
    config_handler = ConfigHandler(config)
    config = config_handler.process()

    # generally select what to plot
    plot_white_list = ['extrapolation_surface_plots', 'deterministic_error_integration_plots']

    # remove the rest
    for plot_type in config['analysis']['plots']:
        if plot_type not in plot_white_list:
            config['analysis']['plots'][plot_type] = list()

    # generally select what to evaluate
    eval_white_list = ['classifier_eval']

    # remove the rest
    for eval_type in list(config['analysis']['evaluation']):
        if eval_type not in eval_white_list:
            del config['analysis']['evaluation'][eval_type]

    # -- ABSOLUTE DEVIATION, LINEAR REGRESSION, 95% PI, 4 EXTRAPOLATION PARAMETERS --

    # -- modify settings
    config['validation']['error_model']['alpha'] = 0.05

    # -- instantiate VVUQ class and run through the process
    vvuq = VVUQ(config)
    vvuq.process()

    # -- create plots
    plot_path = base_path + "/Plots/Plots_AbsDev_LinRegr_95PI_4ExtrapolationParams"
    vvuq_plot_wrapper.create_vvuq_plots(vvuq, config, plot_path)

    # -- perform evaluation of VVUQ results
    eval_path = base_path + "/Evaluation/Eval_AbsDev_LinRegr_95PI_4ExtrapolationParams"
    vvuq_evaluation_wrapper.create_vvuq_evaluation(vvuq, config, eval_path)

    # -- ABSOLUTE DEVIATION, LINEAR REGRESSION, 90% PI, 4 EXTRAPOLATION PARAMETERS --

    # -- modify settings
    config['validation']['error_model']['alpha'] = 0.1

    # -- instantiate VVUQ class and run through the process
    vvuq = VVUQ(config)
    vvuq.process()

    # -- create plots
    plot_path = base_path + "/Plots/Plots_AbsDev_LinRegr_90PI_4ExtrapolationParams"
    vvuq_plot_wrapper.create_vvuq_plots(vvuq, config, plot_path)

    # -- perform evaluation of VVUQ results
    eval_path = base_path + "/Evaluation/Eval_AbsDev_LinRegr_90PI_4ExtrapolationParams"
    vvuq_evaluation_wrapper.create_vvuq_evaluation(vvuq, config, eval_path)


# -- TOLERANCE APPROACH WITHOUT ERROR MODEL ----------------------------------------------------------------------------

if mean_dict['tolerance']:

    # -- read settings from configuration file
    with open('configs/vvuq_config_r79_simpat_mean.json') as json_config_file:
        config = json.load(json_config_file)

    # handle the user configuration
    config_handler = ConfigHandler(config)
    config = config_handler.process()

    # generally select what to plot
    plot_white_list = ['decision_space_plots']

    # remove the rest
    for plot_type in config['analysis']['plots']:
        if plot_type not in plot_white_list:
            config['analysis']['plots'][plot_type] = list()

    # generally select what to evaluate
    eval_white_list = ['classifier_eval']

    # remove the rest
    for eval_type in list(config['analysis']['evaluation']):
        if eval_type not in eval_white_list:
            del config['analysis']['evaluation'][eval_type]

    # -- modify settings
    config['validation']['error_model']['method'] = 'none'
    config['validation']['error_model'].pop('alpha', None)
    config['validation']['error_model'].pop('extrapolation_parameters', None)

    # -- instantiate VVUQ class and run through the process
    vvuq = VVUQ(config)
    vvuq.process()

    # -- create plots
    plot_path = base_path + "/Plots/Plots_Tolerance"
    vvuq_plot_wrapper.create_vvuq_plots(vvuq, config, plot_path)

    # -- perform evaluation of VVUQ results
    eval_path = base_path + "/Evaluation/Eval_Tolerance"
    vvuq_evaluation_wrapper.create_vvuq_evaluation(vvuq, config, eval_path)
