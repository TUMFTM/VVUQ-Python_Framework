"""
This module wrapps the evaluation of the VV&UQ methodology to call the internal methods based on the user config.

It includes a main function called create_vvuq_evaluation. See details in its own documentations.

Contact person: Stefan Riedmaier
Creation date: 08.09.2020
Python version: 3.8
"""

# -- IMPORTS -----------------------------------------------------------------------------------------------------------
# -- built-in imports --
import os
import pathlib
import shutil
import pickle

# -- third-party imports --
import matplotlib.pyplot as plt

# -- custom imports --
from src.evaluation import vvuq_evaluation


# -- FUNCTIONS ---------------------------------------------------------------------------------------------------------

def create_vvuq_evaluation(vvuq, config, eval_path=None):
    """
    This function wraps the internal evaluation methods based on the user config.

    :param vvuq: vvuq object with all results
    :param dict config: config dictionary
    :param str eval_path: (optional) path where the evaluation shall be stored (otherwise in results folder)
    :return:
    """
    if 'analysis' not in config or 'evaluation' not in config['analysis']:
        return

    cfgev = config['analysis']['evaluation']

    if not eval_path:
        # -- create a folder to the store the plots
        # replace the last part of the simulator path with "Evaluation"
        simulator_path = pathlib.Path(config['cross_domain']['simulator']['result_folder'])
        eval_path_parts = simulator_path.parts[:-1] + ('Evaluation',)
        eval_path = str(pathlib.Path('').joinpath(*eval_path_parts))

    # create the folder
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

    # -- Boolean Classifier
    if 'classifier_eval' in cfgev:

        vvuq_evaluation.boolean_classifier(
            idx_dict=cfgev['classifier_eval'],
            decision_system_estimated_da=vvuq.application.decision_system_estimated_da,
            decision_system_da=vvuq.application.decision_system_da,
            decision_model_da=vvuq.application.decision_model_da,
            qois_kpi_system_estimated_da=vvuq.application.qois_kpi_system_estimated_da,
            qois_kpi_system_da=vvuq.application.qois_kpi_system_da,
            save_path=eval_path + '/' + 'boolean_classifier')

    # -- Evaluation Area Metric
    if 'metric_eval' in cfgev:

        vvuq_evaluation.evaluation_area_metric(
            config=config,
            idx_dict=cfgev['metric_eval'],
            qois_kpi_model_da=vvuq.application.qois_kpi_model_da,
            save_path=eval_path + '/Eval_Area_Metric/' + 'eval_area_metric')
