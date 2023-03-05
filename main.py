"""
This module shows a minimal main code.

It reads a config, instantiates a VVUQ object, runs through its process, plots results and evaluates the VVUQ results.

Contact person: Stefan Riedmaier
Creation date: 20.04.2020
Python version: 3.8
"""

# -- IMPORTS -----------------------------------------------------------------------------------------------------------
# -- built-in imports --

# -- third-party imports --
# import toml
import json

# -- custom imports --
from src.commonalities.ConfigHandler import ConfigHandler
from src.domains.VVUQ import VVUQ
from src.plots import vvuq_plot_wrapper
from src.evaluation import vvuq_evaluation_wrapper


# -- VVUQ PROCESS ------------------------------------------------------------------------------------------------------

# read settings from configuration file
with open('configs/vvuq_config_r79_example_deterministic.json') as json_config_file:
    config = json.load(json_config_file)

# handle the user configuration
config_handler = ConfigHandler(config)
config = config_handler.process()

# instantiate VVUQ class and run through the process
vvuq = VVUQ(config)
vvuq.process()

# -- PLOTS -------------------------------------------------------------------------------------------------------------
vvuq_plot_wrapper.create_vvuq_plots(vvuq, config)

# -- EVALUATION --------------------------------------------------------------------------------------------------------
vvuq_evaluation_wrapper.create_vvuq_evaluation(vvuq, config)
