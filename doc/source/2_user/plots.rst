=====
Plots
=====
There are various plot functions that were primarily used to visualize the results of important VVUQ steps and to generate the figures for the publications. Many examples can be found after scrolling through the papers. Plots are also included in the exemplary config files so that they will be generated at the end of the VVUQ process. The code currently offers the following plot functions:

=====================  ================================================================
Plot Type              Description
=====================  ================================================================
Scenario               Samples across the scenario space
Time Series            Quantities over time
Time Series R79        Special time signal constellation of the UNECE-R79 use case
KPI                    Assessment KPIs across the scenario space
CDF                    CDFs to visualize the scatter from non-deterministic simulations
Metric                 Area metric between measurement and simulation CDFs
Extrapolation          Model errors across the scenario space
Uncertainty Expansion  Uncertainty expansion of non-deterministic simulations
Error Integration      Error integration of deterministic simulations
Decision               Binary decision making results across the scenario space
=====================  ================================================================

In addition to the plots, there is a binary classifier implemented under the category "evaluation" (and a new area metric that is not relevant at the moment). The classifier compares the model predictions based on uncertainty with the true values in the application domain, if available, as is the case in [Riedmaier.2021b]_.

