"""
This module wrapps the plotting of the VV&UQ results to call the internal methods based on the user config.

It includes a main function called create_vvuq_plots. See details in its own documentations.

Contact person: Stefan Riedmaier
Creation date: 20.08.2020
Python version: 3.8
"""

# -- IMPORTS -----------------------------------------------------------------------------------------------------------
# -- built-in imports --
import os
import pathlib

# -- third-party imports --
import numpy as np

# -- custom imports --
from src.plots import vvuq_plots


# -- FUNCTIONS ---------------------------------------------------------------------------------------------------------

def create_vvuq_plots(vvuq, config, plot_path=None):
    """
    This function wraps the internal plot methods based on the user config.

    :param vvuq: vvuq object with all results
    :param dict config: config dictionary
    :param str plot_path: path where the plots shall be stored (otherwise automatically generated in results folder)
    """
    if 'analysis' not in config or 'plots' not in config['analysis']:
        return

    cfgpl = config['analysis']['plots']

    if not plot_path:
        # -- create a folder to the store the plots
        # replace the last part of the simulator path with "Plots"
        simulator_path = pathlib.Path(config['cross_domain']['simulator']['result_folder'])
        plot_path_parts = simulator_path.parts[:-1] + ('Plots',)
        plot_path = str(pathlib.Path('').joinpath(*plot_path_parts))

    # create the folder
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # -- set negative distance to line values to zero for plotting
    # this can happen during the error integration, since it subtracts the inferred errors from the model responses
    # the VVUQ results are the same, but the distance to line plots look more natural without negative values
    nested_break_flag = False
    for plot_type_list in cfgpl.values():
        for plot_type_dict in plot_type_list:
            if 'qois' in plot_type_dict and plot_type_dict['qois'] in {'D2LL', 'D2RL', 'D2L'}:
                d2l_qoi = plot_type_dict['qois']

                d2l_data = vvuq.application.qois_kpi_system_estimated_da.loc[{'qois': d2l_qoi}].data
                negative_mask = (d2l_data < 0) & (d2l_data != -np.inf)
                vvuq.application.qois_kpi_system_estimated_da.loc[{'qois': d2l_qoi}].data[negative_mask] = 0

                if hasattr(vvuq.application.error_integration, 'kpi_system_estimated_validation_da') and \
                        vvuq.application.error_integration.kpi_system_estimated_validation_da.dims != ():
                    d2l_data = vvuq.application.error_integration.kpi_system_estimated_validation_da.loc[
                        {'qois': d2l_qoi}].data
                    negative_mask = (d2l_data < 0) & (d2l_data != -np.inf)
                    vvuq.application.error_integration.kpi_system_estimated_validation_da.loc[
                        {'qois': d2l_qoi}].data[negative_mask] = 0

                nested_break_flag = True
                break
        if nested_break_flag:
            break

    # -- Scenario Space Plots --
    for i, plot_type_dict in enumerate(cfgpl['scenario_space_plots']):

        if vvuq.application.scenarios_model.samples_da.ndim == 2:
            # in case of a deterministic simulation w/o uncertainties (2D data), there are only space samples to plot
            vvuq_plots.plot_scenario_space(
                config=config,
                cfgpl=plot_type_dict,
                scenarios_space_validation_da=vvuq.validation.scenarios_model.space_samples_da,
                scenarios_space_application_da=vvuq.application.scenarios_model.space_samples_da,
                save_path=plot_path + '/' + 'scenario_space_' + str(i)
            )
        else:
            vvuq_plots.plot_scenario_space(
                config=config,
                cfgpl=plot_type_dict,
                scenarios_verification_da=vvuq.verification.scenarios.samples_da,
                scenarios_validation_model_da=vvuq.validation.scenarios_model.samples_da,
                scenarios_validation_system_da=vvuq.validation.scenarios_system.samples_da,
                scenarios_application_da=vvuq.application.scenarios_model.samples_da,
                scenarios_space_validation_da=vvuq.validation.scenarios_model.space_samples_da,
                scenarios_space_application_da=vvuq.application.scenarios_model.space_samples_da,
                save_path=plot_path + '/' + 'scenario_space_' + str(i)
            )

    if config['cross_domain']['assessment']['method'] != 'read_csv':

        # -- Time Series Plots --
        for i, plot_type_dict in enumerate(cfgpl['time_series_plots']):
            vvuq_plots.plot_timeseries(
                config=config,
                cfgpl=plot_type_dict,
                qois_ts_da=vvuq.validation.assessment_system.qois_ts_da,
                qois_kpi_da=vvuq.validation.assessment_system.qois_kpi_raw_da,
                save_path=plot_path + '/' + 'timeseries_validation_system_' + str(i)
            )

        # -- Time Series UNECE-R79 Plots --
        for i, plot_type_dict in enumerate(cfgpl['time_series_unecer79_plots']):

            if plot_type_dict['domain'] == 'application' and plot_type_dict['instance'] == 'simulator':
                vvuq_obj = vvuq.application.assessment_model
                save_path = plot_path + '/R79_TimeSeries/' + 'r79_timeseries_application_model_' + str(i)

            elif plot_type_dict['domain'] == 'application' and plot_type_dict['instance'] == 'experiment':
                vvuq_obj = vvuq.application.assessment_system
                save_path = plot_path + '/R79_TimeSeries/' + 'r79_timeseries_application_system_' + str(i)

            elif plot_type_dict['domain'] == 'validation' and plot_type_dict['instance'] == 'simulator':
                vvuq_obj = vvuq.validation.assessment_model
                save_path = plot_path + '/R79_TimeSeries/' + 'r79_timeseries_validation_model_' + str(i)

            elif plot_type_dict['domain'] == 'validation' and plot_type_dict['instance'] == 'experiment':
                vvuq_obj = vvuq.validation.assessment_system
                save_path = plot_path + '/R79_TimeSeries/' + 'r79_timeseries_validation_system_' + str(i)

            else:
                raise ValueError("This plot configuration should not be reachable.")

            fct_dict = {
                'cfgpl': plot_type_dict,
                'qois_ts_da': vvuq_obj.qois_ts_da,
                'quantities_ts_da': vvuq_obj.quantities_ts_da,
                'qois_ts_untrimmed_da': vvuq_obj.qois_ts_untrimmed_da,
                'qois_kpi_da': vvuq_obj.qois_kpi_raw_da,
                'save_path': save_path
            }

            if config['cross_domain']['assessment']['is_event_finder']:
                fct_dict['start_idx'] = vvuq_obj.unece_event_finder.start_idx_longest
                fct_dict['stop_idx'] = vvuq_obj.unece_event_finder.stop_idx_longest
                fct_dict['ay_lower_bound'] = vvuq_obj.unece_event_finder.ay_lower_bounds
                fct_dict['ay_upper_bound'] = vvuq_obj.unece_event_finder.ay_upper_bounds

            vvuq_plots.plot_timeseries_unecer79(**fct_dict)

    # -- KPI Surface Plots --
    for i, plot_type_dict in enumerate(cfgpl['kpi_surface_plots']):

        if plot_type_dict['domain'] == 'application' and plot_type_dict['instance'] == 'simulator':
            # plot the KPI surface of the simulation model in the application domain
            vvuq_plots.plot_kpi_surface(
                config=config,
                cfgpl=plot_type_dict,
                scenarios_da=vvuq.application.scenarios_model.samples_da,
                qois_kpi_raw_da=vvuq.application.assessment_model.qois_kpi_raw_da,
                plot_type=plot_type_dict['type'],
                save_path=plot_path + '/' + 'kpi_surface_application_model_' + str(i)
            )

        elif plot_type_dict['domain'] == 'application' and plot_type_dict['instance'] == 'experiment':
            # plot the KPI surface of the system in the application domain
            vvuq_plots.plot_kpi_surface(
                config=config,
                cfgpl=plot_type_dict,
                scenarios_da=vvuq.application.scenarios_system.samples_da,
                qois_kpi_raw_da=vvuq.application.assessment_system.qois_kpi_raw_da,
                plot_type=plot_type_dict['type'],
                save_path=plot_path + '/' + 'kpi_surface_application_system_' + str(i)
            )

        elif plot_type_dict['domain'] == 'validation' and plot_type_dict['instance'] == 'simulator':
            # plot the KPI surface of the simulation model in the validation domain
            vvuq_plots.plot_kpi_surface(
                config=config,
                cfgpl=plot_type_dict,
                scenarios_da=vvuq.validation.scenarios_model.samples_da,
                qois_kpi_raw_da=vvuq.validation.assessment_model.qois_kpi_raw_da,
                plot_type=plot_type_dict['type'],
                save_path=plot_path + '/' + 'kpi_surface_validation_model_' + str(i)
            )

        elif plot_type_dict['domain'] == 'validation' and plot_type_dict['instance'] == 'experiment':
            # plot the KPI surface of the system in the validation domain
            vvuq_plots.plot_kpi_surface(
                config=config,
                cfgpl=plot_type_dict,
                scenarios_da=vvuq.validation.scenarios_system.samples_da,
                qois_kpi_raw_da=vvuq.validation.assessment_system.qois_kpi_raw_da,
                plot_type=plot_type_dict['type'],
                save_path=plot_path + '/' + 'kpi_surface_validation_system_' + str(i)
            )

    # -- CDF Plots --
    for i, plot_type_dict in enumerate(cfgpl['cdf_plots']):

        vvuq_plots.plot_cdf(
            config=config,
            cfgpl=plot_type_dict,
            qois_kpi_raw_da=vvuq.validation.assessment_model.qois_kpi_raw_da,
            fill_flag=True,
            qois_kpi_da=vvuq.validation.assessment_model.qois_kpi_da,
            save_path=plot_path + '/CDFs/' + 'cdf_' + str(i)
        )

    # -- Metric Plots --
    for i, plot_type_dict in enumerate(cfgpl['metric_plots']):

        vvuq_plots.plot_area_metrics(
            config=config,
            cfgpl=plot_type_dict,
            pbox_y_model=vvuq.validation.metric.pbox_y_model,
            pbox_y_system=vvuq.validation.metric.pbox_y_system,
            pbox_x_model_list=vvuq.validation.metric.pbox_x_model_list,
            pbox_x_system_list=vvuq.validation.metric.pbox_x_system_list,
            metric_da=vvuq.validation.metric_da,
            scenarios_da=vvuq.validation.scenarios_model.samples_da,
            save_path=plot_path + '/Area_Metric/' + 'area_metric_' + str(i)
        )

    # -- Extrapolation Surface Plots --
    for i, plot_type_dict in enumerate(cfgpl['extrapolation_surface_plots']):
        vvuq_plots.plot_extrapolation_surface(
            config=config,
            cfgpl=plot_type_dict,
            scenarios_validation_da=vvuq.validation.scenarios_model.space_samples_da,
            metric_validation_da=vvuq.validation.metric_da,
            scenarios_application_da=vvuq.application.scenarios_model.space_samples_da,
            error_validation_da=vvuq.validation.error_model.error_full_da,
            save_path=plot_path + '/' + 'extrapolation_surface_' + str(i)
        )

    # -- Uncertainty Expansion Plots --
    # non-deterministic uncertainty expansion
    for i, plot_type_dict in enumerate(cfgpl['nondeterministic_uncertainty_expansion_plots']):
        vvuq_plots.plot_uncertainty_expansion_nondeterministic(
            config=config,
            cfgpl=plot_type_dict,
            qois_kpi_model_da=vvuq.application.qois_kpi_model_da,
            qois_kpi_system_estimated_da=vvuq.application.qois_kpi_system_estimated_da,
            qois_kpi_system_estimated_validation_da=
            vvuq.application.error_integration.kpi_system_estimated_validation_da,
            scenarios_model_da=vvuq.application.scenarios_model.samples_da,
            qois_kpi_system_da=vvuq.application.qois_kpi_system_da,
            save_path=plot_path + '/Uncertainty_Expansion/' + 'uncertainty_expansion_' + str(i)
        )

    # deterministic uncertainty expansion
    for i, plot_type_dict in enumerate(cfgpl['deterministic_error_integration_plots']):

        vvuq_plots.plot_error_integration_deterministic(
            config=config,
            cfgpl=plot_type_dict,
            qois_kpi_model_da=vvuq.application.qois_kpi_model_da,
            qois_kpi_system_estimated_da=vvuq.application.qois_kpi_system_estimated_da,
            scenarios_model_da=vvuq.application.scenarios_model.samples_da,
            qois_kpi_system_da=vvuq.application.qois_kpi_system_da,
            save_path=plot_path + '/Uncertainty_Expansion/' + 'uncertainty_expansion_' + str(i)
        )

    # -- Decision Space Plots --
    for i, plot_type_dict in enumerate(cfgpl['decision_space_plots']):

        fct_dict = {
            'config': config,
            'cfgpl': plot_type_dict,
            'scenarios_application_da': vvuq.application.scenarios_model.space_samples_da,
            'decision_application_system_estimated_da': vvuq.application.decision_system_estimated_da,
            'scenarios_validation_da': vvuq.validation.scenarios_model.space_samples_da,
            'save_path': plot_path + '/' + 'scenario_decision_' + str(i)
        }

        # add data to function args depending on user selections
        if plot_type_dict['application_decisions'] == 'uncertainty_model':
            fct_dict['decision_application_model_da'] = vvuq.application.decision_model_da
        elif plot_type_dict['application_decisions'] == 'uncertainty_system':
            fct_dict['decision_application_system_da'] = vvuq.application.decision_system_da

        if plot_type_dict['validation_decisions'] == 'accuracy':
            fct_dict['decision_validation_da'] = vvuq.validation.decision_da
        else:
            if plot_type_dict['validation_decisions'] in {'model_safety', 'model_system_safety'}:
                fct_dict['decision_validation_model_da'] = vvuq.validation.decision_model_da

            if plot_type_dict['validation_decisions'] in {'system_safety', 'model_system_safety'}:
                fct_dict['decision_validation_system_da'] = vvuq.validation.decision_system_da

        vvuq_plots.plot_decision_space(**fct_dict)

    return
