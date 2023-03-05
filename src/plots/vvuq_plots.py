"""
This module is responsible for the plotting the results from the VV&UQ methodology.

It includes several functions that plot different aspects. See details in their own documentations.
Generally, a index dictionary can be used to select only a subset of the data arrays that shall be plotted.

Examples:
    idx_dict = {'quantities': 'Car.ax', 'space_samples': 1, 'timesteps': slice(1000, 3000)}
    idx_dict = {'quantities': 'Car.ax', 'timesteps': slice(1000, 3000)}
    idx_dict = {'quantities': 'Car.ax', 'space_samples': 1}
    idx_dict = {'quantities': 'Car.ax'}

Contact person: Stefan Riedmaier
Creation date: 20.08.2020
Python version: 3.8
"""

# -- IMPORTS -----------------------------------------------------------------------------------------------------------
# -- built-in imports --
import os
import pickle

# -- third-party imports --
import numpy as np
import xarray as xr

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.backends.backend_pgf import FigureCanvasPgf

# -- custom imports --
import src.variants.metrics.area_metrics as am
from src.blocks import Assessment


# -- MODULE-LEVEL VARIABLES --------------------------------------------------------------------------------------------
# create a list with file formats for storage
# the pgf and pdf plot might exceed the tex memory size
format_list = ['png', 'pdf']

# change the backend of mpl so that the plots are visualized in Pycharm Pro in separate windows instead of SciView
# matplotlib.use("TkAgg")

# activate pgf plot option
PGF = False
if PGF:
    matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'text.usetex': True,
        'pgf.rcfonts': False,
    })

    format_list.append('pgf')

# add color definitions according to the TUM CI
colors = {
    'TUMprimaryBlue': [0, 0.4, 0.74],
    'TUMprimaryWhite': [1, 1, 1],
    'TUMprimaryBlack': [0, 0, 0],
    'TUMsecondaryBlue': [0, 0.32, 0.58],
    'TUMsecondaryDarkblue': [0, 0.2, 0.35],
    'TUMaccentOrange': [0.89, 0.45, 0.13],
    'TUMaccentGreen': [0.64, 0.68, 0],
    'TUMaccentLightlightblue': [0.60, 0.78, 0.92],
    'TUMaccentLightblue': [0.39, 0.63, 0.78],
    'TUMIvory': [0.85, 0.84, 0.80],
    'TUMGray2': [0.5, 0.5, 0.5]
}


# -- FUNCTIONS ---------------------------------------------------------------------------------------------------------

def plot_scenario_space(config, cfgpl,
                        scenarios_verification_da=xr.DataArray(None),
                        scenarios_validation_model_da=xr.DataArray(None),
                        scenarios_validation_system_da=xr.DataArray(None),
                        scenarios_application_da=xr.DataArray(None),
                        scenarios_space_validation_da=xr.DataArray(None),
                        scenarios_space_application_da=xr.DataArray(None),
                        save_path=''):
    """
    This function plots scatter points over the coordinates given by scenario points of multiple domains.

    The idx_dict must contain a 'parameters' key.
    The 'parameters'-key must contain a list of either two or three string values: parameter names from scenario arrays.
    Depending on the 'parameters'-key, a two or three dimensional scatter plot will be created.
    The order of the parameter names in the list determines the x, y and z axes in the plot.

    The function arguments are not fully optional.
    Please provide at least one array, so that something can be plotted.

    :param dict config: user configuration
    :param dict cfgpl: user configuration of this plot
    :param xr.DataArray scenarios_verification_da: (optional) verification scenario array
    :param xr.DataArray scenarios_validation_model_da: (optional) validation scenario array of the simulator
    :param xr.DataArray scenarios_validation_system_da: (optional) validation scenario array of the experiment
    :param xr.DataArray scenarios_application_da: (optional) application scenario array
    :param xr.DataArray scenarios_space_validation_da: (optional) nominal validation scenario points
    :param xr.DataArray scenarios_space_application_da: (optional) nominal appl. scenario points
    :param str save_path: (optional) path where the plots shall be saved (otherwise they will be shown)
    """
    idx_dict = cfgpl['idx_dict']

    # -- FORMAT PLOT ---------------------------------------------------------------------------------------------------

    if config['project']['id'] in {'MDPI_PoC_Paper', 'SIMPAT_MMU_Paper'}:
        # adapt font sizes of the plots
        matplotlib.rcParams.update({'font.size': 22})
        plt.rc('font', size=11)  # controls default text sizes
        plt.rc('axes', titlesize=11)  # fontsize of the axes title
        plt.rc('axes', labelsize=11)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=11)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=11)  # fontsize of the tick labels
        plt.rc('legend', fontsize=11)  # legend fontsize
        plt.rc('figure', titlesize=11)  # fontsize of the figure title

    # -- VALIDATE ARGS -------------------------------------------------------------------------------------------------

    # providing no data to plot makes no sense
    if scenarios_verification_da.dims == () and scenarios_validation_model_da.dims == () \
            and scenarios_validation_system_da.dims == () and scenarios_application_da.dims == () \
            and scenarios_space_validation_da.dims == () and scenarios_space_application_da.dims == ():
        raise ValueError("cannot create plot if no data is provided.")

    # set space_flag true if only space samples are provided (different setup of colors and markers)
    space_flag = False
    if scenarios_verification_da.dims == () and scenarios_validation_model_da.dims == () \
            and scenarios_validation_system_da.dims == () and scenarios_application_da.dims == () \
            and scenarios_space_validation_da.dims != () and scenarios_space_application_da.dims != ():
        space_flag = True

    # -- CREATE PLOT ---------------------------------------------------------------------------------------------------

    # create a figure
    fig = plt.figure()

    # distinguish between 2D and 3D scatter plots
    number_space_dimensions = len(idx_dict['parameters'])
    if number_space_dimensions == 2:
        ax = plt.gca()
    elif number_space_dimensions == 3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        raise ValueError("the space plot supports only two or three parameters.")

    # -- INDEX AND PLOT DATA -------------------------------------------------------------------------------------------

    # create a list of dictionaries to index the scenario array to get the parameter for the x, y and optionally z axis
    idx_dict_list = [{'parameters': p} for p in idx_dict['parameters']]

    # -- automatic determination of zorder
    if scenarios_validation_model_da.dims != () and scenarios_validation_system_da.dims != ():
        if scenarios_validation_model_da.ndim >= scenarios_validation_system_da.ndim:
            # if the model has more dims than the system (propagation case), plot the system in front
            zorder_validation_model = 11
            zorder_validation_system = 12
        else:
            # if the model has less dims than the system (mean value case), plot the model in front
            zorder_validation_model = 12
            zorder_validation_system = 11
    else:
        # if just one, the order does not matter
        zorder_validation_model = 11
        zorder_validation_system = 12

    if scenarios_verification_da.dims != ():
        # create tuple for variable 2D / 3D scatter plot
        scenarios_verification_tuple = tuple(
            [scenarios_verification_da.loc[idx_dict_elem] for idx_dict_elem in idx_dict_list])

        # plot the verification scenarios
        ax.scatter(*scenarios_verification_tuple, color=colors['TUMaccentOrange'], marker="x", s=20, zorder=14,
                   label='Nominal Verification Scenario')
    
    if scenarios_space_validation_da.dims != ():
        # create tuple for variable 2D / 3D scatter plot
        scenarios_space_validation_tuple = tuple(
            [scenarios_space_validation_da.loc[idx_dict_elem] for idx_dict_elem in idx_dict_list])

        # plot the validation space scenarios
        if space_flag:
            ax.scatter(*scenarios_space_validation_tuple, color=colors['TUMaccentOrange'], s=20, marker="o",
                       zorder=11, label='Validation Scenarios')
        else:
            ax.scatter(*scenarios_space_validation_tuple, color=colors['TUMsecondaryDarkblue'], s=18, zorder=13,
                       marker="x", label='Nominal Validation Scenario')

    if scenarios_space_application_da.dims != ():
        # create tuple for variable 2D / 3D scatter plot
        scenarios_space_application_tuple = tuple(
            [scenarios_space_application_da.loc[idx_dict_elem] for idx_dict_elem in idx_dict_list])

        # plot the application space scenarios
        if space_flag:
            ax.scatter(*scenarios_space_application_tuple, color=colors['TUMprimaryBlue'], s=20, marker="o",
                       zorder=10, label='Application Scenarios')
        else:
            ax.scatter(*scenarios_space_application_tuple, color=colors['TUMIvory'], s=18, zorder=13, marker="x",
                       label='Nominal Application Scenario')

    if scenarios_validation_model_da.dims != ():
        # create tuple for variable 2D / 3D scatter plot
        scenarios_validation_model_tuple = tuple(
            [scenarios_validation_model_da.loc[idx_dict_elem] for idx_dict_elem in idx_dict_list])

        # plot the validation scenarios
        ax.scatter(*scenarios_validation_model_tuple, color=colors['TUMsecondaryDarkblue'], s=20,
                   zorder=zorder_validation_model, label='Validation Scenarios')

    if scenarios_validation_system_da.dims != ():
        # create tuple for variable 2D / 3D scatter plot
        scenarios_validation_system_tuple = tuple(
            [scenarios_validation_system_da.loc[idx_dict_elem] for idx_dict_elem in idx_dict_list])

        # plot the validation scenarios
        ax.scatter(*scenarios_validation_system_tuple, color=colors['TUMaccentGreen'], alpha=0.3, marker='^', s=20,
                   zorder=zorder_validation_system, label='Validation Uncertainty Samples, System')

    if scenarios_application_da.dims != ():
        # create tuple for variable 2D / 3D scatter plot
        scenarios_application_tuple = tuple(
            [scenarios_application_da.loc[idx_dict_elem] for idx_dict_elem in idx_dict_list])

        # plot the application scenarios
        ax.scatter(*scenarios_application_tuple, color=colors['TUMprimaryBlue'], s=20, zorder=10,
                   label='Application Uncertainty Samples')
    
    # -- FORMAT PLOT ---------------------------------------------------------------------------------------------------

    # -- add plot metadata
    if config['project']['id'] == 'MDPI_PoC_Paper':
        plt.locator_params(axis='x', nbins=12)
        plt.xlim(60, 180)
        plt.locator_params(axis='y', nbins=10)
        plt.ylim(0, 1)
        plt.legend(loc='center', bbox_to_anchor=(0.5, -0.2),
                   framealpha=0.9, ncol=2, handleheight=1, labelspacing=0.025, markerscale=2)
        ax.grid(zorder=0)

    else:
        plt.legend(loc=(-0.1, -0.315), framealpha=0.9, ncol=2, handleheight=1, labelspacing=0.025, markerscale=2)

    # axes labels
    pdict = config['cross_domain']['parameters']['parameters_dict']
    ax.set_xlabel(pdict[idx_dict['parameters'][0]]['axes_label'])
    ax.set_ylabel(pdict[idx_dict['parameters'][1]]['axes_label'])
    if number_space_dimensions == 3:
        ax.set_zlabel(pdict[idx_dict['parameters'][2]]['axes_label'])

    # -- SAVE PLOT -----------------------------------------------------------------------------------------------------

    if save_path:
        # create a sub-folder if it does not already exist
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(os.path.dirname(save_path))

        # save the plot
        pickle.dump(ax, open(save_path + '.pkl', "wb"))
        [plt.savefig(save_path + '.' + fmt, bbox_inches='tight') for fmt in format_list]

        # close the plot if it was automatically opened
        plt.close()
    else:
        # visualize the plot
        plt.show()

    return


def plot_timeseries(config, cfgpl, qois_ts_da, qois_kpi_da=xr.DataArray(None), save_path=''):
    """
    This function plots possibly multiple time series.

    The idx_dict should contain the qoi to be plotted.
    It can include a timesteps key to slice the time dimension.

    :param dict config: user configuration
    :param dict cfgpl: user configuration of this plot
    :param xr.DataArray qois_ts_da: array of qoi time series
    :param xr.DataArray qois_kpi_da: (optional) array of model kpis
    :param str save_path: (optional) path where the plots shall be saved (otherwise they will be shown)
    """
    idx_dict = cfgpl['idx_dict']

    sample_rate = 0.01

    # -- FORMAT PLOT ---------------------------------------------------------------------------------------------------

    if config['project']['id'] in {'MDPI_PoC_Paper', 'SIMPAT_MMU_Paper'}:
        # adapt font sizes of the plots
        matplotlib.rcParams.update({'font.size': 22})
        plt.rc('font', size=13)  # controls default text sizes
        plt.rc('axes', titlesize=13)  # fontsize of the axes title
        plt.rc('axes', labelsize=13)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=13)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=13)  # fontsize of the tick labels
        plt.rc('legend', fontsize=13)  # legend fontsize
        plt.rc('figure', titlesize=13)  # fontsize of the figure title

    # -- INDEX DATA ----------------------------------------------------------------------------------------------------

    # check the timesteps-dimension
    # (the steps of a slice argument whould be omitted)
    if 'timesteps' in cfgpl['idx_dict']:
        sample_start = cfgpl['idx_dict']['timesteps'].start
        sample_end = cfgpl['idx_dict']['timesteps'].stop
    else:
        sample_start = 0
        sample_end = qois_ts_da.shape[qois_ts_da.dims.index('timesteps')]

    # convert from samples to seconds
    time = np.arange(sample_start * sample_rate, sample_end * sample_rate, sample_rate)

    # extract the selected time series data array from the complete data array
    qois_ts_sel_da = qois_ts_da.loc[cfgpl['idx_dict']]

    # exclude the time steps from the iteration (wo_ts for without time series)
    dim_idx_ts = qois_ts_sel_da.dims.index('timesteps')
    shape_wo_ts = qois_ts_sel_da.shape[:dim_idx_ts] + qois_ts_sel_da.shape[dim_idx_ts + 1:]
    dims_wo_ts = qois_ts_sel_da.dims[:dim_idx_ts] + qois_ts_sel_da.dims[dim_idx_ts + 1:]

    # -- PLOT DATA -----------------------------------------------------------------------------------------------------

    # create the figure with the generic settings
    fig = plt.figure()
    legend_str = 'Time Signal'

    # loop through the shape of the data array (w/o time dimension), if multiple time series were selected
    for idx_wo_ts in np.ndindex(shape_wo_ts):
        # add the time steps for indexing
        idx = idx_wo_ts[:dim_idx_ts] + (slice(None),) + idx_wo_ts[dim_idx_ts + 1:]

        # extract a single time series
        qois_ts_sel_array = qois_ts_sel_da[idx].data

        # exclude nan values from plotting
        nan_mask = ~np.isnan(qois_ts_sel_array)

        # create the legend
        if len(dims_wo_ts) > 0:
            legend_str = ''
            for (dim_name, dim_idx) in zip(dims_wo_ts, idx_wo_ts):
                legend_str = legend_str + dim_name + ': ' + str(dim_idx) + ', '
            # remove the last ', '
            legend_str = legend_str[:-2]

        # add the time series to the figure
        if nan_mask.sum():
            plt.plot(time[nan_mask], qois_ts_sel_array[nan_mask], label=legend_str)

        # add the calculated KPI
        if qois_kpi_da.dims != ():
            idx_dict_wo_ts = cfgpl['idx_dict'].copy()
            idx_dict_wo_ts.pop('timesteps', None)
            kpi_index = np.where(qois_ts_sel_array == qois_kpi_da.loc[idx_dict_wo_ts][idx_wo_ts].data)

            # only plot the KPI label once
            if idx_wo_ts in {(), (0,), (0, 0)}:
                plt.scatter(time[kpi_index[0][0]], qois_kpi_da.loc[idx_dict_wo_ts][idx_wo_ts].data,
                            marker='X', color=colors['TUMaccentOrange'], s=55, zorder=3, label='KPI')
            else:
                plt.scatter(time[kpi_index[0][0]], qois_kpi_da.loc[idx_dict_wo_ts][idx_wo_ts].data,
                            marker='X', color=colors['TUMaccentOrange'], s=55, zorder=3)

    # -- FORMAT PLOT ---------------------------------------------------------------------------------------------------

    plt.xlabel('Time (s)')
    qdict = config['cross_domain']['assessment']['qois_dict']
    plt.ylabel(qdict[idx_dict['qois']]['axes_label'])
    plt.grid(True)

    # add the legend information
    plt.legend(loc='upper center', borderpad=1.2)

    # -- SAVE PLOT -----------------------------------------------------------------------------------------------------

    if save_path:
        # create a sub-folder if it does not already exist
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(os.path.dirname(save_path))

        # save the plot
        pickle.dump(fig, open(save_path + '.pkl', "wb"))
        [plt.savefig(save_path + '.' + fmt, bbox_inches='tight') for fmt in format_list]

        # close the plot if it was automatically opened
        plt.close()
    else:
        # visualize the plot
        plt.show()

    return


def plot_timeseries_unecer79(cfgpl, qois_ts_da, quantities_ts_da, qois_ts_untrimmed_da,
                             qois_kpi_da=xr.DataArray(None), start_idx=None, stop_idx=None,
                             ay_lower_bound=None, ay_upper_bound=None, save_path=''):
    """
    This function plots a specific time series of lateral acceleration and distance to line for the UNECE-R79.

    In the upper subplot, it visualizes the check against the stationary ay-conditions to extract the valid events.
    In the lower subplot, it visualizes the actual distance to line qoi signal during the valid event.
    The idx_dict should not contain the actual qoi.

    If the start and stop indices are provided, it visualizes vertical lines as event borders.
    If the ay bounds are provided, it visualizes horizontal bounds as stationary conditions.

    If the KPI is provided, it will be highlighted as point in the time series of the distance to line.

    :param dict cfgpl: user configuration of this plot
    :param xr.DataArray qois_ts_da: array of qoi time series
    :param xr.DataArray quantities_ts_da: array of quantity time series
    :param xr.DataArray qois_ts_untrimmed_da: array of untrimmed qoi time series
    :param tuple(np.ndarray) start_idx: (optional) tuple of integer index arrays with the event start indices
    :param tuple(np.ndarray) stop_idx: (optional) tuple of integer index arrays with the event stop indices
    :param np.ndarray ay_lower_bound: (optional) lower ay bound to be plotted as horizontal line
    :param np.ndarray ay_upper_bound: (optional) upper ay bound to be plotted as horizontal line
    :param xr.DataArray qois_kpi_da: (optional) array of model kpis
    :param str save_path: (optional) path where the plots shall be saved (otherwise they will be shown)
    """
    idx_dict = cfgpl['idx_dict']

    sample_rate = 0.01

    # -- VALIDATE ARGS -------------------------------------------------------------------------------------------------

    # set the time information
    if 'timesteps' in idx_dict:
        raise ValueError("The UNECE-R79 time series plot does not support slicing the timesteps dimension.")

    # -- INDEX DATA ----------------------------------------------------------------------------------------------------

    sample_start = 0
    sample_end = qois_ts_da.shape[qois_ts_da.dims.index('timesteps')]
    sample_end_untrimmed = quantities_ts_da.shape[quantities_ts_da.dims.index('timesteps')]
    time_untrimmed = np.arange(sample_start * sample_rate, sample_end_untrimmed * sample_rate, sample_rate)

    # convert from samples to seconds
    time = np.arange(sample_start * sample_rate, sample_end * sample_rate, sample_rate)

    # distinguish between different types of distance to lines
    qoi_names_list = qois_ts_untrimmed_da.qois.values.tolist()
    if 'D2LL' in qoi_names_list:
        d2l = 'D2LL'
    elif 'D2L' in qoi_names_list:
        d2l = 'D2L'
    else:
        raise ValueError("distance to line signal required for UNECE time series plot")

    # extract the selected time series data array from the complete data array
    qois_ts_sel_da = qois_ts_da.loc[idx_dict]
    quantities_ts_sel_da = quantities_ts_da.loc[idx_dict]
    qois_ts_untrimmed_sel_da = qois_ts_untrimmed_da.loc[idx_dict]
    qois_kpi_sel_da = None
    if qois_kpi_da.dims != ():
        qois_kpi_sel_da = qois_kpi_da.loc[idx_dict]

    # exclude the qois and the time steps from the iteration
    shape_dict = dict(zip(qois_ts_sel_da.dims, qois_ts_sel_da.shape))
    shape_wo_qois_ts_dict = shape_dict.copy()
    shape_wo_qois_ts_dict.pop('timesteps')
    shape_wo_qois_ts_dict.pop('qois')
    shape_wo_qois_ts = tuple(shape_wo_qois_ts_dict.values())
    dims_wo_qois_ts = tuple(shape_wo_qois_ts_dict.keys())

    # loop through the shape of the data array (w/o time dimension), if multiple time series were selected
    for idx_wo_qois_ts in np.ndindex(shape_wo_qois_ts):

        # create the dictionary for indexing (use a full slice of the qois and the timesteps dimension)
        loop_idx_dict = dict(zip(dims_wo_qois_ts, idx_wo_qois_ts))

        # index the data arrays
        qois_ts_loop_da = qois_ts_sel_da.loc[loop_idx_dict]
        quantities_ts_loop_da = quantities_ts_sel_da.loc[loop_idx_dict]
        qois_ts_untrimmed_loop_da = qois_ts_untrimmed_sel_da.loc[loop_idx_dict]

        # check if there is a valid signal to plot (not only nan values)
        ay_nan_mask = ~np.isnan(quantities_ts_loop_da.loc[{'quantities': 'Car.ay'}]).data
        d2l_nan_mask = ~np.isnan(qois_ts_loop_da.loc[{'qois': d2l}]).data
        if not ay_nan_mask.sum() or not d2l_nan_mask.sum():
            continue

        # PLOT TIME DATA -----------------------------------------------------------------------------------------------

        # create a new figure with two subplots arranged one above the other
        fig, (ax1, ax2) = plt.subplots(2, 1)

        # plot the absolute lateral acceleration in the upper subplot
        ax1.plot(time_untrimmed, np.abs(quantities_ts_loop_da.loc[{'quantities': 'Car.ay'}]),
                 zorder=1.7, color=colors['TUMprimaryBlue'])
        ax1.grid(zorder=1)

        # plot the distance to line in the lower subplot
        ax2.plot(time_untrimmed, qois_ts_untrimmed_loop_da.loc[{'qois': d2l}],
                 zorder=1.7, color=colors['TUMprimaryBlue'])
        # ax2.set_ylim(1.7, 2.0)
        ax2.grid(zorder=1)

        # -- PLOT HORIZONTAL BOUNDS ------------------------------------------------------------------------------------

        # if available, plot horizontal lines as bounds for the lateral acceleration
        if ay_lower_bound is not None:
            # create indexing tuple
            ay_idx = tuple(loop_idx_dict.values())

            # get the first and last not-nan time index of the ay signal
            first_valid_ay_time_idx = np.argmax(ay_nan_mask)
            last_valid_ay_time_idx = len(ay_nan_mask) - np.argmax(ay_nan_mask[::-1]) - 1

            # plot a horizontal line
            ax1.plot([time_untrimmed[first_valid_ay_time_idx], time_untrimmed[last_valid_ay_time_idx]],
                     [ay_lower_bound[ay_idx], ay_lower_bound[ay_idx]],
                     label='a_{y} - Bound', color=colors['TUMsecondaryDarkblue'], linewidth=1.5)

        if ay_upper_bound is not None:
            # create indexing tuple
            ay_idx = tuple(loop_idx_dict.values())

            # get the first and last not-nan time index of the ay signal
            first_valid_ay_time_idx = np.argmax(ay_nan_mask)
            last_valid_ay_time_idx = len(ay_nan_mask) - np.argmax(ay_nan_mask[::-1]) - 1

            # plot a horizontal line
            ax1.plot([time_untrimmed[first_valid_ay_time_idx], time_untrimmed[last_valid_ay_time_idx]],
                     [ay_upper_bound[ay_idx], ay_upper_bound[ay_idx]],
                     label='a_{y} - Bound', color=colors['TUMsecondaryDarkblue'], linewidth=1.5)

        # -- PLOT VERTICAL BOUNDS --------------------------------------------------------------------------------------

        if start_idx:
            # compare the loop index (extended by one dim) with the start index tuple (except the last actual index)
            eq_mask = np.array(np.array(idx_wo_qois_ts)[:, None] == np.array(start_idx[:-1]))

            # sum up the boolean mask and check that there is exactly one match where all values are True
            eq_mask = np.array(np.sum(eq_mask, axis=0) == eq_mask.shape[0])
            if eq_mask.sum() != 1:
                raise ValueError("there must be exactly one match in the start_idx for the selected data.")

            # get the index of the one true element
            eq_idx = eq_mask.nonzero()[0][0]

            # get the actual start index
            vertical_start = start_idx[-1][eq_idx]

            # plot vertical lines
            ax1.axvline(x=time_untrimmed[vertical_start], ymin=-1.2, ymax=1,
                        color=colors['TUMsecondaryDarkblue'], lw=1.5, zorder=2, clip_on=False, label='Trim-Line')
            ax2.axvline(x=time_untrimmed[vertical_start], ymin=0, ymax=1,
                        color=colors['TUMsecondaryDarkblue'], lw=1.5, zorder=2, clip_on=False)

        if stop_idx:
            # compare the loop index (extended by one dim) with the start index tuple (except the last actual index)
            eq_mask = np.array(np.array(idx_wo_qois_ts)[:, None] == np.array(stop_idx[:-1]))

            # sum up the boolean mask and check that there is exactly one match where all values are True
            eq_mask = np.array(np.sum(eq_mask, axis=0) == eq_mask.shape[0])
            if eq_mask.sum() != 1:
                raise ValueError("there must be exactly one match in the stop_idx for the selected data.")

            # get the index of the one true element
            eq_idx = eq_mask.nonzero()[0][0]

            # get the actual start index
            vertical_stop = stop_idx[-1][eq_idx]

            # plot vertical lines
            ax1.axvline(x=time_untrimmed[vertical_stop], ymin=-1.2, ymax=1,
                        color=colors['TUMsecondaryDarkblue'], lw=1.5, zorder=2, clip_on=False)
            ax2.axvline(x=time_untrimmed[vertical_stop], ymin=0, ymax=1,
                        color=colors['TUMsecondaryDarkblue'], lw=1.5, zorder=2, clip_on=False)

        # -- PLOT KPI --------------------------------------------------------------------------------------------------

        # -- plot the KPI into the time series of the distance to line qoi
        if qois_kpi_da.dims != ():
            qois_kpi_loop_da = qois_kpi_sel_da.loc[loop_idx_dict]
            kpi_index = np.where(qois_ts_untrimmed_loop_da.loc[{'qois': d2l}].data ==
                                 qois_kpi_loop_da.loc[{'qois': d2l}].data)[0][0]
            ax2.scatter(time[kpi_index], qois_kpi_loop_da.loc[{'qois': d2l}].data,
                        marker='X', color=colors['TUMaccentOrange'], s=40, label='KPI', zorder=3)

        # -- FORMAT PLOT -----------------------------------------------------------------------------------------------

        # add labels and legend
        # ax1.legend()
        ax2.set_xlabel('Time (s)')
        ax1.set_ylabel('Lateral Acceleration (m/s$^2$)')
        ax2.set_ylabel('Distance to Line (m)')

        # -- SAVE PLOT -------------------------------------------------------------------------------------------------

        if save_path:
            # create a sub-folder if it does not already exist
            if not os.path.exists(os.path.dirname(save_path)):
                os.mkdir(os.path.dirname(save_path))

            # save the plot
            idx_string = '_'.join(str(idx) for idx in idx_wo_qois_ts)
            pickle.dump(fig, open(save_path + '_' + idx_string + '.pkl', "wb"))
            [plt.savefig(save_path + '_' + idx_string + '.' + fmt, bbox_inches='tight') for fmt in format_list]

            # close the plot if it was automatically opened
            plt.close()
        else:
            # visualize the plot
            plt.show()

    return


def plot_kpi_surface(config, cfgpl, scenarios_da, qois_kpi_raw_da, plot_type='', save_path=''):
    """
    This function plots a response surface of KPIs across the scenario space.

    The idx_dict must contain a 'qois' and a 'parameters' key.
    The 'qois'-key must contain exactly one string value: qoi name from the decision arrays.
    The 'parameters'-key must contain a list of either two or three string values: parameter names from scenario arrays.
    The order of the parameter names in the list determines the x and y (and z) axes in the plot.
    The idx_dict can contain a 'space_samples' key to index only a subset of the whole data, e.g. slice(None, None, 10).
    This is important for high-dimensional spaces to only visualize one layer in the 2D scatter plot.
    The idx_dict can contain the dimensions 'repetitions', 'aleatory_samples' or 'epistemic_samples' to index a subset.

    It creates a surface plot if the selected data is one-dimensional.
    Otherwise it creates a scatter plot.
    Depending on the 'parameters'-key, the scatter plot is two or three dimensional.

    :param dict config: user configuration
    :param dict cfgpl: user configuration of this plot
    :param xr.DataArray scenarios_da: array with scenarios
    :param xr.DataArray qois_kpi_raw_da: array with KPIs
    :param str plot_type: (optional) type of the plot: surface, scatter or stem (depending on dimensionality!)
    :param str save_path: (optional) path where the plots shall be saved (otherwise they will be shown)
    """
    idx_dict = cfgpl['idx_dict']

    # -- FORMAT PLOT ---------------------------------------------------------------------------------------------------

    if config['project']['id'] in {'MDPI_PoC_Paper', 'SIMPAT_MMU_Paper'}:
        # adapt font sizes of the plots
        matplotlib.rcParams.update({'font.size': 16})
        plt.rc('font', size=14)  # controls default text sizes
        plt.rc('axes', titlesize=14)  # fontsize of the axes title
        plt.rc('axes', labelsize=14)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=14)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=14)  # fontsize of the tick labels
        plt.rc('legend', fontsize=13.5)  # legend fontsize
        plt.rc('figure', titlesize=14)  # fontsize of the figure title

    # -- INDEX DATA ----------------------------------------------------------------------------------------------------

    number_space_dimensions = len(idx_dict['parameters'])

    # create dictionaries to index the x and y parameter and the output KPI
    scenario_idx_dict = idx_dict.copy()
    scenario_idx_dict.pop('qois')
    scenario_x_idx_dict = scenario_idx_dict.copy()
    scenario_x_idx_dict['parameters'] = scenario_x_idx_dict['parameters'][0]
    scenario_y_idx_dict = scenario_idx_dict.copy()
    scenario_y_idx_dict['parameters'] = scenario_y_idx_dict['parameters'][1]
    if number_space_dimensions == 3:
        scenario_z_idx_dict = scenario_idx_dict.copy()
        scenario_z_idx_dict['parameters'] = scenario_z_idx_dict['parameters'][2]
    kpi_idx_dict = idx_dict.copy()
    kpi_idx_dict.pop('parameters')
    if 'repetitions' in kpi_idx_dict and 'repetitions' not in qois_kpi_raw_da.dims:
        kpi_idx_dict.pop('repetitions')

    # extract the data
    x = scenarios_da.loc[scenario_x_idx_dict].data
    y = scenarios_da.loc[scenario_y_idx_dict].data
    if number_space_dimensions == 3:
        # noinspection PyUnboundLocalVariable
        z = scenarios_da.loc[scenario_z_idx_dict].data
    kpi_array = qois_kpi_raw_da.loc[kpi_idx_dict].data

    # -- CREATE PLOT ---------------------------------------------------------------------------------------------------

    # create the surface plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cbar = 0

    if x.ndim == 1:
        # -- space scenarios without uncertainties

        if not plot_type:
            # set the surface plot as default if not specified by the user
            plot_type = 'surface'

        if plot_type == 'surface':
            # revert the coolwarm colormap so that red values symbolize low values (e.g. for distance to line)
            # coolwarm_reverted = ListedColormap(cm.coolwarm.colors[::-1])

            # create a surface plot
            ax.plot_trisurf(x, y, kpi_array, cmap=cm.coolwarm)

        elif plot_type == 'scatter':
            # create a scatter plot
            ax.scatter(x, y, kpi_array, color=colors['TUMprimaryBlue'], s=10)

        elif plot_type == 'stem':
            # manually create a stem3d plot since matplotlib does not provide it

            # draw a vertical line from zero to the z value of each scenario result
            for xx, yy, zz in zip(x, y, kpi_array):
                ax.plot([xx, xx], [yy, yy], [0, zz], '-')

            # add the typical scatter plot as point on top of the vertical line
            ax.scatter(x, y, kpi_array, color=colors['TUMprimaryBlue'], s=10)

        else:
            raise ValueError("This plot type is not available for this data.")

    else:
        # -- scenarios with uncertainties

        if number_space_dimensions == 2:
            # -- two scenario parameters: we can use the third axis for the KPIs

            if not plot_type:
                # set the scatter plot as default if not specified by the user
                plot_type = 'scatter'

            if plot_type == 'scatter':
                # create a 3D scatter plot with constant color
                ax.scatter(x, y, kpi_array, color=colors['TUMprimaryBlue'], s=5)

            elif plot_type == 'mean_surface_stem_uncertainty':

                # calculate the min and max value across all repetitions per scenario
                kpi_min_array = np.nanmin(kpi_array, axis=1)
                kpi_max_array = np.nanmax(kpi_array, axis=1)

                # draw a vertical line from the min value to the max value of each scenario result as uncertainty
                for xx, yy, zz_min, zz_max in zip(x[:, 0], y[:, 0], kpi_min_array, kpi_max_array):
                    ax.plot([xx, xx], [yy, yy], [zz_min, zz_max], '-')

                # add the typical scatter plot as points on the vertical line
                ax.scatter(x, y, kpi_array, color=colors['TUMprimaryBlue'], s=5)

                # calculate mean for debugging
                kpi_mean_array = np.nanmean(kpi_array, axis=1)

                # create a surface plot based on the mean value
                ax.plot_trisurf(x[:, 0], y[:, 0], kpi_mean_array, alpha=0.3, cmap=cm.coolwarm)

            else:
                raise ValueError("This plot type is not available for this data.")

        elif number_space_dimensions == 3:
            # -- three scenario parameters: we have to encode the KPIs in the color dimension

            if not plot_type:
                # set the scatter plot as default if not specified by the user
                plot_type = 'scatter'

            if plot_type == 'scatter':
                # create a 3D scatter plot and encode the KPI in the color dimension
                # noinspection PyUnboundLocalVariable
                sc = ax.scatter(x, y, z, c=kpi_array.flatten(), s=5)
                cbar = plt.colorbar(sc)
            else:
                raise ValueError("This plot type is not available for this data.")

        else:
            raise ValueError("The kpi surface plot supports only two or three parameters.")

    # -- FORMAT PLOT ---------------------------------------------------------------------------------------------------

    if config['project']['id'] == 'SIMPAT_MMU_Paper':
        # set perspective of the 3D plot
        ax.view_init(azim=130, elev=30)
        ax.dist = 11.5

        # for phd thesis
        ax.set_zlim(0, 0.5)

    if config['project']['id'] == 'MDPI_PoC_Paper':
        ax.set_zlim(0, 0.7)

    # axes labels
    pdict = config['cross_domain']['parameters']['parameters_dict']
    qdict = config['cross_domain']['assessment']['qois_dict']
    ax.set_xlabel(pdict[idx_dict['parameters'][0]]['axes_label'])
    ax.set_ylabel(pdict[idx_dict['parameters'][1]]['axes_label'])
    if (x.ndim == 1) or (number_space_dimensions == 2):
        ax.set_zlabel(qdict[idx_dict['qois']]['axes_label'])
    elif number_space_dimensions == 3:
        ax.set_zlabel(pdict[idx_dict['parameters'][2]]['axes_label'])
        cbar.ax.set_ylabel(qdict[idx_dict['qois']]['axes_label'], rotation=270)
    else:
        raise ValueError("The kpi surface plot supports only two or three parameters.")

    # title
    if config['project']['id'] not in {'MDPI_PoC_Paper', 'SIMPAT_MMU_Paper'}:
        ax.set_title('Distribution of model KPIs across the application space')

    # -- SAVE PLOT -----------------------------------------------------------------------------------------------------

    if save_path:
        # create a sub-folder if it does not already exist
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(os.path.dirname(save_path))

        # save the plot
        pickle.dump(ax, open(save_path + '.pkl', "wb"))
        [plt.savefig(save_path + '.' + fmt, bbox_inches='tight') for fmt in format_list]

        # close the plot if it was automatically opened
        plt.close()
    else:
        # visualize the plot
        plt.show()

    return


def plot_cdf(config, cfgpl, qois_kpi_raw_da, fill_flag=False, qois_kpi_da=None, save_path=''):
    """
    This function plots an Empirical (step function) Cumulative Distribution Function (ECDF).

    :param dict config: user configuration
    :param dict cfgpl: user configuration of this plot
    :param xr.DataArray qois_kpi_raw_da: array with KPIs, in the raw form without CDFs
    :param bool fill_flag: flag to fill the area between CDFs
    :param xr.DataArray qois_kpi_da: array with KPIs, in the form of a pbox
    :param str save_path: path where the plots shall be saved (otherwise they will be shown)
    :return:
    """
    idx_dict = cfgpl['idx_dict']

    # -- INDEX DATA ----------------------------------------------------------------------------------------------------

    # check if only selected space samples shall be plotted
    if 'space_samples' in idx_dict:
        space_start_idx = idx_dict['space_samples'].start
        space_end_idx = idx_dict['space_samples'].stop
    else:
        space_start_idx = 0
        space_end_idx = qois_kpi_raw_da.shape[qois_kpi_raw_da.dims.index('space_samples')]

    # recalculate CDFs
    qois_kpi_cdf_da = Assessment.Assessment.create_ecdf(qois_kpi_raw_da)

    # loop through all the selected space samples
    for i in range(space_start_idx, space_end_idx):

        # create the figure
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # extract the selected part of the data array
        qois_kpi_raw_sel_da = qois_kpi_cdf_da.loc[{'qois': idx_dict['qois'], 'space_samples': i}]

        # -- PLOT DATA -------------------------------------------------------------------------------------------------

        # check if a single or multiple CDFs are choosen
        if qois_kpi_raw_sel_da.dims == 'aleatory_samples':
            # -- Single CDF
            # store a legend information
            legend_str = ('Epistemic Sample: ' + str(idx_dict['epistemic_samples']))

            # plot a step function for the CDF
            plt.step(qois_kpi_raw_sel_da, qois_kpi_cdf_da.probs, where='post',
                     color=colors['TUMprimaryBlue'], label=legend_str)

        elif set(qois_kpi_raw_sel_da.dims) == {'epistemic_samples', 'aleatory_samples'}:
            # -- Multiple CDFs

            # loop through the CDFs
            for j in range(qois_kpi_raw_da.epistemic_samples.shape[0]):
                # store a legend information
                legend_str = ('Epistemic Sample: ' + str(j))

                # plot a step function for each CDF
                plt.step(qois_kpi_raw_sel_da[j], qois_kpi_cdf_da.probs, where='post', label=legend_str)
        else:
            raise IndexError("The selected array dimensions do not match the CDF plot.")

        if fill_flag:
            # loop through the aleatory samples
            for p in range(1, qois_kpi_da.aleatory_samples.shape[0]):
                # extract both pbox edges
                qois_kpi_left_da = qois_kpi_da.loc[{'qois': idx_dict['qois'], 'space_samples': i,
                                                   'pbox_edges': 'left', 'aleatory_samples': p}]
                qois_kpi_right_da = qois_kpi_da.loc[{'qois': idx_dict['qois'], 'space_samples': i,
                                                    'pbox_edges': 'right', 'aleatory_samples': p}]

                # fill rectangular areas of the step functions
                rect1 = matplotlib.patches.Rectangle((qois_kpi_left_da.data, qois_kpi_da.probs[p-1]),
                                                     (qois_kpi_right_da - qois_kpi_left_da).data,
                                                     qois_kpi_da.probs[p] - qois_kpi_da.probs[p-1],
                                                     color=colors['TUMprimaryBlue'], alpha=0.3, linewidth=0)
                ax.add_patch(rect1)

        # -- FORMAT PLOT -----------------------------------------------------------------------------------------------

        # add labels, grid, and legend
        qdict = config['cross_domain']['assessment']['qois_dict']
        plt.xlabel(qdict[idx_dict['qois']]['axes_label'])
        plt.ylabel('Cumulative Probability')
        plt.grid(True)
        plt.legend()

        # -- SAVE PLOT -------------------------------------------------------------------------------------------------

        if save_path:
            # create a sub-folder if it does not already exist
            if not os.path.exists(os.path.dirname(save_path)):
                os.mkdir(os.path.dirname(save_path))

            # save the plot
            pickle.dump(ax, open(save_path + '_' + str(i) + '.pkl', "wb"))
            [plt.savefig(save_path + '_' + str(i) + '.' + fmt, bbox_inches='tight') for fmt in format_list]

            # close the plot if it was automatically opened
            plt.close()
        else:
            # visualize the plot
            plt.show()

    return


def plot_area_metrics(config, cfgpl, pbox_y_model, pbox_y_system, pbox_x_model_list, pbox_x_system_list, metric_da,
                      scenarios_da, save_path=''):
    """
    This function plots the area between CDFs or pboxes (see area validation metrics).

    :param dict config: user configuration
    :param dict cfgpl: user configuration of this plot
    :param np.ndarray pbox_y_model: y vector of the model
    :param np.ndarray pbox_y_system: y vector of the system
    :param list[np.ndarray, np.ndarray] pbox_x_model_list: list with left and right x vector of the model
    :param list[np.ndarray, np.ndarray] pbox_x_system_list: list with left and right x vector of the system
    :param xr.DataArray metric_da: array with validation metric results
    :param xr.DataArray scenarios_da: array with scenarios
    :param str save_path: path where the plots shall be saved (otherwise they will be shown)
    :return:
    """
    idx_dict = cfgpl['idx_dict']

    # -- FORMAT PLOT ---------------------------------------------------------------------------------------------------

    if config['project']['id'] in {'MDPI_PoC_Paper', 'SIMPAT_MMU_Paper'}:
        # adapt font sizes of the plots
        matplotlib.rcParams.update({'font.size': 16})
        plt.rc('font', size=17)  # controls default text sizes
        plt.rc('axes', titlesize=17)  # fontsize of the axes title
        plt.rc('axes', labelsize=17)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=17)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=17)  # fontsize of the tick labels
        plt.rc('legend', fontsize=15)  # legend fontsize
        plt.rc('figure', titlesize=17)  # fontsize of the figure title

    # -- INDEX AND PLOT DATA -------------------------------------------------------------------------------------------

    # get the index of the selected qoi
    qoi_idx = np.where(metric_da.coords['qois'] == idx_dict['qois'])[0][0]

    # check if only selected space samples shall be plotted
    if 'space_samples' in idx_dict:
        space_start_idx = idx_dict['space_samples'].start
        space_end_idx = idx_dict['space_samples'].stop
    else:
        space_start_idx = 0
        space_end_idx = metric_da.shape[metric_da.dims.index('space_samples')]

    # loop through all the selected space samples
    for i in range(space_start_idx, space_end_idx):

        # create a new figure
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # extract the selected qoi and one space sample from the x arrays
        sp_pbox_x_model_list = [x[qoi_idx, i, :] for x in pbox_x_model_list]
        sp_pbox_x_system_list = [x[qoi_idx, i, :] for x in pbox_x_system_list]

        # exclude the nan values representing invalid cdf steps in case of varying repetitions
        sp_pbox_x_model_list = [x[~np.isnan(x)] for x in sp_pbox_x_model_list]
        sp_pbox_x_system_list = [x[~np.isnan(x)] for x in sp_pbox_x_system_list]

        if isinstance(pbox_y_model, np.ndarray) and isinstance(pbox_y_system, np.ndarray):
            # -- case: equal number of cdf steps in 1D arrays pbox_y_model and pbox_y_system
            sp_pbox_y_model = pbox_y_model[:]
            sp_pbox_y_system = pbox_y_system[:]

        elif isinstance(pbox_y_model, list) and isinstance(pbox_y_system, list):
            # -- case: variable number of cdf steps in list of 1D arrays pbox_y_model and pbox_y_systems
            sp_pbox_y_model = pbox_y_model[i]
            sp_pbox_y_system = pbox_y_system[i]

        else:
            raise ValueError("The argument pbox_y_model should either be of type np.ndarray or of type list.")

        # plot the left and right ECDFs of the model pbox (step functions)
        if np.sum(sp_pbox_x_model_list[0] != sp_pbox_x_model_list[1]):
            # -- full pbox case
            plt.step(sp_pbox_x_model_list[0], sp_pbox_y_model, where='post',
                     zorder=2, color=colors['TUMprimaryBlue'], linewidth=4, label='Model Input Uncertainty')
            plt.step(sp_pbox_x_model_list[1], sp_pbox_y_model, where='post',
                     zorder=2, color=colors['TUMprimaryBlue'], linewidth=4)
        else:
            # -- degenerate single ECDF case
            plt.step(sp_pbox_x_model_list[0], sp_pbox_y_model, where='post',
                     zorder=2, color=colors['TUMprimaryBlue'], linewidth=4, label='Model')

        # plot the left and right ECDFs of the system pbox (step functions)
        plt.step(sp_pbox_x_system_list[0], sp_pbox_y_system, where='post',
                 zorder=3, color='black', linewidth=4, label='System')
        plt.step(sp_pbox_x_system_list[1], sp_pbox_y_system, where='post',
                 zorder=3, color='black', linewidth=4)

        # -- FILL AREAS ------------------------------------------------------------------------------------------------

        # re-merge the ecdf functions
        dim_idx = -1
        y_unique, sp_pbox_model_merged, sp_pbox_system_merged = am.merge_ecdf_functions(
            sp_pbox_y_model, sp_pbox_y_system, sp_pbox_x_model_list, sp_pbox_x_system_list, dim_idx)

        # loop through the aleatory samples
        for p in range(1, len(sp_pbox_model_merged[0])):
            # fill the rectangular areas between the CDFs
            rect1 = matplotlib.patches.Rectangle(
                (sp_pbox_model_merged[0][p], y_unique[p - 1]), sp_pbox_model_merged[1][p] - sp_pbox_model_merged[0][p],
                y_unique[p] - y_unique[p - 1],
                zorder=1, color=colors['TUMprimaryBlue'], alpha=0.3, linewidth=0)
            ax.add_patch(rect1)

        # -- left area
        # get each step of left areas
        x1 = sp_pbox_model_merged[0]
        x2 = sp_pbox_system_merged[0]
        idx = np.nonzero(x1 > x2)

        # if left areas exist
        if idx[0].size != 0:
            # loop through each step
            # label = 'Model-form Uncertainty'
            label = 'Left area'
            for p in range(x1[idx].shape[0]):
                # fill the rectangular areas
                rect1 = matplotlib.patches.Rectangle(
                    (x1[idx][p], y_unique[idx[0] - 1][p]),
                    x2[idx][p] - x1[idx][p], y_unique[idx][p] - y_unique[idx[0] - 1][p],
                    zorder=0, color=colors['TUMaccentGreen'], alpha=0.3, linewidth=0, label=label)
                label = None
                ax.add_patch(rect1)

        # -- right area
        # get each step of right areas
        x1 = sp_pbox_model_merged[1]
        x2 = sp_pbox_system_merged[0]
        idx = np.nonzero(x1 < x2)

        # if right areas exist
        if idx[0].size != 0:
            # loop through each step
            # label = 'Model-form Uncertainty'
            label = 'Right area'
            for p in range(x1[idx].shape[0]):
                # fill the rectangular areas
                rect1 = matplotlib.patches.Rectangle(
                    (x1[idx][p], y_unique[idx[0] - 1][p]),
                    x2[idx][p] - x1[idx][p], y_unique[idx][p] - y_unique[idx[0] - 1][p],
                    zorder=0, color=colors['TUMaccentOrange'], alpha=0.3, linewidth=0, label=label)
                label = None
                ax.add_patch(rect1)

        # in case of the interval area metric, there are two more areas (four in total)
        if config['validation']['metric']['metric'] == 'iavm':
            # -- left worst case area
            # get each step of left worst case areas
            x1 = sp_pbox_model_merged[0]
            x2 = sp_pbox_system_merged[0]
            idx = np.nonzero(x1 > x2)

            # if left worst case areas exist
            if idx[0].size != 0:
                # loop through each step
                for p in range(x1[idx].shape[0]):
                    # fill the rectangular areas
                    rect1 = matplotlib.patches.Rectangle(
                        (x1[idx][p], y_unique[idx[0] - 1][p]),
                        x2[idx][p] - x1[idx][p], y_unique[idx][p] - y_unique[idx[0] - 1][p],
                        zorder=0, color=colors['TUMaccentGreen'], alpha=0.3, linewidth=0)
                    ax.add_patch(rect1)

            # -- right worst case area
            # get each step of right worst case areas
            x1 = sp_pbox_model_merged[1]
            x2 = sp_pbox_system_merged[1]
            idx = np.nonzero(x1 < x2)

            # if right worst case areas exist
            if idx[0].size != 0:
                # loop through each step
                for p in range(x1[idx].shape[0]):
                    # fill the rectangular areas
                    rect1 = matplotlib.patches.Rectangle(
                        (x1[idx][p], y_unique[idx[0] - 1][p]),
                        x2[idx][p] - x1[idx][p], y_unique[idx][p] - y_unique[idx[0] - 1][p],
                        zorder=0, color=colors['TUMaccentOrange'], alpha=0.3, linewidth=0)
                    ax.add_patch(rect1)

        # -- ADD AREA VALUES AS TEXT BOX -------------------------------------------------------------------------------

        # -- get the plot settings accociated with each validation metric
        if config['validation']['metric']['metric'] == 'avm':
            # extract the selected metrics
            metric_sel_array = metric_da.loc[{'qois': idx_dict['qois'], 'space_samples': i, 'interval': 'left'}].data

            # create text, title and legend
            textstr = ('Area Metric = ' + "%.2f" % metric_sel_array)
            title = 'Area Metric - '

        elif config['validation']['metric']['metric'] == 'mavm':
            # extract the selected metrics
            metric_left_array = metric_da.loc[{'qois': idx_dict['qois'], 'space_samples': i, 'interval': 'left'}].data
            metric_right_array = metric_da.loc[{'qois': idx_dict['qois'], 'space_samples': i, 'interval': 'right'}].data

            # create text, title and legend
            textstr = ('Left Area = ' + "%.2f" % metric_left_array +
                       "\n" + 'Right Area = ' + "%.2f" % metric_right_array)
            title = 'Modified Area Metric - '

        elif config['validation']['metric']['metric'] == 'iavm':
            # extract the selected metrics
            metric_left_array = metric_da.loc[{'qois': idx_dict['qois'], 'space_samples': i, 'interval': 'left'}].data
            metric_right_array = metric_da.loc[{'qois': idx_dict['qois'], 'space_samples': i, 'interval': 'right'}].data

            # create text, title and legend
            textstr = ('Intervall Area = [' + "%.2f" % metric_left_array + ' ; ' + "%.2f" % metric_right_array + ']')
            title = 'Intervall Area Metric - '
        else:
            raise ValueError("this area validation metric is not available.")

        # add legend and area metric values as text box
        props = dict(boxstyle='square', facecolor='white', alpha=0.5)
        if config['project']['id'] == 'SIMPAT_MMU_Paper':
            plt.legend(loc=(-0.1, -0.32), framealpha=0.9, ncol=2, handleheight=1, labelspacing=0.025)
            ax.text(0.35, 0.95, textstr, transform=ax.transAxes, fontsize=15, verticalalignment='top', bbox=props)

        else:
            plt.legend(loc='center', bbox_to_anchor=(0.5, -0.3),
                       framealpha=0.9, ncol=2, handleheight=1, labelspacing=0.025, markerscale=2)
            ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=15,
                    verticalalignment='bottom', horizontalalignment='right', bbox=props)

        # -- FORMAT PLOT -----------------------------------------------------------------------------------------------

        # add scenario point to plot title
        if config['project']['id'] not in {'MDPI_PoC_Paper', 'SIMPAT_MMU_Paper'}:
            title_scenario = ''
            for k in range(scenarios_da.attrs['number_space_parameters']):
                title_add = ("%.2f" % scenarios_da[dict(space_samples=i, epistemic_samples=0,
                                                        aleatory_samples=0, parameters=k)].data + '/')
                title_scenario = title_scenario + title_add
            title_compl = title + title_scenario[:-1]

            # add the title
            plt.title(title_compl)

        # add labels, grid, and limits
        plt.ylabel('Cumulative Probability')
        plt.grid(True)
        plt.ylim(0, 1)

        qdict = config['cross_domain']['assessment']['qois_dict']
        plt.xlabel(qdict[idx_dict['qois']]['axes_label'])

        # -- SAVE PLOT -------------------------------------------------------------------------------------------------

        if save_path:
            # create a sub-folder if it does not already exist
            if not os.path.exists(os.path.dirname(save_path)):
                os.mkdir(os.path.dirname(save_path))

            # save the plot
            pickle.dump(ax, open(save_path + '_' + str(i) + '.pkl', "wb"))
            [plt.savefig(save_path + '_' + str(i) + '.' + fmt, bbox_inches='tight') for fmt in format_list]

            # close the plot if it was automatically opened
            plt.close()
        else:
            # visualize the plot
            plt.show()

    return


def plot_extrapolation_surface(config, cfgpl, scenarios_validation_da, metric_validation_da, scenarios_application_da,
                               error_validation_da, save_path=''):
    """
    This function plots the response surface of the error model including prediction interval and the metric results.

    The idx_dict must contain a 'qois' and a 'parameters' key.
    The 'qois'-key must contain exactly one string value: qoi name from the decision arrays.
    The 'parameters'-key must contain a list of two string values: parameter names from scenario arrays.
    The order of the parameter names in the list determines the x, y and z axes in the plot.
    The idx_dict can contain a 'space_samples' key to index only a subset of the whole data, e.g. slice(None, None, 10).
    This is important for high-dimensional spaces to only visualize one layer in the 2D scatter plot.

    In case of non-deterministic data, the idx_dict should contain an 'interval'-key.
    Then, either the left or the right metric results are plotted (scatter points) and three surfaces for the lower and
    upper prediction interval and the regression estimate.
    In case of deterministic data, the idx_dict should not contain an 'interval'-key.
    Then, the metric results are plotted (scatter points) and the error estimation with both bounds (three surfaces).

    :param dict config: user configuration
    :param dict cfgpl: user configuration of this plot
    :param xr.DataArray scenarios_validation_da: array with validation scenarios
    :param xr.DataArray metric_validation_da: array with validation metric results
    :param xr.DataArray scenarios_application_da: array with application scenarios
    :param xr.DataArray error_validation_da: array with inferred errors across the application domain
    :param str save_path: (optional) path where the plots shall be saved (otherwise they will be shown)
    """
    idx_dict = cfgpl['idx_dict']

    # -- FORMAT PLOT ---------------------------------------------------------------------------------------------------

    if config['project']['id'] in {'MDPI_PoC_Paper', 'SIMPAT_MMU_Paper'}:
        # adapt font sizes of the plots
        matplotlib.rcParams.update({'font.size': 16})
        plt.rc('font', size=14)  # controls default text sizes
        plt.rc('axes', titlesize=14)  # fontsize of the axes title
        plt.rc('axes', labelsize=14)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=14)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=14)  # fontsize of the tick labels
        plt.rc('legend', fontsize=14)  # legend fontsize
        plt.rc('figure', titlesize=14)  # fontsize of the figure title

    # -- INDEX METRIC DATA ---------------------------------------------------------------------------------------------

    # create dictionaries to index the scenario array to get the parameter for the x and y axis
    scenarios_x_idx_dict = idx_dict.copy()
    scenarios_x_idx_dict.pop('qois')
    scenarios_x_idx_dict.pop('interval', None)
    scenarios_x_idx_dict['parameters'] = idx_dict['parameters'][0]
    scenarios_y_idx_dict = scenarios_x_idx_dict.copy()
    scenarios_y_idx_dict['parameters'] = idx_dict['parameters'][1]

    # create dictionary to index the metric arrays
    metric_idx_dict = idx_dict.copy()
    metric_idx_dict.pop('parameters')

    # -- PLOT METRIC DATA ----------------------------------------------------------------------------------------------

    # create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # create the scatter plot
    ax.scatter(scenarios_validation_da.loc[scenarios_x_idx_dict],
               scenarios_validation_da.loc[scenarios_y_idx_dict],
               metric_validation_da.loc[metric_idx_dict].data,
               color=colors['TUMaccentOrange'], s=60, label='Validation Metric', depthshade=0)

    # -- INDEX INFERENCE DATA ------------------------------------------------------------------------------------------

    # extract the application scenarios
    x = scenarios_application_da.loc[scenarios_x_idx_dict].data
    y = scenarios_application_da.loc[scenarios_y_idx_dict].data

    # -- plot three surfaces in non-deterministic case (see docstring)

    # extract the estimated error
    error_idx_dict = idx_dict.copy()
    error_idx_dict.pop('parameters')
    error_sel_da = error_validation_da.loc[error_idx_dict]

    # extract the prediction intervals
    lower_pi = error_sel_da.loc[{'prediction_interval': 'lower'}].data
    upper_pi = error_sel_da.loc[{'prediction_interval': 'upper'}].data
    regression = error_sel_da.loc[{'prediction_interval': 'regression'}].data

    # -- PLOT INFERENCE DATA -------------------------------------------------------------------------------------------

    # plot the response surfaces
    surf_1 = ax.plot_trisurf(x, y, lower_pi, alpha=0.6, color=colors['TUMprimaryBlue'], label='Lower Bound')
    surf_3 = ax.plot_trisurf(x, y, regression, alpha=0.4, color=colors['TUMprimaryBlue'], label='Inferred Error')
    surf_2 = ax.plot_trisurf(x, y, upper_pi, alpha=0.6, color=colors['TUMprimaryBlue'], label='Upper Bound')

    # workaround for matplotlib issue: 'Poly3DCollection' object has no attribute '_facecolors2d'
    # https://github.com/matplotlib/matplotlib/issues/4067
    # noinspection PyProtectedMember
    surf_1._facecolors2d, surf_1._edgecolors2d = surf_1._facecolor3d, surf_1._edgecolor3d
    # noinspection PyProtectedMember
    surf_2._facecolors2d, surf_2._edgecolors2d = surf_2._facecolor3d, surf_2._edgecolor3d
    # noinspection PyProtectedMember
    surf_3._facecolors2d, surf_3._edgecolors2d = surf_3._facecolor3d, surf_3._edgecolor3d

    # -- FORMAT PLOT ---------------------------------------------------------------------------------------------------

    plt.grid(True)

    # axes labels
    pdict = config['cross_domain']['parameters']['parameters_dict']
    qdict = config['cross_domain']['assessment']['qois_dict']
    ax.set_xlabel(pdict[idx_dict['parameters'][0]]['axes_label'])
    ax.set_ylabel(pdict[idx_dict['parameters'][1]]['axes_label'])
    ax.set_zlabel('Error ' + qdict[idx_dict['qois']]['axes_label'])

    if config['project']['id'] == 'SIMPAT_MMU_Paper':
        ax.legend(loc=(0.12, -0.2), framealpha=0.9, ncol=2, handleheight=1, labelspacing=0.025)
        ax.view_init(elev=20)
        ax.dist = 11.5

    else:
        plt.legend(loc='center', bbox_to_anchor=(0.5, -0.2),
                   framealpha=0.9, ncol=2, handleheight=1, labelspacing=0.025)

    # -- SAVE PLOT -----------------------------------------------------------------------------------------------------

    if save_path:
        # create a sub-folder if it does not already exist
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(os.path.dirname(save_path))

        # save the plot
        pickle.dump(ax, open(save_path + '.pkl', "wb"))
        [plt.savefig(save_path + '.' + fmt, bbox_inches='tight') for fmt in format_list]

        # close the plot if it was automatically opened
        plt.close()
    else:
        # visualize the plot
        plt.show()

    return


def plot_uncertainty_expansion_nondeterministic(config, cfgpl,
                                                qois_kpi_model_da,
                                                qois_kpi_system_estimated_da,
                                                qois_kpi_system_estimated_validation_da,
                                                scenarios_model_da,
                                                qois_kpi_system_da=xr.DataArray(None),
                                                save_path=''):
    """
    This function plots the three non-deterministic model uncertainties, the true system uncertainty and the regulation.

    :param dict config: user configuration
    :param dict cfgpl: user configuration of this plot
    :param xr.DataArray qois_kpi_model_da: array with model kpis in the application domain
    :param xr.DataArray qois_kpi_system_estimated_da: array with estimated system responses
    :param xr.DataArray qois_kpi_system_estimated_validation_da: array with estimated system responses, only validation
    :param xr.DataArray scenarios_model_da: array with application scenarios
    :param xr.DataArray qois_kpi_system_da: (optional) array with the true system kpis in the application domain
    :param str save_path: (optional) path where the plots shall be saved (otherwise they will be shown)
    :return:
    """
    idx_dict = cfgpl['idx_dict']

    # -- FORMAT PLOT ---------------------------------------------------------------------------------------------------

    if config['project']['id'] in {'MDPI_PoC_Paper', 'SIMPAT_MMU_Paper'}:
        # adapt font sizes of the plots
        matplotlib.rcParams.update({'font.size': 16})
        plt.rc('font', size=15)  # controls default text sizes
        plt.rc('axes', titlesize=15)  # fontsize of the axes title
        plt.rc('axes', labelsize=15)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=15)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=15)  # fontsize of the tick labels
        plt.rc('legend', fontsize=14)  # legend fontsize
        plt.rc('figure', titlesize=15)  # fontsize of the figure title

    # -- CREATE PLOT ---------------------------------------------------------------------------------------------------

    # check if only selected space samples shall be plotted
    if 'space_samples' in idx_dict:
        space_start_idx = idx_dict['space_samples'].start
        space_end_idx = idx_dict['space_samples'].stop
    else:
        space_start_idx = 0
        space_end_idx = qois_kpi_model_da.shape[qois_kpi_model_da.dims.index('space_samples')]

    # loop through the space samples
    for i in range(space_start_idx, space_end_idx):

        # create a new figure
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # -- INDEX INPUT UNCERTAINTY -----------------------------------------------------------------------------------

        # extract the selected pbox edges
        qois_kpi_model_left_sel_da = qois_kpi_model_da.loc[
            {'qois': idx_dict['qois'], 'space_samples': i, 'pbox_edges': 'left'}]
        qois_kpi_model_right_sel_da = qois_kpi_model_da.loc[
            {'qois': idx_dict['qois'], 'space_samples': i, 'pbox_edges': 'right'}]

        # -- INDEX AND PLOT MODEL-FORM UNCERTAINTY ---------------------------------------------------------------------

        # extract the pbox edges of the model-form uncertainties
        qois_kpi_system_estimated_validation_left_sel_da = qois_kpi_system_estimated_validation_da.loc[
            {'qois': idx_dict['qois'], 'space_samples': i, 'pbox_edges': 'left'}]
        qois_kpi_system_estimated_validation_right_sel_da = qois_kpi_system_estimated_validation_da.loc[
           {'qois': idx_dict['qois'], 'space_samples': i, 'pbox_edges': 'right'}]

        # plot the pbox edges of the model-form uncertainties
        plt.step(qois_kpi_system_estimated_validation_left_sel_da, qois_kpi_model_da.probs, where='post',
                 color=colors['TUMaccentGreen'], linewidth=4, label='Model-form Uncertainty')
        plt.step(qois_kpi_system_estimated_validation_right_sel_da, qois_kpi_model_da.probs, where='post',
                 color=colors['TUMaccentGreen'], linewidth=4)

        # determine the deviations to fill the area in the plots
        deviation_left_sel_da = qois_kpi_model_left_sel_da - qois_kpi_system_estimated_validation_left_sel_da
        deviation_right_sel_da = qois_kpi_system_estimated_validation_right_sel_da - qois_kpi_model_right_sel_da

        # loop through the aleatory samples
        for p in range(1, len(qois_kpi_model_left_sel_da)):
            # fill the area of the model-form uncertainty
            rect_left = matplotlib.patches.Rectangle(
                (qois_kpi_system_estimated_validation_left_sel_da[{'aleatory_samples': p}].data,
                 qois_kpi_model_da.probs[p - 1]),
                deviation_left_sel_da[{'aleatory_samples': p}].data,
                qois_kpi_model_da.probs[p] - qois_kpi_model_da.probs[p - 1],
                color=colors['TUMaccentGreen'], alpha=0.3, linewidth=0)
            rect_right = matplotlib.patches.Rectangle(
               (qois_kpi_model_right_sel_da[{'aleatory_samples': p}].data,
                qois_kpi_model_da.probs[p - 1]),
               deviation_right_sel_da[{'aleatory_samples': p}].data,
               qois_kpi_model_da.probs[p] - qois_kpi_model_da.probs[p - 1],
               color=colors['TUMaccentGreen'], alpha=0.3, linewidth=0)
            ax.add_patch(rect_left)
            ax.add_patch(rect_right)

        # -- PLOT INPUT UNCERTAINTY ------------------------------------------------------------------------------------

        # plot both pbox edges
        plt.step(qois_kpi_model_left_sel_da, qois_kpi_model_da.probs, where='post',
                 color=colors['TUMprimaryBlue'], linewidth=4, label='Model Input Uncertainty')
        plt.step(qois_kpi_model_right_sel_da, qois_kpi_model_da.probs, where='post',
                 color=colors['TUMprimaryBlue'], linewidth=4)

        # loop through the aleatory samples
        for p in range(1, len(qois_kpi_model_left_sel_da)):
            # fill the pbox area
            rect1 = matplotlib.patches.Rectangle(
                (qois_kpi_model_left_sel_da[{'aleatory_samples': p}].data, qois_kpi_model_da.probs[p - 1]),
                (qois_kpi_model_right_sel_da[{'aleatory_samples': p}] -
                 qois_kpi_model_left_sel_da[{'aleatory_samples': p}]).data,
                qois_kpi_model_da.probs[p] - qois_kpi_model_da.probs[p - 1],
                color=colors['TUMprimaryBlue'], alpha=0.3, linewidth=0)
            ax.add_patch(rect1)

        # -- INDEX AND PLOT NUMERICAL UNCERTAINTY ----------------------------------------------------------------------

        numerical_uncertainty_flag = False
        if numerical_uncertainty_flag:
            # -- extract and plot the numerical uncertainties
            qois_kpi_system_estimated_left_sel_da = qois_kpi_system_estimated_da.loc[
                {'qois': idx_dict['qois'], 'space_samples': i, 'pbox_edges': 'left'}]
            qois_kpi_system_estimated_right_sel_da = qois_kpi_system_estimated_da.loc[
                {'qois': idx_dict['qois'], 'space_samples': i, 'pbox_edges': 'right'}]
            plt.step(qois_kpi_system_estimated_left_sel_da, qois_kpi_model_da.probs, where='post',
                     color=colors['TUMaccentOrange'], linewidth=4)
            plt.step(qois_kpi_system_estimated_right_sel_da, qois_kpi_model_da.probs, where='post',
                     color=colors['TUMaccentOrange'], linewidth=4)

            # determine the deviations to fill the area in the plots
            deviation_left_sel_da =\
                qois_kpi_system_estimated_validation_left_sel_da - qois_kpi_system_estimated_left_sel_da
            deviation_right_sel_da =\
                qois_kpi_system_estimated_right_sel_da - qois_kpi_system_estimated_validation_right_sel_da

            # loop through the aleatory samples
            for p in range(1, len(qois_kpi_model_left_sel_da)):
                # fill the area of the numerical uncertainty
                rect_left = matplotlib.patches.Rectangle(
                    (qois_kpi_system_estimated_left_sel_da[{'aleatory_samples': p}].data,
                     qois_kpi_model_da.probs[p - 1]),
                    deviation_left_sel_da[{'aleatory_samples': p}].data,
                    qois_kpi_model_da.probs[p] - qois_kpi_model_da.probs[p - 1],
                    color=colors['TUMaccentOrange'], alpha=0.3, linewidth=0)
                rect_right = matplotlib.patches.Rectangle(
                    (qois_kpi_system_estimated_validation_left_sel_da[{'aleatory_samples': p}].data,
                     qois_kpi_model_da.probs[p - 1]),
                    deviation_right_sel_da[{'aleatory_samples': p}].data,
                    qois_kpi_model_da.probs[p] - qois_kpi_model_da.probs[p - 1],
                    color=colors['TUMaccentOrange'], alpha=0.3, linewidth=0)
                ax.add_patch(rect_left)
                ax.add_patch(rect_right)

        # -- INDEX AND PLOT REGULATION ---------------------------------------------------------------------------------

        # get the indices of the selected qois
        cfgdm = config['application']['decision_making']
        indices = [k for k, x in enumerate(cfgdm['qois_name_list']) if x == idx_dict['qois']]
        for m in range(len(indices)):
            if cfgdm['qois_type_list'][m] == 'absolute':
                # plot the regulation threshold
                plt.plot([cfgdm['qois_lower_threshold_list'][m],
                          cfgdm['qois_lower_threshold_list'][m]], [0, 1],
                         color='red', linewidth=4, label='Regulation')
            elif cfgdm['qois_type_list'][m] == 'relative':
                raise NotImplementedError('plotting of relative thresholds not yet implemented.')
            else:
                raise ValueError('This threshold type does not exist.')

        # -- INDEX AND PLOT SYSTEM ECDF (GROUND TRUTH)

        if qois_kpi_system_da.dims != ():

            # extract the selected pbox edges
            qois_kpi_system_left_da = qois_kpi_system_da.loc[
                {'qois': idx_dict['qois'], 'space_samples': i, 'pbox_edges': 'left'}]
            qois_kpi_system_right_da = qois_kpi_system_da.loc[
                {'qois': idx_dict['qois'], 'space_samples': i, 'pbox_edges': 'right'}]

            # plot both pbox edges
            plt.step(qois_kpi_system_left_da, qois_kpi_system_da.probs, where='post',
                     color='black', linewidth=4, label='System')
            plt.step(qois_kpi_system_right_da, qois_kpi_system_da.probs, where='post',
                     color='black', linewidth=4)

            # loop through the aleatory samples
            for p in range(1, len(qois_kpi_system_left_da)):
                # fill the pbox area
                rect1 = matplotlib.patches.Rectangle(
                    (qois_kpi_system_left_da[{'aleatory_samples': p}].data, qois_kpi_system_da.probs[p - 1]),
                    (qois_kpi_system_right_da[{'aleatory_samples': p}] -
                     qois_kpi_system_left_da[{'aleatory_samples': p}]).data,
                    qois_kpi_system_da.probs[p] - qois_kpi_system_da.probs[p - 1],
                    color='black', alpha=0.3, linewidth=0)
                ax.add_patch(rect1)

        # -- FORMAT PLOT -----------------------------------------------------------------------------------------------

        # add scenario point to title
        if config['project']['id'] not in {'MDPI_PoC_Paper', 'SIMPAT_MMU_Paper'}:
            title_scenario = ''
            for k in range(scenarios_model_da.attrs['number_space_parameters']):
                title_add = ("%.2f" % scenarios_model_da[dict(space_samples=i, epistemic_samples=0,
                                                              aleatory_samples=0, parameters=k)].data + '/')
                title_scenario = title_scenario + title_add
            title_compl = 'Uncertainty Expansion - ' + title_scenario[:-1]

            # add title
            plt.title(title_compl)

        # add grid, legend, labels, and limits
        plt.grid(True)
        plt.ylabel('Cumulative Probability')
        plt.ylim(0, 1)
        plt.legend(loc=(0.05, -0.3), framealpha=0.9, ncol=2, handleheight=1, labelspacing=0.025)

        if config['project']['id'] == 'SIMPAT_MMU_Paper':
            plt.legend(loc=(0.05, -0.3), framealpha=0.9, ncol=2, handleheight=1, labelspacing=0.025)
            plt.xlim(-0.05, 0.65)

        # axes labels
        qdict = config['cross_domain']['assessment']['qois_dict']
        ax.set_xlabel(qdict[idx_dict['qois']]['axes_label'])

        # -- SAVE PLOT -------------------------------------------------------------------------------------------------

        if save_path:
            # create a sub-folder if it does not already exist
            if not os.path.exists(os.path.dirname(save_path)):
                os.mkdir(os.path.dirname(save_path))

            # save the plot
            pickle.dump(ax, open(save_path + '_' + str(i) + '.pkl', "wb"))
            [plt.savefig(save_path + '_' + str(i) + '.' + fmt, bbox_inches='tight') for fmt in format_list]

            # close the plot if it was automatically opened
            plt.close()
        else:
            # visualize the plot
            plt.show()

    return


def plot_error_integration_deterministic(config, cfgpl, qois_kpi_model_da, qois_kpi_system_estimated_da,
                                         scenarios_model_da, qois_kpi_system_da=xr.DataArray(None), save_path=''):
    """
    This function plots the deterministic model kpis with uncertainty, the true system kpis and the regulation.

    :param dict config: user configuration
    :param dict cfgpl: user configuration of this plot
    :param xr.DataArray qois_kpi_model_da: array with model kpis in the application domain
    :param xr.DataArray qois_kpi_system_estimated_da: array with estimated system values in the application domain
    :param xr.DataArray scenarios_model_da: array with application scenarios
    :param xr.DataArray qois_kpi_system_da: (optional) array with the true system kpis in the application domain
    :param str save_path: (optional) path where the plots shall be saved (otherwise they will be shown)
    :return:
    """
    idx_dict = cfgpl['idx_dict']

    # -- FORMAT PLOT ---------------------------------------------------------------------------------------------------

    if config['project']['id'] in {'MDPI_PoC_Paper', 'SIMPAT_MMU_Paper'}:
        matplotlib.rcParams.update({'font.size': 16})
        plt.rc('font', size=15)  # controls default text sizes
        plt.rc('axes', titlesize=15)  # fontsize of the axes title
        plt.rc('axes', labelsize=15)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=15)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=15)  # fontsize of the tick labels
        plt.rc('legend', fontsize=14)  # legend fontsize
        plt.rc('figure', titlesize=15)  # fontsize of the figure title

    # -- CREATE PLOT ---------------------------------------------------------------------------------------------------

    # check if only selected space samples shall be plotted
    if 'space_samples' in idx_dict:
        space_start_idx = idx_dict['space_samples'].start
        space_end_idx = idx_dict['space_samples'].stop
    else:
        # all space samples require the full range
        space_start_idx = 0
        space_end_idx = qois_kpi_model_da.shape[qois_kpi_model_da.dims.index('space_samples')]

    # loop through the space samples
    for i in range(space_start_idx, space_end_idx):

        # create a new figure
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # -- INDEX AND PLOT MODEL KPIS ---------------------------------------------------------------------------------

        # -- extract and plot the deterministic model kpis
        qois_kpi_model_sel_da = qois_kpi_model_da.loc[{'qois': idx_dict['qois'], 'space_samples': i}]
        plt.plot([qois_kpi_model_sel_da, qois_kpi_model_sel_da], [0, 1],
                 zorder=4, color=colors['TUMprimaryBlue'], linewidth=4, label='Model')

        # -- INDEX AND PLOT MODEL UNCERTAINTY --------------------------------------------------------------------------

        # -- extract and plot the inferred model uncertainty
        qois_kpi_system_estimated_left_sel_da = qois_kpi_system_estimated_da.loc[
            {'qois': idx_dict['qois'], 'space_samples': i, 'interval': 'left'}]
        qois_kpi_system_estimated_right_sel_da = qois_kpi_system_estimated_da.loc[
            {'qois': idx_dict['qois'], 'space_samples': i, 'interval': 'right'}]
        plt.plot([qois_kpi_system_estimated_left_sel_da, qois_kpi_system_estimated_left_sel_da], [0, 1],
                 zorder=0, color=colors['TUMaccentOrange'], linewidth=4, label='Left Uncertainty')
        plt.plot([qois_kpi_system_estimated_right_sel_da, qois_kpi_system_estimated_right_sel_da], [0, 1],
                 zorder=0, color=colors['TUMaccentGreen'], linewidth=4, label='Right Uncertainty')

        # determine the deviations to fill the area in the plots
        # can deviate from error_validation_da in case of uncertainty expansion (not bias correction)
        deviation_right_sel_da = qois_kpi_model_sel_da - qois_kpi_system_estimated_right_sel_da
        deviation_left_sel_da = qois_kpi_model_sel_da - qois_kpi_system_estimated_left_sel_da

        # fill the model uncertainty area in the plot
        rect1 = matplotlib.patches.Rectangle(
            (qois_kpi_system_estimated_left_sel_da.data, 0), deviation_left_sel_da.data, 1,
            zorder=1, color=colors['TUMaccentOrange'], alpha=0.4, linewidth=0)
        rect2 = matplotlib.patches.Rectangle(
            (qois_kpi_system_estimated_right_sel_da.data, 0), deviation_right_sel_da.data, 1,
            zorder=1, color=colors['TUMaccentGreen'], alpha=0.4, linewidth=0)
        ax.add_patch(rect1)
        ax.add_patch(rect2)

        # -- INDEX AND PLOT REGULATION ---------------------------------------------------------------------------------

        # -- plot the regulation threshold
        # get the indices of the selected qoi
        cfgdm = config['application']['decision_making']
        indices = [k for k, x in enumerate(cfgdm['qois_name_list']) if x == idx_dict['qois']]
        for m in range(len(indices)):
            if cfgdm['qois_type_list'][m] == 'absolute':
                # plot the absolute regulation threshold
                plt.plot([cfgdm['qois_lower_threshold_list'][m],
                          cfgdm['qois_lower_threshold_list'][m]], [0, 1],
                         zorder=2, color='red', linewidth=4, label='Regulation')
            elif cfgdm['qois_type_list'][m] == 'relative':
                raise ValueError("plotting of relative thresholds is currently not supported.")
            else:
                raise ValueError("this threshold type is not available.")

        # -- INDEX AND PLOT SYSTEM KPIS --------------------------------------------------------------------------------

        # -- extract and plot the deterministic system kpis (ground truth values)
        if qois_kpi_system_da.dims != ():
            qois_kpi_system_sel_da = qois_kpi_system_da.loc[{'qois': idx_dict['qois'], 'space_samples': i}]
            plt.plot([qois_kpi_system_sel_da, qois_kpi_system_sel_da], [0, 1],
                     zorder=3, color='black', linewidth=4, label='System')

        # -- FORMAT PLOT -----------------------------------------------------------------------------------------------

        # add scenario point to plot title
        if config['project']['id'] not in {'MDPI_PoC_Paper', 'SIMPAT_MMU_Paper'}:
            title_scenario = ''
            for k in range(scenarios_model_da.attrs['number_space_parameters']):
                title_add = ("%.2f" % scenarios_model_da[dict(space_samples=i, parameters=k)].data + '/')
                title_scenario = title_scenario + title_add
            title_compl = 'Uncertainty Expansion Determinsitic - ' + title_scenario[:-1]

            # add the title
            plt.title(title_compl)

        # add further plot settings
        plt.grid(True)
        plt.ylabel('Cumulative Probability')
        plt.ylim(0, 1)

        if config['project']['id'] == 'SIMPAT_MMU_Paper':
            plt.legend(loc=(0.05, -0.3), framealpha=0.9, ncol=2, handleheight=1, labelspacing=0.025)
            plt.xlim(-0.05, 0.65)

        else:
            plt.legend(loc='center', bbox_to_anchor=(0.5, -0.3),
                       framealpha=0.9, ncol=2, handleheight=1, labelspacing=0.025, markerscale=2)

        # axes labels
        qdict = config['cross_domain']['assessment']['qois_dict']
        ax.set_xlabel(qdict[idx_dict['qois']]['axes_label'])

        # -- SAVE PLOT -------------------------------------------------------------------------------------------------

        if save_path:
            # create a sub-folder if it does not already exist
            if not os.path.exists(os.path.dirname(save_path)):
                os.mkdir(os.path.dirname(save_path))

            # save the plot
            pickle.dump(ax, open(save_path + '_' + str(i) + '.pkl', "wb"))
            [plt.savefig(save_path + '_' + str(i) + '.' + fmt, bbox_inches='tight') for fmt in format_list]

            # close the plot if it was automatically opened
            plt.close()
        else:
            # visualize the plot
            plt.show()

    return


def plot_decision_space(config, cfgpl,
                        scenarios_application_da=xr.DataArray(None),
                        decision_application_system_estimated_da=xr.DataArray(None),
                        decision_application_system_da=xr.DataArray(None),
                        decision_application_model_da=xr.DataArray(None),
                        scenarios_validation_da=xr.DataArray(None),
                        decision_validation_da=xr.DataArray(None),
                        decision_validation_system_da=xr.DataArray(None),
                        decision_validation_model_da=xr.DataArray(None),
                        save_path=''):
    """
    This function plots the binary decisions across the scenario space by coloring the points.

    The idx_dict must contain a 'qois' and a 'parameters' key.
    The 'qois'-key must contain exactly one string value: qoi name from the decision arrays.
    The 'parameters'-key must contain a list of either two or three string values: parameter names from scenario arrays.
    Depending on the 'parameters'-key, a two or three dimensional scatter plot will be created.
    The order of the parameter names in the list determines the x, y and z axes in the plot.
    The idx_dict can contain a 'space_samples' key to index only a subset of the whole data, e.g. slice(None, None, 10).
    This is important for high-dimensional spaces to only visualize one layer in the 2D scatter plot.

    The function arguments are not fully optional.
    Please provide either application or validation data, so that something can be plotted.
    Please provide the scenario and decision data together, since they cannot be plotted standalone.

    The decision_system_da and decision_model_da arguments are fully optional.
    However, please provide not both at the same time.

    :param dict config: user configuration
    :param dict cfgpl: user configuration of this plot
    :param xr.DataArray scenarios_application_da: (optional) application scenario array
    :param xr.DataArray decision_application_system_estimated_da: (optional) decisions of the system estimator in appl.
    :param xr.DataArray decision_application_system_da: (optional) decisions of the actual system in the appl. domain
    :param xr.DataArray decision_application_model_da: (optional) decisions of the model in the application domain
    :param xr.DataArray scenarios_validation_da: (optional) validation scenario array
    :param xr.DataArray decision_validation_da: (optional) model accuracy decisions in the validation domain
    :param xr.DataArray decision_validation_system_da: (optional) safety decisions of the system in validation domain
    :param xr.DataArray decision_validation_model_da: (optional) safety decisions of the model in the validation domain
    :param str save_path: (optional) path where the plots shall be saved (otherwise they will be shown)
    """
    idx_dict = cfgpl['idx_dict']

    # -- FORMAT PLOT ---------------------------------------------------------------------------------------------------

    if config['project']['id'] in {'MDPI_PoC_Paper', 'SIMPAT_MMU_Paper'}:
        # adapt font sizes of the plots
        matplotlib.rcParams.update({'font.size': 22})
        plt.rc('font', size=12)  # controls default text sizes
        plt.rc('axes', titlesize=12)  # fontsize of the axes title
        plt.rc('axes', labelsize=12)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
        plt.rc('legend', fontsize=11)  # legend fontsize
        plt.rc('figure', titlesize=12)  # fontsize of the figure title

    # -- VALIDATE ARGS -------------------------------------------------------------------------------------------------

    # providing no data to plot makes no sense
    if scenarios_application_da.dims == () and scenarios_validation_da.dims == ():
        raise ValueError("cannot create plot if no data is provided.")

    # checks within the application domain
    if decision_application_system_estimated_da.dims != () and scenarios_application_da.dims != ():
        # if application decision shall be plotted, decision_application_system_estimated_da is currently required
        # the system decisions and model decisions are optional, but not possible at the same time (only 4 combinations)
        if decision_application_model_da.dims != () and decision_application_system_da.dims != ():
            raise ValueError("The system and model decisions in the application domain cannot be plotted together.")
    elif decision_application_system_estimated_da.dims == () and scenarios_application_da.dims == ():
        # providing no decisions for the application domain is fine -> check passed
        pass
    else:
        raise ValueError("The scenario and decision array have to be provided together.")

    # checks within the validation domain
    if decision_validation_da.dims != () and scenarios_validation_da.dims != ():
        if decision_validation_model_da.dims != () or decision_validation_system_da.dims != ():
            raise ValueError("The accuracy decisions cannot be combined with safety decisions in the validation domain")
    elif (decision_validation_model_da.dims != () or decision_validation_system_da.dims != ()) and (
            scenarios_validation_da.dims != ()):
        if decision_validation_da.dims != ():
            raise ValueError("The accuracy decisions cannot be combined with safety decisions in the validation domain")
    elif decision_validation_da.dims == () and scenarios_validation_da.dims != () and (
            decision_validation_model_da.dims == () and decision_validation_system_da.dims == ()):
        # providing no decisions for the validation domain is fine -> check passed
        pass
    else:
        raise ValueError("The scenario and decision array have to be provided together.")

    # -- CREATE PLOT ---------------------------------------------------------------------------------------------------

    # create a figure
    fig = plt.figure()

    # distinguish between 2D and 3D scatter plots
    number_space_dimensions = len(idx_dict['parameters'])
    if number_space_dimensions == 2:
        ax = plt.gca()
    elif number_space_dimensions == 3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        raise ValueError("the space plot supports only two or three parameters.")

    # -- INDEX AND PLOT DATA -------------------------------------------------------------------------------------------

    # create dictionaries to index the scenario array to get the parameter for the x and y axis
    scenarios_idx_dict = idx_dict.copy()
    scenarios_idx_dict.pop('qois')
    scenarios_x_idx_dict = scenarios_idx_dict.copy()
    scenarios_x_idx_dict['parameters'] = idx_dict['parameters'][0]
    scenarios_y_idx_dict = scenarios_idx_dict.copy()
    scenarios_y_idx_dict['parameters'] = idx_dict['parameters'][1]
    if number_space_dimensions == 3:
        scenarios_z_idx_dict = scenarios_idx_dict.copy()
        scenarios_z_idx_dict['parameters'] = idx_dict['parameters'][2]

    # create dictionary to index the decision arrays
    decision_idx_dict = idx_dict.copy()
    decision_idx_dict.pop('parameters')

    # initialize the legend elements
    legend_elements = []

    # if application decisions shall be plotted (then uncertainty decisions are always included)
    if decision_application_system_estimated_da.dims != ():

        # select the desired qoi (and space samples)
        decision_application_system_estimated_sel_da = decision_application_system_estimated_da.loc[decision_idx_dict]

        # select the desired parameters (and space samples)
        scenarios_application_x_da = scenarios_application_da.loc[scenarios_x_idx_dict]
        scenarios_application_y_da = scenarios_application_da.loc[scenarios_y_idx_dict]
        if number_space_dimensions == 3:
            # noinspection PyUnboundLocalVariable
            scenarios_application_z_da = scenarios_application_da.loc[scenarios_z_idx_dict]

        # create a color list: green if "passed", orange if "failed"
        color_list = []
        for i in range(len(decision_application_system_estimated_sel_da)):
            if decision_application_system_estimated_sel_da[{'space_samples': i}]:
                color_list.append(colors['TUMaccentGreen'])
            else:
                color_list.append(colors['TUMaccentOrange'])

        if decision_application_system_da.dims == () and decision_application_model_da.dims == ():
            # -- case only uncertainty decisions in application domain

            # plot the application scenarios with circles colored in red or green based on the decisions
            if number_space_dimensions == 2:
                # create a 2D scatter plot
                ax.scatter(scenarios_application_x_da, scenarios_application_y_da, color=color_list, s=40, marker="o")
            else:
                # create a 3D scatter plot
                # noinspection PyUnboundLocalVariable
                ax.scatter(scenarios_application_x_da, scenarios_application_y_da, scenarios_application_z_da,
                           color=color_list, s=40, marker="o")

            # store the corresponding legend information
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Regulation Passed',
                                          markerfacecolor=colors['TUMaccentGreen'], markersize=10))
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Regulation Failed',
                                          markerfacecolor=colors['TUMaccentOrange'], markersize=10))

        else:
            # -- this function offers coding the decisions of either the system GT or the model into the plot symbols

            # select the markers
            marker1 = "^"
            marker2 = "x"

            if decision_application_system_da.dims != () and decision_application_model_da.dims == ():
                # -- case system (ground truth) decisions (and uncertainty decisions) in application domain

                # select the desired qoi
                decision_application_system_sel_da = decision_application_system_da.loc[decision_idx_dict]

                # marker 1 will be used if the system agrees with the estimator, marker 2 if it disagrees
                marker1_mask_da = decision_application_system_estimated_sel_da == decision_application_system_sel_da

                # store the corresponding legend information
                legend_elements.append(Line2D([0], [0], marker=marker1, color='black', label='Approval Correct',
                                              markerfacecolor='black', markersize=10))
                legend_elements.append(Line2D([0], [0], color=colors['TUMaccentGreen'], label='Approval Passed'))
                # legend_elements.append(Line2D([0], [0], marker="s", color='w', label='Model Valid',
                #                               markerfacecolor=colors['TUMaccentGreen'], markersize=10))
                legend_elements.append(Line2D([0], [0], marker=marker2, color='black', label='Approval Wrong',
                                              markerfacecolor='black', markersize=10))
                legend_elements.append(Line2D([0], [0], color=colors['TUMaccentOrange'], label='Approval Failed'))
                # legend_elements.append(Line2D([0], [0], marker="s", color='w', label='Model Invalid',
                #                               markerfacecolor=colors['TUMaccentOrange'], markersize=10))

            elif decision_application_model_da.dims != () and decision_application_system_da.dims == ():
                # -- case model decisions (and uncertainty decisions) in application domain

                # select the desired qoi
                decision_application_model_sel_da = decision_application_model_da.loc[decision_idx_dict]

                # marker 1 will be used if the model passes, marker 2 if it fails
                marker1_mask_da = decision_application_model_sel_da

                # store the corresponding legend information
                legend_elements.append(Line2D([0], [0], color=colors['TUMaccentGreen'], marker=marker1,
                                              label='Appl.: Estimator Pass, XiL Pass'))
                legend_elements.append(Line2D([0], [0], color=colors['TUMaccentOrange'], marker=marker1,
                                              label='Appl.: Estimator Fail, XiL Pass'))
                legend_elements.append(Line2D([0], [0], color=colors['TUMaccentGreen'], marker=marker2,
                                              label='Appl.: Estimator Pass, XiL Fail'))
                legend_elements.append(Line2D([0], [0], color=colors['TUMaccentOrange'], marker=marker2,
                                              label='Appl.: Estimator Fail, XiL Fail'))

            else:
                raise ValueError("The system and model decisions in the application domain cannot be plotted together.")

            # matplotlib does currently not support a list of markers (compared to a list of colors)
            # https://github.com/matplotlib/matplotlib/issues/11155
            # so we have to split the data in two groups to perform two separate plot calls later
            # put all objects where marker1_mask_da is True in one group and
            # put all objects where marker1_mask_da is False in another group

            # split the scenarios
            scenarios_application_x_true_da = scenarios_application_x_da[{'space_samples': marker1_mask_da}]
            scenarios_application_x_false_da = scenarios_application_x_da[{'space_samples': ~marker1_mask_da}]
            scenarios_application_y_true_da = scenarios_application_y_da[{'space_samples': marker1_mask_da}]
            scenarios_application_y_false_da = scenarios_application_y_da[{'space_samples': ~marker1_mask_da}]
            if number_space_dimensions == 3:
                # noinspection PyUnboundLocalVariable
                scenarios_application_z_true_da = scenarios_application_z_da[{'space_samples': marker1_mask_da}]
                scenarios_application_z_false_da = scenarios_application_z_da[{'space_samples': ~marker1_mask_da}]

            # split the color list
            color_true_list = [color for (color, m1) in zip(color_list, marker1_mask_da) if m1]
            color_false_list = [color for (color, m1) in zip(color_list, marker1_mask_da) if not m1]

            # -- plots
            # plot the application scenarios with green and orange color and two markers (4 combinations)
            if number_space_dimensions == 2:
                # create 2D scatter plots
                ax.scatter(scenarios_application_x_true_da, scenarios_application_y_true_da,
                           color=color_true_list, s=40, marker=marker1, zorder=1)
                ax.scatter(scenarios_application_x_false_da, scenarios_application_y_false_da,
                           color=color_false_list, s=40, marker=marker2, zorder=3)

            else:
                # create 3D scatter plots
                # noinspection PyUnboundLocalVariable
                ax.scatter(scenarios_application_x_true_da, scenarios_application_y_true_da,
                           scenarios_application_z_true_da, color=color_true_list, s=40, marker=marker1)
                # noinspection PyUnboundLocalVariable
                ax.scatter(scenarios_application_x_false_da, scenarios_application_y_false_da,
                           scenarios_application_z_false_da, color=color_false_list, s=40, marker=marker2)

    # if validation decisions shall be plotted
    if decision_validation_da.dims != () or decision_validation_system_da.dims != () or (
            decision_validation_model_da.dims != ()):

        # select the desired parameters (and space samples)
        scenarios_validation_x_da = scenarios_validation_da.loc[scenarios_x_idx_dict]
        scenarios_validation_y_da = scenarios_validation_da.loc[scenarios_y_idx_dict]
        if number_space_dimensions == 3:
            # noinspection PyUnboundLocalVariable
            scenarios_validation_z_da = scenarios_validation_da.loc[scenarios_z_idx_dict]

        if decision_validation_da.dims != () and decision_validation_system_da.dims == () and (
                decision_validation_model_da.dims == ()):
            # -- case (only) accuracy decisions in validation domain

            # select the desired qoi
            decision_validation_sel_da = decision_validation_da.loc[decision_idx_dict]

            # create a color list: green if "valid" model, orange if "invalid" model
            color_list = []
            for i in range(len(decision_validation_sel_da)):
                if decision_validation_sel_da[{'space_samples': i}]:
                    color_list.append(colors['TUMaccentGreen'])
                else:
                    color_list.append(colors['TUMaccentOrange'])

            # plot the application scenarios with rectangles in green for valid models and in red for invalid models
            if number_space_dimensions == 2:
                # create a 2D scatter plot
                ax.scatter(scenarios_validation_x_da, scenarios_validation_y_da, color=color_list, s=40, marker="s")
            else:
                # create a 3D scatter plot
                # noinspection PyUnboundLocalVariable
                ax.scatter(scenarios_validation_x_da, scenarios_validation_y_da, scenarios_validation_z_da,
                           color=color_list, s=40, marker="s")

            # store the corresponding legend information
            legend_elements.append(Line2D([0], [0], marker="s", color='w', label='Model Valid',
                                          markerfacecolor=colors['TUMaccentGreen'], markersize=10))
            legend_elements.append(Line2D([0], [0], marker="s", color='w', label='Model Invalid',
                                          markerfacecolor=colors['TUMaccentOrange'], markersize=10))

        elif decision_validation_da.dims == () and (
                decision_validation_system_da.dims != () or decision_validation_model_da.dims != ()):
            # -- case (only) safety decisions in validation domain

            # select the markers
            marker1 = "o"
            marker2 = "*"

            # init color list
            color_list = []

            if decision_validation_system_da.dims != ():
                # -- case at least system safety decisions in validation domain

                # select the desired qoi
                decision_validation_system_sel_da = decision_validation_system_da.loc[decision_idx_dict]

                # create a color list: green if system "passed", orange if system "failed"
                for i in range(len(decision_validation_system_sel_da)):
                    if decision_validation_system_sel_da[{'space_samples': i}]:
                        color_list.append(colors['TUMprimaryBlue'])
                    else:
                        color_list.append(colors['TUMprimaryBlack'])

                if decision_validation_model_da.dims == ():
                    # -- case only system safety decisions in validation domain

                    # store the corresponding legend information
                    legend_elements.append(Line2D([0], [0], color=colors['TUMprimaryBlue'], marker="s",
                                                  label='Validation System Passes'))
                    legend_elements.append(Line2D([0], [0], color=colors['TUMprimaryBlack'], marker="s",
                                                  label='Validation System Fails'))

                    # plot the application scenarios with circles colored in red or green based on the decisions
                    if number_space_dimensions == 2:
                        # create a 2D scatter plot
                        ax.scatter(scenarios_validation_x_da, scenarios_validation_y_da, color=color_list, s=40,
                                   marker="s")
                    else:
                        # create a 3D scatter plot
                        # noinspection PyUnboundLocalVariable
                        ax.scatter(scenarios_validation_x_da, scenarios_validation_y_da, scenarios_validation_z_da,
                                   color=color_list, s=40, marker="s")

            else:
                # -- case no system safety and only model safety decisions in validation domain
                # use green as default color
                color_list.append(colors['TUMaccentGreen'])

            if decision_validation_model_da.dims != ():
                # -- case at least model safety decisions in validation domain

                # select the desired qoi
                decision_validation_model_sel_da = decision_validation_model_da.loc[decision_idx_dict]

                # marker 1 will be used if the system passes, marker 2 if it fails
                marker1_mask_da = decision_validation_model_sel_da

                # split the scenarios
                scenarios_validation_x_true_da = scenarios_validation_x_da[{'space_samples': marker1_mask_da}]
                scenarios_validation_x_false_da = scenarios_validation_x_da[{'space_samples': ~marker1_mask_da}]
                scenarios_validation_y_true_da = scenarios_validation_y_da[{'space_samples': marker1_mask_da}]
                scenarios_validation_y_false_da = scenarios_validation_y_da[{'space_samples': ~marker1_mask_da}]
                if number_space_dimensions == 3:
                    # noinspection PyUnboundLocalVariable
                    scenarios_validation_z_true_da = scenarios_validation_z_da[{'space_samples': marker1_mask_da}]
                    scenarios_validation_z_false_da = scenarios_validation_z_da[{'space_samples': ~marker1_mask_da}]

                # split the color list
                color_true_list = [color for (color, m1) in zip(color_list, marker1_mask_da) if m1]
                color_false_list = [color for (color, m1) in zip(color_list, marker1_mask_da) if not m1]

                # store the corresponding legend information
                legend_elements.append(Line2D([0], [0], color=colors['TUMprimaryBlue'], marker=marker1,
                                              label='Val.: System Pass, XiL Pass'))
                legend_elements.append(Line2D([0], [0], color=colors['TUMprimaryBlack'], marker=marker1,
                                              label='Val.: System Fail, XiL Pass'))
                legend_elements.append(Line2D([0], [0], color=colors['TUMprimaryBlue'], marker=marker2,
                                              label='Val.: System Pass, XiL Fail'))
                legend_elements.append(Line2D([0], [0], color=colors['TUMprimaryBlack'], marker=marker2,
                                              label='Val.: System Fail, XiL Fail'))

                # -- plots
                # plot the validation scenarios with green and orange color and two markers (4 combinations)
                if number_space_dimensions == 2:
                    # create 2D scatter plots
                    ax.scatter(scenarios_validation_x_true_da, scenarios_validation_y_true_da,
                               color=color_true_list, s=40, marker=marker1, zorder=0)
                    ax.scatter(scenarios_validation_x_false_da, scenarios_validation_y_false_da,
                               color=color_false_list, s=40, marker=marker2, zorder=2)

                else:
                    # create 3D scatter plots
                    # noinspection PyUnboundLocalVariable
                    ax.scatter(scenarios_validation_x_true_da, scenarios_validation_y_true_da,
                               scenarios_validation_z_true_da, color=color_true_list, s=40, marker=marker1)
                    # noinspection PyUnboundLocalVariable
                    ax.scatter(scenarios_validation_x_false_da, scenarios_validation_y_false_da,
                               scenarios_validation_z_false_da, color=color_false_list, s=40, marker=marker2)

    # -- FORMAT PLOT ---------------------------------------------------------------------------------------------------

    plt.grid(True)

    if config['project']['id'] == 'MDPI_PoC_Paper':
        plt.locator_params(axis='x', nbins=12)
        plt.xlim(60, 180)
        plt.locator_params(axis='y', nbins=10)
        plt.ylim(0, 1)
        plt.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.3),
                   framealpha=0.9, ncol=2, handleheight=1, labelspacing=0.025)

    else:
        plt.legend(handles=legend_elements, loc=(0.155, -0.31),
                   framealpha=0.9, ncol=2, handleheight=1, labelspacing=0.025)

    # axes labels
    pdict = config['cross_domain']['parameters']['parameters_dict']
    ax.set_xlabel(pdict[idx_dict['parameters'][0]]['axes_label'])
    ax.set_ylabel(pdict[idx_dict['parameters'][1]]['axes_label'])
    if number_space_dimensions == 3:
        ax.set_zlabel(pdict[idx_dict['parameters'][2]]['axes_label'])

    # -- SAVE PLOT -----------------------------------------------------------------------------------------------------

    if save_path:
        # create a sub-folder if it does not already exist
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(os.path.dirname(save_path))

        # save the plot
        pickle.dump(ax, open(save_path + '.pkl', "wb"))
        [plt.savefig(save_path + '.' + fmt, bbox_inches='tight') for fmt in format_list]

        # close the plot if it was automatically opened
        plt.close()
    else:
        # visualize the plot
        plt.show()

    return
