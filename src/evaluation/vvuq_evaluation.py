"""
This module is responsible for the evaluation of the VV&UQ methodology itself.

It includes several functions that analyze different aspects. See details in their own documentations.

Contact person: Stefan Riedmaier
Creation date: 20.08.2020
Python version: 3.8
"""

# -- IMPORTS -----------------------------------------------------------------------------------------------------------
# -- built-in imports --
import os
import pickle
import csv

# -- third-party imports --
import numpy as np
import xarray as xr
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch

# -- custom imports --


# -- MODULE-LEVEL VARIABLES --------------------------------------------------------------------------------------------
# create a list with file formats for storage
format_list = ['png', 'pdf']

PGF = False
if PGF:
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

def boolean_classifier(idx_dict, decision_system_estimated_da, decision_system_da, decision_model_da,
                       qois_kpi_system_estimated_da, qois_kpi_system_da, save_path='', percentage_flag=False):
    """
    This function compares binary decision making results to their corresponding ground truth values.

    It compares the decision from the nominal model with the GT and the decision from the VVUQ methodology with the GT.
    The binary classifier creates a confusion matrix for the comparisons.

    It checks whether the GT lies within the bounds from uncertainty expansion.

    It offers saving the results via pickle and in form of tables via further file formats.

    :param dict idx_dict: dictionary to index the data arrays
    :param xr.DataArray decision_system_estimated_da: array of boolean decisions from the system estimation
    :param xr.DataArray decision_system_da: array of boolean decisions from the system
    :param xr.DataArray decision_model_da: array of boolean decisions from the nominal model
    :param xr.DataArray qois_kpi_system_estimated_da: array of responses from the system estimation
    :param xr.DataArray qois_kpi_system_da: array of resonses from the system
    :param str save_path: (optional) path where the plots shall be saved (otherwise they will be shown)
    :param bool percentage_flag: (optional) select whether the classifier results shall be given as percentage ratios
    """

    # extract the selected qois
    decision_system_sel_da = decision_system_da.loc[idx_dict]
    decision_system_estimated_sel_da = decision_system_estimated_da.loc[idx_dict]
    decision_model_sel_da = decision_model_da.loc[idx_dict]
    kpi_idx_dict = idx_dict.copy()
    if 'pbox_edges' in qois_kpi_system_da.dims:
        # skip the first -inf value from the aleatory step function
        kpi_idx_dict['aleatory_samples'] = slice(1, None)
    qois_kpi_system_estimated_sel_da = qois_kpi_system_estimated_da.loc[kpi_idx_dict]
    qois_kpi_system_sel_da = qois_kpi_system_da.loc[kpi_idx_dict]

    # -- create the confusion matrix of the VVUQ results in the application domain
    classifier_dict = dict()
    # we define a True Positive (TP) as system fails and model-based fails
    classifier_dict['VVUQ_TP'] = ~decision_system_sel_da.data & ~decision_system_estimated_sel_da.data
    # we define a True Negative (TN) as system passes and model-based passes
    classifier_dict['VVUQ_TN'] = decision_system_sel_da.data & decision_system_estimated_sel_da.data
    # we define a False Positive (FP) as system passes and model-based fails (convicting an innocent, Type I error)
    classifier_dict['VVUQ_FP'] = decision_system_sel_da.data & ~decision_system_estimated_sel_da.data
    # we define a False Negative (FN) as system fails and model-based passes (acquitting a criminal, Type II error)
    classifier_dict['VVUQ_FN'] = ~decision_system_sel_da.data & decision_system_estimated_sel_da.data

    # -- create the confusion matrix of the ideal model results in the application domain
    classifier_dict['Model_TP'] = ~decision_system_sel_da.data & ~decision_model_sel_da.data
    classifier_dict['Model_TN'] = decision_system_sel_da.data & decision_model_sel_da.data
    classifier_dict['Model_FP'] = decision_system_sel_da.data & ~decision_model_sel_da.data
    classifier_dict['Model_FN'] = ~decision_system_sel_da.data & decision_model_sel_da.data

    # -- check whether the Ground Truth lies within the model-based uncertainties
    if 'pbox_edges' in qois_kpi_system_da.dims:
        # check whether both the left GT edge is greater than the left model-based edge and
        # the right GT edge is smaller than the right model-based edge
        is_greater_left_da = qois_kpi_system_sel_da.loc[{'pbox_edges': 'left'}] > qois_kpi_system_estimated_sel_da.loc[
            {'pbox_edges': 'left'}]
        is_smaller_right_da =\
            qois_kpi_system_sel_da.loc[{'pbox_edges': 'right'}] < qois_kpi_system_estimated_sel_da.loc[
                {'pbox_edges': 'right'}]
        al_idx = is_greater_left_da.dims.index('aleatory_samples')
        is_within_bounds_da = np.all(is_greater_left_da, axis=al_idx) & np.all(is_smaller_right_da, axis=al_idx)

    elif 'interval' in qois_kpi_system_estimated_da.dims:
        # check whether the deterministic GT value lies within the model-based interval bounds
        is_greater_left_da = qois_kpi_system_sel_da > qois_kpi_system_estimated_sel_da.loc[{'interval': 'left'}]
        is_smaller_right_da = qois_kpi_system_sel_da < qois_kpi_system_estimated_sel_da.loc[{'interval': 'right'}]
        is_within_bounds_da = is_greater_left_da & is_smaller_right_da

    else:
        raise ValueError("within bounds check just possible for pboxes or intervals.")

    # store the bounds check in the dictionary
    classifier_dict['GT_in_bounds'] = is_within_bounds_da.data

    # -- analyze the GT-check in relation to the VVUQ confusion matrix
    classifier_dict['GT_in_bounds_TP'] = classifier_dict['GT_in_bounds'] & classifier_dict['VVUQ_TP']
    classifier_dict['GT_in_bounds_TN'] = classifier_dict['GT_in_bounds'] & classifier_dict['VVUQ_TN']
    classifier_dict['GT_in_bounds_FP'] = classifier_dict['GT_in_bounds'] & classifier_dict['VVUQ_FP']
    classifier_dict['GT_in_bounds_FN'] = classifier_dict['GT_in_bounds'] & classifier_dict['VVUQ_FN']

    # save the negative counterparts for completeness
    classifier_dict['GT_not_in_bounds'] = ~classifier_dict['GT_in_bounds']
    classifier_dict['GT_not_in_bounds_TP'] = ~classifier_dict['GT_in_bounds'] & classifier_dict['VVUQ_TP']
    classifier_dict['GT_not_in_bounds_TN'] = ~classifier_dict['GT_in_bounds'] & classifier_dict['VVUQ_TN']
    classifier_dict['GT_not_in_bounds_FP'] = ~classifier_dict['GT_in_bounds'] & classifier_dict['VVUQ_FP']
    classifier_dict['GT_not_in_bounds_FN'] = ~classifier_dict['GT_in_bounds'] & classifier_dict['VVUQ_FN']

    # -- determine the amount of classifications for each dict element
    classifier_sum_dict = dict()
    for key, value in classifier_dict.items():
        classifier_sum_dict[key] = np.sum(value)

    # if the classifier results shall be calculated as percentage ratios
    if percentage_flag:
        # determine the denominator values
        number_space_samples = decision_model_da.shape[decision_model_da.dims.index('space_samples')]
        classifier_max_dict = dict.fromkeys(classifier_sum_dict, number_space_samples)
        classifier_max_dict['GT_in_bounds_TP'] = classifier_sum_dict['VVUQ_TP']
        classifier_max_dict['GT_not_in_bounds_TP'] = classifier_sum_dict['VVUQ_TP']
        classifier_max_dict['GT_in_bounds_TN'] = classifier_sum_dict['VVUQ_TN']
        classifier_max_dict['GT_not_in_bounds_TN'] = classifier_sum_dict['VVUQ_TN']
        classifier_max_dict['GT_in_bounds_FP'] = classifier_sum_dict['VVUQ_FP']
        classifier_max_dict['GT_not_in_bounds_FP'] = classifier_sum_dict['VVUQ_FP']
        classifier_max_dict['GT_in_bounds_FN'] = classifier_sum_dict['VVUQ_FN']
        classifier_max_dict['GT_not_in_bounds_FN'] = classifier_sum_dict['VVUQ_FN']

        for key in classifier_sum_dict.keys():
            # calculate the ratios
            if classifier_max_dict[key] != 0:
                classifier_sum_dict[key] = classifier_sum_dict[key] / classifier_max_dict[key]
            else:
                classifier_sum_dict[key] = 100

    # -- determine further measures based on the confusion matrix
    # the recall or sensitivity is defined as the ratio of TPs to the sum of TPs and FNs
    classifier_sum_dict['VVUQ_recall'] =\
        classifier_sum_dict['VVUQ_TP'] / (classifier_sum_dict['VVUQ_TP'] + classifier_sum_dict['VVUQ_FN'])
    # the precision is defined as the ratio of TPs to the sum of TPs and FPs
    classifier_sum_dict['VVUQ_precision'] = \
        classifier_sum_dict['VVUQ_TP'] / (classifier_sum_dict['VVUQ_TP'] + classifier_sum_dict['VVUQ_FP'])

    # -- Visualisation of the VVUQ Results
    ax = plt.figure()
    plt.table(cellText=[[str(classifier_sum_dict['VVUQ_TP']) +
                         ' (In: ' + str(classifier_sum_dict['GT_in_bounds_TP']) +
                         ' / Out: ' + str(classifier_sum_dict['GT_not_in_bounds_TP']) + ')',
                         str(classifier_sum_dict['VVUQ_FP']) +
                         ' (In: ' + str(classifier_sum_dict['GT_in_bounds_FP']) +
                         ' / Out: ' + str(classifier_sum_dict['GT_not_in_bounds_FP']) + ')'],
                        [str(classifier_sum_dict['VVUQ_FN']) +
                         ' (In: ' + str(classifier_sum_dict['GT_in_bounds_FN']) +
                         ' / Out: ' + str(classifier_sum_dict['GT_not_in_bounds_FN']) + ')',
                         str(classifier_sum_dict['VVUQ_TN']) +
                         ' (In: ' + str(classifier_sum_dict['GT_in_bounds_TN']) +
                         ' / Out: ' + str(classifier_sum_dict['GT_not_in_bounds_TN']) + ')']],
              colLabels=('System Fails', 'System Passes'),
              rowLabels=('VVUQ Fails', 'VVUQ Passes'), loc='center',
              colWidths=[0.25, 0.25], rowLoc='right', cellLoc='center')
    plt.title('Binary Classifier - Methodology')

    # -- save plot
    if save_path:
        # create a sub-folder if it does not already exist
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(os.path.dirname(save_path))

        # save the plot
        pickle.dump(ax, open(save_path + '_vvuq' + '.pkl', "wb"))
        [plt.savefig(save_path + '_vvuq' + '.' + fmt, bbox_inches='tight') for fmt in format_list]

        # close the plot if it was automatically opened
        plt.close()
    else:
        # visualize the plot
        plt.show()

    # -- Visualisation of the Model Results
    ax = plt.figure()
    plt.table(cellText=[[str(classifier_sum_dict['Model_TP']), str(classifier_sum_dict['Model_FP'])],
                        [str(classifier_sum_dict['Model_FN']), str(classifier_sum_dict['Model_TN'])]],
              colLabels=('System Fails', 'System Passes'),
              rowLabels=('Model Fails', 'Model Passes'), loc='center',
              colWidths=[0.25, 0.25], rowLoc='right', cellLoc='center')
    plt.title('Binary Classifier - Input')

    if save_path:
        # save the plot
        pickle.dump(ax, open(save_path + '_model' + '.pkl', "wb"))
        [plt.savefig(save_path + '_model' + '.' + fmt, bbox_inches='tight') for fmt in format_list]

        # close the plot if it was automatically opened
        plt.close()
    else:
        # visualize the plot
        plt.show()

    if save_path:
        # save the dictionary in a csv file
        with open(save_path + '.csv', 'w') as csv_file:
            # write dict column-wise: first column for keys, second column for values
            writer = csv.writer(csv_file)
            for row in classifier_sum_dict.items():
                writer.writerow(row)

            # # write dict row-wise: first row for keys, second row for values
            # w = csv.DictWriter(csv_file, classifier_sum_dict.keys())
            # w.writeheader()
            # w.writerow(classifier_sum_dict)

    return


def evaluation_area_metric(config, idx_dict, qois_kpi_model_da, save_path=''):
    """
    This function calculates and visualizes the area between the regulation and the model p-box to evaluate the buffer.

    :param dict config: user configuration
    :param dict idx_dict: dictionary to index the data arrays
    :param xr.DataArray qois_kpi_model_da: array with model kpis in the application domain
    :param str save_path: path where the plots shall be saved (otherwise they will be shown)
    :return: average area between regulation and pbox across all space samples
    """
    cfgdm = config['application']['decision_making']

    # check if only selected space samples shall be plotted
    if 'space_samples' in idx_dict:
        space_start_idx = idx_dict['space_samples'].start
        space_end_idx = idx_dict['space_samples'].stop
    else:
        space_start_idx = 0
        space_end_idx = qois_kpi_model_da.shape[qois_kpi_model_da.dims.index('space_samples')]

    # extract the regulation treshold of the selected qoi
    qoi_idx = cfgdm['qois_name_list'].index(idx_dict['qois'])
    if idx_dict['evaluation_direction'] == 'left':
        threshold = cfgdm['qois_lower_threshold_list'][qoi_idx]
    else:
        threshold = cfgdm['qois_upper_threshold_list'][qoi_idx]

    # create a 1D threshold array to match the aleatory samples
    number_aleatory_samples = qois_kpi_model_da.shape[qois_kpi_model_da.dims.index('aleatory_samples')]
    thresh_array = np.ones(number_aleatory_samples) * threshold
    thresh_array[0] = np.NINF

    # get the probabilities of the aleatory samples
    probs = qois_kpi_model_da.probs

    # iterate through the space samples
    evaluation_metric = np.zeros(space_end_idx)
    for i in range(space_start_idx, space_end_idx):

        # create a new figure
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # extract the selected data so that only the aleatory samples of the pbox edges remain
        pbox_left = qois_kpi_model_da.loc[{'qois': idx_dict['qois'], 'space_samples': i, 'pbox_edges': 'left'}].data
        pbox_right = qois_kpi_model_da.loc[{'qois': idx_dict['qois'], 'space_samples': i, 'pbox_edges': 'right'}].data

        if idx_dict['evaluation_direction'] == 'left':
            pbox_edge = pbox_left

            # get the indices where the threshold is on the left of the ECDF
            idx = np.nonzero(pbox_edge > thresh_array)

            # calc the area to the left
            area_1d = (pbox_edge[idx] - thresh_array[idx]) * (probs[idx[-1]] - probs[idx[-1] - 1])
        else:
            pbox_edge = pbox_right

            # get the indices where the threshold is on the right of the ECDF
            idx = np.nonzero(pbox_edge < thresh_array)

            # calc the area to the right
            area_1d = (thresh_array[idx] - pbox_edge[idx]) * (probs[idx[-1]] - probs[idx[-1] - 1])

        # sum up the area for all aleatory samples
        evaluation_metric[i] = np.sum(area_1d)

        # plot the pbox edges
        plt.step(pbox_left, probs, where='post', color=colors['TUMprimaryBlue'], label='Input Uncertainty')
        plt.step(pbox_right, probs, where='post', color=colors['TUMprimaryBlue'])

        # fill the pbox area between the pbox edges
        for p in range(1, len(pbox_left)):
            rect1 = matplotlib.patches.Rectangle(
                (pbox_left[p], probs[p - 1]), pbox_right[p] - pbox_left[p], probs[p] - probs[p - 1],
                color=colors['TUMprimaryBlue'], alpha=0.3, linewidth=0)
            ax.add_patch(rect1)

        # plot the regulation threshold
        plt.plot([threshold, threshold], [0, 1], color='black', label='Regulation')

        # fill the evaluation area between the regulation threshold and the selected pbox edge
        if idx_dict['evaluation_direction'] == 'left':
            for p in range(pbox_left[idx].shape[0]):
                rect1 = matplotlib.patches.Rectangle(
                    (thresh_array[idx][p], probs[idx[0]-1][p]),
                    pbox_edge[idx][p] - thresh_array[idx][p], probs[idx][p] - probs[idx[0]-1][p],
                    color=colors['TUMaccentOrange'], alpha=0.3, linewidth=0)
                ax.add_patch(rect1)
        else:
            for p in range(pbox_right[idx].shape[0]):
                # fill the rectangular areas
                rect1 = matplotlib.patches.Rectangle(
                    (pbox_edge[idx][p], probs[idx[0]-1][p]),
                    thresh_array[idx][p] - pbox_edge[idx][p], probs[idx][p] - probs[idx[0]-1][p],
                    color=colors['TUMaccentOrange'], alpha=0.3, linewidth=0)
                ax.add_patch(rect1)

        # add the evaluation area metric value to the plot
        textstr = ('Evaluation Area Metric = ' + "%.2f" % evaluation_metric[i])
        props = dict(boxstyle='square', facecolor='white', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=props)

        # forma the plot
        legend_elements = [Patch(facecolor=colors['TUMaccentOrange'], edgecolor=colors['TUMaccentOrange'],
                                 alpha=0.3, label='Evaluation Area Metric'),
                           Patch(facecolor=colors['TUMprimaryBlue'], edgecolor=colors['TUMprimaryBlue'],
                                 alpha=0.3, label='P-Box'),
                           Patch(facecolor='black', edgecolor='black', label='Regulation Threshold')]
        ax.set_ylim(0, 1)
        ax.legend(handles=legend_elements)
        ax.grid(True)

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

    # average the evaluation area metrics
    evaluation_metric_average = np.average(evaluation_metric)

    return evaluation_metric_average
