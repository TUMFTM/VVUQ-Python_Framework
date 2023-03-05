"""
This module is responsible for saving and loading informations via csv files.

It includes one parent class CsvHandler and several child classes for different framework blocks.
See details in their respective documentations and the main documentation in the doc folder.

Contact person: Stefan Riedmaier
Creation date: 12.06.2020
Python version: 3.8
"""

# -- IMPORTS -----------------------------------------------------------------------------------------------------------
# -- built-in imports --
import os
from pathlib import Path

# -- third-party imports --
import numpy as np
import pandas as pd
import xarray as xr

# -- custom imports --


# -- CLASSES -----------------------------------------------------------------------------------------------------------

class CsvHandler:
    """
    This class is a parent class for csv handlers.
    """

    def __init__(self, config, domain, instance):
        """
        This method initializes a new class instance.

        :param dict config: configuration dictionary
        :param str domain: type of VVUQ domain
        :param str instance: test instance
        """

        self.config = config
        self.domain = domain

        self.csv_path = config['cross_domain'][instance]['result_folder'] + '/' + domain + '/parameter_erg_mapping.csv'


class ScenarioCsvHandler(CsvHandler):
    """
    This class handles the scenario informations in the csv files.
    """

    def save(self, samples_da, space_samples_da):
        """
        This function stores the sample array to a csv file for archivation or for reloading it later.

        :param xr.DataArray samples_da: array with scenarios to be saved
        :param xr.DataArray space_samples_da: array with ideal space samples to be saved
        :return:
        """

        if self.config[self.domain]['scenarios']['method'] in {"read_csv", "post"}:
            # if we already read from the csv, we dont need to save it again
            return

        # -- CREATE CSV HEADER -----------------------------------------------------------------------------------------
        # use the term "Parameter" in the first header line
        parameter_list = ['Parameter'] * samples_da.shape[samples_da.dims.index('parameters')]

        # use the type of parameters in the second header line
        type_list = samples_da.parameter_type_list

        # use the parameter names in the third header line
        name_list = samples_da.parameters.values.tolist()

        # use multiindex columns for the header consisting of multiple rows
        columns = pd.MultiIndex.from_tuples(zip(parameter_list, type_list, name_list))

        # -- CREATE CSV INDEX ------------------------------------------------------------------------------------------
        # the index_col has the syntax sp:ep:ap: (if the parameter type exists, respectively)
        # e.g. for 2 space samples and 3 aleatory samples: 1:1:, 1:2:, 1:3:, 2:1:, 2:2:, 2:3:

        # create the range of the space samples
        number_space_samples = samples_da.shape[samples_da.dims.index('space_samples')]
        index_space_array = np.arange(1, number_space_samples + 1)

        if set(samples_da.dims) == {'space_samples', 'parameters'}:
            # -- deterministic

            # add trailing colons
            # index = np.char.add(index_space.astype(str), ':')
            index_samples_list = [str(sp) + ':' for sp in index_space_array]

        elif set(samples_da.dims) == {'space_samples', 'repetitions', 'parameters'} or \
                set(samples_da.dims) == {'space_samples', 'epistemic_samples', 'parameters'} or \
                set(samples_da.dims) == {'space_samples', 'aleatory_samples', 'parameters'}:
            # -- single-loop sampling

            if 'repetitions' in samples_da.dims:
                # single-loop sampling with aleatory and epistemic parameters
                # single loop sampling with unknown epistemic and aleatory parameters
                number_repetitions = samples_da.shape[samples_da.dims.index('repetitions')]
            elif 'epistemic_samples' in samples_da.dims:
                # single-loop sampling with epistemic and without aleatory parameters
                number_repetitions = samples_da.shape[samples_da.dims.index('epistemic_samples')]
            else:
                # single-loop sampling with aleatory and without epistemic parameters
                number_repetitions = samples_da.shape[samples_da.dims.index('aleatory_samples')]

            # create permutations of parameter types
            index_space_total_array = np.repeat(index_space_array, number_repetitions)
            index_repetitions_array = np.arange(1, number_repetitions + 1)
            index_repetitions_total_array = np.tile(index_repetitions_array, number_space_samples)

            # concatenate sequence with colons
            index_samples_list = [str(sp) + ':' + str(ep) + ':' for (sp, ep) in zip(
                index_space_total_array, index_repetitions_total_array)]

        elif set(samples_da.dims) == {'space_samples', 'epistemic_samples', 'aleatory_samples', 'parameters'}:
            # -- two-loop sampling

            # create the range of the epistemic and aleatory samples
            number_epistemic_samples = samples_da.shape[samples_da.dims.index('epistemic_samples')]
            number_aleatory_samples = samples_da.shape[samples_da.dims.index('aleatory_samples')]
            index_epistemic_array = np.arange(1, number_epistemic_samples + 1)
            index_aleatory_array = np.arange(1, number_aleatory_samples + 1)

            # create permutations of parameter types
            index_space_total_array = np.repeat(index_space_array, number_epistemic_samples * number_aleatory_samples)
            index_epistemic_total_array = np.repeat(index_epistemic_array, number_aleatory_samples)
            index_epistemic_total_array = np.tile(index_epistemic_total_array, number_space_samples)
            index_aleatory_total_array = np.tile(index_aleatory_array, number_space_samples * number_epistemic_samples)

            # concatenate sequence with colons
            index_samples_list = [str(sp) + ':' + str(ep) + ':' + str(ap) + ':' for (sp, ep, ap) in
                                  zip(index_space_total_array, index_epistemic_total_array, index_aleatory_total_array)]

        else:
            raise ValueError("The names of the array dimensions have unexpected values.")

        if set(samples_da.dims) == {'space_samples', 'parameters'}:
            # in the deterministic case, the samples and space samples are identical
            index_list = index_samples_list
            samples_array = samples_da.samples_2d

        else:
            # -- otherwise, we append the ideal space samples at the bottom of the csv file

            # add trailing commas to get the index of the space samples (1:, 2:, 3:, etc.)
            index_space_samples_list = [str(sp) + ':' for sp in index_space_array]

            # concatenate the index lists of the samples and space samples
            index_list = index_samples_list + index_space_samples_list

            # concatenate the actual sample data
            samples_array = np.concatenate((samples_da.samples_2d, space_samples_da.data))

        # create the pandas index for the first column of the csv file
        index = pd.Index(index_list)

        # -- CREATE PANDAS DATAFRAME -----------------------------------------------------------------------------------
        # combine the header, index and array in a pandas dataframe
        df = pd.DataFrame(data=samples_array, columns=columns, index=index)

        # -- WRITE DATAFRAME TO CSV ------------------------------------------------------------------------------------
        # create the directories if they do not exist
        Path.mkdir(Path(os.path.dirname(self.csv_path)), parents=True, exist_ok=True)

        df.to_csv(self.csv_path, index=True)

    def load(self):
        """
        This function reads scenarios from a csv file.

        It is aimed at the use case where operations have been performed in advanced and should be reloaded here in the
        same format to reproduce results or to save processing time.

        :return: scenario array, ideal space scenario array
        :rtype: tuple(xr.DataArray, xr.DataArray)
        """

        # -- CSV HEADER HANDLING ---------------------------------------------------------------------------------------
        # read only the header from the csv file
        df = pd.read_csv(self.csv_path, sep=',', header=[0, 1, 2], nrows=0)

        # convert header to a 2d numpy array of strings
        header_array = np.array([*df.columns])

        # extract the first header line to check which columns are parameters
        is_parameter = header_array[:, 0] == 'Parameter'

        # extract the second header line to check the parameter types
        parameter_type_array = header_array[:, 1][is_parameter]
        epistemic_parameter_mask_array = (parameter_type_array == 'epistemic') | (parameter_type_array == 'mixed')
        aleatory_parameter_mask_array = (parameter_type_array == 'aleatory') | (parameter_type_array == 'mixed')

        # determine the number of paramters of each type
        number_space_parameters = np.sum(is_parameter)
        number_epistemic_parameters = np.sum(epistemic_parameter_mask_array)
        number_aleatory_parameters = np.sum(aleatory_parameter_mask_array)

        # check whether the csv file includes space parameters
        if number_space_parameters < 1:
            raise ValueError("invalid csv file header")

        # extract the third header line with the parameter names
        parameter_names = header_array[is_parameter, 2]

        # -- CSV DATA HANDLING -----------------------------------------------------------------------------------------
        # read only the relevant data columns (without multi-index header)
        df = pd.read_csv(self.csv_path, sep=',', header=0, skiprows=[0, 1], index_col=0,
                         usecols=np.arange(is_parameter.sum() + 1))

        # -- CSV INDEX HANDLING ----------------------------------------------------------------------------------------
        # analyze the index_col specifying the number of samples per parameter type in the syntax "sp:ep:al:"
        # split the number of samples at each column (excluding the trailing one)
        number_samples_list = [row[:-1].split(':') for row in df.index.to_list()]
        number_distinct_samples = len(number_samples_list[0])
        number_space_samples = int(number_samples_list[-1][0])

        # -- DATA RESHAPE ----------------------------------------------------------------------------------------------
        if number_distinct_samples == 1:
            # -- deterministic

            # check that a deterministic simulation contains no epistemic and aleatory parameters
            if number_epistemic_parameters > 1 or number_aleatory_parameters > 1:
                raise ValueError("invalid csv file header")

            # create dataarray without reshaping in the 2D deterministic case
            samples_da = xr.DataArray(
                df.values, dims=('space_samples', 'parameters'), coords={'parameters': parameter_names})

            # in the deterministic case, the samples and space samples are identical
            space_samples_da = samples_da

        elif number_distinct_samples in {2, 3}:

            # at first, we extract the space samples from the bottom of the csv file and put them in a data array
            space_samples_array = df.values[-number_space_samples:, :]
            space_samples_da = xr.DataArray(
                space_samples_array, dims=('space_samples', 'parameters'), coords={'parameters': parameter_names})

            # then, we remove the space samples part from the dataframe and number_samples_list
            df = df.iloc[:-number_space_samples, :]
            number_samples_list = number_samples_list[:-number_space_samples]

            # -- finally, we can focus on the actual sample data
            if number_distinct_samples == 2:
                # -- single-loop sampling

                # specify the dimensions of the data array
                if number_epistemic_parameters > 0 and number_aleatory_parameters > 0:
                    # single-loop sampling with aleatory and epistemic parameters
                    dims = ('space_samples', 'repetitions', 'parameters')
                elif number_epistemic_parameters == 0 and number_aleatory_parameters > 0:
                    # single-loop sampling with aleatory and without epistemic parameters
                    dims = ('space_samples', 'aleatory_samples', 'parameters')
                elif number_epistemic_parameters > 0 and number_aleatory_parameters == 0:
                    # single-loop sampling with epistemic and without aleatory parameters
                    dims = ('space_samples', 'epistemic_samples', 'parameters')
                elif number_epistemic_parameters == 0 and number_aleatory_parameters == 0:
                    # single loop sampling with unknown epistemic and aleatory parameters
                    dims = ('space_samples', 'repetitions', 'parameters')
                else:
                    raise ValueError("invalid csv file header")

                # extract the number of repetitions from the index list
                number_repetitions_list = [
                    int(number_samples_list[i - 1][1]) for i in range(1, len(number_samples_list))
                    if number_samples_list[i][0] != number_samples_list[i - 1][0]]
                number_repetitions_list += [int(number_samples_list[-1][1])]

                # check whether all space samples have the same number of repetitions
                if all(rep == number_repetitions_list[0] for rep in number_repetitions_list):

                    # reshape the 2d array to a 3d array and create the data array
                    shape = (number_space_samples, number_repetitions_list[0], df.values.shape[-1])
                    samples_da = xr.DataArray(df.values.reshape(shape), dims=dims,
                                              coords={'parameters': parameter_names})

                else:
                    # use the maximum number of repetitions to determine the shape of the data array
                    shape = (number_space_samples, max(number_repetitions_list), df.values.shape[-1])

                    # initialize an array with nan values
                    data_array = np.ones(shape=shape) * np.nan

                    # create lists from the csv index column to index the array dimensions
                    space_idx_list = [int(idx[0]) - 1 for idx in number_samples_list]
                    repetition_idx_list = [int(idx[1]) - 1 for idx in number_samples_list]

                    # replace the nan values with the actual ones
                    data_array[space_idx_list, repetition_idx_list, :] = df.values

                    # create the data array
                    samples_da = xr.DataArray(data_array, dims=dims, coords={'parameters': parameter_names})

            else:
                # -- two-loop sampling

                # check that a full non-deterministic simulation contains epistemic and aleatory parameters
                if number_epistemic_parameters < 1 and number_aleatory_parameters < 1:
                    raise ValueError("invalid csv file header")

                # reshape the 2d array to a 4d array
                number_epistemic_samples = int(number_samples_list[-1][1])
                number_aleatory_samples = int(number_samples_list[-1][2])
                shape = (number_space_samples, number_epistemic_samples, number_aleatory_samples, df.values.shape[-1])
                samples_da = xr.DataArray(df.values.reshape(shape),
                                          dims=('space_samples', 'epistemic_samples', 'aleatory_samples', 'parameters'),
                                          coords={'parameters': parameter_names})

        else:
            raise ValueError("invalid csv file header")

        # store important metadata in the attributes dictionary of the data array
        samples_da.attrs['samples_2d'] = df.values
        samples_da.attrs['space_parameters_name_list'] = parameter_names.tolist()
        samples_da.attrs['epistemic_parameters_name_list'] = parameter_names[epistemic_parameter_mask_array].tolist()
        samples_da.attrs['aleatory_parameters_name_list'] = parameter_names[aleatory_parameter_mask_array].tolist()
        samples_da.attrs['number_space_parameters'] = number_space_parameters
        samples_da.attrs['number_epistemic_parameters'] = number_epistemic_parameters
        samples_da.attrs['number_aleatory_parameters'] = number_aleatory_parameters
        samples_da.attrs['space_parameters_mask_list'] = [True] * number_space_parameters
        samples_da.attrs['is_epistemic_parameter_list'] = epistemic_parameter_mask_array.tolist()
        samples_da.attrs['is_aleatory_parameter_list'] = aleatory_parameter_mask_array.tolist()
        samples_da.attrs['parameter_type_list'] = parameter_type_array.tolist()

        return samples_da, space_samples_da


class SimulatorCsvHandler(CsvHandler):
    """
    This class handles the simulator informations in the csv files.
    """

    def save(self, result_path_list):
        """
        This function stores the result paths for each scenario parameter combination in a csv file.

        :param list[str] result_path_list: list of (relative) file paths for each scenario
        :return:
        """

        # read the csv file to a pandas dataframe
        df = pd.read_csv(self.csv_path, sep=',', header=[0, 1, 2], index_col=0)

        if len(result_path_list) != df.shape[0]:
            # -- non-deterministic cases with ideal space samples
            # add a '-' for the space samples, representing no actual simulations and thus no results
            addendum_list = ['-'] * (df.shape[0] - len(result_path_list))
            result_path_list += addendum_list

        # append a column for the erg paths
        df['Filepath', 'Filepath', 'Filepath'] = result_path_list

        # write the dataframe back to the csv file
        df.to_csv(self.csv_path, index=True)

    def load(self):
        """
        This function reads result paths from a csv file.

        It is aimed at the use case where operations have been performed in advanced and should be reloaded here in the
        same format to reproduce results or to save processing time.

        :return: a list containing the paths to the result files
        :type: list[str]
        """

        # -- CSV HEADER HANDLING ---------------------------------------------------------------------------------------
        # read only the header from the csv file
        df = pd.read_csv(self.csv_path, sep=',', header=[0, 1, 2], index_col=0, nrows=0)

        # convert header to a 2d numpy array of strings
        header_array = np.array([*df.columns])

        # analyze the firsts header line to get the result path column
        is_result = header_array[:, 0] == 'Filepath'

        # -- CSV DATA HANDLING -----------------------------------------------------------------------------------------
        # read only the relevant erg path column (without multi-index header)
        df = pd.read_csv(self.csv_path, sep=',', header=0, skiprows=[0, 1],
                         usecols=header_array[is_result, 0])

        # convert the object type array to a list of strings
        rel_path_list = df['Filepath'].to_list()

        if '-' in rel_path_list:
            # -- non-deterministic cases with ideal space samples
            # drop the dummy '-' strings from the list
            rel_path_list = [rel_path for rel_path in rel_path_list if rel_path != '-']

        # generate absolute paths
        base_path = os.path.dirname(self.csv_path)
        py2res_path_list = [base_path + '/' + rel_path for rel_path in rel_path_list]

        return rel_path_list, py2res_path_list


class AssessmentCsvHandler(CsvHandler):
    """
    This class handles the assessment informations in the csv files.
    """

    def save(self, kpi_da):
        """
        This function stores the kpis for each scenario parameter combination in a csv file.

        :param xr.DataArray kpi_da: kpi array with kpis for each scenario
        :return:
        """

        # -- CREATE CSV HEADER -----------------------------------------------------------------------------------------
        # use the term "KPI" in the first header line
        kpi_list = ['KPI'] * len(self.config['cross_domain']['assessment']['qois_name_list'])

        # use the type of KPI in the second header line
        type_list = self.config['cross_domain']['assessment']['qois_kpi_list']

        # use the qoi names in the third header line
        name_list = self.config['cross_domain']['assessment']['qois_name_list']

        # -- CREATE PANDAS DATAFRAME -----------------------------------------------------------------------------------
        # use multiindex columns for the header consisting of multiple rows
        columns = pd.MultiIndex.from_tuples(zip(kpi_list, type_list, name_list))

        # -- READ, CONCATENATE AND WRITE DATAFRAME TO CSV
        # read the csv file to a pandas dataframe
        base_df = pd.read_csv(self.csv_path, sep=',', header=[0, 1, 2], index_col=0)

        # convert header to a 2d numpy array of strings
        header_array = np.array([*base_df.columns])

        # analyze the firsts header line to check whether there are already KPIs
        is_kpi = header_array[:, 0] == 'KPI'

        # if there are already KPIs, remove these columns to replace them afterwards
        base_df = base_df.iloc[:, ~is_kpi]

        # extract the index
        index_list = base_df.index.to_list()

        # analyze the index_col specifying the number of samples per parameter type in the syntax "sp:ep:al:"
        # split the number of samples at each column (excluding the trailing one)
        number_samples_list = [row[:-1].split(':') for row in index_list]
        number_distinct_samples = len(number_samples_list[0])

        if number_distinct_samples == 2:
            # -- case experimental repetitions

            # remove the space samples from the index list
            number_samples_list = [number_samples for number_samples in number_samples_list if len(number_samples) == 2]

            if 'repetitions' in kpi_da.dims:
                # -- case repetitions with possibly variable number

                # get the maximum number of repetitions
                max_reps = kpi_da.shape[kpi_da.dims.index('repetitions')]

                # determine the indices of valid data (excluding the nan values to fill up to the max. repetitions)
                valid_idx_list = [(int(number_samples_list[i][0]) - 1) * max_reps + int(number_samples_list[i][1]) - 1
                                  for i in range(len(number_samples_list))]

                # exclude the invalid data
                kpi_da.attrs['array2d'] = kpi_da.array2d[valid_idx_list, :]

            else:
                # -- case mean value of experimental repetitions

                # extract the number of repetitions from the index list
                number_repetitions_list = [
                    int(number_samples_list[i - 1][1]) for i in range(1, len(number_samples_list))
                    if number_samples_list[i][0] != number_samples_list[i - 1][0]]
                number_repetitions_list += [int(number_samples_list[-1][1])]

                # repeat the mean value for each repetition
                kpi_da.attrs['array2d'] = np.repeat(kpi_da.array2d, number_repetitions_list, axis=0)

        if kpi_da.array2d.shape[0] != base_df.shape[0]:
            # -- non-deterministic cases with ideal space samples
            # add zeros for the space samples, representing no actual simulations and thus no KPIs
            kpi_array = np.zeros((base_df.shape[0], kpi_da.array2d.shape[1]))

            # replace the first zeros with the actual values
            kpi_array[:kpi_da.array2d.shape[0], :] = kpi_da.array2d
        else:
            kpi_array = kpi_da.array2d

        # combine the header and array in a pandas dataframe
        append_df = pd.DataFrame(data=kpi_array, columns=columns, index=base_df.index)

        # concatenate both data frames
        new_df = pd.concat([base_df, append_df], axis=1)

        # write the dataframe back to the csv file
        new_df.to_csv(self.csv_path, index=True)

    def load(self, responses_da):
        """
        This function reads kpis from a csv file.

        It is aimed at the use case where operations have been performed in advanced and should be reloaded here in the
        same format to reproduce results or to save processing time.

        :param xr.DataArray responses_da: data array with the (possibly dummy) responses from the simulator
        :return: kpi array with kpis for each scenario
        :type: np.ndarray
        """

        # -- CSV HEADER HANDLING ---------------------------------------------------------------------------------------
        # read only the header from the csv file
        df = pd.read_csv(self.csv_path, sep=',', header=[0, 1, 2], index_col=0, nrows=0)

        # convert header to a 2d numpy array of strings
        header_array = np.array([*df.columns])

        # extract the first header line to check which columns are KPIs
        is_kpi = header_array[:, 0] == 'KPI'

        # -- CSV DATA HANDLING -----------------------------------------------------------------------------------------
        # combine the kpi columns with the index col to index only the relevant columns
        kpi_idx_list = [0] + (is_kpi.nonzero()[0] + 1).tolist()

        # read only the relevant KPI columns (without multi-index header)
        df = pd.read_csv(self.csv_path, sep=',', header=0, skiprows=[0, 1], index_col=0, usecols=kpi_idx_list)

        # -- CSV HEADER HANDLING FOR RESHAPE ---------------------------------------------------------------------------
        # replace the 'quantities'-dimension with a 'qois'-dimension and skip the 'timesteps'-dimension
        dims = ('qois',) + responses_da[{'quantities': 0, 'timesteps': 0}].dims
        shape = (is_kpi.sum(),) + responses_da[{'quantities': 0, 'timesteps': 0}].shape

        # use the names of the kpis as coordinates of the qois-dimension
        coords = {'qois': header_array[is_kpi, 2]}

        # extract the index
        index_list = df.index.to_list()

        # analyze the index_col specifying the number of samples per parameter type in the syntax "sp:ep:al:"
        # split the number of samples at each column (excluding the trailing one)
        number_samples_list = [row[:-1].split(':') for row in index_list]
        number_distinct_samples = len(number_samples_list[0])

        if number_distinct_samples in {2, 3}:
            # -- non-deterministic cases with ideal space samples
            # drop the dummy KPIs of the space samples without actual simulations
            is_no_space_sample_list = [len(i) == number_distinct_samples for i in number_samples_list]
            df = df.iloc[is_no_space_sample_list, :]

            # remove the space samples from the index list
            number_samples_list = [number_samples for number_samples in number_samples_list if len(number_samples) == 2]

        if set(dims) == {'qois', 'space_samples', 'repetitions'}:
            # -- case variable number of repetitions requires generation of the data array with nan values

            # extract the number of repetitions from the index list
            number_repetitions_list = [
                int(number_samples_list[i - 1][1]) for i in range(1, len(number_samples_list))
                if number_samples_list[i][0] != number_samples_list[i - 1][0]]
            number_repetitions_list += [int(number_samples_list[-1][1])]

            # check whether all space samples have the same number of repetitions
            if all(rep == number_repetitions_list[0] for rep in number_repetitions_list):
                # create the data array
                kpi_da = xr.DataArray(df.values.T.reshape(shape), dims=dims, coords=coords)

            else:
                # create lists from the csv index column to index the array dimensions
                space_idx_list = [int(idx[0]) - 1 for idx in number_samples_list]
                repetition_idx_list = [int(idx[1]) - 1 for idx in number_samples_list]

                # create the data array with initial nan values
                kpi_array = np.ones(shape=shape) * np.nan

                # fill in the actual values
                # it currently assumes the order 'space_samples' followed by 'repetitions'
                kpi_array[:, space_idx_list, repetition_idx_list] = df.values.T.flatten()

                # create the data array
                kpi_da = xr.DataArray(kpi_array, dims=dims, coords=coords)

        else:
            # create the data array
            kpi_da = xr.DataArray(df.values.T.reshape(shape), dims=dims, coords=coords)

        # -- in case of experimental repetitions, we need to check whether a mean propagation was stored
        if set(kpi_da.dims) == {'qois', 'space_samples', 'repetitions'}:
            # -- in the mean case, the KPIs of all repetitions are equal

            # extract all the first aleatory samples as references for the equality check
            kpi_first_aleatory = kpi_da[{'repetitions': 0}].data

            # create a new axis for the dropped 'aleatory_samples'
            al_idx = kpi_da.dims.index('repetitions')
            idx_list = [slice(None)] * kpi_da.ndim
            idx_list[al_idx] = np.newaxis
            kpi_first_aleatory = kpi_first_aleatory[tuple(idx_list)]

            # repeat the first aleatory elements along the aleatory dimension
            kpi_first_aleatory = np.repeat(kpi_first_aleatory, kpi_da.shape[al_idx], axis=al_idx)

            # check if each element matches the respective first element (nan for variable number of repetitions)
            equality_mask = (kpi_da.data == kpi_first_aleatory) | np.isnan(kpi_da.data)

            # if all elements are equal, it is the mean-case and we must drop the aleatory-dimension with equal values
            if np.all(equality_mask):
                kpi_da = kpi_da[{'repetitions': 0}]

        return kpi_da
