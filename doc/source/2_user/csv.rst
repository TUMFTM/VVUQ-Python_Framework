====
Data
====
During the VVUQ process there can be steps where we load data from files as well as steps where we want to save data to files. For example, we may store simulation data and load measurement data, load both from previous runs, or even load already processed KPIs. Therefore, we introduce central csv files that contain important data values and metadata with links to external storage paths for each test scenario. This keeps the measurement files relatively flexible and the csv files the first port of call, following some conventions and therefore the focus of this page.

The purpose of the csv files is threefold:

1. We provide the meta information of the measurement files with respect to the scenario parameter values.
2. We archive processed data from the code in the csv file.
3. We can reload processed data from the csv file to avoid time-consuming re-computations.

The file format should be human-readiable. Therefore, we select csv files and not, e.g., pickle. The csv cells must be separated by commas as usual.

Assumptions and Conventions
---------------------------
The initial structure of the measurement files can vary greatly between applications. On the one hand, there are the output quantities from the system. On the other hand, there are the external scenario parameters and/or internal model parameters. They might generally be of different formats such as scalar values, time vectors, or characteristic maps. They might require different measurement systems with different physical principles, samples sizes, file naming conventions, etc. The data might be stored in one large file or separated across several files. The VVUQ framework itself cannot deal with every conceivable situation. Therefore, we introduced a couple of assumptions and conventions that are in line with the use case of the publications:

#. The scenario parameters (or equally the model parameters) are scalar values.
#. The output quantities are time signals.
#. The output time signals will be processed to scalar KPIs in the assessment block of the framework.
#. The scenario parameter values are provided within csv files.
#. The output quantities of interest (selected by the user via the config) are provided within the measurement files.
#. The measurement files can contain further quantities.
#. The output KPIs are added to the csv files.
#. There is one csv file per framework domain and per testing environment.
#. There is one measurement file per scenario that contains all the output signals.
#. The csv file contains a mapping between the scenario parameter values, the measurement file paths, and the KPIs.
#. The naming of the measurement files is free.
#. The csv files are currently named *parameter_erg_mapping.csv*.
#. The data directory follows a defined structure explained later.
#. The csv files follow a defined structure explained later.

The first three points are assumptions that will not be met by every application. The framework would have to be extended to change this. For example, there is yet no (re-)simulation of scenario time signals. The remaining points are conventions that require some preparations, but they should not require changes in the code. For example, a large file has to be splitted and the scenario parameter values have to be determined. The values should be measured or estimated as close to the true value as possible. A purely data-driven extraction of the scenario information from the measurement files themselves was thought of as one option in the assessment, but is is not fully implemented yet and only interesting for certain use cases. At the moment, we assume one measurement file per test scenario with the csv file linking the scenario parameter values to the respective file path. This avoids incorporating the information in the file name itself, which would require a fixed file name convention, possibly renamings, and can reach its limits for multiple scenario parameters.

Framework Blocks
----------------
The csv file handling currently affects the framework blocks scenarios, simulator, and assessment. The code contains a parent class CsvHandler with one child class for each of those framework blocks. Each class has one method for loading/reading and one method for saving/archving/writing. The following list presents an overview:

#. The scenario parameter values and the measurement file paths have to be provided for the experiment.
#. The scenario parameter values and the simulation file paths can be provided if a previous simulation run should be re-loaded. Otherwise, the scenario design and the simulations take place during the actual VVUQ process and will be stored in a new csv file.
#. The assessment KPIs can be provided in addition to the scenario parameter values and the file paths for both the experiment or the simulation if they should be re-loaded. Otherwise, the VVUQ framework calculates them and adds them to the csv file.
#. If the scenario parameter values, file paths, and/or KPIs are missing, the VVUQ framework computes and adds them for archiving.

There are several combinations possible. If a simulation is performed from scratch, the VVUQ framework creates and fills the csv file step-by-step from scratch with the generated scenarios, file paths, and calculated KPIs. In case of experiment recordings, the VVUQ framework reads the csv file, calculates the KPIs, and adds them to the existing csv file. In case of stored KPIs of an experiment or simulation, the VVUQ framework directly loads the scenario parameters and the KPIs without even loading the actual recordings.

The reason behind the focus on the scenario, simulator, and assessment block lies in the computational load, which varies between the different framework blocks. The most time-consuming task in the software is by far the simulation, since it involves executing several test scenarios, possibly in real-time depending on the requirements. The second most time-consuming task is the assessment of the recordings, since it involves many measurement files with many quantities and possibly large signals. Nevertheless, this is optimized for speed and relatively fast. The other framework blocks mostly rely on vectorized computations and are very fast. Therefore, it makes sense to always store the scenario information with the result file path and the processed KPIs to provide the option to reload them later. This avoids repeating simulations and re-calculating KPIs from recordings. In addition, it suits the column format of the csv files.

In contrast, we refrain from saving calculated metrics or binary decisions, since they are re-calculated very fast and they would involve some empty or redundant cells in the csv files. Nevertheless, this could be extended in the future.

Data Folder Structure
---------------------
The data folder can contain subfolders with data from separate projects. Each such subfolder must follow a specific folder structure:

::

   .../data/.../
   |-- Experiment
   |   |-- application
   |   |   |-- parameter_erg_mapping.csv
   |   |-- validation
   |   |   |-- parameter_erg_mapping.csv
   |-- Simulator
   |   |-- application
   |   |   |-- parameter_erg_mapping.csv
   |   |-- validation
   |   |   |-- parameter_erg_mapping.csv
   |   |-- verification
   |   |   |-- parameter_erg_mapping.csv

The structure must be met exactly so that the re-loading functionalities work. The missing parts are automatically generated by the VVUQ framework as described above. Strictly speaking, the framework reuses the data folder to archive further elements. They do not intefere with the re-loading. The testrun and vehicle config from the CarMaker project are saved to a folder named *CarMaker* at the level of the framework domains. CarMaker testseries files are archived at the level of the csv files. In case the plots and evaluation methods are used, the generated plots and evaluation measures are often stored in folders named *Plots* and *Evaluation* at the level of the test environments.

.. warning::
   If the user selects to perform new simulations, all domain subfolders within the given directory are emptied! This erases old data if the given directory is not empty!

The automatic generation of the result file paths in case of new simulations follows a specific structure, as well. This is handled by the *create_erg_filepaths* method in the simulator class. It distinguishes:

1. deterministic simulations,
2. simulations with repetitions sampling uncertainties in a single loop, and
3. simulations with nested uncertainty propagation of epistemic and aleatory samples.

The file and folder names contain a list of the parameters with their type, a consecutive number, and their value. Regarding the type, *sp* stand for a nominal scenario parameter, *eaps* for epistemic and aleatory parameters, *ep* for epistemic parameter, and *aps* for aleatory parameters. The values contain up to two digits after the comma and can be negative. The verification domain is an exception with up to five digits to support small step sizes. The type and number are not separated, while each value is separated with underscores from the previous number and the subsequent type. The possible commas are also represented as underscores.

.. warning::
   If a parameter variation takes place at the third digit or deeper, the CarMaker simulator would constantly overwrite the erg result files. In this exceptional case, additional digits would have to be added by editing the code.

In the first case, the storage conventions are as follows:

- The measurement files lie directly at the level of the csv file.
- Their file names contain the values of the scenario parameters.

In the second case, the storage conventions are as follows:

- Each nominal scenario gets one subfolder.
- Their folder names contain the values of the scenario parameters.
- The measurement files lie within the subfolders.
- Their file names contain a consecutive number of eaps1, etc.

In the third case, the storage conventions are as follows:

- Each nominal scenario gets one subfolder.
- Their folder names contain the values of the scenario parameters.
- Each epistemic parameter gets one subsubfolder.
- Their folder names contain the values of the epistemic parameters.
- The measurement files lie within the subsubfolders.
- Their file names contain a consecutive number of aps1, etc.

We illustrate the principle using an example:

- We have 2 scenario parameters. The first one ranges between 85 and 150 and the second one between -0.3 and 0.3.
- We have 10 repetitions.
- We have one epistemic parameter ranging from -2.0 to 2.0 and 100 aleatory parameters.

The example results in the following directory structure for the three cases:

::

   .../data/.../domain/
   |-- parameter_erg_mapping.csv
   |-- sp1_85_0_sp2_-0_3.erg
   |-- ...
   |-- sp1_150_0_sp2_0_3.erg

::

   .../data/.../domain/
   |-- parameter_erg_mapping.csv
   |-- sp1_85_0_sp2_-0_3
   |   |-- eaps1.erg
   |   |-- ...
   |   |-- eaps10.erg
   |-- ...
   |-- sp1_150_0_sp2_0_3
   |   |-- eaps1.erg
   |   |-- ...
   |   |-- eaps10.erg

::

   .../data/.../domain/
   |-- parameter_erg_mapping.csv
   |-- sp1_85_0_sp2_-0_3
   |   |-- ep1_-2_0
   |   |   |-- aps1.erg
   |   |   |-- ...
   |   |   |-- aps100.erg
   |   |-- ...
   |   |-- ep1_2_0
   |   |   |-- aps1.erg
   |   |   |-- ...
   |   |   |-- aps100.erg

CSV File Structure
------------------
The csv files have a multi-line header consisting of three rows and one leading index column as meta information. The first columns are used for the scenario parameters, the second ones for the result paths, and the last ones for the assessment KPIs. This follows the order of the VVUQ process. Depending on the current execution location of the framework and the user input, the csv files may not exist yet, exist in parts, or already fully exist.

The first header row can only contain the following terms to specify the general content, which indicates the framework block, of each column:

1. "Parameter" to indicate columns with scenario parameters,
2. "Filepath" to indicate the column of results paths, and
3. "KPI" to indicate the columns with KPIs.

The second header row specifies the type of the scenario parameters and KPIs (see config documentation):

1. the parameter types: "deterministic", "aleatory", "epistemic", and "mixed",
2. again the term "Filepath" as no additional information is required, and
3. the KPI types: "min", "max", "mean", "min_mean", "max_mean", "mean_mean".

The third header row includes the names of the parameters and quantities of interest (QOIs):

1. the parameter names,
2. again the term "Filepath" as no additional information is required, and
3. the QOI names.

The index column enumerates the scenario samples. It depends on the three cases from the previous section (deterministic simulation, simulation with repetitions, nested uncertainty propagation):

- In the first case, it only contains one consecutive number for each sample of the scenario parameters.
- In the second case, it contains one consecutive number for each sample of the scenario parameters and one consecutive number for each repetition of each scenario.
- In the third case, it contains one consecutive number for each sample of the scenario parameters, one consecutive number for each epistemic sample, and one consecutive number for each aleatory sample.

Thus, the index column contains one number in the first case, two numbers in the seoncd case, and three number in the full third case. A colon follows after each number as separator and delimiter. In combination with the second header line, this allows to restore the scenario information when loading the csv file. Afterwards, for each index column, the corresponding scenario information, result file paths, and KPIs follow.

There are a few exceptions. In the "_mean" case, the KPI values of each repetition of the same scenario are the same. We actually save these same values multiple times in the rows of each repetition so that we can identify this case again by checking equality during loading the KPI information. In the exceptional case that a KPI cannot be calculated or does not exist (e.g. no overshooting), the cell can be kept empty. Furthermore, there are use cases where the number of repetitions varies between the scenarios. This cannot be configured in the config by the user. However, it is supported when loading measurement files. The numbering principle stays the same, while the repetition number just does not always end with the same digit.

In the case of deterministic simulations, there are only independent scenarios that will be executed directly. However, in case of tests with repetitions or uncertainty propagation, we obtain samples in the neighborhood of each nominal scenario. If we would only store the scenario information of the performed tests, the nominal scenario information would be lost. However, this information is required for the error model and some plots and it is generally useful for archiving. Therefore, we add a second section at the bottom of the csv file to store the nominal data so that it can be re-loaded. This is only relevant to the mentioned cases and not to the deterministic one, since there is no need for a distinction. The index column cells of the nominal scenarios contain per definition only one number of the scenario parameters. Then, the nominal parameters values follow. We fill the remaining columns with dummy values: "-" for the result paths and 0.0 for the KPIs.

Example:

    ,Parameter,Parameter,Parameter,Parameter,Parameter,Parameter,Filepath,KPI,KPI
    ,deterministic,deterministic,deterministic,deterministic,deterministic,deterministic,Filepath,min_mean,max_mean
    ,$Ego_Init_Velocity,_ay_norm,$Slope,Env.Wind.Velocity,VehicleLoad.0.mass,$stepsize,Filepath,D2LL,Car.Jerk
    1:,100.0,0.85,0.0,0.0,0.0,0.001,sp1_100_0_sp2_0_85_sp3_0_0_sp4_0_0_sp5_0_0_sp6_0_001.erg,0.327,0.156
    2:,100.0,0.85,0.0,0.0,0.0,0.0005,sp1_100_0_sp2_0_85_sp3_0_0_sp4_0_0_sp5_0_0_sp6_0_0005.erg,0.327,0.156
    3:,100.0,0.85,0.0,0.0,0.0,0.00025,sp1_100_0_sp2_0_85_sp3_0_0_sp4_0_0_sp5_0_0_sp6_0_00025.erg,0.327,0.156