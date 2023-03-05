===============
Data Structures
===============
The VVUQ framework heavily relies on multi-dimensional data. We decided to use `xarrays <http://xarray.pydata.org/en/stable/>`_ as the main data structure in the code to support multi-dimensional arrays, fast vectorized operations, descriptive naming, and meta infos. The xarrays can be interpreted as N-D versions of the infamous 2D pandas dataframes. Both go beyond classic numpy arrays by offering to add descriptive names to each column and row and in case of the xarrays even to the dimensions themselves. This allows accessing the data by the descriptive names and increases the flexibility and readability of the code. The three packages support each other so that it is often possible to use functions of the other packages.

The xarray.DataArrays have the following key attributes that are all heavily used by our VVUQ framework:

- values: an numpy.ndarray containing the actual data. This can be accessed directly and even modified in-place so that all the numpy functionalities are enabled. The remaining attributes can be seen as a wrapper around a numpy array.
- dims: a tuple containing the names of each dimension
- coords: a dict-like container containing the names of the individual coordinates of each dimension
- attrs: a dictionary holding arbitrary meta information

The major methods of each framework block, which are called on the level of the framework domains, all use xarrays for their input arguments and their return values. Thus, the entire VVUQ process is realized based on xarrays. The plotting functionalities offered afterwards also rely on the filled xarrays. Even many internal methods continue to work with xarrays or numpy arrays.

Composition of the Arrays
-------------------------
We first explain some general aspects that affect the xarrays of several framework blocks so that we do not have to repeat them separately for each block later. Our xarrays consist of three parts making up the dimensions:

- The data can be of different data types such as time vectors or probabalities.
- We do not store the data of each parameter or quantity separately but combine them in one array for efficient processing.
- We do not store the data of each sampled test separately but combine them again in the same array.

The second point is quite easy. It only takes one dimension with the name 'parameters', 'quantities', or 'qois'. The corresponding coordinates contain their names, which are taken from the user config or the csv files. The other two points are more complex and affected by the :ref:`framework manifestations <framework-manifestations>`. They rely on the following types:

1. deterministic simulations
2. experiments with repetitions
3. probabilistic simulations
4. interval simulations
5. non-deterministic simulations

.. warning:: We should keep in mind that the degenerate probabilistic and interval simulations are yet only implemented partly and not tested.

These types affect the dimensions of the xarrays with respect to the sampling structure in the following way:

1. The deterministic simulations contain one dimension called 'space_samples' for each nominal scenario.
2. The experiments with repetitions contain one dimension called 'space_samples' and one dimension called 'repetitions' to align the repetitions with the respective nominal scenario.
3. The probabilistic simulations contain one dimension called 'space_samples' and one dimension called 'aleatory_samples' to align the aleatory samples with the respective nominal scenario.
4. The interval simulations contain one dimension called 'space_samples' and one dimension called 'epistemic_samples' to align the epistemic samples with the respective nominal scenario.
5. The non-deterministic simulations contain one dimension called 'space_samples', one dimension called 'epistemic_samples', and one dimension called 'aleatory_samples' to align the epistemic samples with the respective nominal scenario and again the aleatory samples with the respective epistemic sample of the outer uncertainty propagation loop.

Thus, we have 1D for the samples in the determinisitic case, 3D in the non-determinisitic case, and 2D in the other cases. We do not use specific coordinates for these dimensions, since consecutive numbers are enough for the samples.

.. note:: To deduce the type from a given xarray, we can check with an if-elif-else structure which of the predefined tuples the actual dim-tuple of the given xarray matches. It is a bit easier to just check for the number of dimensions. We sometimes did this, since we focused on the 2D deterministic case, 3D repetition case, and 4D non-deterministic case. However, when also considering the degenerate cases, 3 dimensions are no longer unambiguous. Then, it is important to inspect the code to ensure that all checks involve the full tuples with the dimension names.

Data Types
----------
Our data can be of several types:

- scalar values: Scalar values are single points and do not need an additional dimension for the data type itself.
- time vectors: Time vectors require one dimension with the name 'timesteps'.
- intervals: An interval is limited by the two interval boundaries. So, it can be represented as one dimension in form of a vector with two elements. As coordinates, we use the terms 'left' and 'right' for the boundaries.
- empirical cumulative distribution functions (ECDFs): We represent our probability distributions as ECDFs, since the CDF form is required for the area validation metrics and since we have several empirical steps due to the sampling instead of a smooth theoretical function. The ECDFs consist of the actual values along the "x"-axis and the probability information along the "y"-axis. Both of them have to be sorted. It is either possible to start with a value of -inf and a probability of 0 or end with a value of inf and a probability of 1. Sometimes one of them is set as convention, while sometimes the user can specify it. For example, the matplotlib.pyplot.steps function gives the user a flag to state where the steps should be placed. We select a leading -inf as convention in our code. The x-component is directly stored in the xarray, whereas the y-component goes beyond the multi-dimensional xarray structure. We store the y-component in the attributes dictionary of the xarray to keep both together and still have a xarray as the top-level data structure for consistent processing. The y-component is currently stored under the key 'probs' as a 1D numpy array, since the number of repetitions or aleatory samples is often constant. However, this does not cover an arbitrary ECDF and might require an extension to full data arrays for the y-component in the future.
- probability-boxes: A p-box is a mixture of an ECDF and an interval. It has two boundaries like an interval but in form of ECDFs instead of scalar values. Thus, it requires two dimensions to store two separate ECDFs. We use the terms 'left' and 'right' for the dimension with the two boundaries and yet assume both have the same number of x and y values.

These data types appear at diffent points in the framework. For example, the values of the parameters and KPIs are scalar values. The outputs from the system and model are time signals. The p-boxes only occur if non-deterministic simulations are performed. The ECDFs can occur for experiments with repetitions and probabilistic simulations. Intervals occur frequently, since an area validation metric or an uncertainty expansion of a deterministic result also leads to intervals.

An array/matrix/table always assumes that each row/column has the same number of elements. This assumption is mostly satisfied in our VVUQ framework and enables the fast vectorized operations. However, there are a few exceptions possible. The number of repetitions, aleatory samples, or time steps can vary between the nominal scenarios. This would violate the assumption and loose the efficiency. However, we can still preserve the xarrays with a common trick. We allow NaN values in certain cases to fill missing array cells. We allow a varying number of repetitions in the case of loading from csv files and we allow that different measurement files have different durations. The shape of the xarray is determined then by the scenario with the maximum number of repetitions or time steps. A varying number of repetitions also affects the probabilties of the ECDF. This means we have to replace the single numpy vector with a list of numpy vectors.

Data Structures of the Framework Blocks
---------------------------------------
In the following, we will go step-by-step through the xarrays of each framework block:

- The scenario block combines the sampling dimensions with a 'parameters'-dimension. It does not require a separate dimension for the data type, since we assume scalar parameter values. The resulting xarrays can be either 2D, 3D, or 4D depending on the simulation type. This results in dim-tuples such as ('space_samples', 'parameters') or ('space_samples', 'epistemic_samples', 'aleatory_samples', 'parameters').
- The simulator and experiment block combine a 'quantities'-dimension with the sampling dimensions and a 'timesteps'-dimension. The resulting xarrays can be either 3D, 4D, or 5D. This results in dim-tuples such as ('quantities', 'space_samples', 'epistemic_samples', 'aleatory_samples', 'timesteps').
- The assessment block contains several steps that transform the previous xarray. It starts calculating the quantities of interest from the recorded quantities. Thereby, it renames the 'quantities'-dimension to 'qois' and updates their coordinates with the QoI names. Then, the KPI calculation follows and removes the 'timesteps'-dimension. It is basically a mandatory step at the moment. It would be possible to reach the metric block and calculate, e.g., an RMSE when skipping it, but this causes many dependencies (csv files, ECDF, p-boxes) in the code that are not fully implemented and not tested at all. In case of aleatory samples or repetitions, ECDFs are created in the next step. This preservers the dim-tuple. In case of epistemic samples, p-boxes are created in the final step. This renames the 'epistemic_samples'-dimension to 'pbox_edges'. The resulting arrays can be either 2D, 3D, or 4D. This results in the dim-tuples ('qois', 'space_samples'), ('qois', 'space_samples', 'aleatory_samples'/'repetitions'), or ('qois', 'space_samples', 'pbox_edges', 'aleatory_samples').
- The metric block compares the xarrays from the assessment of the simulation and experiment. The data type is either a scalar value in case of a determinisitic model-form error or an interval in case of a model-form uncertainty with the dimension name 'interval' at the end of the dim-tuple. The resulting xarrays can be either 2D or 3D. This results in the dim-tuples ('qois', 'space_samples') or ('qois', 'space_samples', 'interval').
- The validation decision making transforms the metric xarray. It always leads to binary scalar values and a 2D xarray with the dim-tuple ('qois', 'space_samples'). In addition, it provides one global decision in a 1D xarray with the dim-tuple ('qois').
- The training of the error model is an exception. It returns the error model itself and no xarray.
- The inference of the error model predicts metrics. Thus, it has the same type of xarrays as described for the metric block.
- The error integration offers two techniques. In case of a bias correction of deterministic simulations, we obtain a 2D xarray with the dim-tuple ('qois', 'space_samples'). In case of a uncertainty expansion of a deterministic (or interval) simulation, we obtain a 3D xarray with the dim-tuple ('qois', 'space_samples', 'interval'). In case of an uncertainty expansion of a non-deterministic (or probabilistic) simulation, we obtain a 4D xarray with the dim-tuple ('qois', 'space_samples', 'pbox_edges', 'aleatory_samples').
- The application decision making transforms the previous xarray. It always leads to binary scalar values and a 2D xarray with the dim-tuple ('qois', 'space_samples'). In addition, it provides one global decision in a 1D xarray with the dim-tuple ('qois').

Variable Naming
---------------
We follow the PEP8 style guide in our code. In addition, we often mention the data structure of a variable at the end of its name. We append a

- *_list* for lists,
- *_dict* for dictionaries,
- *_tuple* for tuples,
- *_na* for numpy arrays,
- *_pd* for pandas DataFrames, and
- *_da* for xarray DataArrays.

We sometimes refrain from this to enable shorter names if it is clear or if a certain part of the code is mostly based on the same data structure.
