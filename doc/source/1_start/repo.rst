==========
Repository
==========
This repository contains the following folders and subfolders:

configs
   This folder contains configuration files for the VVUQ framework. It contains a json schema definition and several json config files. More information can be found in the :doc:`../2_user/config`.

data
   This folder contains external data. The structure of each subfolder has to follow specific definitions explained in :doc:`../2_user/csv`. Each subfolder has multiple csv files that contain an overview of the most important infos. They are usually accompanied by actual measurement recordings. However, this causes large data subfolders that should be ignored by version control or stored outside of the repository. Therefore, this folder only contains a small example to get started.

doc
   This folder contains the Sphinx documentation of the VVUQ framework.

sim
   This folder contains a simulator project for the UNECE regulation 79 use case of the examples and papers.

testing
   This folder contains a few unit tests. They are, however, exceptions due to the time resources of this small project.

src
   This folder contains the actual python code of the VVUQ framework.

src/applications/UNECE_R79
   This subfolder contains a class and a couple of scripts that were required for [Riedmaier.2021c]_. The class provides the methods for the assessment according to the UNECE-R79 use case targeting the lane-keeping behavior of vehicles. The scripts are independent of the framework and acted as a link from a co-author's code to this framework.

src/blocks
   This subfolder contains one class for each framework block. More information can be found at :doc:`../1_start/intro`.

src/commonalities
   This subfolder contains code that is shared by several framework blocks. This includes the config loader and the csv file handling. More information can be found at :doc:`../2_user/config` and :doc:`../2_user/csv`.

src/domains
   This subfolder contains one class for each framework domain. More information can be found at :doc:`../1_start/intro`.

src/evaluation
   This subfolder contains specific modules used for the validation of the VVUQ framework itself based on ground truth data in [Riedmaier.2021b]_. The ground truth is not available during a normal application of the framework.

src/helpers
   This subfolders contains helpful modules that generally go beyond this VVUQ framework. This includes, for example, file parsers or signal processing algorithms.

src/plots
   This subfolder contains all the code that deals with the visualization of the data. All the important steps of the VVUQ process are represented in a plot. These plots were frequently used in the publications.

src/variants
   This subfolder contains modules that offer various options for the configuration of a framework block. It is used to outsource code that would be too comprehensive for the major class of the block, that offers too many separate methods, or that is generally helpful in a separate module. Examples are sampling techniques, validation metrics, meta-modeling techniques, or the simulator control.