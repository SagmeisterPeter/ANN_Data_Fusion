# ANN_Data_Fusion

The folder "models" contains the best models obtained in the training process.

The folder "ANN_NMR" contains the python code for generating artifical spectra and training the NMR ANN.

The folder "ANN_datafusion_NMR_UVvis" contains the python code for training the datafusion ANN.

The data sets can be found on:

https://doi.org/10.5281/zenodo.6066166

Data set of low-field NMR spectra and UV/vis spectra for the synthesis of mesalazine intermediates, which were used as training or validation data for data processing with artificial neural networks development

Low-field NMR spectra for the nitration step:

The pure component spectrum of 2ClBA, 3N-2ClBA, and 5N-2ClBA are marked as NMR_pure_spectrum. The concentration levels for 2ClBA, 3N-2ClBA and 5N-2ClBA are in row 1, 2, and 3, respectively.

The data sets marked as NMR_ represents low-field NMR-spectra recorded. The reference values for 2ClBA, 3N-2ClBA and 5N-2ClBA are in column 1, 2, and 3, respectively.

Datafusion data sets for the hydrolysis and nitration step

The NMR data are either recorded or simulated from the pure NMR spectrum of each individual component. The reference values for 2ClBA, 3N-2ClBA, 5N-2ClBA, 3-NSA and 5-NSA are either assigned with UHPLC measurements or calculated from the prepared solutions.

The NMR spectra are depicted in datafusion_NMR_training. The reference values for 2ClBA, 3N-2ClBA and 5N-2ClBA are in column 1, 2, and 3, respectively.

The UV/vis spectra are depicted in datafusion_UVvis_training. The reference values for 2ClBA, 3N-2ClBA, 5N-2ClBA, 3-NSA and 5-NSA are in column 1, 2, 3, 4, and 5, respectively.

Process data

The NMR spectra for the stability run and the run with dynamic changes are depicted in process_NMR_. The first column is the time stamp.

The UV/vis spectra for the stability run and the run with dynamic changes are depicted in process_UV_. The first column is the time stamp.
