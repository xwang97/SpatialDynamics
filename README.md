# SpatialDynamics

## Structures

### Functional scripts

| **Filename**          | **Description**                                                                 |
|-----------------------|---------------------------------------------------------------------------------|
| **models.py**         | Definition of the main model, with some dependent classes and loss functions.    |
| **main.py**           | Definition of modeling trainng in command line. |
| **data_builder.py**   | The TimeSeriesBuilder class, which builds pseudo time-series samples based on neighborhood information. |
| **training.py**       | Definition of train and test functions. |
| **utils.py**          | Small functions called in other scripts.|
| **plotting.R**        | R functions for visualizing the results and plotting figures.  |

### Simulation

| **Filename**          | **Description**                                                                 |
|-----------------------|---------------------------------------------------------------------------------|
| **simulate.ipynb**    | The cell simulator for molecule drift-diffusion process, with an example for data genration. |
| **run_simu.ipynb**    | Run the SPADE model on simulated data.  |

### Analysis

| **Filename**       | **Description**                                                                 |
|--------------------|---------------------------------------------------------------------------------|
| **burst.ipynb**    | Analysis of transcription burst.                                                |
