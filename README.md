# SpatialDynamics

## 📝 Structured files

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

| **Filename**          | **Description**                                                                 |
|-----------------------|---------------------------------------------------------------------------------|
| **burst.ipynb**       | Analysis of transcription burst.                                                |


## 🎮 Simulation

This section introduces how to generate simulated data with the Cell simulator, and run experiments for model validation and visualization.

### 1. Generate simulated datasets

There is an example at the end of the jupyter notebook Simulation/simulate.ipynb. Here we introduce this process in more details.

Firstly we need to set the parameters of the cell (a cell is modeled as a sphere, so we need to set the center coordinates and radius) and the molecule movements (drift and diffusion components).

```python
# set the hyper-parameters
resolution = 0.1    # difference between two time steps
center = (0, 0, 0)
radius = 5
velocity = 0.1
brownian = 0.05
```

To model the transcription process, we need to set the time steps when new molecules are generated, which is achieved by the scheduler function. Here the resolution has been set to 0.1 in the previous step, and the total time T is set to 20, so we will have 200 time steps. We can then set the transcription rates at different phases within the 200 time steps. The scheduler function will use these rates as input, and return the time steps when the transcription happens by a Poisson process model.
```python
# set the total time and transcription rates for each interval
T = 20
rates = [10]*30 + [0]*70 + [10]*100 
# rates = [5]*120 + [0]*80
schedule = scheduler(T, rates, 0.1)
```

Now that we have the transcription schedule, we can start to generate snythetic data. For each sample, we create a Cell object, and simulate the transcription and drift-diffusion process based on the schedule (you can see this process in real time by uncommenting the plotcell function). The build_data function is used to convert the cells into feature vectors (num_dists defines the dimension of the feature vectors, see Figure. 1 in our paper for the definition of a feature vector). To save the data, we flatten each sample into a 1*(time_steps*dim_feature) vector, and append it to a list. You can then save the list to csv format for later usage.
```python
# generate data
num_samples = 200
num_dists = 50
data = []
cells = []
for i in range(num_samples):
    print(f"\rProgress: {i+1}", end='', flush=True)
    cell = Cell(center, radius, resolution, velocity, brownian)
    cell.dynamic(T, schedule)
    # plotcell(cell.trajectory, center)
    cells.append(cell)
    sample = build_data(cell.trajectory, radius, num_dists)
    data.append(sample.flatten())
```

### 2. Training
Please see the run_simu.ipynb notebook saved in Simulation for the whole process of modeling training and validation. Basically, this is what you need to do:
- Set the hyper-parameters.
- Load the data and start training.
- Run the test function after training, and compare the inferred transcription rates with ground truth.
