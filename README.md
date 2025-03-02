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

There is an example at the end of [this notebook](Simulation/simulate.ipynb). Here we introduce this process in more details.

Firstly we need to set the parameters of the cell (a cell is modeled as a sphere, so we need to set the `center` coordinates and `radius`) and the molecule movements (drift and diffusion components).

```python
# set the hyper-parameters
resolution = 0.1    # difference between two time steps
center = (0, 0, 0)
radius = 5
velocity = 0.1
brownian = 0.05
```

To model the transcription process, we need to set the time steps when new molecules are generated, which is achieved by the [scheduler](Simulation/simulator.py#L11-L29) function. Here the `resolution` has been set to 0.1 in the previous step, and the total time `T` is set to 20, so we will have 200 time steps. We can then set the transcription `rates` at different phases within the 200 time steps. The scheduler function will use these rates as input, and return the time steps when the transcription happens by a Poisson process model.
```python
# set the total time and transcription rates for each interval
T = 20
rates = [10]*30 + [0]*70 + [10]*100 
# rates = [5]*120 + [0]*80
schedule = scheduler(T, rates, 0.1)
```

Now that we have the transcription `schedule`, we can start to generate snythetic data. For each sample, we create a [Cell](Simulation/simulator.py#L86-L155) object, and simulate the transcription and drift-diffusion process based on the schedule (you can see this process in real time by uncommenting the plotcell function). The [build_data](Simulation/simulator.py#L62-L72) function is used to convert the cells into feature vectors (`num_dists` defines the dimension of the feature vectors, see Figure. 1 in our paper for the definition of a feature vector). To save the data, we flatten each `sample` into a 1*(time_steps*dim_feature) vector, and append it to a list. You can then save the list to csv format for later usage.
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
Please see [this notebook](Simulation/run_simu.ipynb) for the whole process of modeling training and validation. Basically, this is what you need to do:
- Set the hyper-parameters.
- Load the data and start training.
- Run the test function after training, and compare the inferred transcription rates with ground truth.


## 🚀 Real-world Pipeline
This section introduces how to run the whole pipeline of SPADE. We will start by the standard data format (the output bundle generated by public platforms like Xenium), go through every step in applying the model, and end by several analysis supported by our tool.

### 1. Preprocessing
For Xenium data, prepare the standard output bundle from their website. Make sure to extract cells.csv (find in cells.csv.gz) and features.tsv (find in cell_feature_matrix.tar.gz) from the zipped files. 

Run the automatic preprocessing with the following code:
```python
from preprocess import pre_xenium

folder = "/path_to_your_data_folder/"
pre_xenium(folder)
```

The preprocessing is includes the following steps:
- **Filtering**. Remove the transcripts with low qualities (qv score less than 20); remove the transcripts of controlled code words (not in the gene list); remove transcripts that are not assigned to any cells.
- **Compute distance**. For each transcript, check the assigned cell_id and find out the centroid coordinates of that cell in cells.csv. Compute the relative locations of the transcript compared to the centroids, and then compute the distance.
- **Saving**. Once finished, a new parquet file named as **transcripts_processed.parquet** will be saved to the current folder. Meanwhile, a new folder named **MoleculesPerGene** will be created, and the transcripts of each gene will be saved as a single csv file in this folder.

### 2. Generate training sets for single genes
For each single gene, we need to build a set of pseudo-time-series samples so that we can train the model and do some downstream analysis. The following code provides an automatic way to run this for all the genes. Before running this, make sure to extract the clusters.csv file to your data folder (you can find it in analysis.tar.gz, graph-based clustering results is recommended).
```python
from data_builder import TimeSeriesBuilder, read_trans, read_labels

# load transcripts and cell types
folder = '/path_to_your_data_folder/'
save_path = folder + 'TimeSeries/'
# cell_types = read_labels(filename=folder + 'Cell_Barcode_Type_Matrices.xlsx', sheet='Xenium R1 Fig1-5 (supervised)')
cell_types = read_labels(filename=folder + 'clusters.csv')
gene_list = sorted([f[:-4] for f in os.listdir(folder+'MoleculesPerGene') if f.endswith('.csv')])
for i in range(len(gene_list)):
    gene = gene_list[i]
    clear_output(wait=True)
    print(f'Processing the {i+1}-th gene: {gene}')
    transcripts = read_trans(folder + 'MoleculesPerGene/' + gene + '.csv')
    num_cells = transcripts['cell_id'].nunique()
    # Build time series
    tsb = TimeSeriesBuilder(transcripts, cell_types)
    tsb.run(num_samples=int(num_cells/10), save_path=save_path, gene=gene)
```
After this step, a new folder named **TimeSeries** will be created in your data folder. For each gene, three csv files will be save in this new folder: gene_data.csv contains the feature vectors of the generated time-series samples, gene_ids.csv contains the cell_ids in each sample, gene_locs.csv are the spatial coordinates of the cells.

### 3. Model training and validation for single genes
After step 2, we will have our time-series data set for training and testing the model. In this section, we will train the SPADE model for each single gene, and do some validation and visualization.

**(1) Import modules, set hyper-parameters**
```python
from utils import read_data, set_seed, heuristic_alpha
from models import Model
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from training import train, test
from tqdm import tqdm

# Set parameters
SEQ_LEN = 10
dim_inputs = 10
hidden_size = 50
latent_size = 50
batch_size = 1024
base_lr = 0.01
lr_step = 10
num_epochs = 100
```
**(2) Set up your working folders**
```python
folder = "/path_to_your_data_folder/"
data_folder = folder + 'TimeSeries/'
model_folder = folder + 'Models/'
molecules_folder = folder + 'MoleculesPerGene/'
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
```
**(3) Train a model for each gene (No modification needed for this part)**
```python
# Fetch the list of genes
gene_list = [f.split('_')[0] for f in os.listdir(data_folder) if f.endswith('.csv')]
# Remove duplicates
gene_list = sorted(list(set(gene_list)))

for i in range(len(gene_list)):
    gene = gene_list[i]
    clear_output(wait=True)
    print(f'Training gene: {gene}')
    # data loading and training
    data_path = data_folder + gene + '_data.csv'
    locs_path = data_folder + gene + '_locs.csv'
    data, locs = read_data(data_path, locs_path, SEQ_LEN, dim_inputs)
    training_size = int(data.shape[0] * 0.8)  # 80% of the data is used for training
    train_data = data[:training_size]
    train_locs = locs[:training_size]
    set_seed(42)
    alpha = heuristic_alpha(molecules_folder + gene + '.csv')
    print(f'alpha: {alpha}')
    net, _, _ = train(train_data, train_locs, alpha, batch_size, base_lr, lr_step, num_epochs, hidden_size, latent_size, SEQ_LEN)
    # save the trained model
    torch.save(net.state_dict(), model_folder + gene + '_model.pth')
```
**(4) Test and plot the MSE (no modification needed)**
```python
mse_per_gene = {}
for gene in tqdm(gene_list, desc='Testing'):
    # load test data
    data_path = data_folder + gene + '_data.csv'
    locs_path = data_folder + gene + '_locs.csv'
    data, locs = read_data(data_path, locs_path, SEQ_LEN, dim_inputs)
    test_size = int(data.shape[0] * 0.1)
    test_data = data[-test_size:]
    test_locs = locs[-test_size:]
    # load the trained model and start test
    model_path = model_folder + gene + '_model.pth'
    net = Model(dim_inputs, hidden_size, latent_size, SEQ_LEN)
    net.load_state_dict(torch.load(model_path))
    prediction, generation, trans_status, loss_recon = test(test_data, test_locs, net)
    mse_per_gene[gene] = loss_recon.item()
import seaborn as sns
# Get the MSE values
mse_values = list(mse_per_gene.values())
# Plot the density plot
sns.kdeplot(mse_values)
plt.xlabel('MSE')
plt.ylabel('Density')
plt.title('Density Plot of MSE')
plt.show()
```