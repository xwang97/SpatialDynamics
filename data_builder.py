import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy import stats

def read_trans(filename):
    transcripts = pd.read_csv(filename)
    return transcripts

def pair_ks(base, neighbor):
    """
    Compare the ks distance between the ecdf of two cells (the ecdf is not built by all the molecules, 
    we will only use a subset of the molecules)
    Input:
        base: a list of molecule distances of the base cell
        neighbor: a list of molecule distances of the neighbor cell
    Ouput:
        The ks distance
    """
    base = np.sort(np.array(base))
    neighbor = np.sort(np.array(neighbor))
    view_max = np.quantile(base, 0.75)
    view_min = np.min(base)
    sub_base = base[base <= view_max]  # We only care about the curve from start to view_max
    ks_list = []
    for i in range(3):
        sub_base += 1  # Move the curve to the right by 1 unit each time
        view_max += 1
        view_min += 1
        sub_neighbor = neighbor[np.where(np.logical_and(neighbor>=view_min, neighbor<=view_max))[0]]
        if sub_neighbor.shape[0] == 0:
            ks_list.append(1)
        else:
            ks_list.append(stats.ks_2samp(sub_base, sub_neighbor).statistic)
        # stats.ecdf(sub_base).cdf.plot()
    return min(ks_list)

class TimeSeriesBuilder:
    def __init__(self, transcripts):
        self.transcripts = transcripts  # transcripts matrix
        self.cell_dists = {}  # key: cell_id, value: a list of molecule distances
        self.cell_centers = {} # key: cell_id, value: a list containing x and y of the center
        self.cell_neighbors = {} # key: cell_id, value: a list contianing the cell's k neareast neighbors
        self.cell_features = {}  # key: cell_id, value: feature vector of the cell
        self.cell_probs = {}     # key: cell_id, value: probabilities of transiting to each neighbor
    
    def build_dict(self):
        """
        Build dictionaries for each cell: 1) distances of all the molecules; 2) center coordinates
        Input:
            transcripts: each row contains information (such as coordinates) of a single molecule
        Output:
            cell_dists: dict, key is cell_id, value is a list of molecule distances in this cell
            cell_centers: dict, key is cell_id, value is a list of molecule coordinates
        """
        for index, row in self.transcripts.iterrows():
            if row['cell_id'] not in self.cell_dists:
                self.cell_dists[row['cell_id']] = [row['distance']]
                self.cell_centers[row['cell_id']] = [row['x_centroid'], row['y_centroid']]
            else:
                self.cell_dists[row['cell_id']].append(row['distance'])
        # filter out cells with less than 5 molecules
        self.cell_dists = {key: value for key, value in self.cell_dists.items() if len(value) >= 5}
        self.cell_centers = {key: value for key, value in self.cell_centers.items() if key in self.cell_dists}
    
    def find_neighbors(self, k_neighbors=20):
        """
        This function finds the k-nearest spatial neighbors for each cell.
        Input:
            cell_centers: the cell_centers dictionary which contains the coordinates of cell centers
            k_neighbors: number of nearest neighbors
        Output:
            k_nearest_neighbors: a dictionary, keys are cell_ids of each cell, values are the cell_ids 
            of that cell's neighbors
        """
        coordinates_to_ids = {tuple(coord): cell_id for cell_id, coord in self.cells_centers.items()}
        cell_coordinates = np.array(list(self.cell_centers.values()))
        nbrs = NearestNeighbors(n_neighbors=k_neighbors+1, algorithm='auto').fit(cell_coordinates)
        distances, indices = nbrs.kneighbors(cell_coordinates)
        for cell_id, neighbor_indices in zip(self.cell_centers.keys(), indices):
            neighbors = [neighbor_id for neighbor_id in neighbor_indices if neighbor_id != cell_id]
            neighbor_ids = [coordinates_to_ids[tuple(cell_coordinates[neighbor_id])] for neighbor_id in neighbors]
            self.cell_neighbors[cell_id] = neighbor_ids[1:]
    
    def build_features(self, stride=3):
        """
        This function is used to build feature vectors for each cell.
        Input:
            transcripts: the transcripts matrix where each row is a molecule
            cells_dists: the cell_dists dictionary which contains the dists of all molecules in each cell
            stride: controls how to discretize the cell (the distance between two circles)
        Output:
            cell_features: a dictionary, keys: cell_id, values: feature vector of the cell, which is the molecule counts at each distance
        """
        num_strides = int(np.floor(max(self.transcripts['distance'])) / stride) + 1
        for id, dists in self.cell_dists.items():
            self.cell_features[id] = np.zeros(num_strides)
            for d in dists:
                j = int(d / stride)
                self.cell_features[id][j] += 1
    
    def cal_probs(self, alpha=2):
        """
        This function computes transition probabilities from each cell to its neighbors.
        Input:
            cell_neighbors: the nearest neighbor dictionary
            alpha: scale parameter used to control the smoothness of the probabilities
        Output:
            cell_probs: a dictionary, key: cell_id, value: probabilities of transiting to each neighbor
        """
        for id, neighbors in self.cell_neighbors.items():
            num_nbrs = len(neighbors)
            ks_dists = np.zeros(num_nbrs)
            base = self.cell_dists[id]
            for i in range(num_nbrs):
                nbr_id = neighbors[i]
                neighbor = self.cell_dists[nbr_id]
                ks_dists[i] = pair_ks(base, neighbor)
            probs = np.exp(-alpha * ks_dists)
            probs /= np.sum(probs)
            self.cell_probs[id] = probs

    def walk(self, start, length, dim_features):
        """
        Start at a cell and generate a path by random walk
        Input:
            start: cell_id of the start cell
            length: length of the series
            dim_features: dimension of feature vectors
        Output:
            series: a length*di_features matrix, each row is the fetures of a local pseudo-time step
            selected_ids: a vector, each element is a cell_id of this series
        """
        series = np.zeros((length, dim_features), dtype=float)
        selected_ids = np.zeros(length)
        id = start
        for i in range(length):
            series[i] = self.cell_features[id] # save the features
            selected_ids[i] = id               # save the cell_id
            nbrs = self.cell_neighbors[id]     # find the neighbors of the current cell
            probs = self.cell_probs[id]        # find the corresponding probs
            next_index = np.random.choice(len(probs), p=probs) # select the next cell based on the probs
            next_cell_id = nbrs[next_index]
            id = next_cell_id             # jump to the next cell
        return series, selected_ids
    
    def build_dataset_base(self, num_samples, seq_len=20):
        """
        Build a dataset by random walk
        Input:
            num_samples: number of samples in the dataset
            seq_len: length of each series
        Output:
            data: num_samples * (seq_len * dim_features) matrix, each row is the flattened features of a local series
            locations: the locations of each start cell
            cell_ids: num_samples * seq_len matrix, each row is the cell_ids of a local series
        """
        data = []
        locations = []
        cell_ids = []  # save the cell ids of each path
        # all_ids = np.unique(transcripts['cell_id'])
        all_ids = np.array(list(self.cell_dists.keys()))
        num_ids = all_ids.shape[0]
        dim_features = self.cell_features[all_ids[0]].shape[0]
        for i in range(num_samples):
            rand_index = np.random.randint(num_ids)
            while len(self.cell_neighbors[all_ids[rand_index]]) <= 1:
                rand_index = np.random.randint(num_ids)
            start = all_ids[rand_index]
            sample, selected_ids = self.walk(start, seq_len, dim_features)
            data.append(sample.flatten())
            locations.append(self.cell_centers[start])
            cell_ids.append(selected_ids)
            if i % 1000 == 0:
                print(i)
        return np.array(data), np.array(locations), np.array(cell_ids)
    
    def build_dataset_refer(self, cell_ids):
        """
        Build a dataset for a gene based on a reference gene. If we have already built a dataset for some gene,
        we must have saved the cell_ids of each series of each sample. We will use these well-built series to build
        a dataset for a new gene.
        """
        data = []
        locations = []
        reference_index = []
        all_ids = np.array(list(self.cell_dists.keys()))
        dim_features = self.cell_features[all_ids[0]].shape[0]
        num_samples = cell_ids.shape[0]
        seq_len = cell_ids.shape[1]
        for i in range(num_samples):
            series = np.zeros((seq_len, dim_features), dtype=float)
            save = 1
            if cell_ids[i, 0] not in all_ids:
                save = 0
            else:
                for j in range(seq_len):
                    if cell_ids[i ,j] in all_ids:
                        series[j] = self.cell_features[cell_ids[i, j]]
                    else:
                        continue
            if save:
                data.append(series.flatten())
                locations.append(self.cell_centers[cell_ids[i, 0]])
                reference_index.append(i)
        return np.array(data), np.array(locations), np.array(reference_index)
    
    def run(self, num_samples, gene, method='base', reference_ids = None):
        """
        RUn the functions above, and save the time series samples.
        """
        self.build_dict()
        self.find_neighbors()
        self.build_features()
        self.cal_probs()
        if method == 'base':
            data, locations, cell_ids = self.build_dataset_base(num_samples)
            np.savetxt(gene+'_data.csv', data, delimiter=',')
            np.savetxt(gene+'_locs.csv', locations, delimiter=',')
            np.savetxt(gene+'GATA3_ids.csv', cell_ids, delimiter=',')
        else:
            data, locations, reference_index = self.build_dataset_refer(reference_ids)
            np.savetxt(gene+'_data.csv', data, delimiter=',')
            np.savetxt(gene+'_locs.csv', locations, delimiter=',')
            np.savetxt(gene+'_reference.csv', reference_index, delimiter=',')

