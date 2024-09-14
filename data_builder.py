import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np

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
    
    def cal_probs3(self, alpha=2):
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

