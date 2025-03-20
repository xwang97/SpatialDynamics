import numpy as np
import os
from tqdm import tqdm

class FeaturesWrapper:
    """
    A wrapper class for building features at one time point.
    """
    def __init__(self, transcripts, time_stamp):
        self.time_stamp = time_stamp  # time point
        self.transcripts = transcripts  # transcripts matrix
        self.cell_dists = {}  # key: cell_id, value: a list of molecule distances
        self.cell_centers = {} # key: cell_id, value: a list containing x and y of the center
        self.cell_features = {}  # key: cell_id, value: feature vector of the cell
        self.build_dict()
        self.build_features()
    
    def build_dict(self):
        """
        Build dictionaries for each cell: 1) distances of all the molecules; 2) center coordinates
        Input:
            transcripts: each row contains information (such as coordinates) of a single molecule
        Output:
            cell_dists: dict, key is cell_id, value is a list of molecule distances in this cell
            cell_centers: dict, key is cell_id, value is a list of molecule coordinates
        """
        print("Building cell dictionaries")
        # build cell_dists and cell_centers
        for index, row in self.transcripts.iterrows():
            if row['cell_id'] not in self.cell_dists:
                self.cell_dists[row['cell_id']] = [row['distance']]
                self.cell_centers[row['cell_id']] = [row['x_centroid'], row['y_centroid']]
            else:
                self.cell_dists[row['cell_id']].append(row['distance'])
    
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
        print("Build cell feature vectors")
        # num_strides = int(np.floor(max(self.transcripts['distance'])) / stride) + 1
        num_strides = 10
        for id, dists in self.cell_dists.items():
            self.cell_features[id] = np.zeros(num_strides)
            for d in dists:
                j = min(9, int(d / stride))
                self.cell_features[id][j] += 1


class TimeSeriesBuilder:
    def __init__(self, obs_list):
        self.obs_list = sorted(obs_list, key=lambda x: x.time_stamp)  # sort the list of observations by time_stamp
        self.num_time_points = len(self.obs_list)
        self.cell_ids = list(self.obs_list[0].cell_features.keys())
    
    def build_sample(self, cell_id):
        dim_features = self.obs_list[0].cell_features[self.cell_ids[0]].shape[0]
        sample = np.zeros((len(self.obs_list), dim_features))
        for i in range(self.num_time_points):
            if cell_id in self.obs_list[i].cell_features:
                sample[i] = self.obs_list[i].cell_features[cell_id]
        return sample

    def build_dataset(self):
        data = []
        locations = []
        for cell_id in tqdm(self.cell_ids):
            sample = self.build_sample(cell_id)
            data.append(sample.flatten())
            locations.append(self.obs_list[0].cell_centers[cell_id])
        return np.array(data), np.array(locations), np.array(self.cell_ids)
    
    def run(self, save_path, gene):
        if len(self.cell_ids) < 100:
            print(f"Less than 100 cells for {gene}, skip")
            return
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        data, locations, cell_ids = self.build_dataset()
        num_samples = data.shape[0]
        np.savetxt(save_path+gene+'_data.csv', data, delimiter=',')
        np.savetxt(save_path+gene+'_locs.csv', locations, delimiter=',')
        np.savetxt(save_path+gene+'_ids.csv', cell_ids, delimiter=',', fmt='%s')
        print(f"{num_samples} time-series samples of {gene} generated")