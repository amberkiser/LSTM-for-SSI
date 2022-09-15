import logging
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class LoadNNData:
    """
        Load train, validation, or test data.
        Changes numpy data into DataLoader object.
    """
    def __init__(self, X_data, y_data, batch_size):
        self.X = X_data.values
        self.y = y_data.values
        self.batch_size = batch_size
        self.loader = self.create_loader_object()

    def create_loader_object(self):
        dataset = TensorDataset(torch.from_numpy(self.X), torch.from_numpy(self.y))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        logging.info('Data loaded.')
        return loader

    def find_pos_weight(self):
        """
            Find weight for positive class used in training only.
            pos_weight = negative cases / positive cases
        """
        pos_weight = torch.tensor([(len(self.y) - np.sum(self.y)) / np.sum(self.y)])
        return pos_weight
