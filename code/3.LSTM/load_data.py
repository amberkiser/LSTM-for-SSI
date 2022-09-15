import logging
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class LoadPyTorchData:
    """
        Load train, validation, or test data.
        Changes numpy data into DataLoader object.
    """

    def __init__(self, path, batch_size, test_flag=False, debug=False):
        self.test_flag = test_flag
        self.X = self.read_in_data(path, 'X', debug)
        self.y = self.read_in_data(path, 'y', debug)
        self.ids = self.read_in_data(path, 'ids', debug)
        self.batch_size = batch_size
        self.loader = self.create_loader_object()

    def read_in_data(self, path, variables, debug):
        if (self.test_flag is False) & (variables == 'ids'):
            return None

        with open('%s_%s.pkl' % (path, variables), 'rb') as f:
            dataset = pickle.load(f)

        if debug:
            dataset = dataset[:50]
        return dataset

    def create_loader_object(self):
        if self.test_flag:
            dataset = TestIdDataset(self.X, self.y, self.ids)
        else:
            dataset = TensorDataset(torch.from_numpy(self.X), torch.from_numpy(self.y))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        logging.info('Data loaded.')
        return loader

    def find_pos_weight(self):
        """
            Find weight for positive class used in training only.
            pos_weight = negative cases / positive cases
        """
        pos_weight = torch.tensor([(len(self.y[:, 0]) - np.sum(self.y[:, 0])) / np.sum(self.y[:, 0])])
        return pos_weight


class TestIdDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, ids):
        self.ids = ids.astype(np.str)
        self.X_data = torch.tensor(X, dtype=torch.float32)
        self.y_data = torch.tensor(y, dtype=torch.int64)

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        inputs = self.X_data[idx]
        labels = self.y_data[idx]
        pt_key = self.ids[idx]
        return inputs, labels, pt_key
