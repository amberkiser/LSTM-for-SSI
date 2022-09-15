import logging
import pickle
import pandas as pd

from load_data import LoadNNData
from utils import *
from nn_models import *

# Train model
logging.basicConfig(filename='training_NN.log', level=logging.INFO,
                    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S %Z')
logging.info('SCRIPT STARTED...')
device = find_device()
logging.info('Device: %s' % device)

# Best hyperparameters
batch_size = 256
lr = 0.01

number_of_features = 50
input_size = number_of_features
h1_size = 32
h2_size = 16
output_size = 1
epochs = 1000

X_train = pd.read_pickle('../../data/train_X_agg.pkl')
y_train = pd.read_pickle('../../data/train_y_agg.pkl')
X_val = pd.read_pickle('../../data/val_X_agg.pkl')
y_val = pd.read_pickle('../../data/val_y_agg.pkl')

# load data
train_loader = LoadNNData(X_train, y_train, batch_size)
val_loader = LoadNNData(X_val, y_val, batch_size)

# instantiate the model
model = NeuralNetModule(input_size, h1_size, h2_size, output_size)
model = model.double()
model.to(device)

# set training variables
pos_weight = train_loader.find_pos_weight()
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
sigmoid = nn.Sigmoid()
bestValAp = 0.0
bestValEpoch = 0
patience = 10

# epoch loop
for ep in range(1, epochs + 1):
    logging.info('Epoch %d' % ep)
    start = time.time()

    # batch loop
    n_iter = 0
    for inputs, labels in train_loader.loader:
        n_iter += 1
        logging.info('Batch %d' % n_iter)
        train_batch(model, inputs, labels, device, criterion, optimizer)

    train_time = time_since(start)

    val_start = time.time()
    train_ap, val_ap = validate_nn(model, device, train_loader, val_loader, sigmoid)
    val_time = time_since(val_start)

    logging.info('EPOCH #%d: TRAIN AP %f, VAL AP %f, TRAIN TIME %s, VAL TIME %s' % (ep, train_ap, val_ap,
                                                                                    train_time, val_time))

    # early stopping check
    if val_ap > bestValAp:
        bestValAp = val_ap
        bestValEpoch = ep
        best_model = model
    if ep - bestValEpoch > patience:
        logging.info('STOPPING...BEST EPOCH #%d, BEST VAL AP %f' % (bestValEpoch, bestValAp))
        break

# save model & parameters
torch.save(best_model, '../../models/NN_model.pth')
torch.save(best_model.state_dict(), '../../models/NN_model.st')
logging.info('SCRIPT DONE')
