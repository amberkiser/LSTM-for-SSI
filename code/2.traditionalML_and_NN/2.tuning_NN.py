import logging
from sklearn.model_selection import StratifiedKFold
import pandas as pd

from load_data import LoadNNData
from utils import *
from nn_models import *


# hyperparameter tuning for neural network
# hyperparameter(s) include: learning rate, batch size
logging.basicConfig(filename='hyperparameter_tuning_NN.log', level=logging.INFO,
                    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S %Z')
logging.info('Started script...')

device = find_device()
logging.info('Device: %s' % device)

batch_size_list = [128, 256, 512]
lr_list = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]

number_of_features = 50
input_size = number_of_features
h1_size = 32
h2_size = 16
output_size = 1
epochs = 1000
cv_num = 10

X_train = pd.read_pickle('../../data/train_X_agg.pkl')
y_train = pd.read_pickle('../../data/train_y_agg.pkl')

k_folds = StratifiedKFold(n_splits=cv_num)
for batch_size in batch_size_list:
    logging.info('BATCH SIZE: %d' % batch_size)
    for lr in lr_list:
        logging.info('LR: %f' % lr)
        # perform 10-fold cross validation with 5 repeats - 2 for loops
        fold = 0
        for train_index, val_index in k_folds.split(X_train, y_train):
            fold += 1
            logging.info('Fold %d' % fold)
            X_trn, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_trn, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
            for rep in range(1, 6):
            # for rep in range(1, 3):
                logging.info('Fold %d; Rep %d' % (fold, rep))

                # load data
                train_loader = LoadNNData(X_trn, y_trn, batch_size)
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
                for ep in range(1, epochs+1):
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
                        logging.info('STOPPING...BEST EPOCH #%d' % bestValEpoch)
                        break
