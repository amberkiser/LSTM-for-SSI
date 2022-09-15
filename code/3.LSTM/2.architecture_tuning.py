import logging

from load_data import LoadPyTorchData
from utils import *
from lstm_models import *


# architecture tuning for each: model (12) = 12 total files
logging.basicConfig(filename='logs/architecture_tuning.log', level=logging.INFO,
                    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S %Z')
logging.info('Started script...')

device = find_device()
logging.info('Device: %s' % device)

batch_size = 512
number_of_features = 50
input_dim = number_of_features + 1
hidden_dim = 32
n_layers = 2
epochs = 1000

# perform 10-fold cross validation with 5 repeats - 2 for loops
for k in range(1, 11):
    logging.info('Fold %d' % k)
    for rep in range(1, 6):
        logging.info('Fold %d; Rep %d' % (k, rep))

        # load data
        train_loader = LoadPyTorchData('../../data/cv_data/train_%d' % k, batch_size)
        val_loader = LoadPyTorchData('../../data/cv_data/val_%d' % k, batch_size)

        # instantiate the model
        model = LSTMNetBasic(input_dim, hidden_dim, n_layers, batch_size)
        model = model.double()
        model.to(device)

        # set training variables
        lr = 0.001  # tune this
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
            train_ap, val_ap = validate(model, device, train_loader, val_loader, sigmoid)
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
