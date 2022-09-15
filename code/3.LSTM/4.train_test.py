import logging
import pickle
import pandas as pd
from sklearn.metrics import recall_score, confusion_matrix

from load_data import LoadPyTorchData
from utils import *
from lstm_models import *

# Train and test model
logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S %Z')
logging.info('SCRIPT STARTED...')
device = find_device()
logging.info('Device: %s' % device)

batch_size = 256
lr = 0.005

number_of_features = 50
input_dim = number_of_features + 1
hidden_dim = 32
n_layers = 2
epochs = 1000

# load data
train_loader = LoadPyTorchData('../../data/train', batch_size, test_flag=False)
val_loader = LoadPyTorchData('../../data/val', batch_size, test_flag=False)
test_loader = LoadPyTorchData('../../data/test', batch_size, test_flag=True)

# instantiate the model
model = LSTMNetBasic(input_dim, hidden_dim, n_layers, batch_size)
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
        logging.info('STOPPING...BEST EPOCH #%d, BEST VAL AP %f' % (bestValEpoch, bestValAp))
        break

# test model
logging.info('TESTING MODEL...')
y_true, y_prob, pt_keys = get_test_predictions(best_model, device, test_loader, sigmoid)

# save predictions
predictions = {'y_true': y_true, 'y_prob': y_prob, 'pt_keys': pt_keys}
with open('predictions_.pkl', 'wb') as f:
    pickle.dump(predictions, f)

# performance metrics
threshold = 0.5
# Max probability
y_prob = np.amax(y_prob, axis=1)
y_pred = np.where(y_prob > threshold, 1, 0)

ap = average_precision_score(y_true, y_prob)
auc = roc_auc_score(y_true, y_prob)
sensitivity = recall_score(y_true, y_pred)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
specificity = tn / (tn + fp)

metrics = pd.DataFrame({'AP': [ap], 'AUC': [auc], 'SENSITIVITY': [sensitivity], 'SPECIFICITY': [specificity]})
metrics.to_csv('../../results/performance_metrics_LSTM.csv', index=False)

# save model & parameters
torch.save(best_model, '../../models/LSTM_model.pth')
torch.save(best_model.state_dict(), '../../models/LSTM_model.st')

logging.info('SCRIPT DONE')
