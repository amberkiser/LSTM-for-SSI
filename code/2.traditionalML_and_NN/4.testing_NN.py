import logging
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix, average_precision_score

from load_data import LoadNNData
from utils import *
from nn_models import *

# Test model
logging.basicConfig(filename='testing_NN.log', level=logging.INFO,
                    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S %Z')
logging.info('SCRIPT STARTED...')
device = find_device()
logging.info('Device: %s' % device)

# Load model
best_model = torch.load('../../models/NN_model.pth')
best_model.load_state_dict(torch.load('../../models/NN_model.st'))

batch_size = 256

# load data
X_test = pd.read_pickle('../../data/test_X_agg.pkl')
y_test = pd.read_pickle('../../data/test_y_agg.pkl')
test_loader = LoadNNData(X_test, y_test, batch_size)

sigmoid = nn.Sigmoid()

# test model
logging.info('TESTING MODEL...')
y_true, y_prob = get_predictions(best_model, device, test_loader, sigmoid)

# save predictions
predictions = {'y_true': y_true, 'y_prob': y_prob}
with open('predictions_NN.pkl', 'wb') as f:
    pickle.dump(predictions, f)

# Performance metrics
threshold = 0.5
y_pred = np.where(y_prob > threshold, 1, 0)

ap = average_precision_score(y_true, y_prob)
auc = roc_auc_score(y_true, y_prob)
sensitivity = recall_score(y_true, y_pred)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
specificity = tn / (tn + fp)

# Collect metrics in dataframe
metrics = pd.DataFrame({'AP': [ap], 'AUC': [auc], 'SENSITIVITY': [sensitivity], 'SPECIFICITY': [specificity]})
metrics.to_csv('../../results/performance_metrics_NN.csv', index=False)

logging.info('SCRIPT DONE')
