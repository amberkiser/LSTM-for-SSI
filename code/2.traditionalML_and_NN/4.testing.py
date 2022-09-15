import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix, average_precision_score


class EvaluateModel(object):
    def __init__(self, X_data, y_data, model):
        self.X = X_data
        self.y = y_data
        self.y_pred = None
        self.y_prob = None
        self.clf = model

    def run_test(self):
        self.y_pred = self.clf.predict(self.X)
        y_prob = self.clf.predict_proba(self.X)
        self.y_prob = y_prob[:, 1]

    def save_evaluation(self, algorithm, threshold=0.5):
        y_true = self.y.values
        y_prob = self.y_prob
        y_pred = np.where(y_prob > threshold, 1, 0)

        ap = average_precision_score(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        sensitivity = recall_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)

        metrics = pd.DataFrame({'AP': [ap], 'AUC': [auc], 'SENSITIVITY': [sensitivity], 'SPECIFICITY': [specificity]})
        metrics.to_csv('../../results/performance_metrics_%s.csv' % algorithm, index=False)


X_data = pd.read_pickle('../../data/test_X_agg.pkl')
y_data = pd.read_pickle('../../data/test_y_agg.pkl')

with open('../../models/SVM_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)
with open('../../models/RF_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('../../models/XGB_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

# SVM
final_model = EvaluateModel(X_data, y_data, svm_model)
final_model.run_test()
final_model.save_evaluation('SVM')

# Random Forest
final_model = EvaluateModel(X_data, y_data, rf_model)
final_model.run_test()
final_model.save_evaluation('RF')

# XGBoost
final_model = EvaluateModel(X_data, y_data, xgb_model)
final_model.run_test()
final_model.save_evaluation('XGB')
