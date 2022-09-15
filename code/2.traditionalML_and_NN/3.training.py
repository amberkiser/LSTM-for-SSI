import time
import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

start_time = time.time()


class TrainFinalModel(object):
    def __init__(self, X_data, y_data, clf):
        self.X = X_data
        self.y = y_data
        self.clf = clf

    def train_model(self):
        self.clf.fit(self.X, self.y)
        return None

    def save_model(self, path_to_save_models):
        with open(path_to_save_models, 'wb') as f:
            pickle.dump(self.clf, f)
        return None


X_data = pd.read_pickle('../../data/train_X_agg.pkl')
y_data = pd.read_pickle('../../data/train_y_agg.pkl')

# SVM
svm_clf = SVC(probability=True,
              gamma='scale',
              class_weight='balanced',
              kernel='linear',
              C=1)

path_to_save_models = '../../models/SVM_model.pkl'

final_model = TrainFinalModel(X_data, y_data, svm_clf)
final_model.train_model()
final_model.save_model(path_to_save_models)

print('SVM Done!')


# Random forest
rf_clf = RandomForestClassifier(class_weight='balanced',
                                n_estimators=200,
                                max_depth=10,
                                criterion='entropy')

path_to_save_models = '../../models/RF_model.pkl'

final_model = TrainFinalModel(X_data, y_data, rf_clf)
final_model.train_model()
final_model.save_model(path_to_save_models)

print('RF Done!')


# XGBoost
xgb_clf = xgb.XGBClassifier(eval_metric='aucpr',
                            use_label_encoder=False,
                            scale_pos_weight=((len(y_data) - y_data.sum()) / y_data.sum()),
                            n_estimators=200,
                            max_depth=200,
                            learning_rate=0.1)

path_to_save_models = '../../models/XGB_model.pkl'

final_model = TrainFinalModel(X_data, y_data, xgb_clf)
final_model.train_model()
final_model.save_model(path_to_save_models)

print('XGB Done!')


end_time = time.time()
total_time = end_time - start_time
print('Total run time: %f seconds' % total_time)
