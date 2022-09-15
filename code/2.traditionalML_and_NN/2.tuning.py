import time
import pandas as pd
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

start_time = time.time()


class HyperparameterTuning(object):
    def __init__(self, X_data, y_data, base_clf, grid_param, cv_num):
        self.X = X_data
        self.y = y_data
        self.base_clf = base_clf
        self.parameter_combos = ParameterGrid(grid_param)
        self.k_folds = StratifiedKFold(n_splits=cv_num)
        self.cv_scores = pd.DataFrame()

    def tune_parameters(self):
        for i in range(len(self.parameter_combos)):
            self.run_cv(i)

    def run_cv(self, iteration):
        fold = 0
        for rep in range(1, 6):
            for train_index, val_index in self.k_folds.split(self.X, self.y):
                fold += 1
                X_train, X_val = self.X.iloc[train_index], self.X.iloc[val_index]
                y_train, y_val = self.y.iloc[train_index], self.y.iloc[val_index]

                clf = clone(self.base_clf)
                clf.set_params(**self.parameter_combos[iteration])
                clf.fit(X_train, y_train)

                y_train_prob = clf.predict_proba(X_train)
                y_train_prob = y_train_prob[:, 1]
                train_scores = self.evaluate_cv_results(y_train, y_train_prob, iteration)

                y_test_prob = clf.predict_proba(X_val)
                y_test_prob = y_test_prob[:, 1]
                val_scores = self.evaluate_cv_results(y_val, y_test_prob, iteration)

                eval_df = self.create_scores_dataframe(train_scores, val_scores, fold)
                self.cv_scores = pd.concat([self.cv_scores, eval_df])

        return None

    def evaluate_cv_results(self, y_true, y_prob, iteration):
        scores = {'Parameter_combo': [], 'AP': [], 'AUC': []}

        scores['Parameter_combo'].append(self.parameter_combos[iteration])
        scores['AP'].append(average_precision_score(y_true, y_prob))
        scores['AUC'].append(roc_auc_score(y_true, y_prob))

        return scores

    def create_scores_dataframe(self, train_dict, val_dict, fold):
        train_df = pd.DataFrame(train_dict)
        train_df['dataset'] = 'train'
        train_df['fold'] = fold

        val_df = pd.DataFrame(val_dict)
        val_df['dataset'] = 'val'
        val_df['fold'] = fold
        eval_df = pd.concat([train_df, val_df]).reset_index(drop=True)
        return eval_df

    def process_and_save_results(self, algorithm):
        self.cv_scores['Parameter_combo'] = self.cv_scores['Parameter_combo'].apply(lambda x: str(x))
        results_agg = self.cv_scores.drop(columns='fold').groupby(['dataset', 'Parameter_combo']).agg('mean').reset_index()

        results_train = results_agg.loc[results_agg.dataset == 'train'].drop(columns='dataset')
        new_names = {}
        for col in results_train.columns:
            if col != 'Parameter_combo':
                new_names[col] = col + '_train'
        results_train = results_train.rename(columns=new_names)

        results_val = results_agg.loc[results_agg.dataset == 'val'].drop(columns='dataset').reset_index(drop=True)
        new_names = {}
        for col in results_val.columns:
            if col != 'Parameter_combo':
                new_names[col] = col + '_val'
        results_val = results_val.rename(columns=new_names)
        results_val['AP_rank'] = results_val['AP_val'].rank(axis=0, method='dense', ascending=False)
        results_val['AUC_rank'] = results_val['AUC_val'].rank(axis=0, method='dense', ascending=False)

        results_combined = results_train.merge(results_val, on='Parameter_combo')
        results_combined = results_combined[['Parameter_combo', 'AP_val', 'AP_rank', 'AUC_val',
                                             'AUC_rank', 'AP_train', 'AUC_train']]
        results_combined.to_csv('hyperparameter_tuning_%s.csv' % algorithm)
        return None


X_data = pd.read_pickle('../../data/train_X_agg.pkl')
y_data = pd.read_pickle('../../data/train_y_agg.pkl')

# SVM
base_clf = SVC(probability=True, gamma='scale', class_weight='balanced')
parameters = {
    'C': [0.01, 0.1, 0.25, 0.50, 1, 2],
    'kernel': ['linear', 'rbf']
}
tune = HyperparameterTuning(X_data, y_data, base_clf, parameters, 10)
tune.tune_parameters()
tune.process_and_save_results('SVM')
print('SVM - Done!')

# Random forest
base_clf = RandomForestClassifier(class_weight='balanced')
parameters = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 2, 10, 20],
    'criterion': ['gini', 'entropy']
}
tune = HyperparameterTuning(X_data, y_data, base_clf, parameters, 10)
tune.tune_parameters()
tune.process_and_save_results('RF')
print('RF - Done!')

# XGBoost
# scale_pos_weight = total_negative / total_positive
base_clf = xgb.XGBClassifier(eval_metric='aucpr', use_label_encoder=False,
                             scale_pos_weight=((len(y_data) - y_data.sum()) / y_data.sum()))
parameters = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [10, 50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1]
}
tune = HyperparameterTuning(X_data, y_data, base_clf, parameters, 10)
tune.tune_parameters()
tune.process_and_save_results('XGB')
print('XGB - Done!')

end_time = time.time()
total_time = end_time - start_time
print('Total run time: %f seconds' % total_time)
