import numpy as np
import pandas as pd

train_set = pd.read_csv('data/adult.data', header=None)
test_set = pd.read_csv('data/adult.test', skiprows=1, header=None)

col_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
              'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
             'wage_class']
train_set.columns = col_labels
test_set.columns = col_labels

train_nomissing = train_set.replace('?', np.nan).dropna()
test_nomissing = test_set.replace('?', np.nan).dropna()

test_nomissing['wage_class'] = test_nomissing.wage_class.replace({'<=50K.': '<=50K', '>50K.': '>50K'})

combined_set = pd.concat([train_nomissing, test_nomissing], axis=0)
for feature in combined_set.columns:
    if combined_set[feature].dtype == 'object':
        combined_set[feature] = pd.Categorical(combined_set[feature]).codes

final_train = combined_set[:train_nomissing.shape[0] +1]
final_test = combined_set[train_nomissing.shape[0]:]
y_train = final_train.pop('wage_class')
y_test = final_test.pop('wage_class')

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
cv_params = {'max_depth': [3, 5, 7], 'min_child_weight': [1, 3, 5]}
ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8,
             'objective': 'binary:logistic'}
optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), cv_params, scoring='accuracy', cv=5, n_jobs=-1)
optimized_GBM.fit(final_train, y_train)
print(optimized_GBM.grid_scores_)
