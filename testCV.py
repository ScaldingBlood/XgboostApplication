from xgboost.sklearn import XGBClassifier
from sklearn.grid_search import GridSearchCV
import xgboost

other_param = {}
xgb_model = XGBClassifier(other_param)

test_param = {}
model = GridSearchCV(estimator=xgb_model, param_grid=test_param)

train = xgboost.DMatrix()
target = xgboost.DMatrix()
model.fit(train, target)
print(model.best_params_)


