import numpy as np
import datetime as dt
import pandas as pd 
import os
import numpy as np
import time 
import joblib
import datetime 
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

from evaluation_metrics import CW_test, DM_test, R_squared_OSXS
from consts import DATAROOT, used_characteristics
from ml_helper_functions import get_data_between, train_validation_test_split


# load data 
start = time.time()
data_file = "all_characteristics"
option_with_feature = pd.read_csv(os.path.join(DATAROOT, f"{data_file}.csv"), nrows=1000000)
OUTLIER = 2000
option_with_feature = option_with_feature[option_with_feature.option_ret.abs() <= OUTLIER]
option_with_feature = option_with_feature[~option_with_feature.option_ret.isna()]
option_with_feature["date"] = pd.to_datetime(option_with_feature["date"])
option_with_feature.replace([np.inf, -np.inf], np.nan, inplace=True)
print(f"finished loading data, used {time.time() - start} seconds")
print("------------------------------------------------------")
# load model 
model_root = "./models_all_characteristics"
model_name = "Ridge_alpha0.1"
year = 1996
model = joblib.load(Path(model_root, f"{model_name}_{year}.pkl"))
print(model)
with open(Path(model_root, f"{model_name}_{year}.txt"), "r") as fh:
    used_characteristics = list(
        map(lambda x: x[1:-1], 
            fh.readline()[1:-1].split(", "))
    )
# split data 
training_data, validation_data, test_data = train_validation_test_split(option_with_feature, year)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(training_data[used_characteristics])
training_data.loc[:, used_characteristics] = imp.transform(training_data[used_characteristics])
test_data.loc[:, used_characteristics] = imp.transform(test_data[used_characteristics])
X_test = test_data[used_characteristics + ["date", "optionid"]]
y_test = test_data['option_ret']
# make prediction and analysis
y_pred = model.predict(X_test[used_characteristics])
mean_squared_error_ = mean_squared_error(y_test, y_pred)
r2_score_ = r2_score(y_test, y_pred)
R2_score_squared_OSXS_ = R_squared_OSXS(y_test, y_pred)

print(mean_squared_error_, r2_score_, R2_score_squared_OSXS_)
