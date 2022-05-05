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
from ml_helper_functions import get_data_between, train_validation_test_split, backtesting

if __name__ == '__main__':
    # load data 
    start = time.time()
    data_file = "all_characteristics"
    option_with_feature = pd.read_csv(Path(DATAROOT, f"{data_file}.csv"))
    OUTLIER = 2000
    option_with_feature = option_with_feature[option_with_feature.option_ret.abs() <= OUTLIER]
    option_with_feature = option_with_feature[~option_with_feature.option_ret.isna()]
    option_with_feature["date"] = pd.to_datetime(option_with_feature["date"])
    option_with_feature.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f"finished loading data, used {time.time() - start} seconds")
    print("------------------------------------------------------")

    CW_scores = []
    for year in range(1996, 2012 + 1):
        # load model 
        model_root = "./models_all_characteristics"
        model_name = "Ridge_alpha0.1"
        model = joblib.load(Path(model_root, f"{model_name}_{year}.pkl"))
        print(model.get_params())
        with open(Path(model_root, f"{model_name}_{year}.txt"), "r") as fh:
            used_characteristics = list(
                map(lambda x: x[1:-1], 
                    fh.readline()[1:-1].split(", "))
            )

        training_data, validation_data, test_data = train_validation_test_split(option_with_feature, year)
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(training_data[used_characteristics])
        training_data.loc[:, used_characteristics] = imp.transform(training_data[used_characteristics])
        test_data.loc[:, used_characteristics] = imp.transform(test_data[used_characteristics])

        # compute CW score
        true_pred_return = pd.DataFrame({
            'optionid': test_data.optionid, 
            'time': test_data.date, 
            'true_return': test_data.option_ret, 
            'pred_return': model.predict(test_data[used_characteristics])
        })
        CW_score = CW_test(true_pred_return)
        CW_scores.append(CW_score)
        print(f"CW_score for {model_name} in {year+8}", CW_score)
    summary_df = pd.DataFrame({
        "year": list(map(lambda x: x + 7, range(1996, 2012 + 1))),
        "CW_score": CW_scores
    })
    print(summary_df)
    summary_df.to_csv(Path("analysis_results", f"CW_score_{model_name}_year.csv"))
