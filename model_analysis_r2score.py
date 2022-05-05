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


def report_r2_score_and_OS(model, option_with_feature, used_characteristics, year):
    '''
    report the performance of model on year + 7 of option_with_feature, 
    using used_characteristics
    '''
    # split data 
    training_data, validation_data, test_data = train_validation_test_split(option_with_feature, year)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(training_data[used_characteristics])
    training_data.loc[:, used_characteristics] = imp.transform(training_data[used_characteristics])
    test_data.loc[:, used_characteristics] = imp.transform(test_data[used_characteristics])
    X_test = test_data[used_characteristics + ["date", "optionid"]]
    y_test = test_data['option_ret']
    X_test_call = X_test[X_test["C"] == 1]
    y_test_call = y_test[X_test["C"] == 1]
    X_test_put = X_test[X_test["C"] == 0]
    y_test_put = y_test[X_test["C"] == 0]

    def get_r2_score_and_OS(X_test, y_test):
        y_pred = model.predict(X_test[used_characteristics])
        mean_squared_error_ = mean_squared_error(y_test, y_pred)
        r2_score_ = r2_score(y_test, y_pred)
        R2_score_squared_OSXS_ = R_squared_OSXS(y_test, y_pred)
        return r2_score_, R2_score_squared_OSXS_

    # make prediction and analysis
    # for all options
    result = {}
    r2_score_, R2_score_squared_OSXS_ = get_r2_score_and_OS(X_test, y_test)    
    result["all"] = {
        "r2_score": r2_score_, 
        "R2_score_squared_OSXS": R2_score_squared_OSXS_
    }
    r2_score_, R2_score_squared_OSXS_ = get_r2_score_and_OS(X_test_call, y_test_call) 
    result["call"] = {
        "r2_score": r2_score_, 
        "R2_score_squared_OSXS": R2_score_squared_OSXS_
    }
    r2_score_, R2_score_squared_OSXS_ = get_r2_score_and_OS(X_test_put, y_test_put)
    result["put"] = {
        "r2_score": r2_score_, 
        "R2_score_squared_OSXS": R2_score_squared_OSXS_
    }
    return result


if __name__ == "__main__":
    # load data 
    start = time.time()
    data_file = "all_characteristics"
    option_with_feature = pd.read_csv(os.path.join(DATAROOT, f"{data_file}.csv"))
    OUTLIER = 2000
    option_with_feature = option_with_feature[option_with_feature.option_ret.abs() <= OUTLIER]
    option_with_feature = option_with_feature[~option_with_feature.option_ret.isna()]
    option_with_feature["date"] = pd.to_datetime(option_with_feature["date"])
    option_with_feature.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f"finished loading data, used {time.time() - start} seconds")
    print("------------------------------------------------------")
    results = []
    for year in range(1996, 2012 + 1):
        print("year:", year)
        # load model 
        model_root = "./models_all_characteristics"
        model_name = "Ridge_alpha0.1"
        # year = 1996
        model = joblib.load(Path(model_root, f"{model_name}_{year}.pkl"))
        print(model)
        with open(Path(model_root, f"{model_name}_{year}.txt"), "r") as fh:
            used_characteristics = list(
                map(lambda x: x[1:-1], 
                    fh.readline()[1:-1].split(", "))
            )

        result = report_r2_score_and_OS(
            model, option_with_feature, 
            used_characteristics, year
        )
        results.append(result)
        print(result)
    print(results)
    summary = dict()
    summary['test_year'] = list(map(lambda x: x + 7, range(1996, 2012 + 1)))
    for result in results:
        for put_or_call, scores in result.items():
            for score_name, score_value in scores.items():
                if summary.get(put_or_call + score_name, -1) == -1:
                    summary[put_or_call + score_name] = []
                summary[put_or_call + score_name].append(score_value)

    summary_df = pd.DataFrame(summary)
    print(summary_df)
    summary_df.to_csv(Path("analysis_results", f"{model_name}_r2.csv"))
