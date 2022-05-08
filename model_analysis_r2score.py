import numpy as np
import datetime as dt
import pandas as pd 
import os
import numpy as np
import time 
import joblib
import datetime 
from pathlib import Path
import psutil 
import gc

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
from consts import DATAROOT, used_characteristics_all
from ml_helper_functions import get_data_between, train_validation_test_split


def report_r2_score_and_OS(model, X_test, y_test, X_test_call, y_test_call,
                           X_test_put, y_test_put, used_characteristics, year):
    '''
    report the performance of model on year + 7 of option_with_feature, 
    using used_characteristics
    '''
    def get_r2_score_and_OS(X_test, y_test):
        y_pred = model.predict(X_test[used_characteristics])
        mean_squared_error_ = mean_squared_error(y_test, y_pred)
        r2_score_ = r2_score(y_test, y_pred)
        R2_score_squared_OSXS_ = R_squared_OSXS(y_test, y_pred)
        return r2_score_, R2_score_squared_OSXS_

    start = time.time()
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
    print("time used for making prediction and analysis:", time.time() - start)

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

    def compute_r2_and_save(model_root, model_names, start_year, end_year):
        model2results = []
        for year in range(start_year, end_year + 1):
            print("year:", year)
            # load used_characteristics
            with open(Path(model_root, f"{model_names[0]}_{year}.txt"), "r") as fh:
                used_characteristics = list(
                    map(lambda x: x[1:-1], 
                        fh.readline()[1:-1].split(", "))
                )

            # split data 
            start = time.time()
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
            print("time used for spliting and imputing data:", time.time() - start)

            model2result = {}
            # load model 
            for model_name in model_names:
                model = joblib.load(Path(model_root, f"{model_name}_{year}.pkl"))
                print(model)
                result = report_r2_score_and_OS(
                    model, 
                    X_test, y_test, 
                    X_test_call, y_test_call,
                    X_test_put, y_test_put,
                    used_characteristics, year
                )
                model2result[model_name] = result

            model2results.append(model2result)
            print(model2result)

            with open(f"analysis_results/nonsparse_features/r2_score_dict_{year}.txt", "w") as fh:
                fh.write(str(model2result))

            # memory management
            print("virtual memory availability before del: ", psutil.virtual_memory().available / psutil.virtual_memory().total * 100)
            del training_data, validation_data, test_data 
            gc.collect()
            print("virtual memory availability after del: ", psutil.virtual_memory().available / psutil.virtual_memory().total * 100)

        print(model2results)
        with open("analysis_results/nonsparse_features/r2_score_dict.txt", "w") as fh:
            fh.write(str(model2results))
        summary_df = pd.DataFrame()
        summary_df['test_year'] = list(map(lambda x: x + 7, range(start_year, end_year + 1)))
        for i, model_name in enumerate(model_names):
            summary_df[model_name] = [
                model2result[model_name] for model2result in model2results
            ]

        print(summary_df)
        summary_df.to_csv(Path("analysis_results/nonsparse_features", f"{model_name}_r2.csv"))

    model_root = "./models_nonsparse"
    model_names = ["Lasso_alpha0.1", "Ridge_alpha0.1", 
                   "GBR_n100", "RF_n100"]
    compute_r2_and_save(model_root, model_names, 1996, 2013)
