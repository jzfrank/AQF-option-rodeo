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
from consts import DATAROOT, OUTLIER
from ml_helper_functions import get_data_between, train_validation_test_split, backtesting

if __name__ == '__main__':
    # load data 
    start = time.time()
    data_file = "all_characteristics"
    option_with_feature = pd.read_csv(Path(DATAROOT, f"{data_file}.csv"))
    option_with_feature = option_with_feature[option_with_feature.option_ret.abs() <= OUTLIER]
    option_with_feature = option_with_feature[~option_with_feature.option_ret.isna()]
    option_with_feature["date"] = pd.to_datetime(option_with_feature["date"])
    option_with_feature.replace([np.inf, -np.inf], np.nan, inplace=True)
    # merge with weights
    weight_info = pd.read_csv(Path(DATAROOT, "weight_info.csv"))
    weight_info = weight_info[["date", "optionid", "dollar_open_interest"]]
    weight_info["date"] = pd.to_datetime(weight_info["date"])
    weight_info = weight_info.rename(columns={"dollar_open_interest": "open_interest"})
    print("option_with_feature.shape before merging:", option_with_feature.shape)
    option_with_feature = pd.merge(option_with_feature, weight_info, on=["date", "optionid"], how="inner")
    print("option_with_feature.shape after merging:", option_with_feature.shape)
    print(f"finished loading data, used {time.time() - start} seconds")
    print("------------------------------------------------------")

    def run_backtest_and_save(model_root, model_names, 
                              saved_folder="analysis_results/all_features", start_year=1996, end_year=2013):
        for year in range(start_year, end_year + 1):
            # all models use the same characteristics in the same year 
            with open(Path(model_root, f"{model_names[0]}_{year}.txt"), "r") as fh:
                used_characteristics = list(
                    map(lambda x: x[1:-1], 
                        fh.readline()[1:-1].split(", "))
                )
            # load data 
            training_data, validation_data, test_data = train_validation_test_split(option_with_feature, year)
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            imp.fit(training_data[used_characteristics])
            training_data.loc[:, used_characteristics] = imp.transform(training_data[used_characteristics])
            test_data.loc[:, used_characteristics] = imp.transform(test_data[used_characteristics])

            for model_name in model_names:
                print(model_root, model_name, saved_folder)
                # load model 
                model = joblib.load(Path(model_root, f"{model_name}_{year}.pkl"))
                print(model.get_params())

                dates, gain_from_hedges = backtesting(test_data, model, used_characteristics)
                summary_df = pd.DataFrame({
                    "dates": dates,
                    "gain_from_hedges": gain_from_hedges
                })
                print(summary_df)
                summary_df.to_csv(Path(saved_folder, f"backtest_{model_name}_{year}.csv"))

            # memory management
            print("virtual memory availability before del: ", psutil.virtual_memory().available / psutil.virtual_memory().total * 100)
            del training_data, validation_data, test_data 
            gc.collect()
            print("virtual memory availability after del: ", psutil.virtual_memory().available / psutil.virtual_memory().total * 100)

    model_root = "./models_nonsparse"
    model_names = ["Lasso_alpha0.1", "Ridge_alpha0.1", 
                   "GBR_n100", "RF_n100"]
    model_names = ["GBR_n100", "RF_n100"]
    saved_folder = "analysis_results/nonsparse_features"
    if not os.path.exists(saved_folder):
        os.mkdir(saved_folder)
    run_backtest_and_save(model_root, model_names, saved_folder, 1996, 2013)
