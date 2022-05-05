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

    model_root = "./models_all_characteristics"
    model_name = "Ridge_alpha0.1"

    def run_backtest_and_save(model_root, model_name):
	    for year in range(1996, 2012 + 1):
	        # load model 
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

	        dates, gain_from_hedges = backtesting(test_data, model, used_characteristics)
	        summary_df = pd.DataFrame({
	            "dates": dates,
	            "gain_from_hedges": gain_from_hedges
	        })
	        print(summary_df)
	        summary_df.to_csv(Path("./analysis_results", f"backtest_{model_name}_{year}.csv"))

	model_root = "./models_all_characteristics"
	model_names = ["Lasso_alpha0.1", "Ridge_alpha0.1", "ElasticNet_alpha0.1", 
					"GBR_n100", "RF_n100"]
	for model_name in model_names:
		run_backtest_and_save(model_root, model_name)
