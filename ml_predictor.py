import numpy as np
import datetime as dt
import pandas as pd 
import os
import numpy as np
import time 
import joblib
import datetime 

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
from consts import DATAROOT, used_characteristics, used_characteristics_nonsparse
from ml_helper_functions import get_data_between, train_validation_test_split


def run_regression_and_save(reg, saved_folder, model_name, option_with_feature, used_characteristics, year):
    '''
    run regression and save model to f"{saved_folder}/{model_name}{year}.pkl" file
    Note: use joblib.dump(reg, filename) to save model
          use joblib.load(filename) to load model
    '''
    print(datetime.datetime.strftime(datetime.datetime.now(), "%m-%d %H:%M"))
    start = time.time()
    print(f"iteration, year: {year}, running regression...")
    training_data, validation_data, test_data = train_validation_test_split(option_with_feature, year)
    non_useable_feature = set()
    for df in [training_data, validation_data, test_data]:
        for key, val in dict(df.isna().sum()).items():
            if val == df.shape[0]:
                print(key)
                non_useable_feature.add(key)

    used_characteristics = list(set(used_characteristics) - set(non_useable_feature))
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(training_data[used_characteristics])
    training_data.loc[:, used_characteristics] = imp.transform(training_data[used_characteristics])
    validation_data.loc[:, used_characteristics] = imp.transform(validation_data[used_characteristics])
    test_data.loc[:, used_characteristics] = imp.transform(test_data[used_characteristics])

    X_train = training_data[used_characteristics + ["date", "optionid"]]
    y_train = training_data['option_ret']

    X_val = validation_data[used_characteristics + ["date", "optionid"]]
    y_val = validation_data['option_ret']

    X_test = test_data[used_characteristics + ["date", "optionid"]]
    y_test = test_data['option_ret']

    # # tuning hyperparameter on validation set
    # # it costs a long time, so we just select an empirical result later instead of 
    # # running it every time
    # best_alpha = 0.1
    # best_r2_score = None
    # for alpha in [0.01, 0.1, 1]:
    #     reg = linear_model.Lasso(random_state=0, alpha=best_alpha)
    #     reg.fit(X_train[used_characteristics], y_train)
    #     y_pred = reg.predict(X_val[used_characteristics])
    #     r2_score_ = r2_score(y_val, y_pred)
    #     if best_r2_score is None or r2_score_ > best_r2_score:
    #         best_alpha = alpha 
    #         best_r2_score = r2_score_
    # print(f"best_alpha: {best_alpha}")

    # regression    
    reg.fit(X_train[used_characteristics], y_train)
    # save model
    joblib.dump(reg, f"{saved_folder}/{model_name}_{year}.pkl")
    with open(f"{saved_folder}/{model_name}_{year}.txt", "w") as f:
        f.write(str(used_characteristics))

    # y_pred = reg.predict(X_test[used_characteristics])
    # mean_squared_errors.append(mean_squared_error(y_test, y_pred))
    # r2_scores.append(r2_score(y_test, y_pred))
    # R_squared_OSXS_s.append(R_squared_OSXS(y_test, y_pred))
    # print(year, mean_squared_errors[-1], r2_scores[-1], R_squared_OSXS_s[-1])

    # dates, gain_from_hedges = backtesting(test_data, reg)
    # pd.DataFrame({
    #     "dates": dates,
    #     "gain_from_hedges": gain_from_hedges
    # }).to_csv(f"{model_name}_backtesting_{year}.csv")

    print(f"finished one iteration, used {time.time() - start} seconds")
    print("------------------------------------------------------")


def run_regression_from_to(reg, saved_folder, model_name, option_with_feature, used_characteristics, start_year=1996, end_year=2012):
    '''
    run regression from start_year to end_year (inclusive)
    Note: [year, year + 7] is one training-validation-test iteration
    '''
    print(f"for {model_name}\n")
    for year in range(start_year, end_year + 1):
        run_regression_and_save(reg, saved_folder, model_name, option_with_feature, used_characteristics, year)


if __name__ == "__main__":
    start = time.time()
    # load data 
    data_file = "all_characteristics"
    option_with_feature = pd.read_csv(os.path.join(DATAROOT, f"{data_file}.csv"))
    OUT_LIER = 2000
    option_with_feature = option_with_feature[option_with_feature.option_ret.abs() <= OUT_LIER]
    option_with_feature = option_with_feature[~option_with_feature.option_ret.isna()]
    option_with_feature["date"] = pd.to_datetime(option_with_feature["date"])
    option_with_feature.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f"finished loading data, used {time.time() - start} seconds")
    print("------------------------------------------------------")
    # suppress the warning of 
    #   "A value is trying to be set on a copy of a slice from a DataFrame. 
    #   Try using .loc[row_indexer,col_indexer] = value instead"
    pd.options.mode.chained_assignment = None 

    saved_folder = "./models_nonsparse"
    if not os.path.exists(saved_folder):
        os.mkdir(saved_folder)

    # ----- Linear models ------
    # Lasso
    best_alpha = 0.1  # empirically result from validation
    reg = linear_model.Lasso(random_state=0, alpha=best_alpha)
    run_regression_from_to(reg, saved_folder, "Lasso_alpha0.1", option_with_feature, used_characteristics_nonsparse, 1996, 2012)
    # Ridge
    best_alpha = 0.1  # empirically result from validation
    reg = linear_model.Ridge(random_state=0, alpha=best_alpha)
    run_regression_from_to(reg, saved_folder, "Ridge_alpha0.1", option_with_feature, used_characteristics_nonsparse, 1996, 2012)
    # Elastic
    elastic_reg = ElasticNet(alpha=0.1)
    run_regression_from_to(reg, saved_folder, "ElasticNet_alpha0.1", option_with_feature, used_characteristics_nonsparse, 1996, 2012)

    # ----- Nonlinear models ----- 
    # GBR
    reg = GradientBoostingRegressor(
        n_estimators=100, 
        random_state=0,
        loss='huber',
        verbose=1
    )
    run_regression_from_to(reg, saved_folder, "GBR_n100", option_with_feature, used_characteristics_nonsparse, 1996, 2012)
    # RF
    reg = RandomForestRegressor(
        n_estimators=100,
        max_depth=3, 
        random_state=0, 
        verbose=1)
    run_regression_from_to(reg, saved_folder, "RF_n100", option_with_feature, used_characteristics_nonsparse, 1996, 2012)
