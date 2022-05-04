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
from consts import DATAROOT, used_characteristics


def get_data_between(option_with_feature, start_year, end_year):
    '''
    return option_with_feature between {start_year}-01-01 and {end_year}-12-31
    (inclusive: [start_hear, end_year])
    '''
    begin_date = dt.datetime.strptime(f"{start_year}-01-01", "%Y-%m-%d")
    end_date = dt.datetime.strptime(f"{end_year}-12-31", "%Y-%m-%d")
    return option_with_feature[
        option_with_feature.date.between(begin_date, end_date)
    ]


def train_validation_test_split(option_with_feature, year):
    '''
    split option_with_feature into train, validation, test dataset.
    year is the starting year. 
    training: [year, year + 4]
    validation: [year + 5. year + 6]
    test: [year + 7, year + 7]
    '''
    training_data = get_data_between(option_with_feature, year, year + 4)
    validation_data = get_data_between(option_with_feature, year + 5, year + 6)
    test_data = get_data_between(option_with_feature, year + 7, year + 7)
    return training_data, validation_data, test_data 


def backtesting(test_data, regressor, used_characteristics):
    '''
    implement long-short portfolio based on regressor. 
    test_data should contain one year's option_with_feature data 
    regressor predits the option return 
    used_characteristics is a list recording the characteristics used for prediction
    '''
    dates = sorted(list(set(test_data.date)))
    gain_from_hedges = []
    for i in range(len(dates)):
        df = test_data[test_data.date == dates[i]]
        df = pd.DataFrame(
            {
                "secid": df.secid,
                "adj_spot": df.adj_spot,
                "strike_price": df.strike_price,
                "delta": df.delta,
                "mid_price": df.mid_price,
                "option_ret_real": df.option_ret,
                "option_ret_pred": regressor.predict(df[used_characteristics])
            }
        )
        df["hedge_cost"] = df["mid_price"] - df["adj_spot"] * df["delta"]
        df["hedge_gain"] = abs(df["hedge_cost"]) * df["option_ret_real"] 
        long_portfolio = df.sort_values(by="option_ret_pred").head(10)
        short_portfolio = df.sort_values(by="option_ret_pred").tail(10)
        short_portfolio.hedge_cost = - short_portfolio.hedge_cost
        gain_from_hedge = (sum(long_portfolio["hedge_cost"]) + sum(short_portfolio["hedge_cost"]) 
                           + sum(long_portfolio["hedge_gain"]) + sum(short_portfolio["hedge_gain"]))
        gain_from_hedges.append(gain_from_hedge)
    return dates, gain_from_hedges


def run_regression_and_save(reg, model_name, option_with_feature, used_characteristics, year):
    '''
    run regression and save model to "models/{model_name}{year}.pkl" file
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
    joblib.dump(reg, f"models/{model_name}_{year}.pkl")
    with open(file=f"models/{model_name}_{year}.txt") as f:
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


def run_regression_from_to(reg, model_name, option_with_feature, used_characteristics, start_year=1996, end_year=2012):
    '''
    run regression from start_year to end_year (inclusive)
    Note: [year, year + 7] is one training-validation-test iteration
    '''
    print(f"for {model_name}\n {reg.get_params()}")
    for year in range(start_year, end_year + 1):
        run_regression_and_save(reg, model_name, option_with_feature, used_characteristics, year)


if __name__ == "__main__":
    start = time.time()
    # load data 
    option_with_feature = pd.read_csv(os.path.join(DATAROOT, "all_characteristics.csv"))
    option_with_feature = option_with_feature[~option_with_feature.option_ret.isna()]
    option_with_feature["date"] = pd.to_datetime(option_with_feature["date"])
    option_with_feature.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f"finished loading data, used {time.time() - start} seconds")
    print("------------------------------------------------------")
    # suppress the warning of 
    #   "A value is trying to be set on a copy of a slice from a DataFrame. 
    #   Try using .loc[row_indexer,col_indexer] = value instead"
    pd.options.mode.chained_assignment = None 

    # # ----- Linear models ------
    # Lasso
    best_alpha = 0.1  # empirically result from validation
    reg = linear_model.Lasso(random_state=0, alpha=best_alpha)
    run_regression_from_to(reg, "Lasso_alpha0.1", option_with_feature, used_characteristics, 1996, 2012)
    # Ridge
    best_alpha = 0.1  # empirically result from validation
    reg = linear_model.Ridge(random_state=0, alpha=best_alpha)
    run_regression_from_to(reg, "Ridge_alpha0.1", option_with_feature, used_characteristics, 1996, 2012)
    # Elastic
    elastic_reg = ElasticNet(alpha=0.1)
    run_regression_from_to(reg, "ElasticNet_alpha0.1", option_with_feature, used_characteristics, 1996, 2012)

    # ----- Nonlinear models ----- 
    # GBR
    reg = GradientBoostingRegressor(
        n_estimators=100, random_state=0,
        loss='huber',
        verbose=1
    )
    run_regression(reg, "GBR_n100")
    # RF
    reg = RandomForestRegressor(
        n_estimators=100,
        max_depth=3, random_state=0, 
        verbose=1)
    run_regression(reg, "RF_n100", 1996, 2012)
