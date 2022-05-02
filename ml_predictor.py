import numpy as np
import datetime as dt
import pandas as pd 
import os
import numpy as np
import time 

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt
from evaluation_metrics import CW_test, DM_test, R_squared_OSXS


used_characteristics = ['volume', 'ReturnSkew', 'MaxRet', 'delta', 'PriceDelaySlope', 'strike_price', 
                        'IdioVol3F', 'ReturnSkew3F', 'ir_rate', 'mid_price', 'forwardprice', 
                        'zerotradeAlt1', 'theta', 'cfadj', 'zerotrade', 'best_bid', 'spotprice', 'VolMkt', 
                        'IdioRisk', 'days_to_exp', 'PriceDelayTstat', 'High52', 'Coskewness', 'BidAskSpread', 
                                    'Beta', 'days_no_trading', 'open_interest', 'impl_volatility', 'PriceDelayRsq', 'IdioVolAHT', 
                                    'adj_spot', 'vega', 'gamma', 'best_offer', 'DolVol', 'cp_flag_encoded']


def train_validation_test_split(option_with_feature, year):
    begin_date = dt.datetime.strptime(f"{year}-01-01", "%Y-%m-%d")
    end_date = dt.datetime.strptime(f"{year+4}-12-31", "%Y-%m-%d")
    training_data = option_with_feature[
        option_with_feature.date_x.between(begin_date, end_date)
    ]

    begin_date = dt.datetime.strptime(f"{year+5}-01-01", "%Y-%m-%d")
    end_date = dt.datetime.strptime(f"{year+6}-12-31", "%Y-%m-%d")
    validation_data = option_with_feature[
        option_with_feature.date_x.between(begin_date, end_date)
    ]

    begin_date = dt.datetime.strptime(f"{year+7}-01-01", "%Y-%m-%d")
    end_date = dt.datetime.strptime(f"{year+7}-12-31", "%Y-%m-%d")
    test_data = option_with_feature[
        option_with_feature.date_x.between(begin_date, end_date)
    ]

    return training_data, validation_data, test_data 


def backtesting(test_data, regressor):
    dates = sorted(list(set(test_data.date_x)))
    gain_from_hedges = []
    for i in range(len(dates)):
        df = test_data[test_data.date_x == dates[i]]
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


if __name__ == "__main__":
    start = time.time()
    # load data 
    # for running in euler server 
    DATAROOT = "/cluster/scratch/zhajin/data/"
    option_with_feature = pd.read_csv(os.path.join(DATAROOT, "option_with_nonsparse_features.csv"))
    option_with_feature = option_with_feature[~option_with_feature.option_ret.isna()]
    option_with_feature["date_x"] = option_with_feature.date_x.apply(
        lambda x: dt.datetime.strptime(x.split()[0], "%Y-%m-%d")).copy()
    option_with_feature["cp_flag_encoded"] = option_with_feature["cp_flag"].apply(lambda x: {"P": 0, "C": 1}[x])
    print(f"finished loading data, used {time.time() - start} seconds")
    print("------------------------------------------------------")
    # suppress the warning of 
    #   "A value is trying to be set on a copy of a slice from a DataFrame. 
    #   Try using .loc[row_indexer,col_indexer] = value instead"
    pd.options.mode.chained_assignment = None 

    def run_regression(reg, file_name):
        print(f"for {file_name}\n {reg.get_params()}")
        mean_squared_errors = []
        r2_scores = []
        R_squared_OSXS_s = []

        for year in range(1996, 2020 - 7):
            start = time.time()
            print(f"iteration, year: {year}, running regression...")
            training_data, validation_data, test_data = train_validation_test_split(option_with_feature, year)

            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            imp.fit(training_data[used_characteristics])
            training_data.loc[:, used_characteristics] = imp.transform(training_data[used_characteristics])
            validation_data.loc[:, used_characteristics] = imp.transform(validation_data[used_characteristics])
            test_data.loc[:, used_characteristics] = imp.transform(test_data[used_characteristics])

            X_train = training_data[used_characteristics + ["date_x", "optionid"]]
            y_train = training_data['option_ret']

            X_val = validation_data[used_characteristics + ["date_x", "optionid"]]
            y_val = validation_data['option_ret']

            X_test = test_data[used_characteristics + ["date_x", "optionid"]]
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
            # best_alpha = 0.1  # empirically result from validation
            # reg = linear_model.Lasso(random_state=0, alpha=best_alpha)
            reg.fit(X_train[used_characteristics], y_train)
            y_pred = reg.predict(X_test[used_characteristics])
            mean_squared_errors.append(mean_squared_error(y_test, y_pred))
            r2_scores.append(r2_score(y_test, y_pred))
            R_squared_OSXS_s.append(R_squared_OSXS(y_test, y_pred))
            print(year, mean_squared_errors[-1], r2_scores[-1], R_squared_OSXS_s[-1])

            dates, gain_from_hedges = backtesting(test_data, reg)
            pd.DataFrame({
                "dates": dates,
                "gain_from_hedges": gain_from_hedges
            }).to_csv(f"{file_name}_backtesting_{year}.csv")

            print(f"finished one iteration, used {time.time() - start} seconds")
            print("------------------------------------------------------")

        pd.DataFrame({
            "start_year": list(range(1996, 2020 - 7)), 
            "mean_squared_error": mean_squared_errors,
            "r2_score": r2_scores,
            "R_squared_OSXS": R_squared_OSXS_s
        }).to_csv(f"{file_name}.csv")

    # ----- Linear models ------
    # Lasso
    best_alpha = 0.1  # empirically result from validation
    reg = linear_model.Lasso(random_state=0, alpha=best_alpha)
    run_regression(reg, "results/Lasso_alpha0.1")
    # Ridge
    best_alpha = 0.1  # empirically result from validation
    reg = linear_model.Ridge(random_state=0, alpha=best_alpha)
    run_regression(reg, "results/Ridge_alpha0.1")
    # Elastic
    elastic_reg = ElasticNet(alpha=0.1)
    run_regression(reg, "results/Elastic_alpha0.1")

    # ----- Nonlinear models ----- 
    # GBR
    reg = GradientBoostingRegressor(
        n_estimators=100, random_state=0,
        loss='huber',
        verbose=1
    )
    run_regression(reg, "results/GBR_n100")
    # RF
    reg = RandomForestRegressor(
        n_estimators=100,
        max_depth=3, random_state=0, 
        verbose=1)
    run_regression(reg, "results/RF_n100")
