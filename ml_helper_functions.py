import numpy as np
import datetime as dt
import pandas as pd 
import os
import numpy as np
import time 
import joblib
import datetime 
from pathlib import Path


def get_data_between(option_with_feature, start_year, end_year):
    '''
    return option_with_feature between {start_year}-01-01 and {end_year}-12-31
    (inclusive: [start_hear, end_year])
    '''
    begin_date = dt.datetime.strptime(f"{start_year}-01-01", "%Y-%m-%d")
    end_date = dt.datetime.strptime(f"{end_year}-12-31", "%Y-%m-%d")
    print("option_with_feature.date.between(begin_date, end_date).shape:", 
          option_with_feature.date.between(begin_date, end_date).shape)
    return option_with_feature[
        option_with_feature.date.between(begin_date, end_date)
    ].copy(deep=True)


def train_validation_test_split(option_with_feature, year):
    '''
    split option_with_feature into train, validation, test dataset.
    year is the starting year. 
    training: [year, year + 4]
           or [year, year + 6]
    validation: [year + 5. year + 6]
    test: [year + 7, year + 7]
    '''
    training_data = get_data_between(option_with_feature, year, year + 6)
    # since we are not using validation for resources constraint, need to extend trainig data period
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
