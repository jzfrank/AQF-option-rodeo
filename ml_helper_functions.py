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
    print("option_with_feature.date.between(begin_date, end_date).sum():", 
          option_with_feature.date.between(begin_date, end_date).sum())
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

    Algorithm:

        1. Assume we have:
         model, all_characteristics (test_data)
        2. Then we can get:
         [optionid, date], option_ret_pred, option_ret, open_interest
        3. Then sort by option_ret_pred.
          Take head 10% and tail 10% rows, label them as 
        long_portfolio and short_portfolio
          How to construct long-short portfolio:
           Suppose we long 1 dollar, short 1 dollar (the amount does not matter, 
        since we only care about the return).
        for long_portfolio, each option is weighted by its open interest, 
        buy wi of option-delta-hedge-i, get option_ret_i as return
        for short_portfolio, each option is weighted by its open interest, 
        sell wi of option-delta-hedge-i, get negative option_ret_i as return
        The ultimate return is given by:
            (return from long + return from short) / (1+1)
            = 1/2 (sum_{i in {long_porfolio}}w_i r_i 
            - sum_{j in {short_porfolio}}w_j r_j)
    '''
    dates = sorted(list(set(test_data.date)))
    gain_from_hedges = []
    for i in range(len(dates)):
        df = test_data[test_data.date == dates[i]]
        df = pd.DataFrame(
            {
                "secid": df.secid,
                # "adj_spot": df.adj_spot,
                # "strike_price": df.strike_price,
                # "delta": df.delta,
                # "mid_price": df.mid_price,
                "option_ret_real": df.option_ret,
                "option_ret_pred": regressor.predict(df[used_characteristics]), 
                "open_interest": df.open_interest
            }
        )
        # df["hedge_cost"] = df["mid_price"] - df["adj_spot"] * df["delta"]
        # df["hedge_gain"] = abs(df["hedge_cost"]) * df["option_ret_real"] 
        one_decile = int(0.1 * df.shape[0])
        long_portfolio = df.sort_values(by="option_ret_pred").head(one_decile)
        short_portfolio = df.sort_values(by="option_ret_pred").tail(one_decile)
        gain_from_hedge = 1 / 2 * (
            (long_portfolio.option_ret_real * long_portfolio.open_interest).sum() / long_portfolio.open_interest.sum()
            - (short_portfolio.option_ret_real * short_portfolio.open_interest).sum() / short_portfolio.open_interest.sum()
        )

        # short_portfolio.hedge_cost = - short_portfolio.hedge_cost
        # gain_from_hedge = (sum(long_portfolio["hedge_cost"]) + sum(short_portfolio["hedge_cost"]) 
        #                    + sum(long_portfolio["hedge_gain"]) + sum(short_portfolio["hedge_gain"]))
        gain_from_hedges.append(gain_from_hedge)
    return dates, gain_from_hedges
