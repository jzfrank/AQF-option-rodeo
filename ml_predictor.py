import numpy as np
import datetime as dt
import pandas as pd 
import os
from evaluation_metrics import CW_test, DM_test, R_squared_OSXS
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    # load data 
    DATAROOT = "/cluster/scratch/zhajin/data/"  # for running in euler server 
    option_with_feature = pd.read_csv(os.path.join(DATAROOT, "option_with_nonsparse_features.csv"))
    option_with_feature = option_with_feature[~option_with_feature.option_ret.isna()]
    option_with_feature["date_x"] = option_with_feature.date_x.apply(
        lambda x: dt.datetime.strptime(x.split()[0], "%Y-%m-%d")).copy()
    option_with_feature["cp_flag_encoded"] = option_with_feature["cp_flag"].apply(lambda x: {"P": 0, "C": 1}[x])
    print("finished loading data")
    # suppress the warning of 
    #   "A value is trying to be set on a copy of a slice from a DataFrame. 
    #   Try using .loc[row_indexer,col_indexer] = value instead"
    pd.options.mode.chained_assignment = None 

    mean_squared_errors = []
    r2_scores = []
    R_squared_OSXS_s = []

    for year in range(1996, 2020 - 7):
        print(f"iteration, year: {year}, running regression...")
        training_data, validation_data, test_data = train_validation_test_split(option_with_feature, year)

        used_characteristics = ['volume', 'ReturnSkew', 'MaxRet', 'delta', 'PriceDelaySlope', 'strike_price', 
                                'IdioVol3F', 'ReturnSkew3F', 'ir_rate', 'mid_price', 'forwardprice', 
                                'zerotradeAlt1', 'theta', 'cfadj', 'zerotrade', 'best_bid', 'spotprice', 'VolMkt', 
                                'IdioRisk', 'days_to_exp', 'PriceDelayTstat', 'High52', 'Coskewness', 'BidAskSpread', 
                                'Beta', 'days_no_trading', 'open_interest', 'impl_volatility', 'PriceDelayRsq', 'IdioVolAHT', 
                                'adj_spot', 'vega', 'gamma', 'best_offer', 'DolVol', 'cp_flag_encoded']

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
        best_alpha = 0.1  # empirically result from validation
        reg = linear_model.Lasso(random_state=0, alpha=best_alpha)
        reg.fit(X_train[used_characteristics], y_train)
        y_pred = reg.predict(X_test[used_characteristics])
        mean_squared_errors.append(mean_squared_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))
        R_squared_OSXS_s.append(R_squared_OSXS(y_test, y_pred))
        print(year, mean_squared_errors[-1], r2_scores[-1], R_squared_OSXS_s[-1])

        pd.DataFrame({
            "start_year": list(range(1996, 2020 - 7)), 
            "mean_squared_error": mean_squared_errors,
            "r2_score": r2_scores,
            "R_squared_OSXS": R_squared_OSXS_s
        }).to_csv("Lasso_alpha0.1.csv")
