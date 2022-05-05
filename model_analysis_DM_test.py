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
    print(f"finished loading data, used {time.time() - start} seconds")
    print("------------------------------------------------------")

    # load model 
    model_root = "./models_all_characteristics"
    model_names = ["Lasso_alpha0.1", "Ridge_alpha0.1", "ElasticNet_alpha0.1", "GBR_n100", "RF_n100"]

    def load_model_and_characteristics(model_name):
        model = joblib.load(Path(model_root, f"{model_name}_{year}.pkl"))
        print(model.get_params())
        with open(Path(model_root, f"{model_name}_{year}.txt"), "r") as fh:
            used_characteristics = list(
                map(lambda x: x[1:-1], 
                    fh.readline()[1:-1].split(", "))
            )
        return model, used_characteristics

    results = []
    for year in range(1996, 2012 + 1):
        training_data, validation_data, test_data = train_validation_test_split(option_with_feature, year)

        result = dict()
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1_name, model2_name = model_names[i], model_names[j]
                model1, used_characteristics1 = load_model_and_characteristics(model1_name)
                model2, used_characteristics2 = load_model_and_characteristics(model2_name)

                assert set(used_characteristics1) == set(used_characteristics2)
                used_characteristics = used_characteristics1

                imp = SimpleImputer(missing_values=np.nan, strategy='mean')
                imp.fit(training_data[used_characteristics])
                training_data.loc[:, used_characteristics] = imp.transform(training_data[used_characteristics])
                test_data.loc[:, used_characteristics] = imp.transform(test_data[used_characteristics])

                true_pred_1vs2_return = pd.DataFrame(
                    {
                        "optionid": test_data.optionid,
                        "time": test_data.date,
                        "true_return": test_data.option_ret,
                        "pred_return1": model1.predict(test_data[used_characteristics]),
                        "pred_return2": model2.predict(test_data[used_characteristics])
                    }
                )

                DM_score = DM_test(true_pred_1vs2_return)
                key = f"{model1_name} vs {model2_name}"
                result[key] = DM_score
                print(f"DM_score, {model1_name} vs {model2_name} in year {year}", DM_score)
                print(f"finished running DM_test comparison script, total time used: {time.time() - start} seconds")
        results.append(result)
    # save to df 
    summary_df = pd.DataFrame({
        "year": list(map(lambda x: x + 7, range(1996, 2012 + 1)))
    })
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            model1_name, model2_name = model_names[i], model_names[j]
            key = f"{model1_name} vs {model2_name}"
            summary_df[key] = list(
                result[key] for result in results
            )
    print(summary_df)
    summary_df.to_csv(Path("analysis_results", "DM_test_all_models.csv"))
