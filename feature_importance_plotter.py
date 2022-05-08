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
import matplotlib.pyplot as plt

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


model_root = "models_all_characteristics"
model_name = "RF_n100"

start_year, end_year = 1996, 2013
for year in range(start_year, end_year + 1):
    model = joblib.load(Path(model_root, f"{model_name}_{year}.pkl"))

    with open(Path(model_root, f"{model_name}_{year}.txt"), "r") as fh:
        used_characteristics = list(
            map(lambda x: x[1:-1], 
                fh.readline()[1:-1].split(", "))
        )

    assert len(used_characteristics) == len(model.feature_importances_)
    # plot feature importance 
    topK = 20
    feature_importance = model.feature_importances_[:topK]
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    fig = plt.figure(figsize=(12, 6))
    plt.barh(pos, feature_importance[sorted_idx], align="center")
    plt.yticks(pos, np.array(used_characteristics)[sorted_idx])

    plt.title(f"Feature Importance {model_name} {year}")
    plt.show()
    fig.savefig(Path("analysis_plots/all_features", f"FI_{model_name}_{year}.png"))
