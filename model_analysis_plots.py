import os 
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
from pathlib import Path


def plot_monthly_backtest(model_name, results_folder, plots_folder):
    # read csv 
    years = list(range(1996, 2012 + 1))
    backtest = pd.DataFrame()
    for year in years:
        backtest = pd.concat(
            [backtest, pd.read_csv(Path(results_folder, f"backtest_{model_name}_{year}.csv"))]
        )
    backtest.dates = backtest.dates.apply(lambda x: str(x)[:7])
    # plot and save
    fig = plt.figure()
    plt.plot(backtest.dates, backtest.gain_from_hedges)
    plt.xticks(np.arange(0, len(backtest.dates), 12))
    plt.hlines(0, backtest.dates.iloc[0], backtest.dates.iloc[-1], color="black", linestyle="dashed")
    plt.xlabel("dates")
    plt.ylabel("monthly return")
    plt.title(model_name + " monthly backtest")
    plt.show()
    fig.savefig(Path(plots_folder, f"backtest_{model_name}.png"))


def plot_yearly_backtest(model_name, results_folder, plots_folder):
    # read csv 
    years = list(range(1996, 2012 + 1))
    backtest = pd.DataFrame()
    for year in years:
        backtest = pd.concat(
            [backtest, pd.read_csv(Path(results_folder, f"backtest_{model_name}_{year}.csv"))]
        )
    backtest['year'] = backtest.dates.apply(lambda x: str(x)[:4])
    yearly_returns = []
    for year in years:
        yearly_return = backtest[backtest.year == str(year + 7)].gain_from_hedges.mean()
        yearly_returns.append(yearly_return)
        print(yearly_return)
    backtest_yearly = pd.DataFrame({
        "year": years,
        "yearly_return": yearly_returns
    })
    # plot and save
    fig = plt.figure()
    plt.plot(backtest_yearly.year, backtest_yearly.yearly_return)
    plt.hlines(0, years[0], years[-1], color="black", linestyle="dashed")
    plt.xlabel("year")
    plt.ylabel("yearly return")
    plt.title(model_name + " yearly backtest")
    plt.show()
    fig.savefig(Path(plots_folder, f"backtest_{model_name}_year.png"))


def plot_CW_test(model_name, results_folder, plots_folder):
    CW_score = pd.read_csv(Path(results_folder, f"CW_score_{model_name}_year.csv"))
    # plot and save 
    fig = plt.figure()
    plt.plot(CW_score.year, CW_score.CW_score)
    plt.hlines(0, CW_score.year.iloc[0], CW_score.year.iloc[-1],  
               color="black", linestyle="dashed")
    plt.xlabel("year")
    plt.ylabel("CW_score")
    plt.title(model_name + " CW_score")
    plt.show()
    fig.savefig(Path(plots_folder, f"CW_score_{model_name}.png"))


if __name__ == "__main__":
    results_folder = "analysis_results/all_features"
    plots_folder = "analysis_plots"
    model_names = ["Lasso_alpha0.1", "Ridge_alpha0.1", "ElasticNet_alpha0.1", 
                   "GBR_n100", "RF_n100"]
    for model_name in model_names:
        plot_yearly_backtest(model_name, results_folder, plots_folder)
