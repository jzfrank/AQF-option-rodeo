import os 
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
from pathlib import Path


def plot_monthly_backtest(model_name, results_folder, plots_folder):
    # read csv 
    years = list(range(1996, 2013 + 1))
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
    years = list(range(1996, 2013 + 1))
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


def plot_CW_test(model_names, results_folder, plots_folder):
    model2CW_scores = {model_name: [] for model_name in model_names}
    for year in range(1996, 2013 + 1):
        with open(Path(results_folder, 
                       f"CW_scores_{year}.csv"), "r") as fh:
            CW_score = fh.readline()
            CW_score = eval(CW_score)
        for model_name in model_names:
            model2CW_scores[model_name].append(
                CW_score[model_name])

    print(model2CW_scores)

    D = [val for val in model2CW_scores.values()]

    fig, ax = plt.subplots(figsize=(10, 5))
    VP = ax.boxplot(D, positions=[2, 4, 6, 8], widths=1.5, patch_artist=True,
                    showmeans=False, showfliers=False,
                    labels=model_names,
                    medianprops={"color": "white", "linewidth": 0.5},
                    boxprops={"facecolor": "C0", "edgecolor": "white",
                              "linewidth": 0.5},
                    whiskerprops={"color": "C0", "linewidth": 1.5},
                    capprops={"color": "C0", "linewidth": 1.5})

    ylim_up = 0.03
    ylim_down = 0.1
    ax.set(xlim=(1, 10), xticks=np.arange(2, 10, 2),
           ylim=(-ylim_down, ylim_up), yticks=np.arange(-ylim_down, ylim_up, 0.05))

    plt.hlines(0, -1, 10, linestyle="dashed", color="black")
    plt.show()
    fig.savefig(Path(f"analysis_plots/{choice_of_feature}", "CW_test_boxplot.png"))

    # plot and save 
    # fig, ax = plt.subplots()

    # plt.plot(CW_score.year, CW_score.CW_score)
    # plt.hlines(0, CW_score.year.iloc[0], CW_score.year.iloc[-1],  
    #            color="black", linestyle="dashed")
    # plt.xlabel("year")
    # plt.ylabel("CW_score")
    # plt.title(model_name + " CW_score")
    # plt.show()
    # fig.savefig(Path(plots_folder, f"CW_score_{model_name}.png"))


if __name__ == "__main__":
    choice_of_feature = "nonsparse_features"
    results_folder = f"analysis_results/{choice_of_feature}"
    plots_folder = f"analysis_plots/{choice_of_feature}"
    model_names = ["Lasso_alpha0.1", "Ridge_alpha0.1", 
                   "GBR_n100", "RF_n100"]

    # for model_name in model_names:
    #     plot_yearly_backtest(model_name, results_folder, plots_folder)
    plot_CW_test(model_names, results_folder, plots_folder)
