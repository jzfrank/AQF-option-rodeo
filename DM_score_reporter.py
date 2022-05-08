from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd 


def DM_score_plot_for_all_features():
    # ------- for all features --------- 
    model_names = ["Lasso_alpha0.1", "Ridge_alpha0.1", 
                   "GBR_n100", "RF_n100"]
    DM_score_table = np.zeros((len(model_names), len(model_names)))

    start_year, end_year = 1996, 2013
    year2DM_scores = {year: {} for year in range(start_year, end_year + 1)}
    with open(Path("analysis_results/all_features", "DM_test_all.txt"), "r") as fh:
        results = fh.readlines()
        for year in range(start_year, end_year + 1):
            for result in results:
                if f"in year {year}" not in result:
                    continue
                result = result.strip()
                model1 = result.split("vs")[0].split(",")[-1].strip()
                model2 = result.split("vs")[1].split("in")[0].strip()
                DM_score = float(result.split("vs")[1].split()[-1].strip())
                # models2DM_score = {(model1, model2): DM_score}
                # print(models2DM_score)
                key = (model1, model2)
                val = DM_score
                year2DM_scores[year][key] = val 
    # print(year2DM_scores)

    for i, model_name1 in enumerate(model_names):
        for j, model_name2 in enumerate(model_names[i + 1:]):
            for year in range(start_year, end_year + 1):
                DM_score_table[i][i + 1 + j] += year2DM_scores[year][
                    (model_name1, model_name2)
                ] / (end_year - start_year + 1)

    print('''\\begin{table}[]
    \\centering
    \\begin{tabular}{|l|l|l|l|l|}
    \\hline
     & Lasso & Ridge & GBR & RF \\\\ \\hline''')
    for i in range(4):
        print(model_names[i].split("_")[0], end=" & ")
        for j in range(4):
            if j == 3:
                end = ""
            else:
                end = " & "
            print(round(DM_score_table[i][j], 4) if DM_score_table[i][j] != 0 else 0, 
                  end=end)
        print(" \\\\ \\hline")
    print('''\\end{tabular}
        \\caption{DM test across models}
        \\label{tab:DM_test}
    \\end{table}''')
    print(DM_score_table)
    print(model_names)


def DM_score_plot_for_nonsparse_features():
    model_names = ["Lasso_alpha0.1", "Ridge_alpha0.1", 
                   "GBR_n100", "RF_n100"]
    DM_score_table = np.zeros((len(model_names), len(model_names)))
    index2model = {model: i for i, model in enumerate(model_names)}
    DM_score = pd.read_csv(Path("analysis_results/nonsparse_features", 
                                "DM_test_all_models.csv"))
    models2DM = {}
    for col in DM_score.columns[2:]:
        print(col, DM_score[col].mean())
        model1 = col.split("vs")[0].strip()
        model2 = col.split("vs")[1].strip()
        models2DM[(model1, model2)] = DM_score[col].mean()
        i = index2model[model1]
        j = index2model[model2]
        DM_score_table[i][j] = DM_score[col].mean()

    print(DM_score_table)

    input()
    print('''\\begin{table}[]
    \\centering
    \\begin{tabular}{|l|l|l|l|l|}
    \\hline
     & Lasso & Ridge & GBR & RF \\\\ \\hline''')
    for i in range(4):
        print(model_names[i].split("_")[0], end=" & ")
        for j in range(4):
            if j == 3:
                end = ""
            else:
                end = " & "
            print(round(DM_score_table[i][j], 4) if DM_score_table[i][j] != 0 else 0, 
                  end=end)
        print(" \\\\ \\hline")
    print('''\\end{tabular}
        \\caption{DM test across models}
        \\label{tab:DM_test}
    \\end{table}''')
    print(DM_score_table)
    print(model_names)


if __name__ == "__main__":
    DM_score_plot_for_nonsparse_features()
