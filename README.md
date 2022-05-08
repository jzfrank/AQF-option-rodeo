# AQF-option-rodeo

This repo is used for AQF seminar. We plan to replicate the paper [Bali](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3895984). The goal is to predict option returns using machine learning approches. We have considered Lasso, Ridge, RF, and GBR. 

The current result is not very inspiring. This is probably due to time constraints that we did not tune hyperparameters properly, not do feature selection carefully, and imputing missing values naively. 

## Project Structure

Our project has the following structure: 

(Some folders may be empty in the repository. You should create them manually after cloning the project.)

- **data**: stores the data file used or generated. It includes option return data, stock data, characteristics data, and some linking files. 

- **reference_paper**: stores the paper we used as reference. 

- **models_all_characteristics**: stores models trained using all features

- **models_nonsparse**: stores models trained using nonsparse features 

- **analysis_results**: After performing analysis (r squared, backtest, CW test, DM test), the results will be stored in here. 

- **analysis_plots**: Based on analysis_results, plots will be made to give a better visualization of results. 

- ***Code files***: 

  - **consts.py**: stores environmental variables and used_characteristics 
  - **all_chars_generator.py**: prepares all characteristics (including defining new features, gluing csv files together) in one go. 
  - **ml_predictor.py**: files to train models. The user could change its parameters like [model_names, used_characteristics, start_year, end_year] to train different models using different characteristics. The trained models will be saved into files using joblib.dump and could be loaded later using joblib.load. 
  - **ml_helper_functions.py**: provides useful functions in the machine learning pipeline, like spliting train-validation-test data, backtesting. 
  - **evaluation_metrics.py**: defines evaluation metrics like CW test, DM test. 
  - **model_analysis** 
    - **model_analysis_r2score.py**: calculates r2score based on trained models 
    - **model_analysis_backtesting.py**: running backtesting 
    - **model_analysis_CW_test.py**: running CW test, which measures if the model is doing better than guessing zero 
    - **model_analysis_DM_test.py**: running DM test, which compares across models 
  - **Plotter**
    - **model_analysis_plots.py** 
    - **r2_score_plotter.py**
    - **DM_score_reporter.py**

  The project also includes some ipynb files, they are mainly used for debugging and testing. They are not very well maintained. 




## Data

As for option data, we thank Patrick and Nikola for saving us from the laborous work. 

As for equity characteristics, we mainly use data from [here](https://github.com/OpenSourceAP/CrossSection). 

We also implemented some characteristics in the paper ourselves. See all_chars_generator.py

## Reference paper
[Bali](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3895984)
