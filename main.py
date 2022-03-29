import pandas as pd 

sp500_op_ret = pd.read_csv("sp500_op_ret.csv")
print(sp500_op_ret.columns())
print(sp500_op_ret.head())