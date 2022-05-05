# %% [markdown]
# %%

# %%
from consts import DATAROOT
import pandas as pd
import numpy as np
from pathlib import Path
import datetime

# %% [markdown]
# %%<br>
# Reading data

# %% [markdown]
# %%

# %%
daily_stock = pd.read_csv(Path(DATAROOT, "stock_daily.csv"))

# %% [markdown]
# %%

# %%
monthly_stock_wide = pd.read_csv(Path(DATAROOT, 'signed_predictors_dl_wide.csv'))
monthly_stock_wide.head()

# %% [markdown]
# %%

# %%
daily_stock.columns = (col.lower() for col in daily_stock.columns)
daily_stock['date'] = pd.to_datetime(daily_stock.date, format="%Y%m%d")
daily_stock.head()

# %% [markdown]
# %%

# %%
reference = pd.read_csv(Path(DATAROOT, "CRSP_daily_stock_reference"))
reference['Variable Name'] = reference['Variable Name'].str.slice(0, -1)
reference = reference[reference['Variable Name'].isin(daily_stock.columns)]
reference['description_url'] = "https://wrds-www.wharton.upenn.edu/data-dictionary/form_metadata/crsp_a_stock_dsf_identifyinginformation/" + reference['Variable Name']
reference.head(2)

# %% [markdown]
# %%

# %%
sp500_op_ret = pd.read_csv(Path(DATAROOT, "sp500_op_ret.csv"))
sp500_op_ret['date'] = pd.to_datetime(sp500_op_ret['date'])
sp500_op_ret['exdate'] = pd.to_datetime(sp500_op_ret['exdate'])
sp500_op_ret.head()

# %% [markdown]
# %%

# %%
mapping_table = pd.read_csv(Path(DATAROOT, "mapping_table.csv"))
mapping_table.head(2)
mapping_table['sdate'] = pd.to_datetime(mapping_table.sdate)
mapping_table['edate'] = pd.to_datetime(mapping_table.edate)

# %% [markdown]
# %% [markdown]<br>
# # Add permno to monthly data

# %% [markdown]
# %%

# %%
(mapping_table.groupby('secid').count()[mapping_table.groupby('secid').permno.count() != 1])

# %% [markdown]
# %%

# %%
(mapping_table.groupby('permno').count()[mapping_table.groupby('permno').secid.count() != 1])

# %% [markdown]
# %%

# %%
mapping_table[mapping_table.permno == 10113]

# %% [markdown]
# %%

# %%
mapping_table[mapping_table.secid == 5007]

# %% [markdown]
# %%

# %% [markdown]
# %%

# %%
def dates_overlaps(group):
    if group.shape[0] == 1:
        return False
    else:
        group.sort_values('sdate', inplace=True)
        return ((group.edate - group.edate.shift(1)) < datetime.timedelta(0)).any()

# %%
mapping_table.groupby('secid').apply(dates_overlaps)

# %% [markdown]
# %%

# %%
mapping_table.groupby('secid').apply(dates_overlaps)[mapping_table.groupby('secid').apply(dates_overlaps)]

# %% [markdown]
# %%

# %%
mapping_table[mapping_table.secid == 5505]

# %% [markdown]
# %% [markdown]<br>
# ## Summary on mapping table: if we eliminate the secids 5505 and 9534, each (secid, sdate, edate) triplet gives uniquely a permno, so we shall use this to link the data

# %% [markdown]
# %%

# %%
sp500_op_ret = sp500_op_ret[~sp500_op_ret.secid.isin([5505, 9534])]
sp500_op_ret = sp500_op_ret[sp500_op_ret.date <= mapping_table.edate.max()]

# %% [markdown]
# %%

# %%
def add_permno(group):
    secid = group.secid.iloc[0]
    mapping_group = MAPPING_GROUPED.get_group(secid)
    def find_permno(row):
        try:
            return mapping_group[(mapping_group.sdate <= row.date) & (mapping_group.edate >= row.date)].permno.iloc[0]
        except:
            print('occurs') 
            # sometimes it occurs that the date falls between one edate and one sdate in the mapping table. 
            # This event is rare (happens twice in the initial sample), so will just return nan and drop na later.
            return np.nan
    if mapping_group.shape[0] == 1:
        group['permno'] = mapping_group.permno.iloc[0]
    else:
        group['permno'] = group.apply(find_permno, axis=1)
    return group

# %%
MAPPING_GROUPED = mapping_table.groupby('secid')
grouped = sp500_op_ret.groupby('secid')
sp500_op_ret_w_permno = grouped.apply(add_permno).dropna(subset=['permno'])
sp500_op_ret_w_permno

# %% [markdown]
# %%

# %%
sp500_op_ret_w_permno[sp500_op_ret_w_permno.option_ret.isna()]

# %% [markdown]
# %% [markdown]<br>
# # Grouping daily data with month

# %% [markdown]
# %% [markdown]<br>
# First, we check whether all monthly data are recorded on the last trading date of the natural month.

# %% [markdown]
# %%

# %%
dates = pd.Series(daily_stock.date.dt.strftime('%Y-%m-%d').unique())
end_of_trading_month = dates.groupby(dates.str.slice(0, 7)).max().values

# %% [markdown]
# %% [markdown]<br>
# There is one wierd date.

# %% [markdown]
# %%

# %%
sp500_op_ret_w_permno[~sp500_op_ret_w_permno.date.dt.strftime('%Y-%m-%d').isin(end_of_trading_month)].date.unique()

# %% [markdown]
# %%

# %%
daily_stock[daily_stock.date.dt.strftime('%Y-%m-%d') == '2017-09-30']

# %% [markdown]
# %% [markdown]<br>
# However, we checked that 2017-09-30 is Saturday, and since there is only one record in this regard, we simply drop this record.

# %% [markdown]
# %%

# %%
daily_stock = daily_stock[~(daily_stock.date.dt.strftime('%Y-%m-%d') == '2017-09-30')]

# %% [markdown]
# %% [markdown]<br>
# Now everything's cool.

# %% [markdown]
# %%

# %%
dates = pd.Series(daily_stock.date.dt.strftime('%Y-%m-%d').unique())
end_of_trading_month = dates.groupby(dates.str.slice(0, 7)).max().values
sp500_op_ret_w_permno.date.dt.strftime('%Y-%m-%d').isin(end_of_trading_month).all()

# %% [markdown]
# %%

# %%
end_of_mon_dict = {date[0:7]: date for date in end_of_trading_month}

# %% [markdown]
# %%

# %%
def last_date(group):
    return pd.Series([group.date.max()] * group.shape[0], index=group.index)

# %%
daily_stock['month'] = daily_stock.groupby(daily_stock.date.dt.strftime('%Y-%m')).apply(last_date).droplevel(0)
daily_stock.sort_values('date', inplace=True)

# %% [markdown]
# %%

# %%
daily_stock['ind'] = daily_stock[~(daily_stock.siccd=='Z')].siccd.astype(str).str.slice(0,2).astype(int)


# %%
ind_info = daily_stock[['permno','month','ind']]
eom_ind = ind_info.groupby(['permno', 'month']).apply(lambda group: group.ind.iloc[-1]).dropna().astype(int).astype(str)

# %%
ind_dummies = pd.get_dummies(eom_ind)
ind_dummies.columns = [f'ind_{col}' for col in ind_dummies.columns]
ind_dummies = ind_dummies.reset_index(drop=False)
ind_dummies.columns = [(col if col!='month' else 'date') for col in ind_dummies.columns]
ind_dummies

# %% [markdown]
# %% [markdown]<br>
# Now, when calculating the any monthly data, such as realized volatility, from daily data, we can simply do daily_stock.groupby(['permno', 'month']).apply(aggregation_func)

# %% [markdown]
# %% [markdown]<br>
# # Aggregating calculations

# %% [markdown]
# %% [markdown]<br>
# #### List of attributes to be calculated from the daily data<br>
#     1.Realized volatility of stock price<br>
#     2.Volume of stocks traded<br>
#     3.Dollar volume of stocks traded

# %% [markdown]
# %%

# %%
reference

# %% [markdown]
# %%

# %%
attributes = []

# %% [markdown]
# %%

# %%
daily_stock['dollar_volu'] = daily_stock['vol'] * daily_stock['prc']

# %% [markdown]
# %%

# %%
grouped = daily_stock.groupby(['permno', 'month'])

# %% [markdown]
# %%

# %%
rvol = (grouped['prc'].std() * np.sqrt(12)).rename('rvol')
attributes += [rvol]

# %% [markdown]
# %%

# %%
share_volume = grouped['vol'].sum().rename('share_volume')
attributes += [share_volume]

# %% [markdown]
# %%

# %%
dollar_volume = grouped['dollar_volu'].sum().rename('dollar_volume')
attributes += [dollar_volume]

# %% [markdown]
# %%

# %%
attributes_from_daily_stock = pd.DataFrame(attributes).T.reset_index()
attributes_from_daily_stock

# %% [markdown]
# %%

# %%
sp500_op_ret_w_permno = pd.merge(sp500_op_ret_w_permno, attributes_from_daily_stock, left_on=['permno', 'date'], right_on=['permno', 'month'], how='inner')

# %% [markdown]
# %% [markdown]<br>
# # Revisit Daily option data<br>
# From most of the characteristics in Bucket level, we need some volatility data on the options, so we dived back into the daily option data to hopefully make it possible to calculate more things.<br>
# <br>
# But failed...

# %% [markdown]
# %%<br>
# daily_option = pd.read_csv(Path(DATAROOT, 'daily_option.csv'))<br>
# # reading this takes \\ sadly forever... 

# %% [markdown]
# %%

# %% [markdown]
# %%

# %% [markdown]
# %%

# %% [markdown]
# %% [markdown]<br>
# # Calculating Characteristics

# %% [markdown]
# %% [markdown]<br>
# ## Preliminary Work<br>
# We first add some preliminary columns to the dataframe<br>
# <br>
# Most of the works here are contributed by Natasha

# %% [markdown]
# %%

# %%
sp500_op_ret_w_permno["yrs_to_exp"] = sp500_op_ret_w_permno["days_to_exp"] / 250
sp500_op_ret_w_permno["moneyness"] = sp500_op_ret_w_permno['strike_price'] / sp500_op_ret_w_permno['adj_spot']

# %% [markdown]
# # %%<br>
# months = sp500_op_ret_w_permno['date'].sort_values(ascending=True).unique()<br>
# months<br>
# int_months = [int(month) for month in months]# np.searchsorted works wierdly on dt objects <br>
# int_months

# %% [markdown]
# # %%<br>
# # shift option return<br>
# global I <br>
# I = 0<br>
# global L <br>
# L = len(sp500_op_ret_w_permno.optionid.unique())<br>
# # grouped = sp500_op_ret_w_permno.sort_values('date')[['optionid', 'date', 'option_ret']].groupby('optionid')<br>
# grouped = sp500_op_ret_w_permno.sort_values('date').groupby('optionid')<br>
# def get_all_months(group):<br>
#     s_month, e_month = np.datetime64(group.date.iloc[0]), np.datetime64(group.date.iloc[-1])<br>
#     return months[(months >= s_month) & (months <= e_month)]<br>
#     # return months[np.searchsorted(months, s_month): np.searchsorted(months, e_month)+1]<br>
# def get_prev_option_ret(group):<br>
#     signal_df = pd.DataFrame(get_all_months(group), columns=['date'])<br>
#     filled_group = pd.merge(group, signal_df, on='date', how='outer')<br>
#     filled_group['prev_option_ret'] = filled_group.option_ret.shift(1)<br>
#     global I<br>
#     I+=1<br>
#     print(I/L, end="\r")<br>
#     return filled_group.dropna(subset=['optionid'])

# %% [markdown]
# # %%<br>
# group = sp500_op_ret_w_permno.sort_values('date').groupby('optionid').get_group(10009003)

# %% [markdown]
# # %%<br>
# sp500_op_ret_w_permno = grouped.apply(get_prev_option_ret).reset_index(drop=False)

# %% [markdown]
# %% [markdown]<br>
# ## Stock level

# %% [markdown]
# %% [markdown]<br>
# Start to calculate the option characteristics

# %% [markdown]
# %%

# %%
sp500_op_ret_w_permno.columns

# %% [markdown]
# %%

# %%
stock_level_grouped = sp500_op_ret_w_permno.groupby(['secid', 'date'])

# %% [markdown]
# %%<br>
# fucntions calculating characteristics in part 1

# %% [markdown]
# toi

# %%
def toi(opt_data):
    sp500_op_ret = opt_data
    toi = sp500_op_ret.groupby(["secid", "date"])["open_interest"].sum()
    return toi
# toitest=toi(sp500_op_ret)

# %% [markdown]
# pcrratio

# %%
def pcratio(opt_data):
    sp500_op_ret = opt_data
    put_vol_permnth_underlying = sp500_op_ret.drop(sp500_op_ret[sp500_op_ret.cp_flag == 'C'].index)
    put_vol_permnth_underlying = put_vol_permnth_underlying.groupby(['secid', 'date'])["open_interest"].sum()
    put_vol_permnth_underlying.rename("oi_put", inplace=True)
    total_vol_permnth_underlying = sp500_op_ret.groupby(['secid', 'date'])["open_interest"].sum()
    total_vol_permnth_underlying.rename("oi_total", inplace=True)
    join = pd.concat([total_vol_permnth_underlying, put_vol_permnth_underlying], axis=1)
    join = join.fillna(0)
    ratio = join["oi_put"] / join["oi_total"]
    return ratio
# pcratiotest=pcratio(sp500_op_ret)

# %%
def vol(opt_data):
    sp500_op_ret = opt_data
    vol = sp500_op_ret.groupby(['secid', 'date'])["volume"].sum()
    return vol
# voltest=trading_vol(sp500_op_ret)

# %% [markdown]
# nopt

# %%
def nopt(opt_data):
    sp500_op_ret = opt_data
    nopt = sp500_op_ret.groupby(['secid', 'date'])["volume"].mean()
    return nopt
# nopttest=n_opt(sp500_op_ret)

# %% [markdown]
# dvol group by option id

# %%
def dvol(opt_data):
    sp500_op_ret = opt_data
    dvol = sp500_op_ret["volume"] * sp500_op_ret["mid_price"]
    sp500_op_ret["dvol_temp"] = dvol
    dvol = sp500_op_ret.groupby(['secid', 'date'])["dvol_temp"].sum()
    return dvol
# dvoltest=d_vol(sp500_op_ret)

# %% [markdown]
# ailliq

# %%
def ailliq(opt_data):
    sp500_op_ret = opt_data
    ailliq_a = abs(sp500_op_ret['prev_option_ret']) / sp500_op_ret['dvol_temp']
    sp500_op_ret["ailliq_a"] = ailliq_a                                      
    ailliq_a = sp500_op_ret.groupby(['secid', 'date'])["ailliq_a"].sum()                                          
    ailliq_b = sp500_op_ret.groupby(['secid', 'date'])["volume"].sum()
    ailliq = ailliq_a / ailliq_b
    return ailliq
# ailliqtest=a_illiq(sp500_op_ret)

# %% [markdown]
# pilliq

# %%
def pilliq(opt_data):
    sp500_op_ret = opt_data
    pilliq_a = abs(sp500_op_ret['prev_option_ret']) / sp500_op_ret['dvol_temp'] / sp500_op_ret['mid_price']
    sp500_op_ret["pilliq_a"] = pilliq_a                                      
    pilliq_a = sp500_op_ret.groupby(['secid', 'date'])["pilliq_a"].sum()                                          
    pilliq_b = sp500_op_ret.groupby(['secid', 'date'])["volume"].sum()
    pilliq = pilliq_a / pilliq_b
    return pilliq
# pilliqtest=p_illiq(sp500_op_ret)

# %% [markdown]
# ##

# %%
def pcpv(opt_data):
    def get_PC_parity_vio(group):
        group.sort_values(['exdate', 'cp_flag'], ascending=False, inplace=True)
        s_grouped = group.groupby(['strike_price', 'exdate'])
        candi_key = []
        for key, s_group in s_grouped:  # record keys of such s_group that has length 2
            if s_group.shape[0] == 2:
                candi_key += [key]
        if not candi_key:
            return np.nan
        elif len(candi_key) == 1:
            pair = s_grouped.get_group(candi_key[0])
        else:
            current_spot = group.spotprice.iloc[0]
            moneyness = [abs(key[0] - current_spot) for keys in candi_key]
            pair_idx = candi_key[moneyness.index(min(moneyness))]
            pair = s_grouped.get_group(pair_idx)
        cal = pair.strike_price.iloc[0] * np.exp(-pair.ir_rate.iloc[0] * pair.days_to_exp.iloc[0] / 250) + pair.mid_price.iloc[1] - pair.mid_price.iloc[0]
        L = 100 * np.log(pair.spotprice.iloc[0] / cal)
        return L
    pcpv = opt_data.groupby(['secid', 'date']).apply(get_PC_parity_vio)
    return pcpv

# %% [markdown]
# pcpv=sp500_op_ret.groupby(['secid', 'date']).apply(get_PC_parity_vio)

# %%
def shrtfee(opt_data):
    def shrt_fee(group):
        group.sort_values(['exdate', 'cp_flag'], ascending=False, inplace=True)
        s_grouped = group.groupby(['strike_price', 'exdate'])
        candi_key = []
        for key, s_group in s_grouped:  # record keys of such s_group that has length 2
            if s_group.shape[0] == 2:
                candi_key += [key]
        if not candi_key:
            return np.nan
        elif len(candi_key) == 1:
            pair = s_grouped.get_group(candi_key[0])
        else:
            current_spot = group.spotprice.iloc[0]
            moneyness = [abs(key[0] - current_spot) for keys in candi_key]
            pair_idx = candi_key[moneyness.index(min(moneyness))]
            pair = s_grouped.get_group(pair_idx)
        cal = pair.adj_spot.iloc[0] - pair.strike_price.iloc[0] * np.exp(-pair.ir_rate.iloc[0] * pair.days_to_exp.iloc[0] / 250) + pair.mid_price.iloc[0] - pair.mid_price.iloc[1]
        L = (1 - cal / pair.adj_spot.iloc[0])**(1 / pair.strike_price.iloc[0])
        M = (1 - L) / (1 + pair.ir_rate.iloc[0])
        return M
    shrt_fee = opt_data.groupby(['secid', 'date']).apply(shrt_fee)
    return shrt_fee

# %% [markdown]
# iv_skew

# %%
def skewiv(opt_data):
    sp500_op_ret = opt_data
    put_vals = sp500_op_ret.drop(sp500_op_ret[sp500_op_ret.cp_flag == 'C'].index)
    call_vals = sp500_op_ret.drop(sp500_op_ret[sp500_op_ret.cp_flag == 'P'].index)
    temp_callvals = call_vals.groupby(['secid', 'date'])[["impl_volatility", "moneyness"]].mean()
    temp_putvals = put_vals.groupby(['secid', 'date'])[["impl_volatility", "moneyness"]].mean()
    otmput = temp_putvals[((temp_putvals["moneyness"] < 0.9))]
    atmcall = temp_callvals[
        (temp_callvals["moneyness"] >= 0.9) & (temp_callvals["moneyness"] <= 1.1)]
    atmcall = atmcall["impl_volatility"]
    otmput = otmput["impl_volatility"]
    atmcall.rename("implvolatmcall", inplace=True)
    otmput.rename("implvolotmput", inplace=True)
    skewiv_temp = pd.concat([otmput, atmcall], axis=1)
    skewiv = skewiv_temp["implvolotmput"] - skewiv_temp["implvolatmcall"]
    return skewiv
# skewivtest=iv_skew(sp500_op_ret)

# %% [markdown]
# atm_civpiv

# %%
def atm_civpiv(opt_data):
    sp500_op_ret = opt_data
    atm_options = sp500_op_ret[(sp500_op_ret["moneyness"] >= 0.9) & (sp500_op_ret["moneyness"] <= 1.1)]
    atm_puts = atm_options.drop(atm_options[atm_options.cp_flag == 'C'].index)
    atm_calls = atm_options.drop(atm_options[atm_options.cp_flag == 'P'].index)
    atm_puts = atm_puts.groupby(['secid', 'date'])[["impl_volatility"]].mean()
    atm_puts = atm_puts.rename(columns={"impl_volatility": "implvolput"})
    atm_calls = atm_calls.groupby(['secid', 'date'])[["impl_volatility"]].mean()
    atm_calls = atm_calls.rename(columns={"impl_volatility": "implvolcall"})
    atm_civpiv = pd.concat([atm_calls, atm_puts], axis=1)
    atm_civpiv = atm_civpiv["implvolcall"] - atm_civpiv["implvolput"]
    return atm_civpiv
# atm_civpivtest= atm_civ_piv(sp500_op_ret)

# %% [markdown]
# atm_dcivpiv

# %%
def atm_dcivpiv(opt_data):
    sp500_op_ret = opt_data
    atm_options = sp500_op_ret[(sp500_op_ret["moneyness"] >= 0.9) & (sp500_op_ret["moneyness"] <= 1.1)]
    atm_puts_dpiv = atm_options.drop(atm_options[atm_options.cp_flag == 'C'].index)
    dpiv = atm_puts_dpiv.groupby(['secid', 'date'])[["impl_volatility"]].mean()
    dpiv['diffp'] = dpiv['impl_volatility'].diff()
    dpiv = dpiv['diffp']
    atm_options = sp500_op_ret[(sp500_op_ret["moneyness"] >= 0.9) & (sp500_op_ret["moneyness"] <= 1.1)]
    atm_calls_cpiv = atm_options.drop(atm_options[atm_options.cp_flag == 'P'].index)
    cpiv = atm_calls_cpiv.groupby(['secid', 'date'])[["impl_volatility"]].mean()
    cpiv['diffc'] = cpiv['impl_volatility'].diff()
    cpiv = cpiv['diffc']
    dcivpiv_temp = pd.concat([cpiv, dpiv], axis=1)
    atmdcivpiv = dcivpiv_temp["diffc"] - dcivpiv_temp["diffp"]
    return atmdcivpiv
# atm_dcivpiv=atm_dcivpiv_func(sp500_op_ret)

# %% [markdown]
# dpiv

# %%
def dpiv(opt_data):
    sp500_op_ret = opt_data
    atm_options = sp500_op_ret[(sp500_op_ret["moneyness"] >= 0.9) & (sp500_op_ret["moneyness"] <= 1.1)]
    atm_puts_dpiv = atm_options.drop(atm_options[atm_options.cp_flag == 'C'].index)
    dpiv = atm_puts_dpiv.groupby(['secid', 'date'])[["impl_volatility"]].mean()
    dpiv['diffp'] = dpiv['impl_volatility'].diff()
    dpiv = dpiv['diffp']
    return dpiv
# dpivtest=d_piv(sp500_op_ret)

# %% [markdown]
# dciv

# %%
def dciv(opt_data):
    sp500_op_ret = opt_data
    atm_options = sp500_op_ret[(sp500_op_ret["moneyness"] >= 0.9) & (sp500_op_ret["moneyness"] <= 1.1)]
    atm_calls_cpiv = atm_options.drop(atm_options[atm_options.cp_flag == 'P'].index)
    cpiv = atm_calls_cpiv.groupby(['secid', 'date'])[["impl_volatility"]].mean()
    cpiv['diffc'] = cpiv['impl_volatility'].diff()
    cpiv = cpiv['diffc']
    return cpiv
# dcivtest=d_civ(sp500_op_ret)

# %% [markdown]
# civpiv

# %%
def ntm_civpiv(opt_data):
    sp500_op_ret = opt_data
    ntm_options = sp500_op_ret[(sp500_op_ret["moneyness"] >= 0.8) & (sp500_op_ret["moneyness"] <= 1.2)]
    ntm_puts = ntm_options.drop(ntm_options[ntm_options.cp_flag == 'C'].index)
    ntm_calls = ntm_options.drop(ntm_options[ntm_options.cp_flag == 'P'].index)
    ntm_puts = ntm_puts.groupby(['secid', 'date'])[["impl_volatility"]].mean()
    ntm_puts = ntm_puts.rename(columns={"impl_volatility": "implvolput"})
    ntm_calls = ntm_calls.groupby(['secid', 'date'])[["impl_volatility"]].mean()
    ntm_calls = ntm_calls.rename(columns={"impl_volatility": "implvolcall"})
    ntm_civpiv = pd.concat([ntm_calls, ntm_puts], axis=1)
    ntm_civpiv = ntm_civpiv["implvolcall"] - ntm_civpiv["implvolput"]
    return ntm_civpiv
# civpivtest=ntm_civpiv(sp500_op_ret)

# %%
def ivd(opt_data):
    sp500_op_ret = opt_data
    impl_vol_sq = sp500_op_ret["impl_volatility"]**2
    sp500_op_ret["impl_vol_sq"] = impl_vol_sq
    sp500_op_ret.sort_values(["optionid", "days_to_exp"])
    temp_ivd = sp500_op_ret.groupby(['secid', 'date', 'days_to_exp'])[["impl_vol_sq"]].mean()
    temp_ivd = temp_ivd.reset_index(level=['days_to_exp', 'secid', 'date'])
    temp_ivd_diff = temp_ivd.groupby(['secid', 'date'])["impl_vol_sq"].diff()
    temp_ivd['impl_vol_sq_diff'] = temp_ivd_diff
    temp_ivd['impl_difft'] = temp_ivd['impl_vol_sq_diff'] * temp_ivd['days_to_exp']
    a = temp_ivd.groupby(['secid', 'date'])['impl_difft'].sum()
    b = temp_ivd.groupby(['secid', 'date'])['impl_vol_sq_diff'].sum()
    ivd = a / b
    return ivd
# ivdtest=iv_duration(sp500_op_ret)

# %% [markdown]
# iv_slope

# %%
def ivslope(opt_data):
    sp500_op_ret = opt_data
    atm_options = sp500_op_ret[(sp500_op_ret["moneyness"] >= 0.9) & (sp500_op_ret["moneyness"] <= 1.1)]
    short_term_options = atm_options[(sp500_op_ret["days_to_exp"] <= 90)]
    longterm_options = atm_options[(sp500_op_ret["days_to_exp"] > 90)]
    iv_longterm = longterm_options.groupby(['secid', 'date'])["impl_volatility"].mean()
    iv_shortterm = short_term_options.groupby(['secid', 'date'])["impl_volatility"].mean()
    iv_longterm.rename("ivlong", inplace=True)
    iv_shortterm.rename("ivshort", inplace=True)
    ivslope_temp = pd.concat([iv_longterm, iv_shortterm], axis=1)
    result = ivslope_temp["ivlong"] - ivslope_temp["ivshort"]
    return result
# ivslopetest=iv_slope(sp500_op_ret)

# %% [markdown]
# %%<br>
# funcs_by_Natasha = [toi, pcratio, vol, nopt, dvol, ailliq, pilliq,<br>
#                     pcpv, shrtfee, skewiv, atm_civpiv, atm_dcivpiv, <br>
#                     dpiv, dciv, ntm_civpiv, ivd, ivslope]

# %%
funcs_by_Natasha = [toi, pcratio, vol, nopt, dvol, ailliq, pilliq,
                    pcpv, shrtfee, skewiv,
                    dpiv, dciv, ntm_civpiv, ivd, ivslope]  # deleted the two involving option returns

# %% [markdown]
# %%

# %%
results_Natasha = {}
problems = []
for func in funcs_by_Natasha:
    try:
        results_Natasha[func.__name__] = func(sp500_op_ret_w_permno)
    except Exception as e:
        print(f"""problem occured in {func.__name__}\n
              {e}""")
        problems.append(func.__name__)

# %% [markdown]
# %%

# %%
for key in results_Natasha:
    results_Natasha[key].rename(key, inplace=True)

# %% [markdown]
# %%

# %%
STOCKLEVELCHARS = results_Natasha

# %% [markdown]
# %% [markdown]<br>
# 6. Stock vs. option volume (so)

# %% [markdown]
# %%

# %%
stock_level_grouped = sp500_op_ret_w_permno.groupby(['secid', 'date'])

# %%
def get_so(group):
    agg_volu_option = group.volume.sum()
    return group.share_volume.iloc[0] / agg_volu_option

# %%
so = stock_level_grouped.apply(get_so).rename('so', inplace=True)
STOCKLEVELCHARS['so'] = so

# %% [markdown]
# %% [markdown]<br>
# 7. log of so

# %% [markdown]
# %%

# %%
def get_lso(so):
    return np.log(so)

# %%
lso = get_lso(so).rename('lso', inplace=True)
STOCKLEVELCHARS['lso'] = lso

# %% [markdown]
# %% [markdown]<br>
# 8. dso

# %% [markdown]
# %%

# %%
stock_level_grouped = sp500_op_ret_w_permno.groupby(['secid', 'date'])

# %%
def get_dso(group):
    agg_volu_option = (group.volume * group.mid_price).sum()
    return group.share_volume.iloc[0] / agg_volu_option

# %%
dso = stock_level_grouped.apply(get_dso).rename('dso', inplace=True)
STOCKLEVELCHARS['dso'] = dso

# %% [markdown]
# %% [markdown]<br>
# 9. ldso

# %% [markdown]
# %%

# %%
def get_ldso(dso):
    return np.log(dso)

# %%
ldso = get_ldso(dso).rename('ldso', inplace=True)
STOCKLEVELCHARS['ldso'] = ldso

# %% [markdown]
# %% [markdown]<br>
# 13. Proportional bid-ask spread (pba)

# %% [markdown]
# %%

# %%
stock_level_grouped = sp500_op_ret_w_permno.groupby(['secid', 'date'])

# %%
def get_pba(group):
    return (group.volume * (group.best_offer - group.best_bid) / (0.5 * (group.best_offer - group.best_bid))).sum() / group.volume.sum()

# %%
pba = stock_level_grouped.apply(get_pba).rename('pba', inplace=True)
STOCKLEVELCHARS['pba'] = pba

# %% [markdown]
# %% [markdown]<br>
# 30. Weighted put-call spread (vs_level)

# %% [markdown]
# %%

# %%
stock_level_grouped = sp500_op_ret_w_permno.groupby(['secid', 'date'])

# %%
def get_vs_level(group):
    group = group.sort_values('cp_flag', ascending=True)
    s_grouped = group.groupby(['strike_price', 'exdate'])
    candi_key = []
    for key, s_group in s_grouped:  # record keys of such s_group that has length 2
        if s_group.shape[0] == 2:
            candi_key += [key]
    if not candi_key:
        # print(f"{s_group.shape[0]} shape")
        return np.nan
    else:
        # print(1)
        weights = np.array([s_grouped.get_group(key).open_interest.sum() for key in candi_key])
        values = np.array([s_grouped.get_group(key).impl_volatility.iloc[0]
                           - s_grouped.get_group(key).impl_volatility.iloc[1] 
                           for key in candi_key])
        return sum(weights * values) / sum(weights)

# %%
vs_level = stock_level_grouped.apply(get_vs_level).rename('vs_level')
STOCKLEVELCHARS['vs_level'] = vs_level

# %% [markdown]
# %% [markdown]<br>
# 31. Change in Weighted put-call spread (vs_change)

# %% [markdown]
# %%

# %%
def vs_change(vs_level):
    grouped_by_secid = vs_level.sort_index(level=1).groupby(level=0)
    return grouped_by_secid.apply(lambda x: x - x.shift(1))

# %%
vs_change = vs_change(vs_level).rename('vs_change')
STOCKLEVELCHARS['vs_change'] = vs_change

# %% [markdown]
# %%

# %%
STOCKLEVELCHARS_df = pd.DataFrame(STOCKLEVELCHARS)

# %% [markdown]
# %% [markdown]<br>
# # Bucket level<br>
# <br>
# What we can do here is indeed limited, since daily option data is unhandlable for us

# %% [markdown]
# %%

# %%
BUCKETLEVELCHARS = {}

# %% [markdown]
# %%

# %%
def tag_moneyness(row):
    if row.cp_flag == 'C':
        return 'OTM' if row.moneyness > 1.1 else ('ITM' if row.moneyness < 0.9 else 'ATM')
    else:
        return 'ITM' if row.moneyness > 1.1 else ('OTM' if row.moneyness < 0.9 else 'ATM')

# %%
sp500_op_ret_w_permno['moneyness'] = sp500_op_ret_w_permno.strike_price / sp500_op_ret_w_permno.adj_spot
sp500_op_ret_w_permno['moneyness_class'] = sp500_op_ret_w_permno.apply(tag_moneyness, axis=1)
sp500_op_ret_w_permno['maturity_class'] = sp500_op_ret_w_permno['days_to_exp'].map(lambda n: 'L' if n > 90 else 'S')
sp500_op_ret_w_permno['bucket_class'] = sp500_op_ret_w_permno['moneyness_class'] + ',' + sp500_op_ret_w_permno['maturity_class']

# %% [markdown]
# %% [markdown]<br>
# 14. Open Interest vs. stock volume (oistock)

# %% [markdown]
# %%

# %%
bucket_level_grouped = sp500_op_ret_w_permno.groupby(['secid', 'date', 'bucket_class'])

# %%
def get_oistock(group):
    stock_volu = group['share_volume'].iloc[0]
    return group.open_interest.sum() / stock_volu

# %%
oistock = bucket_level_grouped.apply(get_oistock).rename('iostock')
BUCKETLEVELCHARS['oistock'] = oistock

# %% [markdown]
# %% [markdown]<br>
# 15. Volume (bucket_vol)

# %% [markdown]
# %%

# %%
bucket_vol = bucket_level_grouped['volume'].sum().rename('bucket_vol')
BUCKETLEVELCHARS['bucket_vol'] = bucket_vol

# %% [markdown]
# %% [markdown]<br>
# 16. Dollor volume  (bucket_dvol)

# %% [markdown]
# %%

# %%
bucket_dvol = bucket_level_grouped['dollar_volume'].sum().rename('bucket_dvol')
BUCKETLEVELCHARS['bucket_dvol'] = bucket_dvol

# %% [markdown]
# %% [markdown]<br>
# 17. Relative Volume (bucket_vol_share)

# %% [markdown]
# %%

# %%
stock_vol = STOCKLEVELCHARS['vol']
bucket_vol_share = (bucket_vol / stock_vol).rename('bucket_vol_share')
BUCKETLEVELCHARS['bucket_vol_share'] = bucket_vol_share

# %% [markdown]
# %% [markdown]<br>
# 18. Turnover (turnover)

# %% [markdown]
# %%

# %%
turnover = (bucket_level_grouped['volume'].sum() / bucket_level_grouped['open_interest'].sum()).rename('turnover')
BUCKETLEVELCHARS['turnover'] = turnover

# %% [markdown]
# %%

# %%
BUCKETLEVELCHARS_df = pd.DataFrame(BUCKETLEVELCHARS)
BUCKETLEVELCHARS_df

# %% [markdown]
# %% [markdown]<br>
# # Contract Level<br>
# <br>
# By Natasha

# %% [markdown]
# %%<br>
# ###############<br>
# PART3 CHARACTERISTICS

# %% [markdown]
# call_indicator, put indicator

# %%
call_indicator = np.where((sp500_op_ret_w_permno["cp_flag"] == "C"), 1, 0)
sp500_op_ret_w_permno["C"] = call_indicator
put_indicator = np.where((sp500_op_ret_w_permno["cp_flag"] == "P"), 1, 0)
sp500_op_ret_w_permno["P"] = put_indicator

# %% [markdown]
# opt_spread

# %%
optspread = 2 * (sp500_op_ret_w_permno["best_offer"] - sp500_op_ret_w_permno["best_bid"]) / (sp500_op_ret_w_permno["best_offer"] + sp500_op_ret_w_permno["best_bid"])
sp500_op_ret_w_permno["optspread"] = optspread

# %% [markdown]
# embedlev

# %%
embedlev = sp500_op_ret_w_permno["spotprice"] / sp500_op_ret_w_permno["mid_price"] * abs(sp500_op_ret_w_permno["delta"])
sp500_op_ret_w_permno["embedlev"] = embedlev

# %% [markdown]
# volga calculation

# %%
d1 = np.log(sp500_op_ret_w_permno["spotprice"] / sp500_op_ret_w_permno["strike_price"]) + (sp500_op_ret_w_permno["ir_rate"] + 0.5 * sp500_op_ret_w_permno["impl_volatility"]) * sp500_op_ret_w_permno["yrs_to_exp"] / (sp500_op_ret_w_permno["impl_volatility"] * np.sqrt(sp500_op_ret_w_permno["yrs_to_exp"]))
d2 = d1 - sp500_op_ret_w_permno["impl_volatility"] * np.sqrt(sp500_op_ret_w_permno["yrs_to_exp"])
N = np.exp(-0.5 * d1**2) / 2 / np.pi
volga = np.sqrt(sp500_op_ret_w_permno["yrs_to_exp"]) * N * d1 * d2 / sp500_op_ret_w_permno["impl_volatility"]
sp500_op_ret_w_permno["volga"] = volga

# %% [markdown]
# %%

# %%
sp500_op_ret_w_permno['expiration_month'] = np.where((sp500_op_ret_w_permno["days_to_exp"] <= 21), 1, 0)
sp500_op_ret_w_permno['ttm'] = sp500_op_ret_w_permno["yrs_to_exp"]
sp500_op_ret_w_permno['iv'] = sp500_op_ret_w_permno['impl_volatility']
sp500_op_ret_w_permno['oi'] = sp500_op_ret_w_permno['open_interest']
sp500_op_ret_w_permno['doi'] = sp500_op_ret_w_permno['oi'] * sp500_op_ret_w_permno['mid_price']
sp500_op_ret_w_permno['mid'] = sp500_op_ret_w_permno['mid_price']

# %% [markdown]
# %%

# %%
Level_3_chars = ['C', 'P', 'expiration_month', 'ttm', 'moneyness', 'iv', 'delta',
                 'gamma', 'theta', 'vega', 'volga', 'embedlev', 'oi', 'doi',
                 'mid', 'optspread']  # , 'prev_option_ret']
intrinsics = ['secid', 'permno', 'date', 'exdate', 'optionid', 'bucket_class', 'option_ret']

# %% [markdown]
# %%

# %%
CONTRACTLEVELCHARS_df = sp500_op_ret_w_permno[intrinsics + Level_3_chars]

# %% [markdown]
# %% [markdown]<br>
# # Merging results

# %% [markdown]
# %%

# %%
df0 = pd.merge(CONTRACTLEVELCHARS_df, ind_dummies, on=['permno', 'date'],how='inner')
df1 = pd.merge(df0, BUCKETLEVELCHARS_df.reset_index(), on=['secid', 'date', 'bucket_class'])
result = pd.merge(df1, STOCKLEVELCHARS_df.reset_index(), on=['secid', 'date'])

# %% [markdown]
# 

# %% [markdown]
# %%

# %%
result

# %%


# %% [markdown]
# %%

# %%
result.to_csv(Path(DATAROOT, 'option_characteristics.csv'))

# %% [markdown]
# %%

# %%
result['yyyymm'] = result.date.dt.strftime('%Y%m').astype(int)

# %% [markdown]
# %%

# %%
result['yyyymm'].iloc[1]

# %% [markdown]
# %%

# %%
monthly_stock_wide.yyyymm.iloc[1]

# %% [markdown]
# %%

# %%
all_chars_df = pd.merge(result, monthly_stock_wide, on=['permno', 'yyyymm'], how='inner')

# %% [markdown]
# %%

# %%
all_chars_df.to_csv(Path(DATAROOT, 'all_characteristics.csv'))

# %%
list(all_chars_df.columns)

# %% [markdown]
# %%<br>
# all_chars_df


