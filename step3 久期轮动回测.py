# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 16:00:07 2023

@author: shenlinxiao
"""

#import backtrader as bt
#import pyfolio as pf
#import quantstats as qs
import pandas as pd
import numpy as np
import scipy.stats as st

from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
from PIL import Image
#from datetime import datetime,date
from dateutil.relativedelta import *
import datetime,os,math
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV, KFold, cross_val_score
from sklearn.metrics import make_scorer,accuracy_score
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取数据库
def read_data():
    df_basic = pd.read_excel('./数据读取/preprocessed_data.xlsx',sheet_name='SCH_data')

    df_basic = df_basic.set_index('date')
    
    # BOND
    lg_pred_result_bond = pd.read_excel('./轮动预测结果/日度预测结果.xlsx')#, parse_dates=True)
    lg_pred_result_bond.columns = ['date','outcome_bond']
    lg_pred_result_bond['date'] = pd.to_datetime(lg_pred_result_bond['date'])
    lg_pred_result_bond = lg_pred_result_bond.set_index('date')
    # lg_pred_result_bond.index.name = 'trade_date'
    lg_pred_result_bond.fillna(method='pad')  # 原数据有一个p值缺失，不影响未来使用

    return df_basic, lg_pred_result_bond




# 提取年，月，日
def date_process(df):
    df['date'] = df.index
    df["trade_date"] = df["date"]
    df = df.reset_index(drop=True)

    df['year'] = df['date'].apply(lambda x: x.year)
    df['month'] = df['date'].apply(lambda x: x.month)
    df['day'] = df['date'].apply(lambda x: x.day)
    for i in ['day', 'month', 'year']:
        a = df[i]
        df.drop(labels=[i], axis=1, inplace=True)
        df.insert(0, i, a)

    return df


# 根据LG预测的P设定阈值和权重，以及净值变化
def regression_weight(df, threshold_weight_list_bond, bond_standard_weight=0.3):
    def wei_bond(x):
        if x >= threshold_weight_list_bond[0]:
            #return 0.5
            return threshold_weight_list_bond[1]
        elif x <= 1 - threshold_weight_list_bond[0]:
            return 1 - threshold_weight_list_bond[1]
        else:
            return bond_standard_weight
    # 计算债
    df['SCH003012.SCH_weight'] = df['p_bond'].apply(wei_bond)
    df['SCH003022.SCH_weight'] = 1 - df['SCH003012.SCH_weight']
    return df


def relocate_cycle(df, freq="hm", freq_day=None):
    # df.index = df['date_x']
    df["trade_date"] = df["date"]
    df["trade_year"] = df["trade_date"].dt.year
    df["trade_month"] = df["trade_date"].dt.month
    df['num'] = range(len(df))
    df_window = df[["trade_date", "trade_year", "trade_month"]].drop_duplicates()
    # 从早到晚提取交易日期(几号)
    df_window["date_rank"] = df_window.groupby(["trade_year", "trade_month"])["trade_date"].rank(method="min")
    # 每个月分别有多少个交易日
    date_count = df_window.groupby(["trade_year", "trade_month"])["trade_date"].count().rename("date_count")
    df_window = pd.merge(df_window, date_count, on=["trade_year", "trade_month"], how="left")
    df = pd.merge(df, df_window, on=["trade_date", "trade_year", "trade_month"], how="left")
    # 防止一个月都没有交易日|一个月交易日小于11日的情况
    df["relocate_date_1"] = df.apply(lambda x: 1 if 1 <= x.date_count else x.date_count, axis=1)
    df["relocate_date_2"] = df.apply(lambda x: 10 if 10 <= x.date_count else x.date_count, axis=1)

    # 按固定间隔调仓，获取调仓日前一日的权重
    if freq_day:
        df_weight = df.iloc[::freq_day - 1, :]
    # 按月度|半月度调仓
    else:
        if freq == "m":
            df_weight = df.query('date_rank == relocate_date_1')
        elif freq == "hm":
            df_weight = df.query('date_rank == relocate_date_1 or date_rank == relocate_date_2')

        else:
            raise ValueError('请输入正确调仓周期，调仓周期可选参数：[m, hm]')

    # 根据筛选出来需要调仓的日期创建relocate_signal
    df_weight = df_weight[["trade_date", "weight"]].rename(columns={"weight": "m_weight"})
    df_weight["relocate_signal"] = 1
    df = pd.merge(df, df_weight, on="trade_date", how="left")
    # 这里如果用前值填充的话，不就是每一天都有relocate_signal?
    # 如果每天都有relocate_signal影响的是买入的手续费和卖出的印花税计算
    # 本模型中暂时没考虑手续费和印花税问题
    # df = df.sort_values(["trade_date"]).fillna(method="pad")
    # df[['m_weight']] = df.sort_values(["trade_date"])[['m_weight']].fillna(method='pad')
    df[['m_weight']] = df.sort_values(["trade_date"])[['m_weight']].fillna(method='ffill')
    df["m_weight"] = df["m_weight"].apply(lambda x: np.matrix(x))

    # 因为第一个日期无法获取更早的权重
    # 比如说回测开始时间为2010年1月4日，则无法得知更早的权重
    # 所以用当日收盘后的权重矩阵进行提前计算【会有不可避免的轻微误差】
    df["m_weight_relocate"] = df["m_weight"].shift(1)
    df.loc[0, "m_weight_relocate"] = str(df.loc[0, "weight"])
    df["m_weight_relocate"] = df["m_weight_relocate"].apply(np.asmatrix)
    return df




def back_test_index(df,
                    base_index=None,
                    leverage=1,
                    annual_cost=0,
                    portfolio_name='轮动策略'):
    

    # 计算长短久期债券的累计收益率
    df["SCH003012.SCH"] = (1 + df["SCH003012.SCH_changeRatio"]).cumprod()
    df["SCH003022.SCH"] = (1 + df["SCH003022.SCH_changeRatio"]).cumprod()
    df["return2"] = df["SCH003022.SCH_changeRatio"]*df["SCH003022.SCH_weight"] + df["SCH003012.SCH_changeRatio"]*df["SCH003012.SCH_weight"]
    # 计算债的累计收益率
   
    df["cum_return_bond"] = (1 + df.return2).cumprod()

    # 债的基准组合
    df['baseline_bond_return'] = (0.5 * df["SCH003012.SCH_changeRatio"] + 0.5 * df["SCH003022.SCH_changeRatio"])
    df['return_base'] = (0.5 * df["SCH003012.SCH_changeRatio"] + 0.5 * df["SCH003022.SCH_changeRatio"])
    
    # np.exp(np.log1p(df["baseline_bond_return"]).cumsum())  #50/50策略
    
    # # 基准组合净值初始值
    # df.loc[0, "return_base"] = 0
    df["cum_return_base"] = (1 + df.baseline_bond_return).cumprod()
    # # return1为股，return2为债
    
    df['cum_return'] = (1 + df.return2).cumprod()
    df['cum_return2'] = df['cum_return'] 

    # 计算模型组合alpha
    df["alpha"] = df["return2"] - df['return_base']
    df['cum_alpha'] = df['alpha'].cumsum()
    # 计算年化收益
    # annual_return = (list(df["cum_return"])[-1])**(250 / (len(list(df.index)) - 1)) - 1
    # max_valid_index = df['cum_return'].loc[max(df['cum_return'].index)] - 1
    annual_return = (df['cum_return'].loc[max(df['cum_return'].index)]) ** (250 / (max(df['cum_return'].index) - 1)) - 1
    annual_volatility = df["return2"].std() * np.sqrt(250)
    annual_downside_volatility = df[
                                     df["return2"] < 0]["return2"].std() * np.sqrt(250)
    annual_return_base = (list(
        df["cum_return_base"])[-1]) ** (250 / (len(list(df.index)) - 1)) - 1
    annual_volatility_base = df["return_base"].std() * np.sqrt(250)
    annual_downside_volatility_base = df[
                                          df["return_base"] < 0]["return_base"].std() * np.sqrt(250)
    annual_active_return = annual_return - annual_return_base
    annual_active_volatility = df["alpha"].std() * np.sqrt(250)
    skewness = df["return2"].skew()
    kurtosis = df["return2"].kurt()


    # 计算回撤
    # 计算简单回撤
    # 当天最大回撤
    # 回撤天数占比
    # 模型组合当天回撤位于0%~0.5%|0.5%~1%/>1%天数占比
    draw_down_pct_max = df['return2'].min() * 100
    draw_down_pct = sum(df['return2'] < 0) * 100 / df.shape[0]

    draw_down_pct_max_base = df['return_base'].min() * 100
    draw_down_pct_base = sum(df['return_base'] < 0) * 100 / df.shape[0]

    # 计算动态回撤
    # 动态最大回撤
    # 动态回撤天数占比
    # 模型组合动态回撤位于0%~0.5%|0.5%~1%/>1%天数占比
    df['draw_down_cummax'] = df["cum_return"].cummax().to_numpy()
    df['draw_down_base_cummax'] = df["cum_return_base"].cummax()
    df["draw_down"] = (df["cum_return"] -
                       df["cum_return"].cummax()) / df["cum_return"].cummax()
    df["draw_down_base"] = (
                                   df["cum_return_base"] -
                                   df["cum_return_base"].cummax()) / df["cum_return_base"].cummax()

    max_draw_down = df["draw_down"].min()
    max_draw_down_base = df["draw_down_base"].min()
    # 最大回撤日期
    max_draw_down_date = df.query(
        "draw_down == @max_draw_down").iloc[0]["date"]
   
    max_draw_down_base_date = df.query(
        "draw_down_base == @max_draw_down_base").iloc[0]["date"]
   
    base_max_draw_down_pre_high = df.query("date <= @max_draw_down_base_date")['cum_return_base'].cummax(
    ).iloc[-1] if len(df.query("date <= @max_draw_down_base_date")['cum_return_base'].cummax())!=0 else np.nan
    max_draw_down_restore_base_date = df.query(
        "date > @max_draw_down_base_date & cum_return_base >= @base_max_draw_down_pre_high"
    ).iloc[0]['date'] if len(df.query(
        "date > @max_draw_down_base_date & cum_return_base >= @base_max_draw_down_pre_high"
    )) != 0 else np.nan
    
    # 夏普比例
    annual_sharpe = annual_return / annual_volatility
    annual_calmar = annual_return / abs(max_draw_down)
    annual_sortino = annual_return / annual_downside_volatility
    annual_sharpe_base = annual_return_base / annual_volatility_base
    # annual_calmar_base = annual_return_base / abs(max_draw_down_base)
    annual_sortino_base = annual_return_base / annual_downside_volatility_base
    annual_information_ratio = annual_active_return / annual_active_volatility

    # 持有一年收益
    df["buy_and_hold_return_y"] = (df["cum_return"].shift(-250) -
                                   df["cum_return"]) / df["cum_return"]
    df["buy_and_hold_return_base_y"] = (df["cum_return_base"].shift(-250)-df["cum_return_base"]) / df["cum_return_base"]
    
    # 持有半年收益
    df["buy_and_hold_return_y_halfyear"] = (df["cum_return"].shift(-130) -
                                   df["cum_return"]) / df["cum_return"]
    df["buy_and_hold_return_base_y_halfyear"] = (
                                               df["cum_return_base"].shift(-130) -
                                               df["cum_return_base"]) / df["cum_return_base"]
    
    # 持有3个月收益
    df["buy_and_hold_return_y_3month"] = (df["cum_return"].shift(-63) -
                                   df["cum_return"]) / df["cum_return"]
    df["buy_and_hold_return_base_y_3month"] = (
                                               df["cum_return_base"].shift(-63) -
                                               df["cum_return_base"]) / df["cum_return_base"]
    
    # 持有一年|半年/三个月净值
    df['ramdom_holding_250_cum_return_differnce'] = (
            df['cum_return'] - df['cum_return'].shift(250)).values
    df['ramdom_holding_130_cum_return_differnce'] = (
            df['cum_return'] - df['cum_return'].shift(130)).values
    df['ramdom_holding_63_cum_return_differnce'] = (
            df['cum_return'] - df['cum_return'].shift(63)).values
    df['base_ramdom_holding_250_cum_return_differnce'] = (
            df['cum_return_base'] - df['cum_return_base'].shift(250)).values
    df['base_ramdom_holding_130_cum_return_differnce'] = (
            df['cum_return_base'] - df['cum_return_base'].shift(130)).values
    
    gfg_data = df["buy_and_hold_return_y_3month"].dropna()
    gfg_data1 = df["buy_and_hold_return_y_halfyear"].dropna()
    gfg_data2 = df["buy_and_hold_return_y"].dropna()


    result = pd.DataFrame()
    result.loc[0, '模型年化收益'] = "%.2f%%" % (float(annual_return) * 100)
    # result.loc[0, '基准年化收益'] = "%.2f%%" % (float(annual_return) * 100)
    result.loc[0, '模型年化波动'] = "%.2f%%" % (float(annual_volatility) * 100)
    # result.loc[0, '基准年化波动'] = "%.2f%%" % (float(annual_volatility) * 100)
    result.loc[0, '模型年化下行波动'] = "%.2f%%" % (float(annual_downside_volatility) * 100)
    result.loc[0, '模型偏度'] = "%.2f" % float(skewness)
    result.loc[0, '模型峰度'] = "%.2f" % float(kurtosis)
    result.loc[0, '模型最大回撤'] = "%.2f%%" % (float(max_draw_down) * 100)
    result.loc[0, '模型最大回撤发生在'] = max_draw_down_date
    result.loc[0, '模型夏普比率'] = "%.2f" % float(annual_sharpe)
    result.loc[0, '模型卡玛比率'] = "%.2f" % float(annual_calmar)
    result.loc[0, '模型索提诺比率'] = "%.2f" % float(annual_sortino)
    result.loc[0, '持有一年平均收益'] = "%.2f" % float(df["buy_and_hold_return_y"].mean())
    result.loc[0, '持有半年平均收益'] = "%.2f" % float(df["buy_and_hold_return_y_halfyear"].mean())
    
    
    return df, result.T



def back_test_output(df):
    df_annual = df.copy()
    
    df_annual["annual_rank"] = df_annual.groupby(["year"])["date"].rank(method="max", ascending=False)
    df_annual.query("annual_rank == 1", inplace=True)
    df_annual["cum_return_pre"] = df_annual["cum_return"].shift(1)
    df_annual["cum_return_pre"].fillna(1, inplace=True)
    df_annual["annual_return"] = (df_annual["cum_return"] - df_annual["cum_return_pre"]) / df_annual["cum_return_pre"]

    df_annual["cum_return2_pre"] = df_annual["cum_return2"].shift(1)
    df_annual["cum_return2_pre"].fillna(1, inplace=True)
    df_annual["annual_return2"] = (df_annual["cum_return2"] - df_annual["cum_return2_pre"]) / df_annual[
        "cum_return2_pre"]
    
    df_annual = df_annual.rename(columns={'date': '日期',
                                          'annual_return': '组合年化收益率',
                                        
                                          'annual_return2': '债券收益率',
                                          
                                          'cum_return_pre': '上一年组合累计净值',
                                         
                                          #     '_annual_return1':'股票对组合收益率贡献',
                                          #     '_annual_return2':'债券对组合收益率贡献',
                                          #    'annual_stock_contribution':'股票收益贡献百分比',
                                          #   'annual_bond_contribution':'债券收益贡献百分比',
                                          })
    df_annual.to_excel(r"./模型计算结果/output_annually.xlsx")



###############################################################################

load_data()
from datetime import date

#合并股债预测所得到的excel文件

pred_bond1 = pd.read_excel('./轮动预测结果/日度预测结果.xlsx').set_index('Unnamed: 0').dropna(axis=0)


#读取数据
df_basic, _ = read_data()
df_basic.index = pd.to_datetime(df_basic.index)
df_basic = df_basic.reset_index()



# 设定回测日期 & 资产的阈值和权重
data_start_date = '2016-12-01'
data_end_date = '2023-11-10'
df_basic = df_basic.query("date >= @data_start_date & date <@data_end_date")

df_basic["SCH003012.SCH_changeRatio"] = df_basic["SCH003012.SCH_changeRatio"] / 100
df_basic["SCH003022.SCH_changeRatio"] = df_basic["SCH003022.SCH_changeRatio"] /100

# # 2019/1/1回测

# 设置调仓周期

df_final= df_basic
df_final1 = df_final[df_final["date"] >= pd.to_datetime('2019-01-01')]

# df_final = relocate_cycle(df_final, "hm")
df_final = df_final.set_index('date')
df_final.loc[pred_bond1.index, 'SCH003022.SCH_weight'] = pred_bond1['RF']
df_final['SCH003022.SCH_weight'] = df_final['SCH003022.SCH_weight'].fillna(method='ffill')
df_final['SCH003012.SCH_weight'] = 1 - df_final['SCH003022.SCH_weight']
df_final['SCH003022.SCH_weight'] =  df_final['SCH003022.SCH_weight'].fillna(0.5)
df_final['SCH003012.SCH_weight'] =  df_final['SCH003012.SCH_weight'].fillna(0.5)


df_final, result = back_test_index(df_final.reset_index(), leverage=1, annual_cost=0)



###############################################################################

#做图，回测债+直方图，每个月2次
df_final['alpha_bond'] = df_final['cum_return_bond'] - df_final['cum_return_base']


palette = pyplot.get_cmap('Set1')
plt.rcParams.update({'font.size':22})
fig = plt.figure(figsize=(12, 8))
df_final['trade_date'] = pd.to_datetime(df_final['date']) 
ax1 = fig.add_subplot(111)
plt.bar(df_final["trade_date"], df_final["SCH003022.SCH_weight"], label="SCH003022.SCH占比", color="indianred")
ax1.set_ylabel('proportion')

ax2 = ax1.twinx()
plt.plot(df_final['trade_date'],df_final['cum_return_bond'], color="purple",label='久期轮动策略累计收益率')
plt.plot(df_final['trade_date'],df_final['cum_return_base'], color='grey', label='基准策略累计收益率（长短久期50/50）')

ax2.set_ylabel("net price")

plt.legend()
plt.savefig(f'./模型走势图/久期轮动策略_直方图_仓位[{date.today()}].jpg', bbox_inches='tight', dpi=800)
plt.show()
