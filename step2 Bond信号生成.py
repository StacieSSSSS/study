# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 11:08:09 2023

@author: shenlinxiao
"""

import math
import pandas as pd
import numpy as np
import calendar
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import dates
from datetime import datetime,date
from dateutil import parser
from time import time
from dateutil.relativedelta import *
from sklearn.metrics import make_scorer,accuracy_score, precision_score, recall_score, cohen_kappa_score, confusion_matrix, mutual_info_score,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier,GradientBoostingRegressor
from sklearn import tree
from sklearn.model_selection import train_test_split,GridSearchCV, KFold, cross_val_score
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
from sklearn.metrics import mean_absolute_error

pd.set_option('display.max_columns', None)
np.set_printoptions(suppress=True)

###############################################################################

# data_len = 1[返回第九行开始的数据]|2[返回完整数据]
def data_clean(data_len=2):
    def _outcome(x):
        if x >= 0:
            return 1
        else:
            return 0
    
    def _outcome2(x):
        if x == True:
            return 1
        else:
            return 0

    df_price = pd.read_excel('./数据读取/preprocessed_data.xlsx',sheet_name='SCH_data')
    df_eco = pd.read_excel('./数据读取/preprocessed_data.xlsx',sheet_name='EDB_data')
    df_all_data = pd.merge(df_price,df_eco,on=['year','month','day'],how='inner')
    
    col_selection = [
        'CPI:环比',
        'M1(货币):期末值', 
        'M2(货币和准货币):期末值', 
        'PPI:环比',
        '中国综合PMI:产出指数',
        '制造业PMI:产成品库存', 
        '制造业PMI:原材料库存', 
        '制造业PMI:新订单', 
        '制造业PMI:生产', 
        '南华期货:工业品指数', 
        '国债收益率:10年', 
        '国债收益率:1年', 
        '市盈率:上证指数', 
        '市盈率:沪深300', 
        '房地产开发投资:累计值', 
        '汽车:产量:当月值', 
        '社会消费品零售总额:当月值', 
        '社会融资规模增量:当月值', 
        '社会融资规模增量:政府债券:当月值', 
        '美国:国债收益率:10年', 
        '规模以上工业企业:利润总额:当月值', 
        '规模以上工业增加值:季调:环比', 
        '金融机构:中长期贷款余额', 
        'year', 
        'month', 
        'day', 
        'SCH003012.SCH_close', 
        'SCH003012.SCH_changeRatio', 
        'SCH003012.SCH_up_days', 
        'SCH003012.SCH_breakout_ma', 
        'SCH003012.SCH_breakdown_ma', 
        'SCH003012.SCH_std_3m', 
        'SCH003012.SCH_std_6m', 
        'SCH003012.SCH_BOLL_hband', 
        'SCH003012.SCH_BOLL_lband', 
        'SCH003022.SCH_close', 
        'SCH003022.SCH_changeRatio', 
        'SCH003022.SCH_up_days', 
        'SCH003022.SCH_breakout_ma', 
        'SCH003022.SCH_breakdown_ma', 
        'SCH003022.SCH_std_3m',
        'SCH003022.SCH_std_6m', 
        'SCH003022.SCH_BOLL_hband', 
        'SCH003022.SCH_BOLL_lband'
    ]
    df_all_data = df_all_data[col_selection]

    df_all_data['date'] = pd.to_datetime(df_all_data[['year','month','day']])
    
    # df_all_data['ytm1_rank']=df_all_data.SCH032A_ytm.rank(pct=True)
    # df_all_data['ytm2_rank']=df_all_data.SCH032B_ytm.rank(pct=True)
    # df_all_data['pct1_rank']=df_all_data.SCH032A_pct_chg.rank(pct=True)
    # df_all_data['pct2_rank']=df_all_data.SCH032B_pct_chg.rank(pct=True)
    # df_all_data['pct3_rank']=df_all_data.SCH031A_pct_chg.rank(pct=True)
    # df_all_data['pct4_rank']=df_all_data.SCH031B_pct_chg.rank(pct=True)
    # df_all_data['bond_rank']=df_all_data["10年期国债收益率"].rank(pct=True)
    
    #生成label
    #df_all_data['outcome1'] = df_all_data['ytm1_rank']-df_all_data['ytm2_rank']
    df_all_data['outcome1'] = df_all_data['SCH003022.SCH_close']-df_all_data['SCH003012.SCH_close']
    df_all_data['outcome2'] = df_all_data.outcome1.rolling(6).rank(pct=True)
    #print(df_all_data['outcome2'])
    df_all_data['outcome3'] = df_all_data['outcome2'] -0.8
    #df_all_data['outcome3'] = df_all_data['outcome2'].pct_change(periods=1)-0.4
    df_all_data['outcome'] = df_all_data['outcome3'].apply(_outcome)
    # df_all_data['SCH032A_breakout_ma'] = df_all_data['SCH032A_breakout_ma'].apply(_outcome2)
    # df_all_data['SCH032A_breakdown_ma'] = df_all_data['SCH032A_breakdown_ma'].apply(_outcome2)
    # df_all_data['SCH032B_breakout_ma'] = df_all_data['SCH032B_breakout_ma'].apply(_outcome2)
    # df_all_data['SCH032B_breakdown_ma'] = df_all_data['SCH032B_breakdown_ma'].apply(_outcome2)
    df_all_data['PPI-CPI'] = df_all_data['PPI:环比']-df_all_data['CPI:环比']
    df_all_data['10yr-1yr'] = df_all_data['国债收益率:10年']-df_all_data['国债收益率:1年']
    df_all_data['M2-M1'] = df_all_data['M2(货币和准货币):期末值']-df_all_data['M1(货币):期末值']
    df_all_data['产出缺口'] = df_all_data['制造业PMI:新订单']-df_all_data['制造业PMI:生产']
    df_all_data['去/补库'] = df_all_data['制造业PMI:产成品库存']-df_all_data['制造业PMI:原材料库存']
    df_all_data['中美利差'] = df_all_data['国债收益率:10年']-df_all_data['美国:国债收益率:10年']
    df_all_data['上证指数风险溢价'] = 1/df_all_data['市盈率:上证指数']-df_all_data['国债收益率:10年']
    df_all_data['沪深300风险溢价'] = 1/df_all_data['市盈率:沪深300']-df_all_data['国债收益率:10年']
    df_all_data['std_diff_3m'] = df_all_data['SCH003012.SCH_std_3m']-df_all_data['SCH003022.SCH_std_3m']
    df_all_data['std_diff_6m'] = df_all_data['SCH003012.SCH_std_6m']-df_all_data['SCH003022.SCH_std_6m']
  
    # for i in ["SCH031A_BOLL","SCH031B_BOLL","PMI","社会融资规模:当月值","金融机构:中长期贷款余额",
    #     "进出口金额:当月环比","PMI:生产","PMI:新订单","PMI:原材料库存","M2","M1","CPI:环比",
    #     "社会消费品零售总额:当月值","工业增加值:环比:季调","产量:汽车:当月值",'PPI:全部工业品:环比',
    #     "工业企业:利润总额:当月值","社会融资规模:政府债券:当月值","10年期国债收益率","bond_rank","1年期国债收益率","房地产开发投资完成额:累计值","南华工业品指数"]:
    #     df_all_data[i] = df_all_data[i].astype(float, errors = 'raise')
   
    # df_all_data['SCH032A_slowKD'] = df_all_data['SCH032A_slowKD'].astype(float, errors = 'raise')
    # df_all_data['SCH032B_slowKD'] = df_all_data['SCH032B_slowKD'].astype(float, errors = 'raise')
    # df_all_data['SCH032A_up_days'] = df_all_data['SCH032A_up_days'].astype(float, errors = 'raise')
    # df_all_data['SCH032B_up_days'] = df_all_data['SCH032B_up_days'].astype(float, errors = 'raise')
    
    if data_len == 1:
        return df_all_data.iloc[8:]
    elif data_len == 2:
        return df_all_data
    

    #     #统一口径，原始经济数据处理：统一口径：同比用（当月环比-过去环比均值）
        

    # type1 = ['社会融资规模增量:当月值','金融机构:中长期贷款余额', 
    #          'M2(货币和准货币):期末值', 'M1(货币):期末值','社会消费品零售总额:当月值', '汽车:产量:当月值',
    #         '规模以上工业企业:利润总额:当月值','社会融资规模增量:政府债券:当月值']
    # type2 = ['房地产开发投资:累计值']
    # type3 = ['PPI:环比', 'CPI:环比', '规模以上工业增加值:季调:环比', '进出口金额:当月环比',
    #          '中国综合PMI:产出指数','制造业PMI:生产','制造业PMI:原材料库存','制造业PMI:新订单']

    # for col in df_eco_new.columns:
    # #     print(df_eco_new)
    #     if col in type1:
    # #         continue # todo
    #         # 首先计算环比，然后计算：当月环比减去过去五年环比平均值
    # #         print(col)
    #         df_eco_new[col] = df_eco_new[col].pct_change()
    #         template = df_eco_new.copy()
    #         for i in range(len(df_eco_new[col])):
    #             curr_time = df_eco_new.index[i]
    # #             print(df_eco_new[col])
    #             try:
    #                 df_eco_new.iat[i, list(df_eco_new.columns).index(col)] -= (template.loc['{}-{}'.format(curr_time.year-1, curr_time.month),col].values+
    #                                                                          template.loc['{}-{}'.format(curr_time.year-2, curr_time.month),col].values+
    #                                                                          template.loc['{}-{}'.format(curr_time.year-3, curr_time.month),col].values)/3
    #             except Exception as e:
    # #                 print(e)
    #                 df_eco_new.iat[i, list(df_eco_new.columns).index(col)] = np.nan
    # #             print(type(df_eco_new.index[i]))
    # #             print(df_eco_new.iat[i, list(df_eco_new.columns).index(col)])
    # #             try:
    # #                 df_eco_new.iloc[i, col] = 
    # #             except:
    # #                 df_eco_new.iloc[i, col] = np.nan
    #         pass
    #     elif col in type2:
    # #         continue # todo
    #         # 先计算每个月的增量，然后计算增量环比，接着计算：增量当月环比减去过去五年环比均值
    #         df_eco_new[col] = df_eco_new[col].diff()
    #         df_eco_new[col] = df_eco_new[col].pct_change()
    #         template = df_eco_new.copy()
    #         for i in range(len(df_eco_new[col])):
    #             curr_time = df_eco_new.index[i]
    # #             print(df_eco_new[col])
    #             try:
    #                 df_eco_new.iat[i, list(df_eco_new.columns).index(col)] -= (template.loc['{}-{}'.format(curr_time.year-1, curr_time.month),col].values+
    #                                                                          template.loc['{}-{}'.format(curr_time.year-2, curr_time.month),col].values+
    #                                                                          template.loc['{}-{}'.format(curr_time.year-3, curr_time.month),col].values)/3
    #             except Exception as e:
    # #                 print(e)
    #                 df_eco_new.iat[i, list(df_eco_new.columns).index(col)] = np.nan
    #         pass
    #     elif col in type3:
    #         # 直接减去过去五年环比平均值
    #         template = df_eco_new.copy()
    #         for i in range(len(df_eco_new[col])):
    #             curr_time = df_eco_new.index[i]
    # #             print(df_eco_new[col])
    #             try:
    # #                 print(template.loc['{}-{}'.format(curr_time.year-1, curr_time.month),col].values)
    #                 df_eco_new.iat[i, list(df_eco_new.columns).index(col)] -= (template.loc['{}-{}'.format(curr_time.year-1, curr_time.month),col].values[0]+
    #                                                                          template.loc['{}-{}'.format(curr_time.year-2, curr_time.month),col].values[0]+
    #                                                                          template.loc['{}-{}'.format(curr_time.year-3, curr_time.month),col].values[0])/3
    #             except Exception as e:
    # #                 print(e)
    #                 df_eco_new.iat[i, list(df_eco_new.columns).index(col)] = np.nan
    #         pass
    #     else:
    #         pass


def data_normalization(set_name='reg'):
    df_all_data = data_clean(data_len=2)
    
    # windows> windows_2
    windows = 6
 
    # 对于未更新数据用均值填充处理，回测的时候不加入最新月数据
    for column in list(df_all_data.columns[df_all_data.isnull().sum()>0]):
        df_all_data[column].fillna(df_all_data[column].mean(),inplace=True)
    
    # 日期数据处理
    df_all_data[[i for i in df_all_data['year'].unique()]]=pd.get_dummies(df_all_data['year'])
    df_all_data[[i for i in df_all_data['month'].unique()]]=pd.get_dummies(df_all_data['month'])
    df_all_data[[i for i in df_all_data['day'].unique()]]=pd.get_dummies(df_all_data['day'])
    price_col = ['SCH003012.SCH_close', 
    'SCH003012.SCH_changeRatio', 
    'SCH003012.SCH_std_3m', 
    'SCH003012.SCH_std_6m', 
    'SCH003012.SCH_BOLL_hband', 
    'SCH003012.SCH_BOLL_lband', 
    'SCH003022.SCH_close', 
    'SCH003022.SCH_changeRatio', 
    
    'SCH003022.SCH_std_3m',
    'SCH003022.SCH_std_6m', 
    'SCH003022.SCH_BOLL_hband', 
    'SCH003022.SCH_BOLL_lband']
   
    std_col = ['std_diff_3m','std_diff_6m']
    trend_col = ['SCH003012.SCH_up_days','SCH003022.SCH_up_days']
    eco_col = ['CPI:环比',
    'M1(货币):期末值', 
    'M2(货币和准货币):期末值', 
    'PPI:环比',
    '中国综合PMI:产出指数',
    '制造业PMI:产成品库存', 
    '制造业PMI:原材料库存', 
    '制造业PMI:新订单', 
    '制造业PMI:生产', 
    '南华期货:工业品指数', 
    '国债收益率:10年', 
    '国债收益率:1年', 
    '市盈率:上证指数', 
    '市盈率:沪深300', 
    '房地产开发投资:累计值', 
    '汽车:产量:当月值', 
    '社会消费品零售总额:当月值', 
    '社会融资规模增量:当月值', 
    '社会融资规模增量:政府债券:当月值', 
    '美国:国债收益率:10年', 
    '规模以上工业企业:利润总额:当月值', 
    '规模以上工业增加值:季调:环比', 
    '金融机构:中长期贷款余额', 
    'PPI-CPI','10yr-1yr',   '产出缺口',"去/补库","中美利差","沪深300风险溢价","上证指数风险溢价"]
    liquidity_col = ['M2-M1']
    outcome_col = ['outcome']
    date_col = ['year','month','date']
    tech_col = [
    'SCH003012.SCH_breakout_ma', 
    'SCH003012.SCH_breakdown_ma', 
    'SCH003022.SCH_breakout_ma', 
    'SCH003022.SCH_breakdown_ma', 
                    ]
    col_sets = {'reg':std_col+trend_col+liquidity_col+eco_col+price_col}
    for i in col_sets[set_name]:
        df_all_data[i] = df_all_data[i].fillna(method='ffill')
        df_all_data[i] = (df_all_data[i]  - df_all_data[i].rolling(window = windows).mean())/((df_all_data[i].rolling(window=windows).mean().std()))
    col_sets = {'reg':std_col+trend_col+liquidity_col+date_col+outcome_col+eco_col+price_col+tech_col}
    if set_name:
        return df_all_data[col_sets[set_name]]#返回没有空值开始的数据（因为前八期用于标准化计算）
    
    else:
        raise ValueError('请表明logistics regression需要的feature')




def feature_regression():
    df_all_data = data_normalization()
    year_features = [i for i in df_all_data['year'].unique()]
    month_features = [i for i in df_all_data['month'].unique()]
    day_features = [i for i in df_all_data['day'].unique()]
    price_features = ['SCH032A_close','SCH032B_close','SCH032A_pct_chg','SCH032B_pct_chg'
                     ,'SCH031A_close','SCH031B_close','SCH031A_pct_chg','SCH031B_pct_chg',
                      "SCH031A_BOLL","SCH031B_BOLL",
                      "ytm1_rank","ytm2_rank",
                      "pct1_rank","pct2_rank",
                     "pct3_rank","pct4_rank"]
    eco_features = ["PMI","社会融资规模:当月值","金融机构:中长期贷款余额",
        "进出口金额:当月环比","PMI:生产","PMI:新订单","PMI:原材料库存","CPI:环比",
        "社会消费品零售总额:当月值","工业增加值:环比:季调","产量:汽车:当月值",'PPI:全部工业品:环比',
        "工业企业:利润总额:当月值","社会融资规模:政府债券:当月值",
        "10年期国债收益率","1年期国债收益率","M2","M1",
        'M2-M1','PPI-CPI','10yr-1yr','产出缺口',"房地产开发投资完成额:累计值","去/补库","PMI:产成品库存",
        "中美利差","风险溢价","南华工业品指数","沪深300风险溢价","上证指数风险溢价"]
    tech_features = ['SCH032A_slowKD',
                     'SCH032A_breakout_ma',
                     'SCH032A_breakdown_ma',
                     'SCH032B_slowKD',
                     'SCH032B_breakout_ma',
                     'SCH032B_breakdown_ma',
                     "SCH032A_up_days",
                     "SCH032B_up_days"
                    ]
    regression_feature_sets = year_features+month_features+day_features+price_features+eco_features+tech_features
    return regression_feature_sets


def _classifier(set_name,feature_set, outcome_data, feature_data, param_grid):
    X = feature_data.loc[:, feature_set]
    y = np.array(df_outcome).ravel()
    
    # split the train, test data
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=123)

    model = GridSearchCV(
        estimator = xgb.XGBClassifier(objective='binary:logistic',
                                      subsample=0.9,
                                      colsample_bytree=0.5),
        param_grid = param_grid,
        scoring=make_scorer(accuracy_score),
        verbose=0, # Want to see what Grid Search is doing, set verbose = 2
        cv=5,
        dn_jobs=-1)
    
    model = model.fit(X_train,
                      y_train,
                      early_stopping_rounds=200,
                      eval_metric='aucpr',
                      eval_set=[(X_test,y_test)])
     
    train_pred = model.predict(X_train)
    train_accuracy = 100*accuracy_score(y_train, train_pred)
    test_pred = model.predict(X_test)
    test_accuracy = 100*accuracy_score(y_test, test_pred)

    # Bookkeeping and printing for the reader (not part of the core loop)
    print(f"Results for {set_name}:")
    # print(model.best_params_)
    print(model.best_params_)
    # print(f"Accuracy on the train set: {train_accuracy:.1f}")
    print(f"Accuracy on the test set: {test_accuracy:.1f}")
    return y_test, test_pred, model

# LR regression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import sklearn.svm as svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn import metrics


def regression(regressor, feature_set,outcome_data,feature_data,cv=4):
    def roc_curve_plt(classifier, test_y, pred_y, pos_label = 1):
        fpr, tpr, thresholds = roc_curve(test_y, pred_y, pos_label = 1)
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate', fontsize = 15)
        plt.ylabel('True Positive Rate', fontsize = 15)
        plt.title(classifier)
        #plt.legend()
        plt.show()
        plt.savefig(f'{classifier}_roc.png')
        
    if regressor == 'LR':
        params = {'penalty':[None,'l2']}
        X = feature_data.loc[:,feature_set]
        y = np.array(outcome_data)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        
        log_reg = GridSearchCV(LogisticRegression(solver='newton-cg',max_iter= 100,C=0.1),params,cv=cv,n_jobs=-1,)
        log_reg = log_reg.fit(x_train, y_train.ravel())
        
        # make prediction
        y_pred = log_reg.predict(x_test)
        roc_curve_plt(regressor, y_test, y_pred, pos_label = 1)
        return y_test, y_pred, log_reg.best_estimator_
    
    if regressor =='SVM':
        params =  [{'kernel': ['linear'], 'C': [1, 10]},
        {'kernel': ['rbf'], 'C': [1, 10], 'gamma':[1, 0.1]}]
        
        X = feature_data.loc[:,feature_set]
        y = np.array(outcome_data)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        model = GridSearchCV(svm.SVC(probability=True), 
                        params, refit=True, return_train_score=True, cv=cv)
        model = model.fit(x_train, y_train.ravel())
        y_pred = model.predict(x_test)
        roc_curve_plt(regressor, y_test, y_pred, pos_label = 1)
        return y_test, y_pred, model.best_estimator_
        
    if regressor == 'adaboost':
        
        X = feature_data.loc[:,feature_set]
        y = np.array(outcome_data)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        clf_DTree_adaboost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
        model =clf_DTree_adaboost
        model = model.fit(x_train, y_train.ravel())
        y_pred = model.predict(x_test)
        roc_curve_plt(regressor, y_test, y_pred, pos_label = 1)
        return y_test, y_pred, model
        
    if regressor =='RF':
        params = { 'n_estimators': [i for i in range(20,150,10)],
                    'max_depth':[5,8,10],
                    'criterion':['gini','entropy'],
                    }
        X = feature_data.loc[:,feature_set]
        y = np.array(outcome_data)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        clf_DTree_RandomForest = RandomForestClassifier()
        
        model = GridSearchCV(clf_DTree_RandomForest, 
                        params, cv=cv)
        model = model.fit(x_train, y_train.ravel())
        y_pred = model.predict(x_test)
        roc_curve_plt(regressor, y_test, y_pred, pos_label = 1)
        return y_test, y_pred, model.best_estimator_



def performance(classifier, test_y, pred_y, perf_df, pos_label = 1):
    accuracy = metrics.accuracy_score(test_y, pred_y)
    precision = metrics.precision_score(test_y, pred_y, pos_label = 1) 
    recall = metrics.recall_score(test_y, pred_y, pos_label = 1) 
    print(classifier, ' accuracy:  %.2f%%, precision: %.2f%%, recall: %.2f%%' % (100 * accuracy, 100 * precision, 100 * recall))
    perf_df.loc[classifier, 'accuracy'] = accuracy
    perf_df.loc[classifier, 'precision'] = precision
    perf_df.loc[classifier, 'recall'] = recall
    
    
 

def regression_iter(regressor_name, df_perf, pred, df_all_data_last_date,regression_feature_sets,df_outcome,normalized_df):
    if len(normalized_df)*0.2<10:
        y_test, test_pred, best_model = regression(regressor_name,regression_feature_sets,df_outcome,normalized_df,cv=2)
    else:
        y_test, test_pred, best_model = regression(regressor_name,regression_feature_sets,df_outcome,normalized_df,cv=10)

    accuracy = 100*accuracy_score(y_test, test_pred)
    
    _pred = best_model.predict(pred_data.loc[:,regression_feature_sets]) # 用最优的模型预测下一期的结果
    # # 计算P，成功概率
    # odd_ratio = np.exp(sum(sum(best_model.coef_[0]*pred_data.loc[:,regression_feature_sets].values))+best_model.intercept_)
    # _percentage = odd_ratio/(1+odd_ratio)
    # coef1.append(best_model.coef_[0])
    pred.loc[df_all_data_last_date, regressor_name] = _pred
    df_perf.loc['precisions',regressor_name].append(metrics.precision_score(y_test, test_pred, pos_label =1))
    df_perf.loc['abs_squared_error', regressor_name].append(mean_absolute_error(y_test, test_pred))
    df_perf.loc['recalls',regressor_name].append(metrics.recall_score(y_test, test_pred, pos_label =1) )
    df_perf.loc['r2_scores',regressor_name].append(r2_score(y_test, test_pred))
    df_perf.loc['accuracies',regressor_name].append(accuracy)
    return df_perf, pred

print('666')

###############################################################################

if __name__ == '__main__':
    data_start_date = date(2016,6,1)
    df_test = pd.DataFrame(data_clean(2).query("date >= @data_start_date"))
    df_test["day"] = df_test["day"].astype(int)
    
    last_date_list = list(df_test.groupby(['year','month'])['date'].last())

    # initialize the bookkeeping result

    best = 0
    best_name = None
    df_perf = pd.DataFrame(index = ['precisions','recalls','accuracies',
                                    'r2_scores','abs_squared_error'],
                           columns=['SVM','LR','adaboost','RF']).astype('object')
    for i in df_perf:
        for j in df_perf.index:
            df_perf.loc[j,i] = [0]

    pred = pd.DataFrame(index=last_date_list,columns = ['SVM','LR','adaboost','RF']).astype('object')

    actual = None
    regression_feature_sets =  ['std_diff_3m', 'std_diff_6m', 'SCH003012.SCH_up_days', 
                                'SCH003022.SCH_up_days', 'M2-M1'
                                , 'CPI:环比', 'M1(货币):期末值', 'M2(货币和准货币):期末值', 
                                'PPI:环比', '中国综合PMI:产出指数', '制造业PMI:产成品库存', 
                                '制造业PMI:原材料库存', '制造业PMI:新订单', '制造业PMI:生产', 
                                '南华期货:工业品指数', '国债收益率:10年', 
                                '国债收益率:1年', '市盈率:上证指数', '市盈率:沪深300', 
                                '房地产开发投资:累计值', '汽车:产量:当月值', '社会消费品零售总额:当月值', 
                                '社会融资规模增量:当月值', '社会融资规模增量:政府债券:当月值', '美国:国债收益率:10年', 
                                '规模以上工业企业:利润总额:当月值', '规模以上工业增加值:季调:环比', '金融机构:中长期贷款余额', 
                                'PPI-CPI', '10yr-1yr', '产出缺口', '去/补库', '中美利差', '沪深300风险溢价', 
                                '上证指数风险溢价', 'SCH003012.SCH_close', 'SCH003012.SCH_changeRatio', 
                                'SCH003012.SCH_std_3m', 'SCH003012.SCH_std_6m', 'SCH003012.SCH_BOLL_hband', 
                                'SCH003012.SCH_BOLL_lband', 'SCH003022.SCH_close', 'SCH003022.SCH_changeRatio', 
                                'SCH003022.SCH_std_3m', 'SCH003022.SCH_std_6m', 'SCH003022.SCH_BOLL_hband', 
                                'SCH003022.SCH_BOLL_lband', 'SCH003012.SCH_breakout_ma', 'SCH003012.SCH_breakdown_ma', 
                                'SCH003022.SCH_breakout_ma', 'SCH003022.SCH_breakdown_ma']
    for i in last_date_list[10:]:    

        df_all_data_last_date = i # 模型训练中训练集（特征）数据的截止日期
        normalized_df = data_normalization('reg')
        normalized_df = normalized_df.dropna(axis=0)

        normalized_df = normalized_df.query('date <= @df_all_data_last_date')
        pred_data =  normalized_df.query('date == @df_all_data_last_date')# 需要预测的数据是last_date的数据
        
        
        df_outcome = normalized_df[['date','outcome']].shift(-1).iloc[:-1]#训练集（结果）
        df_outcome = pd.DataFrame(df_outcome)
        normalized_df = normalized_df[normalized_df["date"].isin(df_outcome["date"])]#训练集（特征）
        df_outcome = df_outcome['outcome']
        
        # 分类器循环
        
        for regressor_name in ['SVM','LR','adaboost', 'RF']:
            df_perf, pred = regression_iter(regressor_name, df_perf, pred, df_all_data_last_date,regression_feature_sets,df_outcome,normalized_df)
            
    for i in df_perf:
        for j in df_perf.index:
            df_perf.loc[j,i] = np.mean(df_perf.loc[j,i])
    df_perf.to_excel('模型表现.xlsx')
    result = pd.DataFrame(pred)['RF']

    result.to_excel('./轮动预测结果/日度预测结果.xlsx')
 
###############################################################################

print(accuracies)

###############################################################################

sum(accuracies)/len(accuracies)

###############################################################################

coef2 = pd.DataFrame(coef1)
coef2.to_excel("./轮动结果分析/coefficent_久期轮动_日度.xlsx")

###############################################################################

normalized_df.to_excel("./轮动结果分析/coefficent_久期轮动_日度.xlsx")

###############################################################################

from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
from sklearn.metrics import mean_absolute_error
print(r2_score(y_test,test_pred))
print(mean_squared_error(y_test,test_pred))

###############################################################################

import matplotlib.pyplot as plt

x_=[]
for i in accuracies:
    i/=100   #对数据逐个处理
    x_.append(i)
accuracies1=x_

def draw_number(last_date_list, number_V2V, number_V2I, number_finished):
    plt.rcParams['font.sans-serif'] = ['SimHei']#用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False#用来正常显示负号
    plt.rcParams.update({'font.size':22})
    fig = plt.figure(figsize=(15, 8))
    
    
#    plt.bar(step,cost, color="red")
#    plt.plot(step,cost)
    plt.plot(last_date_list, number_finished, color = "red", label = "准确率")
    plt.plot(last_date_list, number_V2V, color = "blue", label = "误差")
    plt.plot(last_date_list, number_V2I, color = "green", label = "r方")
    plt.axhline(y=0,ls=":",c="red")#添加水平直线
    plt.axhline(y=0.5,ls=":",c="red")#添加水平直线
    
    plt.legend()#显示图例
    plt.xlabel("date")
    plt.ylabel("data")
    plt.title("作图")
    plt.show()#画图

draw_number(last_date_list, abs_squared_error, r2_scores, accuracies1)
