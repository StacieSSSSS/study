# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 11:24:22 2024

@author: 12345
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 00:24:27 2024

@author: 12345
"""

import os
import sys
import time
import numpy as np
import scipy
import pandas as pd
import math
import random
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
sns.set_style('white')

from sklearn.preprocessing import OneHotEncoder


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import r2_score

# Data input
CN_Slope_differential = pd.read_csv("CN_Slope_differential.csv", index_col=0)
DXY = pd.read_csv("DXY.csv", index_col=0)
USDCNY = pd.read_excel("CNY_NDF.xlsx", index_col=0)
USDCNH = pd.read_excel("CNH_fwd.xlsx", index_col=0)
USDTWD = pd.read_excel("TWD_NDF.xlsx", index_col=0)
SSEC = pd.read_csv("SSEC.csv", index_col=0)
HS300 = pd.read_csv("HS300.csv", index_col=0)
TWII = pd.read_csv("TWII.csv", index_col=0)
Brent = pd.read_csv("Brent.csv", index_col=0)
VIX = pd.read_csv("VIX.csv", index_col=0)
CN_10y = pd.read_csv("CN_10y.csv", index_col=0)
US_10y = pd.read_csv("US_10y.csv", index_col=0)
Gold = pd.read_csv("Gold.csv", index_col=0)
DB = pd.read_excel("DB.xlsx", index_col=0)
JPM = pd.read_excel("JPM.xlsx", index_col=0)


#format transformation
DXY.index = pd.to_datetime(DXY.index)
SSEC.index = pd.to_datetime(SSEC.index)
CN_Slope_differential.index = pd.to_datetime(CN_Slope_differential.index)
USDCNY.index = pd.to_datetime(USDCNY.index)
USDCNH.index = pd.to_datetime(USDCNH.index)
USDTWD.index = pd.to_datetime(USDTWD.index)

HS300.index = pd.to_datetime(HS300.index)
TWII.index = pd.to_datetime(TWII.index)
Brent.index = pd.to_datetime(Brent.index)
VIX.index = pd.to_datetime(VIX.index)
CN_10y.index = pd.to_datetime(CN_10y.index)
US_10y.index = pd.to_datetime(US_10y.index)
Gold.index = pd.to_datetime(Gold.index)
CN_Slope_differential = pd.DataFrame(CN_Slope_differential['Shibor_1y'] - CN_Slope_differential['Shibor_1m'])
CN_Slope_differential.columns = ['CN_Slope_differential']
CN_Carry_differential = pd.merge(US_10y, CN_10y, how='outer', left_index=True, right_index=True)
CN_Carry_differential = pd.DataFrame(CN_Carry_differential['US_10y'] - CN_Carry_differential['CN_10y'])
CN_Carry_differential.columns = ['CN_Carry_differential']
DB.index = pd.to_datetime(DB.index)
JPM.index = pd.to_datetime(JPM.index)

# USDCNH['CNH_fwd'] = np.log(USDCNH['CNH_fwd'])

# Modeling Framework
# Step 1 
# s = σF + Xβ + e
# F: DXY, G10 FX Carry Index, JPM Global FX Vol Index
# X: constant, commodities, Local Equities

CNH_dataset = pd.merge(SSEC, DXY, how='outer', left_index=True, right_index=True)
CNH_dataset = CNH_dataset.merge(Gold, how='outer', left_index=True, right_index=True)
CNH_dataset = CNH_dataset.merge(USDCNH, how='outer', left_index=True, right_index=True)
CNH_dataset = CNH_dataset.merge(JPM, how='outer', left_index=True, right_index=True)
CNH_dataset = CNH_dataset.merge(DB, how='outer', left_index=True, right_index=True)
CNH_dataset = CNH_dataset.ffill()
CNH_dataset = CNH_dataset.dropna(axis=0)

y = CNH_dataset['CNH_fwd']
X = CNH_dataset.drop('CNH_fwd', axis=1)

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
e1 = y-y_pred
CNH_dataset['e1'] = e1
CNH_dataset['e1_lag'] =CNH_dataset['e1'].shift(1)


# Step 2 error-correction equation
# d = constant + θe(lagged) + Z +e'
# Z: Carry differential, Slope differential

model2 = LinearRegression()
CNH_dataset = CNH_dataset.merge(VIX, how='outer', left_index=True, right_index=True)
CNH_dataset = CNH_dataset.merge(CN_Slope_differential, how='outer', left_index=True, right_index=True)
CNH_dataset = CNH_dataset.merge(CN_Carry_differential, how='outer', left_index=True, right_index=True)
CNH_dataset = CNH_dataset.ffill()
CNH_dataset = CNH_dataset.dropna(axis=0)


X2 = CNH_dataset.loc[:,['e1_lag', 'VIX', 'CN_Slope_differential', 'CN_Carry_differential']]
y2 = CNH_dataset['CNH_fwd']
model2.fit(X2, y2)

y_pred2 = model2.predict(X2)
y_diff = y_pred2 - y2

# results
print('intercept: w0={}'.format(model2.intercept_))  # w0: intercept
print('coef: w1={}'.format(model2.coef_))  # w1,..wm: coeff

# eval
print('R2：{:.4f}'.format(model2.score(X2, y2)))
print('MSE：{:.4f}'.format(mean_squared_error(y2, y_pred2)))
print('MAE：{:.4f}'.format(mean_absolute_error(y2, y_pred2))) 

from scipy import stats
n= len(X2)
Regression = sum((y_pred2 - np.mean(y2))**2)
Residual = sum((y2 - y_pred2)**2)
F= (Regression / 1) / (Residual / ( n - 2 ))  # F
pf = stats.f.sf(F, 1, n-2)
print(f'p-value = {pf:.4f}')

# compare
surpass_sigma = abs(y_diff)-0.5*y2.std()
surpass_signal = pd.DataFrame(surpass_sigma>0)
surpass_signal.describe()

surpass_signal[surpass_signal==True] = -1
surpass_signal[surpass_signal==False] = 1
surpass_signal = pd.DataFrame(surpass_signal)
y2 = pd.DataFrame(y2)
rtn_dataset = y2.merge(surpass_signal, how='outer', left_index=True, right_index=True)

# compare
y2 = y2/y2.shift(1)-1
# 1. baseline
sum1 = y2.sum()
rtn_dataset['CNH_fwd_x'] = rtn_dataset['CNH_fwd_x']/rtn_dataset['CNH_fwd_x'].shift(1)-1
rtn_dataset['rtn_strat'] = rtn_dataset['CNH_fwd_x']*rtn_dataset['CNH_fwd_y']
sum2 = rtn_dataset['rtn_strat'].sum()

