#%%
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from datetime import date as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import missingno as msno
from pylab import legend
import math

#%%
df = pd.read_csv('sensor_2_training_all.csv')

df = df.iloc[:,2:]
df.columns
#%%
df = df[['hour', 'day', 'weekday', 'MSE', 'SSIM', 'Sound', 'Radar', 'PIR', 'MSE_10', 'MSE_20', 'MSE_30', 'MSE_MA',
       'SSIM_10', 'SSIM_20', 'SSIM_30', 'SSIM_MA', 'Sound_10', 'Sound_20',
       'Sound_30', 'Sound_MA', 'Radar_10', 'Radar_20', 'Radar_30', 'Radar_MA',
       'PIR_10', 'PIR_20', 'PIR_30', 'PIR_MA', 'NoP']]

df
#%%
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

#%%
smote = SMOTE(random_state=0)
X_train_over,y_train_over = smote.fit_sample(X,y)


# %%
df_1 = pd.concat([X_train_over,y_train_over], axis=1)
df_1.to_csv('smotesamplingdata_2.csv', encoding='utf-8-sig')
# %%
df_1["day"].value_counts()
# %%
