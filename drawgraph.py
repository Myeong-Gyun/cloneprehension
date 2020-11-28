#%%
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
### 데이터 불러오기
sensor_1 = pd.read_csv('./processeddata/preprocessed/rawdata_1.csv')
sensor_2 = pd.read_csv('./processeddata/preprocessed/rawdata_2.csv')
#%%
### 칼럼 순서 변경
sensor_1 = sensor_1[['Date', 'Time','mergedt', 'hour', 'day', 'min', 'weekday', 'MSE', 'SSIM', 'Sound', 'Radar', 'PIR',
       'NoP', 'MSE_10', 'MSE_20', 'MSE_30', 'MSE_MA', 'SSIM_10', 'SSIM_20', 'SSIM_30', 'SSIM_MA',
       'Sound_10', 'Sound_20', 'Sound_30', 'Sound_MA', 'Radar_10', 'Radar_20',
       'Radar_30', 'Radar_MA', 'PIR_10', 'PIR_20', 'PIR_30', 'PIR_MA']]
sensor_2 = sensor_2[['Date', 'Time','mergedt', 'hour', 'day', 'min', 'weekday', 'MSE', 'SSIM', 'Sound', 'Radar', 'PIR',
       'NoP', 'MSE_10', 'MSE_20', 'MSE_30', 'MSE_MA', 'SSIM_10', 'SSIM_20', 'SSIM_30', 'SSIM_MA',
       'Sound_10', 'Sound_20', 'Sound_30', 'Sound_MA', 'Radar_10', 'Radar_20',
       'Radar_30', 'Radar_MA', 'PIR_10', 'PIR_20', 'PIR_30', 'PIR_MA']]
#%%
### 데이터 개요
print("'#1 리빙랩' 데이터기간 : " + str(sensor_1["mergedt"].min()) +" ~ " +str(sensor_1["mergedt"].max()) + ", 총 데이터 수 :"+str(len(sensor_1)))
print("'#2 공용주방' 데이터기간 : " + str(sensor_2["mergedt"].min()) +" ~ " +str(sensor_2["mergedt"].max()) + ", 총 데이터 수 :"+str(len(sensor_2)))
print("\n'#1 리빙랩' NoP" )
print(sensor_1["NoP"].value_counts())
print("\n'#2 공용주방' NoP" )
print(sensor_2["NoP"].value_counts())
#%%
### 데이터 사용 범위
sensor_1_range = sensor_1[(sensor_1["hour"]>=8) & (sensor_1["hour"]<=24)]
sensor_2_range = sensor_2[(sensor_2["hour"]>=8) & (sensor_2["hour"]<=24)]
sensor_1_range = sensor_1[(sensor_1["weekday"]<=4)]
sensor_2_range = sensor_2[(sensor_2["weekday"]<=4)]
#%%
### 데이터용 데이터 만들기
graph_sensor_1 = sensor_1_range.copy()
graph_sensor_2 = sensor_2_range.copy()
#%%
### 시간대별 NoP 그래프 
plt.figure(figsize = (20,10))
sensor_1_graph_NoP_Hour = sns.countplot(x = "hour", data = graph_sensor_1, hue = "NoP")
fig_1 = sensor_1_graph_NoP_Hour.get_figure()
fig_1.savefig('./graph/sensor_1_NoP_Hour.png', dpi=300)

plt.figure(figsize = (20,10))
sensor_2_graph_NoP_Hour = sns.countplot(x = "hour", data = graph_sensor_2, hue = "NoP")
fig_2 = sensor_2_graph_NoP_Hour.get_figure()
fig_2.savefig('./graph/sensor_2_NoP_Hour.png', dpi=300)

#%%
### 요일별로 보기
sensor_1_monday = sensor_1_range[sensor_1_range["weekday"] == 3]
sensor_2_monday = sensor_2_range[sensor_2_range["weekday"] == 3]
#%%

plt.figure(figsize = (20,10))
fig = sns.lineplot(x = "hour", y = "NoP",hue = "day", data =sensor_2_monday, palette="bright" )
fig_line = fig.get_figure()
fig_line.savefig('./graph/sensor_1_NoP_목.png', dpi=300)

# %%
### 히트맵 그리기

df_heatmap = pd.pivot_table(data = sensor_2_monday,
                            columns = 'hour',
                            values = 'NoP',
                            )
df_heatmap.sort_index(ascending=False, inplace= True)

plt.figure(figsize=(10,1))
ax = sns.heatmap(df_heatmap, cmap='Blues', vmin = 0, vmax = 2)
ax_save = ax.get_figure()
ax_save.savefig('./graph/sensor_2_heatmap_mon.png', dpi=300)
# %%
