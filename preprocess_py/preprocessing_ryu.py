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
##데이터 불러오기
df_1_1 = pd.read_csv('rawdata/sensor1/9.21~9.24.csv')
df_1_2 = pd.read_csv('rawdata/sensor1/09.25~09.28.csv')
df_1_3 = pd.read_csv('rawdata/sensor1/sensor1_20200929.csv')
df_1_4 = pd.read_csv('rawdata/sensor1/sensor1_20200930.csv')
df_1_5 = pd.read_csv('rawdata/sensor1/sensor1_20201001.csv')
df_1_6 = pd.read_csv('rawdata/sensor1/sensor1_20201002.csv')
df_1_7 = pd.read_csv('rawdata/sensor1/sensor1_20201003.csv')
df_1_8 = pd.read_csv('rawdata/sensor1/sensor1_20201004.csv')
df_1_9 = pd.read_csv('rawdata/sensor1/sensor1_20201005.csv')
df_1_10 = pd.read_csv('rawdata/sensor1/20201006.csv')
df_1_11 = pd.read_csv('rawdata/sensor1/20201007.csv')
df_1_12 = pd.read_csv('rawdata/sensor1/9.7~9.14.csv')

df_2_1 = pd.read_csv('rawdata/sensor2/09.21~09.25.csv')
df_2_2 = pd.read_csv('rawdata/sensor2/20200926.csv')
df_2_3 = pd.read_csv('rawdata/sensor2/20200927.csv')
df_2_4 = pd.read_csv('rawdata/sensor2/20200928.csv')
df_2_5 = pd.read_csv('rawdata/sensor2/20200929.csv')
df_2_6 = pd.read_csv('rawdata/sensor2/20200930.csv')
df_2_7 = pd.read_csv('rawdata/sensor2/20201001.csv')
df_2_8 = pd.read_csv('rawdata/sensor2/20201002.csv')
df_2_9 = pd.read_csv('rawdata/sensor2/20201003.csv')
df_2_10 = pd.read_csv('rawdata/sensor2/20201004.csv')
df_2_11 = pd.read_csv('rawdata/sensor2/20201005.csv')
df_2_12 = pd.read_csv('rawdata/sensor2/20201006.csv')
df_2_13 = pd.read_csv('rawdata/sensor2/20201007.csv')
df_2_14 = pd.read_csv('rawdata/sensor2/9.10~9.13.csv')
#%%
###데이터 합치기
sensor_1 = pd.DataFrame()
sensor_2 = pd.DataFrame()
for i in range (1,13):
    sensor_1 = pd.concat([sensor_1, globals()["df_1_{}".format(i)]])

for i in range (1,15):
    sensor_2 = pd.concat([sensor_2, globals()["df_2_{}".format(i)]])


sensor_1 = sensor_1[['Date', 'Time', 'MSE', 'SSIM', 'Sound', 'Radar', 'PIR', 'NoP']]
sensor_2 = sensor_2[['Date', 'Time', 'MSE', 'SSIM', 'Sound', 'Radar', 'PIR', 'NoP']]

#%%
sensor_1.to_csv("raw1.csv", encoding="utf-8-sig")
sensor_2.to_csv("raw2.csv", encoding="utf-8-sig")

print(len(sensor_1))
#%%

#%%
def nop_n(x):
    if x > 2:
        return 2
    else:
        return x

def ssim_1(x):
    return round((1-float(x)) ,2)

def log_sound(x):
    if int(x) <=0:
        return 0
    else:
        return round(math.log(int(x)),2)

def mse(x):
    return round(float(x),2)

def pirradar(x):
    return int(x)
#%%
### 시간데이터 만들기

sensor_1 = sensor_1.dropna()
sensor_2 = sensor_2.dropna()

sensor_1["NoP"] = sensor_1["NoP"].apply(nop_n)
sensor_2["NoP"] = sensor_2["NoP"].apply(nop_n)

sensor_1["NoP"] = sensor_1["NoP"].astype(int)
sensor_2["NoP"] = sensor_2["NoP"].astype(int)

sensor_1["mergedt"] = sensor_1["Date"] +" " + sensor_1["Time"]
sensor_1["mergedt"] = pd.to_datetime(sensor_1["mergedt"])
sensor_2["mergedt"] = sensor_2["Date"] +" " + sensor_2["Time"]
sensor_2["mergedt"] = pd.to_datetime(sensor_2["mergedt"])

sensor_1["hour"] = sensor_1["mergedt"].dt.hour
sensor_2["hour"] = sensor_2["mergedt"].dt.hour

sensor_1["day"] = sensor_1["mergedt"].dt.day
sensor_2["day"] = sensor_2["mergedt"].dt.day

sensor_1["min"] = sensor_1["mergedt"].dt.strftime("%H:%M")
sensor_2["min"] = sensor_2["mergedt"].dt.strftime("%H:%M")

sensor_1["md"] = sensor_1["mergedt"].dt.strftime("%m/%d")
sensor_2["md"] = sensor_2["mergedt"].dt.strftime("%m/%d")

sensor_1["weekday"] = sensor_1["mergedt"].dt.weekday
sensor_2["weekday"] = sensor_2["mergedt"].dt.weekday

sensor_1.sort_values(by="mergedt", inplace=True)
sensor_2.sort_values(by="mergedt", inplace=True)
sensor_1.reset_index(drop=True, inplace=True)
sensor_2.reset_index(drop=True, inplace=True)
sensor_1

#%%

plt.figure(figsize = (20,10))
sensor_1_graph_NoP_Hour = sns.countplot(x = "md", data = sensor_1)
fig_1 = sensor_1_graph_NoP_Hour.get_figure()
fig_1.savefig('./graph/sensor_1_NoP_Hour.png', dpi=300)

plt.figure(figsize = (20,10))
sensor_2_graph_NoP_Hour = sns.countplot(x = "md", data = sensor_2)
fig_2 = sensor_2_graph_NoP_Hour.get_figure()
fig_2.savefig('./graph/sensor_2_NoP_Hour.png', dpi=300)

#%%
print(sensor_1.count())

print(sensor_1["Date"].value_counts())
print(sensor_2.count())
print(sensor_2["Date"].value_counts())
#%%
### 위의 함수 적용하기
sensor_1_pre = sensor_1.copy()
sensor_1_pre["SSIM"] = sensor_1["SSIM"].apply(ssim_1)
sensor_1_pre["MSE"] = sensor_1["MSE"].apply(mse)
sensor_1_pre["Sound"] = sensor_1["Sound"].apply(log_sound)

sensor_2_pre = sensor_2.copy()
sensor_2_pre["SSIM"] = sensor_2["SSIM"].apply(ssim_1)
sensor_2_pre["MSE"] = sensor_2["MSE"].apply(mse)
sensor_2_pre["Sound"] = sensor_2["Sound"].apply(log_sound)

sensor_1_pre.reset_index(inplace=True, drop=True)
sensor_2_pre.reset_index(inplace=True, drop=True)
# %%
### new column list
l_mse_diff_10 = ['None', 'None', 'None', 'None', 'None']
l_mse_diff_20 = ['None', 'None', 'None', 'None', 'None']
l_mse_diff_30 = ['None', 'None', 'None', 'None', 'None']

l_ssim_diff_10 = ['None', 'None', 'None', 'None', 'None']
l_ssim_diff_20 = ['None', 'None', 'None', 'None', 'None']
l_ssim_diff_30 = ['None', 'None', 'None', 'None', 'None']

l_sound_diff_10 = ['None', 'None', 'None', 'None', 'None']
l_sound_diff_20 = ['None', 'None', 'None', 'None', 'None']
l_sound_diff_30 = ['None', 'None', 'None', 'None', 'None']

l_radar_diff_10 = ['None', 'None', 'None', 'None', 'None']
l_radar_diff_20 = ['None', 'None', 'None', 'None', 'None']
l_radar_diff_30 = ['None', 'None', 'None', 'None', 'None']

l_pir_diff_10 = ['None', 'None', 'None', 'None', 'None']
l_pir_diff_20 = ['None', 'None', 'None', 'None', 'None']
l_pir_diff_30 = ['None', 'None', 'None', 'None', 'None']
#%%
# update previous sensor info
for i in range(5, len(sensor_1_pre)):

    l_mse_diff_10.append(sensor_1_pre['MSE'][i-1])
    l_mse_diff_20.append(sensor_1_pre['MSE'][i-2])
    l_mse_diff_30.append(sensor_1_pre['MSE'][i-3])

    l_ssim_diff_10.append(sensor_1_pre['SSIM'][i-1])
    l_ssim_diff_20.append(sensor_1_pre['SSIM'][i-2])
    l_ssim_diff_30.append(sensor_1_pre['SSIM'][i-3])

    l_sound_diff_10.append(sensor_1_pre['Sound'][i-1])
    l_sound_diff_20.append(sensor_1_pre['Sound'][i-2])
    l_sound_diff_30.append(sensor_1_pre['Sound'][i-3])

    l_radar_diff_10.append(sensor_1_pre['Radar'][i-1])
    l_radar_diff_20.append(sensor_1_pre['Radar'][i-2])
    l_radar_diff_30.append(sensor_1_pre['Radar'][i-3])

    l_pir_diff_10.append(sensor_1_pre['PIR'][i-1])
    l_pir_diff_20.append(sensor_1_pre['PIR'][i-2])
    l_pir_diff_30.append(sensor_1_pre['PIR'][i-3])

# %%
### add t-10, t-20, t-30 columns to dataframe for 
### Moving Average calculation
mse_pre = pd.DataFrame()
mse_pre['MSE_10'] = l_mse_diff_10
mse_pre['MSE_20'] = l_mse_diff_20
mse_pre['MSE_30'] = l_mse_diff_30
sensor_1_pre['MSE_10'] = l_mse_diff_10
sensor_1_pre['MSE_20'] = l_mse_diff_20
sensor_1_pre['MSE_30'] = l_mse_diff_30
sensor_1_pre['MSE_MA'] = mse_pre[5:].mean(axis=1)

ssim_pre = pd.DataFrame()
ssim_pre['SSIM_10'] = l_ssim_diff_10
ssim_pre['SSIM_20'] = l_ssim_diff_20
ssim_pre['SSIM_30'] = l_ssim_diff_30
sensor_1_pre['SSIM_10'] = l_ssim_diff_10
sensor_1_pre['SSIM_20'] = l_ssim_diff_20
sensor_1_pre['SSIM_30'] = l_ssim_diff_30
sensor_1_pre['SSIM_MA'] = ssim_pre[5:].mean(axis=1)

sound_pre = pd.DataFrame()
sound_pre['Sound_10'] = l_sound_diff_10
sound_pre['Sound_20'] = l_sound_diff_20
sound_pre['Sound_30'] = l_sound_diff_30
sensor_1_pre['Sound_10'] = l_sound_diff_10
sensor_1_pre['Sound_20'] = l_sound_diff_20
sensor_1_pre['Sound_30'] = l_sound_diff_30
sensor_1_pre['Sound_MA'] = sound_pre[5:].mean(axis=1)

radar_pre = pd.DataFrame()
radar_pre['Radar_10'] = l_radar_diff_10
radar_pre['Radar_20'] = l_radar_diff_20
radar_pre['Radar_30'] = l_radar_diff_30
sensor_1_pre['Radar_10'] = l_radar_diff_10
sensor_1_pre['Radar_20'] = l_radar_diff_20
sensor_1_pre['Radar_30'] = l_radar_diff_30
sensor_1_pre['Radar_MA'] = radar_pre[5:].mean(axis=1)

pir_pre = pd.DataFrame()
pir_pre['PIR_10'] = l_pir_diff_10
pir_pre['PIR_20'] = l_pir_diff_20
pir_pre['PIR_30'] = l_pir_diff_30
sensor_1_pre['PIR_10'] = l_pir_diff_10
sensor_1_pre['PIR_20'] = l_pir_diff_20
sensor_1_pre['PIR_30'] = l_pir_diff_30
sensor_1_pre['PIR_MA'] = pir_pre[5:].mean(axis=1)

#%%
### 2번 센서 ma
l_mse_diff_10 = ['None', 'None', 'None', 'None', 'None']
l_mse_diff_20 = ['None', 'None', 'None', 'None', 'None']
l_mse_diff_30 = ['None', 'None', 'None', 'None', 'None']

l_ssim_diff_10 = ['None', 'None', 'None', 'None', 'None']
l_ssim_diff_20 = ['None', 'None', 'None', 'None', 'None']
l_ssim_diff_30 = ['None', 'None', 'None', 'None', 'None']

l_sound_diff_10 = ['None', 'None', 'None', 'None', 'None']
l_sound_diff_20 = ['None', 'None', 'None', 'None', 'None']
l_sound_diff_30 = ['None', 'None', 'None', 'None', 'None']

l_radar_diff_10 = ['None', 'None', 'None', 'None', 'None']
l_radar_diff_20 = ['None', 'None', 'None', 'None', 'None']
l_radar_diff_30 = ['None', 'None', 'None', 'None', 'None']

l_pir_diff_10 = ['None', 'None', 'None', 'None', 'None']
l_pir_diff_20 = ['None', 'None', 'None', 'None', 'None']
l_pir_diff_30 = ['None', 'None', 'None', 'None', 'None']
#%%
# update previous sensor info
for i in range(5, len(sensor_2_pre)):

    l_mse_diff_10.append(sensor_2_pre['MSE'][i-1])
    l_mse_diff_20.append(sensor_2_pre['MSE'][i-2])
    l_mse_diff_30.append(sensor_2_pre['MSE'][i-3])

    l_ssim_diff_10.append(sensor_2_pre['SSIM'][i-1])
    l_ssim_diff_20.append(sensor_2_pre['SSIM'][i-2])
    l_ssim_diff_30.append(sensor_2_pre['SSIM'][i-3])

    l_sound_diff_10.append(sensor_2_pre['Sound'][i-1])
    l_sound_diff_20.append(sensor_2_pre['Sound'][i-2])
    l_sound_diff_30.append(sensor_2_pre['Sound'][i-3])

    l_radar_diff_10.append(sensor_2_pre['Radar'][i-1])
    l_radar_diff_20.append(sensor_2_pre['Radar'][i-2])
    l_radar_diff_30.append(sensor_2_pre['Radar'][i-3])

    l_pir_diff_10.append(sensor_2_pre['PIR'][i-1])
    l_pir_diff_20.append(sensor_2_pre['PIR'][i-2])
    l_pir_diff_30.append(sensor_2_pre['PIR'][i-3])

# %%
### add t-10, t-20, t-30 columns to dataframe for 
### Moving Average calculation
mse_pre = pd.DataFrame()
mse_pre['MSE_10'] = l_mse_diff_10
mse_pre['MSE_20'] = l_mse_diff_20
mse_pre['MSE_30'] = l_mse_diff_30
sensor_2_pre['MSE_10'] = l_mse_diff_10
sensor_2_pre['MSE_20'] = l_mse_diff_20
sensor_2_pre['MSE_30'] = l_mse_diff_30
sensor_2_pre['MSE_MA'] = mse_pre[5:].mean(axis=1)

ssim_pre = pd.DataFrame()
ssim_pre['SSIM_10'] = l_ssim_diff_10
ssim_pre['SSIM_20'] = l_ssim_diff_20
ssim_pre['SSIM_30'] = l_ssim_diff_30
sensor_2_pre['SSIM_10'] = l_ssim_diff_10
sensor_2_pre['SSIM_20'] = l_ssim_diff_20
sensor_2_pre['SSIM_30'] = l_ssim_diff_30
sensor_2_pre['SSIM_MA'] = ssim_pre[5:].mean(axis=1)

sound_pre = pd.DataFrame()
sound_pre['Sound_10'] = l_sound_diff_10
sound_pre['Sound_20'] = l_sound_diff_20
sound_pre['Sound_30'] = l_sound_diff_30
sensor_2_pre['Sound_10'] = l_sound_diff_10
sensor_2_pre['Sound_20'] = l_sound_diff_20
sensor_2_pre['Sound_30'] = l_sound_diff_30
sensor_2_pre['Sound_MA'] = sound_pre[5:].mean(axis=1)

radar_pre = pd.DataFrame()
radar_pre['Radar_10'] = l_radar_diff_10
radar_pre['Radar_20'] = l_radar_diff_20
radar_pre['Radar_30'] = l_radar_diff_30
sensor_2_pre['Radar_10'] = l_radar_diff_10
sensor_2_pre['Radar_20'] = l_radar_diff_20
sensor_2_pre['Radar_30'] = l_radar_diff_30
sensor_2_pre['Radar_MA'] = radar_pre[5:].mean(axis=1)

pir_pre = pd.DataFrame()
pir_pre['PIR_10'] = l_pir_diff_10
pir_pre['PIR_20'] = l_pir_diff_20
pir_pre['PIR_30'] = l_pir_diff_30
sensor_2_pre['PIR_10'] = l_pir_diff_10
sensor_2_pre['PIR_20'] = l_pir_diff_20
sensor_2_pre['PIR_30'] = l_pir_diff_30
sensor_2_pre['PIR_MA'] = pir_pre[5:].mean(axis=1)

#%%

sensor_1_pre.dropna(inplace=True)
sensor_2_pre.dropna(inplace=True)
sensor_1_pre.sort_values(by="mergedt")
sensor_2_pre.sort_values(by="mergedt")

print("'#1 리빙랩' 데이터기간 : " + str(sensor_1_pre["mergedt"].min()) +" ~ " +str(sensor_1_pre["mergedt"].max()))
print("'#2 공용주방' 데이터기간 : " + str(sensor_2_pre["mergedt"].min()) +" ~ " +str(sensor_2_pre["mergedt"].max()))
print("\n'#1 리빙랩' NoP" )
print(sensor_1_pre["NoP"].value_counts())
print("\n'#2 공용주방' NoP" )
print(sensor_2_pre["NoP"].value_counts())
#%%
### Raw data 저장하기
sensor_1_pre.to_csv("rawdata_1.csv", encoding = "utf-8-sig")
sensor_2_pre.to_csv("rawdata_2.csv", encoding = "utf-8-sig")

# %%
### 8시 ~ 24시까지 데이터, 평일 데이터만
### 9월 21일~10월 2일: 트레이닝
### 10월 5일 ~ 10월 7일 : 검증

sensor_1_pre = sensor_1_pre[(sensor_1_pre["hour"]>=8) & (sensor_1_pre["hour"]<=24)]
sensor_2_pre = sensor_2_pre[(sensor_2_pre["hour"]>=8) & (sensor_2_pre["hour"]<=24)]

sensor_1_pre = sensor_1_pre[sensor_1_pre["weekday"]<=4]
sensor_2_pre = sensor_2_pre[sensor_2_pre["weekday"]<=4]

sensor_1_training = sensor_1_pre[(sensor_1_pre["day"]>=21) | (sensor_1_pre["day"]==1) | (sensor_1_pre["day"]==2)]
sensor_2_training = sensor_2_pre[(sensor_2_pre["day"]>=21) | (sensor_2_pre["day"]==1) | (sensor_2_pre["day"]==2)]
sensor_1_validation = sensor_1_pre[(sensor_1_pre["day"]<=7) & (sensor_1_pre["day"]>=5)]
sensor_2_validation = sensor_2_pre[(sensor_2_pre["day"]<=7) & (sensor_2_pre["day"]>=5)]
sensor_training = pd.concat([sensor_1_training, sensor_2_training])
sensor_training.reset_index(drop=True, inplace=True)
sensor_validation = pd.concat([sensor_1_validation, sensor_2_validation])
sensor_validation.reset_index(drop=True, inplace=True)
# %%
### csv 저장
sensor_1_training.to_csv('sensor_1_training_all.csv', encoding = "utf-8-sig")
sensor_2_training.to_csv('sensor_2_training_all.csv', encoding = "utf-8-sig")

sensor_training.to_csv('sensor_trainig_all.csv', encoding = 'utf-8-sig')

# %%
sensor_2_validation.to_csv('sensor_2_validation_all.csv', encoding = "utf-8-sig")
sensor_1_validation.to_csv('sensor_1_validation_all.csv', encoding = "utf-8-sig")
sensor_validation.to_csv('sensor_validation_all.csv', encoding = "utf-8-sig")
# %%
