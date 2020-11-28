## local model trial -- version 1.2 with Keras
## created: 2020/10/27
## author: myoenggyun
## last edited: 2020/11/27

#%%
# import required libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn

# import necessary modules
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from math import sqrt 
from scipy.stats import zscore
import joblib

# keras specific
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# %%
# Load dataset.
df = pd.read_csv('data/정리/smote_raw.csv')
df.columns
df["NoP"].value_counts()
# %%
# set target column
target_column = ['NoP']
predictors = ['MSE', 'SSIM', 'Sound', 'Radar', 'PIR', 
        'MSE_MA', 'SSIM_MA', 'Sound_MA',  'Radar_MA',  'PIR_MA']
#%%
# 이상치 제거
## MSE 1000이상 제거 / MSE 1000이상 비율 = 1.4%
print(len(df[df["MSE"]>=3000])/len(df)*100)
df_outlier= df[(df["MSE"]<1000) & (df["MSE_MA"]<1000)]

## SSIM 0.4이상 제거 / SSIM 0.15이상 비율 = 0.17%
print(len(df[df["SSIM"]>=0.15])/len(df)*100)
df_outlier = df_outlier[(df["SSIM"]<0.15) & (df["SSIM_MA"]<0.15)]

## Radar 200이상 제거 / Radar 50이상 비율 = 0.17%
print(len(df[df["Radar"]>=50])/len(df)*100)
df_outlier = df_outlier[(df["Radar"]<50) & (df["Radar_MA"]<50)]

print("제거된 데이터수 : " +str(len(df) - len(df_outlier)) )

df = df_outlier
#%%
# 박스플랏으로 이상치 있나 확인하기
data = df["MSE"].values
plt.figure(figsize = (5,10))
plt.boxplot(data)
#%%
df["NoP"].value_counts()
# %%
# create training and testing datasets, normalize data
X = df[predictors].values
y = df[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=43)


#%%
# scaler 종류
### RobustScaler
# scaler = RobustScaler()

# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

## MinMax
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

### Normalizer
# normalizer = Normalizer().fit(X)

# X_train = normalizer.transform(X_train)
# X_test = normalizer.transform(X_test)


print(X_train.shape)
print(X_test.shape)

# %%
# one-hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

count_classes = y_test.shape[1]
print(count_classes)

#%%
nop_0 = 0
nop_1 = 0
nop_2 = 0
for i in range(0,len(y_train)):
    if y_train[i][0] == 1:
        nop_0 +=1
    elif y_train[i][1] == 1:
        nop_1 +=1
    else:
        nop_2 +=1

print(str(nop_0) + "," + str(nop_1) + "," + str(nop_2))

# %%
model = Sequential()
model.add(Dense(500, activation='relu', input_dim=10))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# %%
# fit the model
# build the model
### v3에서는 50번만 돌려도될듯
model.fit(X_train, y_train, batch_size=32,epochs=10)

# %%
# perform prediction on test data and
# compute evaluation metrics
pred_train = model.predict(X_train)
scores = model.evaluate(X_train, y_train, verbose=0)
print('Accuracy on training data: {}% \n Error on training data: {}'.format(
    scores[1], 1 - scores[1]))

pred_test = model.predict(X_test)
scores2 = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy on test data: {}% \n Error on test data: {}'.format(
    scores2[1], 1 - scores2[1]))

# %%

df_test = df[predictors]
df_test["NoP"] = df["NoP"]

test = df_test.loc[10140]
print(test['NoP'])

test.drop(['NoP'], inplace=True)
testinput_single = np.asarray(test).astype(np.float32)

# %%
probs = model.predict(np.array([testinput_single, ]))
answer = probs.argmax(axis=-1)

print(answer, probs)

# %%
# save model and scaler
model.save('model_2/prehension_v3_minmax_10_outlier')

scaler_filename = 'model_2/scaler_v3_minmax_outlier.save'
joblib.dump(scaler, scaler_filename)

# %%

# %%
