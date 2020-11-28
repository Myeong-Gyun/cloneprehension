# %%
# import required libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn
import joblib

# import necessary modules
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# %%
# Load dataset.
df_raw = pd.read_csv('data/sensor_sampled_ma-2020-08-25T07-49-22.611Z.csv')
df_raw.head()

# %%
# drop unused columns in dataframe
df =df_raw[['MSE', 'SSIM', 'Sound', 'Radar',"NoP", "MSE_MA", "SSIM_MA", "Sound_MA", "Radar_MA"]]
# %%
# load model
model = keras.models.load_model('model/prehension_v1')

# %%
test = df.loc[11806]
print(test['NoP'])
test.drop(['NoP'], inplace=True)
testinput = np.asarray(test).astype(np.float32)

# %%
probs = model.predict(np.array([testinput,]))
answer = probs.argmax(axis=-1)
print(probs, answer)
# %%

# %%
