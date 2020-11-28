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
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

#%%
# Load dataset.
df= pd.read_csv('data/정리/validation.csv')
df.head()

# %%
# prepare test input
predictors = ['MSE', 'SSIM', 'Sound', 'Radar', 'PIR', 
        'MSE_MA', 'SSIM_MA', 'Sound_MA',  'Radar_MA',  'PIR_MA']
# %%
# load model, scaler & create test input
model =keras.models.load_model('model_2/prehension_v3_minmax_10_outlier')
scaler = joblib.load('model_2/scaler_v3_minmax_outlier.save')

#%%
testinput = df[predictors].values
# testinput = scaler.transform(testinput)
# %%
# run by model, check total accuracy
answers = to_categorical(df['NoP'].values, 3)

#%%
pred_test = model.predict(testinput)
probs = model.predict(testinput)
scores2 = model.evaluate(testinput, answers, verbose=0)
print('Accuracy on test data: {}% \n Error on test data: {}'.format(
    scores2[1], 1 - scores2[1]))

# %%
# labeled vs. predicted comparison
result_class = pred_test.argmax(axis=-1)


checkans = pd.DataFrame()
checkans['Labeled'] = df['NoP']
checkans['Predicted'] = result_class
checkans[["Predicted_0","Predicted_1", "Predicted_2"]] = probs
checkans['Time'] = df_raw['Time']
checkans["Date"] = df_raw['Date']

# %%

checkans.to_csv('data/test.csv')
checkans
# %%
NoP = 0
precision_1 = checkans[checkans["Labeled"] == NoP]
precision_1_count = precision_1[precision_1["Predicted"] ==NoP]

print("{} 정확도 : ".format(NoP) +str(len(precision_1_count) / len(precision_1)*100))
# %%

# %%
