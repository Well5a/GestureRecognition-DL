import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib

##
#   Defines and fits an One Hot Encoder to the categories of the dataset.
#   The encoder is then saved on the file system.
#   Load it with: encoder = joblib.load(path)
##


labelListPath = 'C:\\Users\\mwe\\Desktop\\Gesture dataset\\jester-v1-labels.csv'
encoderSavePath = 'C:\\Users\\mwe\\Documents\\GestureRecognition-DL\\Encoder\\oneHotEncoder.pkl'

dataset = pd.read_csv(labelListPath, header=-1)

onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoder.fit(dataset.values)
joblib.dump(onehot_encoder, encoderSavePath)
print("saved encoder to", encoderSavePath)