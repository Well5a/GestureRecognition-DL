import numpy as np
import pandas as pd
from PIL import Image as PIL_Image# used for loading images
import os # used for navigating to image path
import imageio # used for writing images


trainLabelPath = 'DeepLearningModel\\Dataset\\train_label.pkl'
trainFeaturePath = 'DeepLearningModel\\Dataset\\train_feature.npy'


# train_label = pd.read_pickle(trainLabelPath)

# print(train_label)

train_features = np.load(trainFeaturePath)
# print(train_features)

print('0', train_features[0])
print('1', train_features[0][0])
print('2', train_features[0][0][0])