import numpy as np
import pandas as pd
from PIL import Image as PIL_Image# used for loading images
import os # used for navigating to image path
import imageio # used for writing images

trainLabelPath      = 'F:\\Datasets\\Preprocessed\\train_label.pkl'
trainFeaturePath    = 'F:\\Datasets\\Preprocessed\\train_feature.npy'
validateLabelPath   = 'F:\\Datasets\\Preprocessed\\validate_label.pkl'
validateFeaturePath = 'F:\\Datasets\\Preprocessed\\validate_feature.npy'


# train_label = pd.read_pickle(trainLabelPath)
# print(train_label)

train_features = np.load(trainFeaturePath)
validation_Features = np.load(validateFeaturePath)

# print('0', train_features[0])
# print('1', train_features[0][0])
# print('2', train_features[0][0][0])

print(train_features.shape)
print(validation_Features.shape)