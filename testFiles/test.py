import pandas as pd
import numpy as np
from keras.models import load_model
from PIL import Image as PIL_Image# used for loading images
import os # used for navigating to image path
import imageio # used for writing images

trainLabelPath      = 'F:\\Datasets\\Preprocessed\\train_label.pkl'
trainFeaturePath    = 'F:\\Datasets\\Preprocessed\\train_feature.npy'
validateLabelPath   = 'F:\\Datasets\\Preprocessed\\validate_label.pkl'
validateFeaturePath = 'F:\\Datasets\\Preprocessed\\validate_feature.npy'

# Load Paths
encoderPath             = 'DeepLearningModel\\Encoder\\oneHotEncoder.pkl'
videoDataPath           = 'F:\\Datasets\\20bn-jester-v1\\'
trainMetaDataPath       = 'F:\\Datasets\\jester-v1-train.csv'
validateMetaDataPath    = 'F:\\Datasets\\jester-v1-validation.csv'

# train_label = pd.read_pickle(trainLabelPath)
# print(train_label)

# train_features = np.load(trainFeaturePath)
# validation_features = np.load(validateFeaturePath)

# print('0', train_features[0])
# print('1', train_features[0][0])
# print('2', train_features[0][0][0])

# print(train_features.shape)
# print(validation_features.shape)

# a = [[1,2,3],[4,5,6],[7,8,9]]

# indices = np.arange(2)
# print(indices)

metaData = pd.read_csv(trainMetaDataPath, delimiter=';', header=None, index_col=0, names=['gesture'])
