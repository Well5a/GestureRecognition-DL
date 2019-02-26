import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from pathlib import Path
from PIL import Image as PIL_Image
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing import sequence
from keras.preprocessing.image import load_img, img_to_array
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.layers.pooling import GlobalAveragePooling1D
from keras.utils import plot_model

encoderPath         = Path('DeepLearningModel/Encoder/oneHotEncoder.pkl')
videoDataPath       = Path('F:/Datasets/20bn-jester-v1/')
testMetaDataPath    = Path('F:/Datasets/jester-v1-test.csv')
modelSavePath       = Path('DeepLearningModel/Model/model.h5')

model.load(modelSavePath)
