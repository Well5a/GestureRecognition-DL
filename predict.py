import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
from keras.preprocessing import sequence
from keras.utils import plot_model
from keras.models import load_model

encoderPath         = Path('mwe/oneHotEncoder.pkl')
videoDataPath       = Path('dataset/20bn-jester-v1/')
testMetaDataPath    = Path('dataset/jester-v1-test.csv')
modelSavePath       = Path('mwe/model.h5')

prediction_limit = 5
batchsize = 32


def data_generator(featurePath, labelPath, batchsize, labelEncoder):
    metaData = pd.read_csv(labelPath, delimiter=';', header=None, index_col=0, names=['gesture'])
    videoIdIter = iter(metaData.index.values)

    while True:
        features = []
        labels = []

        while len(labels) < batchsize:
            videoId = next(videoIdIter)
            videoDirectory = featurePath / str(videoId)
            if len(os.listdir(videoDirectory)) >= NUM_FRAMES: # don't use if video length is too small
                video = []
                for imageCounter, image in enumerate(os.listdir(videoDirectory), 1):
                    if imageCounter > NUM_FRAMES: break # break if video length is too big
                    image_values = PIL_Image.open(os.path.join(videoDirectory, image))
                    image_values = image_values.convert('RGB') # L: converts to greyscale | RGB
                    image_values = image_values.resize((IMG_WIDTH, IMG_HEIGHT), PIL_Image.ANTIALIAS)
                    video.append(np.asarray(image_values))

                features.append(np.asarray(video))
                labels.append(metaData.at[videoId, 'gesture'])

        features = np.asarray(features)
        labels = labelEncoder.transform(np.asarray(labels).reshape(-1, 1))
        yield(features, labels)

oneHotEncoder = joblib.load(encoderPath)
model = load_model(modelSavePath)
test_batches = data_generator(videoDataPath, testMetaDataPath, batchsize, oneHotEncoder)
predict = model.predict_generator(test_batches, steps=prediciton_limit)

#toDo: Evaluation