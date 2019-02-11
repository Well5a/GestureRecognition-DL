import numpy as np
import pandas as pd
import os
from PIL import Image as PIL_Image


##
#   Creates a dataset that can be used for training from the 20BN-Jester dataset.
#   Two sets are created: a feature set and a label set, both mapped to the correspondend ID.
#   The features are saved as a numpy array, the labels as a pandas dataframe.
#   The train data is ordered on the file system.
#   Every image is located within a folder that is named after the ID of the video.
#   An external CSV file maps id to label of the video (metadata).
#   Labels are one hot encoded with pandas get_dummies function.
##


# Paths
labelListPath = 'F:\\Datasets\\jester-v1-labels.csv'
trainMetaDataPath = 'F:\\Datasets\\jester-v1-train.csv'
trainDataPath = 'F:\\Datasets\\20bn-jester-v1\\'

trainLabelPath = 'DeepLearningModel\\Dataset\\train_label.pkl'
trainFeaturePath = 'DeepLearningModel\\Dataset\\train_feature.npy'

# Image Size
IMG_HEIGHT = 100
IMG_WIDTH = 150

saveLabels = False
saveFeatures = True


def load_videos(videoIds):
    video_set = []

    i = 0 #test
    videoCount = videoIds.shape[0]

    for videoId in videoIds:
        directory = os.path.join(trainDataPath, str(videoId))

        #test
        print('importing video #', i, ' /', videoCount)
        if i == 100:
            break
        i += 1
        #test

        video = []

        for image in os.listdir(directory):
            image_values = PIL_Image.open(os.path.join(directory, image))
            image_values = image_values.convert('L') # L: converts to greyscale
            image_values = image_values.resize((IMG_WIDTH, IMG_HEIGHT), PIL_Image.ANTIALIAS)

            video.append(np.asarray(image_values))
       
        video_set.append([np.asarray(video), videoId])
 
    return np.asarray(video_set) 


trainMetaData = pd.read_csv(trainMetaDataPath, delimiter=';', header=None, index_col=0, names=['gesture'])
train_videoIDs = trainMetaData.index.values

# Labels
train_labels = pd.get_dummies(trainMetaData, prefix=['gesture'])
if(saveLabels):
    train_labels.to_pickle(trainLabelPath)
    print("saved train labels to ", trainLabelPath)

# Features
train_features = load_videos(train_videoIDs)
if(saveFeatures):
    np.save(trainFeaturePath, train_features)
    print("saved train features to ", trainFeaturePath)