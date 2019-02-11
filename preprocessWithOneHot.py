import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from PIL import Image as PIL_Image


##
#   Creates a dataset that can be used for training from the 20BN-Jester dataset.
#   Two sets are created: a feature set and a label set, both mapped to the correspondend ID.
#   The train data is ordered on the file system.
#   Every image is located within a folder that is named after the ID of the video.
#   An external CSV file maps id to label of the video (metadata).
#   Labels are one hot encoded (creates a vector with length 27).
#   Shape of Dataset array: [[video][id/name][label]]
##


# Paths
labelListPath = 'F:\\Datasets\\jester-v1-labels.csv'
trainMetaDataPath = 'F:\\Datasets\\jester-v1-train.csv'
trainDataPath = 'F:\\Datasets\\20bn-jester-v1\\'
labelSetPath = 'DeepLearningModel\\Dataset\\label_set'
featureSetPath = 'DeepLearningModel\\Dataset\\feature_set'

# Encoders
onehot_encoder = OneHotEncoder(sparse=False)

# Image Size
IMG_HEIGHT = 100
IMG_WIDTH = 150

includeVideoID = False


def fitOneHotEncoder():
    dataset = pd.read_csv(labelListPath, header=-1)
    labels = dataset.values
    onehot_encoder.fit(labels)

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
       
        video_set.append([videoId, np.asarray(video)])
        
    return np.asarray(video_set) 


fitOneHotEncoder()

trainMetaData = pd.read_csv(trainMetaDataPath, delimiter=';', header=None, index_col=0, names=['gesture'])


train_videoIDs = trainMetaData.index.values
# print(trainMetaData)

# Labels
train_labels = trainMetaData.values
train_labels = onehot_encoder.transform(train_labels)
train_videoIDs = train_videoIDs.reshape([train_videoIDs.shape[0], 1])
print(train_videoIDs.shape)
print(train_labels.shape)
train_labels = np.concatenate((train_videoIDs, train_labels), axis=1)
train_labels = train_labels[:100] #restrict to 100 samples for test
print(train_labels)

# Features
# train_features = load_videos(train_videoIDs)




#np.save('DeepLearningModel\\Dataset\\train_set', train_set)