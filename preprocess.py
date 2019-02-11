import numpy as np
import pandas as pd
import os
from PIL import Image as PIL_Image
from sklearn.preprocessing import OneHotEncoder


##
#   Creates datasets that can be used for training and validating the 20BN-Jester dataset.
#   Two sets are created each: a feature set and a label set, both mapped to the correspondend ID of the sample.
#   The features are saved as a numpy array, the labels as a pandas dataframe.
#   The data is ordered on the file system in folders named after the sample ID.
#   An external CSV file maps id to label of the video (metadata).
#   Labels are one hot encoded, images are greyscaled and resized to a specified size.
#   Usage of feature set: set[0] -> first sample video data and ID; 
#   set[0][0] -> first sample video data; set[0][1] -> first sample video ID
#   set[0][0][0] -> first image/frame of first sample video
##


restrictImports = True # restricts video import to 100 videos 

# Load Paths
videoDataPath           = 'F:\\Datasets\\20bn-jester-v1\\'
labelListPath           = 'F:\\Datasets\\jester-v1-labels.csv'
trainMetaDataPath       = 'F:\\Datasets\\jester-v1-train.csv'
validateMetaDataPath    = 'F:\\Datasets\\jester-v1-validation.csv'

# Save Paths
trainLabelPath      = 'F:\\Datasets\\Preprocessed\\train_label.pkl'
trainFeaturePath    = 'F:\\Datasets\\Preprocessed\\train_feature.npy'
validateLabelPath   = 'F:\\Datasets\\Preprocessed\\validate_label.pkl'
validateFeaturePath = 'F:\\Datasets\\Preprocessed\\validate_feature.npy'

onehot_encoder = OneHotEncoder(sparse=False)
IMG_HEIGHT = 100
IMG_WIDTH = 150


def fitOneHotEncoder():
    dataset = pd.read_csv(labelListPath, header=-1)
    onehot_encoder.fit(dataset.values)

def load_Metadata(path):
    return pd.read_csv(path, delimiter=';', header=None, index_col=0, names=['gesture'])

def load_videos(videoIds):
    video_set = []
    nVideos = videoIds.shape[0]

    for counter, videoId in enumerate(videoIds, 1):
        if restrictImports and counter == 100:
            break

        print('importing video #', counter, ' /', nVideos)
        directory = os.path.join(videoDataPath, str(videoId))
        video = []        

        for image in os.listdir(directory):
            image_values = PIL_Image.open(os.path.join(directory, image))
            image_values = image_values.convert('L') # L: converts to greyscale
            image_values = image_values.resize((IMG_WIDTH, IMG_HEIGHT), PIL_Image.ANTIALIAS)

            video.append(np.asarray(image_values))
       
        video_set.append([np.asarray(video), videoId])
 
    return np.asarray(video_set) 

def createDataset(metaDataPath, labelPath, featurePath):
    metaData = load_Metadata(metaDataPath)

    label_set = pd.DataFrame(index=metaData.index.values, data=onehot_encoder.transform(metaData.values))
    label_set.to_pickle(labelPath)
    print("saved labels to ", labelPath)

    feature_set = load_videos(metaData.index.values)
    np.save(featurePath, feature_set)
    print("saved features to ", featurePath)


fitOneHotEncoder()

createDataset(trainMetaDataPath, trainLabelPath, trainFeaturePath)
createDataset(validateMetaDataPath, validateLabelPath, validateFeaturePath)