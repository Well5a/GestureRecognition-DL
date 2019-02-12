import numpy as np
import pandas as pd
import os
from PIL import Image as PIL_Image
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib


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

restrictImports = 100 # restricts video import to value (set to high value for all videos >1Mio)

# Load Paths
encoderPath             = 'DeepLearningModel\\Encoder\\oneHotEncoder.pkl'
videoDataPath           = 'F:\\Datasets\\20bn-jester-v1\\'
trainMetaDataPath       = 'F:\\Datasets\\jester-v1-train.csv'
validateMetaDataPath    = 'F:\\Datasets\\jester-v1-validation.csv'

# Save Paths
trainLabelPath      = 'F:\\Datasets\\Preprocessed\\train_label.pkl'
trainFeaturePath    = 'F:\\Datasets\\Preprocessed\\train_feature.npy'
validateLabelPath   = 'F:\\Datasets\\Preprocessed\\validate_label.pkl'
validateFeaturePath = 'F:\\Datasets\\Preprocessed\\validate_feature.npy'

ONEHOT_ENCODER = joblib.load(encoderPath)
IMG_HEIGHT = 100
IMG_WIDTH = 150
NUM_FRAMES = 30


def load_videos(videoIds):
    video_set = []
    nVideos = videoIds.shape[0]

    for counterVideo, videoId in enumerate(videoIds, 1):
        if len(video_set) == restrictImports:
            break

        print('importing video #', counterVideo, ' /', nVideos)
        directory = os.path.join(videoDataPath, str(videoId))

        if len(os.listdir(directory)) >= NUM_FRAMES
    : # don't use if video length is too small        
            video = []    
            for counterImage, image in enumerate(os.listdir(directory), 1):   
                if counterImage > NUM_FRAMES
            : break             
                image_values = PIL_Image.open(os.path.join(directory, image))
                image_values = image_values.convert('L') # L: converts to greyscale
                image_values = image_values.resize((IMG_WIDTH, IMG_HEIGHT), PIL_Image.ANTIALIAS)
                video.append(np.asarray(image_values))

            video_set.append([np.asarray(video), videoId])
        else:
            print("skipped video # %d with length %d" % (counterVideo, len(os.listdir(directory))))
        
    return np.asarray(video_set) 

def createDataset(metaDataPath, labelPath, featurePath):
    metaData = pd.read_csv(metaDataPath, delimiter=';', header=None, index_col=0, names=['gesture'])

    label_set = pd.DataFrame(index=metaData.index.values, data=ONEHOT_ENCODER.transform(metaData.values))
    label_set.to_pickle(labelPath)
    print("saved labels to ", labelPath)

    feature_set = load_videos(metaData.index.values)
    np.save(featurePath, feature_set)
    print("saved features to ", featurePath)


createDataset(trainMetaDataPath, trainLabelPath, trainFeaturePath)
createDataset(validateMetaDataPath, validateLabelPath, validateFeaturePath)