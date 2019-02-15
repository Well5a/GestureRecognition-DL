import numpy as np
import pandas as pd
from PIL import Image # used for loading images
from matplotlib import pyplot
import os # used for navigating to image path
import imageio # used for writing images


##
#   Shows statistics about the images of the data set
#   e.g. average size
##

encoderPath             = 'DeepLearningModel\\Encoder\\oneHotEncoder.pkl'
videoDataPath           = 'F:\\Datasets\\20bn-jester-v1\\'
trainMetaDataPath       = 'F:\\Datasets\\jester-v1-train.csv'
validateMetaDataPath    = 'F:\\Datasets\\jester-v1-validation.csv'

DIR = 'F:\\Datasets\\20bn-jester-v1'
IMG_HEIGHT = 100
IMG_WIDTH = 150

def get_size_statistics():
    heights = []
    widths = []

    for i in range(1, 500):
        directory = DIR + '\\' + str(i)
        print('reading directory #', i)

        for img in os.listdir(directory): 
            path = os.path.join(directory, img)
            data = np.array(Image.open(path)) #PIL Image library
            heights.append(data.shape[0])
            widths.append(data.shape[1])

    avg_height = sum(heights) / len(heights)
    avg_width = sum(widths) / len(widths)
    print("Average Height: " + str(avg_height))
    print("Max Height: " + str(max(heights)))
    print("Min Height: " + str(min(heights)))
    print('\n')
    print("Average Width: " + str(avg_width))
    print("Max Width: " + str(max(widths)))
    print("Min Width: " + str(min(widths)))


def load_training_data(id):
    train_data = []
    directory = DIR + '\\' + str(id)

    for img in os.listdir(directory):
        label = img
        path = os.path.join(directory, img)
        img = Image.open(path)
        img = img.convert('L')
        img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.ANTIALIAS)
        train_data.append([np.array(img), label])

        # Basic Data Augmentation - Horizontal Flipping
        # flip_img = Image.open(path)
        # flip_img = flip_img.convert('L')
        # flip_img = flip_img.resize((IMG_WIDTH, IMG_HEIGHT), Image.ANTIALIAS)
        # flip_img = np.array(flip_img)
        # flip_img = np.fliplr(flip_img)
        # train_data.append([flip_img, label])

    return train_data

def getFrameNumbers():
    maxFrames = 0
    minFrames = 40
    avgFrames = 0

    for i in range(1, 1000):
        directory = DIR + '\\' + str(i)
        current = len(os.listdir(directory))
        avgFrames += current
        if current > maxFrames:
            maxFrames = current
        elif current < minFrames:
            minFrames = current

    avgFrames /= 1000
    print('maxFrames: %d | minFrames: %d | avgFrames: %d' % (maxFrames, minFrames, avgFrames))

# videos that are longer than a certain number of frames
def getUsableVideos(path, limit):
    metaData = pd.read_csv(path, delimiter=';', header=None, index_col=0, names=['gesture'])
    videoIds = metaData.index.values
    nVideos = videoIds.shape[0]
    nVideosOverFrameLimit = 0
    
    for counterVideo, videoId in enumerate(videoIds, 1):
        print('testing video #', counterVideo, ' /', nVideos)
        directory = os.path.join(videoDataPath, str(videoId))
        nFrames = len(os.listdir(directory))
        if nFrames >= limit:
            nVideosOverFrameLimit += 1

    print("usabel videos over limit %d for %s: %d" % (limit, path, nVideosOverFrameLimit))



getUsableVideos(trainMetaDataPath, 30)

#get_size_statistics()

# train_data = load_training_data(13)

# print(train_data[20][0])

# pyplot.imshow(train_data[20][0], cmap = 'gist_gray')
# pyplot.show()

