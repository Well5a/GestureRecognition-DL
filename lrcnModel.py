import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing import sequence
from keras.preprocessing.image import load_img, img_to_array
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.recurrent import LSTM

# Paths
encoderPath             = 'DeepLearningModel\\Encoder\\oneHotEncoder.pkl'
videoDataPath           = 'F:\\Datasets\\20bn-jester-v1\\'
trainMetaDataPath       = 'F:\\Datasets\\jester-v1-train.csv'
validateMetaDataPath    = 'F:\\Datasets\\jester-v1-validation.csv'

# data params
IMG_HEIGHT = 100
IMG_WIDTH = 150
NUM_FRAMES = 30
NUM_CLASSES = 27
NUM_TRAIN_SAMPLES = 116254
NUM_VALIDATION_SAMPLES = 0

# model params
epochs = 0
batchsize = 32
num_train_batches_per_epoch = int(NUM_TRAIN_SAMPLES / batchsize)
num_validation_batches_per_epoch = int(NUM_VALIDATION_SAMPLES / batchsize)


def data_generator(featurePath, labelPath, batchsize, labelEncoder):
    metaData = pd.read_csv(labelPath, delimiter=';', header=None, index_col=0, names=['gesture'])
    videoIDIter = iter(metaData.index.values)
    
    while True:
        features = []
        labels = []               
        
        while len(labels) < batchsize:  
            videoId = next(videoIdsIter)
            directory = os.path.join(featurePath, str(videoId))
            if len(os.listdir(directory)) >= NUM_FRAMES: # don't use if video length is too small        
                video = []    
                for imageCounter, image in enumerate(os.listdir(directory), 1):   
                    if imageCounter > NUM_FRAMES: break # break if video length is too big        
                    image_values = PIL_Image.open(os.path.join(directory, image))
                    image_values = image_values.convert('L') # L: converts to greyscale
                    image_values = image_values.resize((IMG_WIDTH, IMG_HEIGHT), PIL_Image.ANTIALIAS)
                    video.append(np.asarray(image_values))

                features.append(np.asarray(video))
                labels.append(metaData.at[videoId, 'gesture'])
        labels = labelEncoder.transform(np.asarray(labels))

        yield(np.asarray(features, labels))

def build_model():
    model = Sequential()
    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same'), input_shape=(NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH, 1)))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(20, return_sequences=True))
    model.add(TimeDistributed(Dense(NUM_CLASSES)))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

#Initialization
oneHotEncoder = joblib.load(encoderPath)
model = build_model()
num_train_batches_per_epoch = int(NUM_TRAIN_SAMPLES / batchsize)
num_validation_batches_per_epoch = int(NUM_VALIDATION_SAMPLES / batchsize)

# Data
train_batches = data_generator(videoDataPath, trainMetaDataPath, batchsize, oneHotEncoder)
validation_batches = data_generator(videoDataPath, validateMetaDataPath, batchsize, oneHotEncoder)

# Train
# history = model.fit_generator(train_batches, steps_per_epoch=num_train_batches_per_epoch, epochs=epochs, verbose=1, validation_data=validation_batches, validation_steps=num_validation_batches_per_epoch)

# print_result()

print('done')