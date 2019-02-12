import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing import sequence
from keras.preprocessing.image import load_img, img_to_array
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.recurrent import LSTM

# Load Paths
encoderPath             = 'DeepLearningModel\\Encoder\\oneHotEncoder.pkl'
videoDataPath           = 'F:\\Datasets\\20bn-jester-v1\\'
trainMetaDataPath       = 'F:\\Datasets\\jester-v1-train.csv'
validateMetaDataPath    = 'F:\\Datasets\\jester-v1-validation.csv'

# data
IMG_HEIGHT = 100
IMG_WIDTH = 150
NFRAMES = 30
num_classes = 27

# model params
epochs = 0
batchsize = 32

def data_generator(featurePath, labelPath, batchsize, labelEncoder):
    metaData = pd.read_csv(labelPath, delimiter=';', header=None, index_col=0, names=['gesture'])
    videoIds = metaData.index.values
    
    while True:
        features = []
        labels = []               
        for videoId in videoIds:
            while len(labels) < batchsize:  
                directory = os.path.join(featurePath, str(videoId))
                if len(os.listdir(directory)) >= NFRAMES: # don't use if video length is too small        
                    video = []    
                    for counterImage, image in enumerate(os.listdir(directory), 1):   
                        if counterImage > NFRAMES: break # break if video length is too big        
                        image_values = PIL_Image.open(os.path.join(directory, image))
                        image_values = image_values.convert('L') # L: converts to greyscale
                        image_values = image_values.resize((IMG_WIDTH, IMG_HEIGHT), PIL_Image.ANTIALIAS)
                        video.append(np.asarray(image_values))

                    features.append(np.asarray(video))

def build_model():
    model = Sequential()
    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same'), input_shape=(NFRAMES, IMG_HEIGHT, IMG_WIDTH, 1)))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(20, return_sequences=True))
    model.add(TimeDistributed(Dense(num_classes)))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


# import_features()
model = build_model()
# train_model()
history = model.fit_generator(train_batches, steps_per_epoch=train_steps*2, epochs=epochs, verbose=1, validation_data=valid_batches, validation_steps=valid_steps*2)
# print_result()