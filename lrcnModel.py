import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

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


##
#   Trains a Deep Learning model on the 20bn-jester-v1 dataset.
#   The model used is based on the idea of the lrcn, proposed in arXiv:1411.4389 [cs.CV]
#   Because the dataset is large a data generator function is used. It imports the data in batches in parallel to the training and feeds it to the network periodically when new data is needed.
#   Shape of features: [batch, frame, height, width, channel]. The features are restricted to 30 frames per video and each frame is converted into a uniform size and uses RGB values. 
#   Shape of labels: [batch, class]. The class is an one hot encoded vector.
#   The model and training statistics are saved under 'modelSavePath'.
##

# Paths
encoderPath             = 'DeepLearningModel\\Encoder\\oneHotEncoder.pkl'
videoDataPath           = 'F:\\Datasets\\20bn-jester-v1\\'
trainMetaDataPath       = 'F:\\Datasets\\jester-v1-train.csv'
validateMetaDataPath    = 'F:\\Datasets\\jester-v1-validation.csv'
modelSavePath           = 'DeepLearningModel//Model//'

# data params
IMG_HEIGHT = 100
IMG_WIDTH = 150
NUM_FRAMES = 30
NUM_CLASSES = 27
NUM_TRAIN_SAMPLES = 10240 #116254
NUM_VALIDATION_SAMPLES = 640

# model params
epochs = 5
batchsize = 32

def data_generator(featurePath, labelPath, batchsize, labelEncoder):
    metaData = pd.read_csv(labelPath, delimiter=';', header=None, index_col=0, names=['gesture'])
    videoIdIter = iter(metaData.index.values)
    
    while True:
        features = []
        labels = []   
        
        while len(labels) < batchsize:  
            videoId = next(videoIdIter)            
            directory = os.path.join(featurePath, str(videoId))
            if len(os.listdir(directory)) >= NUM_FRAMES: # don't use if video length is too small   
                video = []    
                for imageCounter, image in enumerate(os.listdir(directory), 1):   
                    if imageCounter > NUM_FRAMES: break # break if video length is too big        
                    image_values = PIL_Image.open(os.path.join(directory, image))
                    image_values = image_values.convert('RGB') # L: converts to greyscale | RGB 
                    image_values = image_values.resize((IMG_WIDTH, IMG_HEIGHT), PIL_Image.ANTIALIAS)
                    video.append(np.asarray(image_values))

                features.append(np.asarray(video))
                labels.append(metaData.at[videoId, 'gesture'])

        features = np.asarray(features)
        labels = labelEncoder.transform(np.asarray(labels).reshape(-1, 1))
        yield(features, labels)

def build_model():
    model = Sequential()
    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same'), input_shape=(NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH, 3), name='Conv')) # shape = (frames, width, heigth, channel)
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), name='MaxPooling')))
    model.add(TimeDistributed(Flatten(), name='Flatten_1'))
    model.add(LSTM(20, return_sequences=True, name='LSTM'))
    model.add(TimeDistributed(Dense(NUM_CLASSES, activation='linear'), name='Dense'))
    model.add(GlobalAveragePooling1D(name='average'))
       
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

def plot_history(history):
    plot_model(model, to_file=modelSavePath+'model.png')
    # accuracy
    plt.plot(history.history['acc'],"o-",label="accuracy")
    plt.plot(history.history['val_acc'],"o-",label="val_acc")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="upper right")
    plt.savefig(modelSavePath+'model_accuracy.png')
    # loss
    plt.plot(history.history['loss'],"o-",label="loss",)
    plt.plot(history.history['val_loss'],"o-",label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.savefig(modelSavePath+'model_loss.png')


#Initialization
oneHotEncoder = joblib.load(encoderPath)
model = build_model()

num_train_batches_per_epoch = int(NUM_TRAIN_SAMPLES / batchsize)
num_validation_batches_per_epoch = int(NUM_VALIDATION_SAMPLES / batchsize)

# Data
train_batches = data_generator(videoDataPath, trainMetaDataPath, batchsize, oneHotEncoder)
validation_batches = data_generator(videoDataPath, validateMetaDataPath, batchsize, oneHotEncoder)
 
# Train
history = model.fit_generator(train_batches, steps_per_epoch=num_train_batches_per_epoch, epochs=epochs, verbose=1, validation_data=validation_batches, validation_steps=num_validation_batches_per_epoch)
plot_history(history)

model.save(modelSavePath+'model.h5')
print("saved model at", modelSavePath)