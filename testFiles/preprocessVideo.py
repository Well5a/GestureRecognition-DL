import numpy as np
import os
from PIL import Image as PIL_Image

##
#   Preprocesses video data for the 20bn-jester-v1 dataset.
#   Saves a numpy array with the converted data in the same folder as the images with the name of the ID.
#   Images are converted into an equal width and RGB values. Also length of the video is limited to maximum of frames.
##

videoDataPath = 'F:\\Datasets\\20bn-jester-v1\\'

IMG_HEIGHT = 100
IMG_WIDTH = 150
NUM_FRAMES = 30
    
for videoID in os.listdir(videoDataPath):    
    if videoID == 5: break                    
    videoDirectory = os.path.join(videoDataPath, str(videoID)) 
    video = []    

    for imageCounter, image in enumerate(os.listdir(videoDirectory), 1):   
        if imageCounter > NUM_FRAMES: break # break if video length is too big          
        image_values = PIL_Image.open(os.path.join(videoDirectory, image))
        image_values = image_values.convert('RGB') # L: converts to greyscale | RGB 
        image_values = image_values.resize((IMG_WIDTH, IMG_HEIGHT), PIL_Image.ANTIALIAS)
        video.append(np.asarray(image_values))

    savePath = os.path.join(videoDirectory, str(videoID))
    np.save(savePath, np.asarray(video))
    print('saved', videoID, 'to',  savePath)