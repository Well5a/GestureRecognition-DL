import numpy as np
import pandas as pd
import os
from PIL import Image as PIL_Image

IMG_HEIGHT = 100
IMG_WIDTH = 150
directory = 'F:\\Datasets\\20bn-jester-v1\\1'

imageName = os.listdir(directory)[0]

image = PIL_Image.open(os.path.join(directory, imageName))
image = image.convert('L') # L: converts to greyscale
image = image.resize((IMG_WIDTH, IMG_HEIGHT), PIL_Image.ANTIALIAS)
image = np.asarray(image)
print(image.shape, image)

# image_values = image_values.convert('RGB') # L: converts to greyscale
# image_values = image_values.resize((IMG_WIDTH, IMG_HEIGHT), PIL_Image.ANTIALIAS)