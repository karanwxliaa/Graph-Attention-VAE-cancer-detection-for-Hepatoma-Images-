import numpy as np
from PIL import Image

def create_SNN(img_path, size=(128, 128)):
    '''
    Takes in the segmentation mask's path and returns the 2D SNN in the required format 
    '''
    img = Image.open(img_path).resize(size)
    img = np.array(img)

    # Find coordinates of pixels with value 255 in the resized image
    SNN = np.array(np.column_stack(np.where(img == 255)))

    return SNN
