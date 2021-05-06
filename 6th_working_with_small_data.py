#Hami sanga less data huda...suppose 100 images matra xa
#yo case ma data ta less vayo so
#yesko lage hamle photos lai rotate or different orientation ma change garera
#suppose aautai data lai 4 different orientation ma garem vane ta 100 ko 400 hunxa data

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from tensorflow import keras
import numpy as np


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


#create a generator that transform image
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range = 0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    )

#pick images to transform
test_img = train_images[14] 
img = image.img_to_array(test_img)
img = img.reshape((1,) + img.shape) 

i =0


for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):
    plt.figure(i)
    plot = plt.imshow(image.img_to_array(batch[0]))
    i+=1
    if i > 4:
        break

plt.show()