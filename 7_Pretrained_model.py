#yo cahi k ho vane
#google ani auru tannai company haru sanga last dherai data hunxa train garna. mIllions of hunxa
#jasti data vayo tati ramro ho so
#ani uniharu ko pretrained model haru open source ma xa
#so we can use those models for our model
#pretrained model o first tira ko chai use garni ani last tira labels haru ko thau ma chai afnu lai chaini jasre use garni

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow import keras
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']



# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=train_images[0].shape,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False
base_model.summary()



global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(10)


model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])


model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(train_images, train_labels, epochs=5,
                    validation_data=(test_images, test_labels))
test_loss, test_acc = model.evaluate(test_images,test_labels, verbose = 2)

print("Tested Acc", test_acc)

model.save("Pretrained.h5")

'''
https://www.tensorflow.org/tutorials/images/transfer_learning
'''


