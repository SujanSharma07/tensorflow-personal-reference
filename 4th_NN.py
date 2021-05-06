import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as  pd
import matplotlib.pyplot as plt
import pickle


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()


class_names = ['T-shirt/top','Trousers',"pillover",'Dress','Coat','sandal','Shirt','Sneaker'
                ,'Bag',"ankle boot"]
'''
#lest look at the data format
for i,j in zip(train_images,train_labels):
        
    print('Train_images',i)
    print('Labels',j)
'''

#lest see shape
print(train_images.shape)
print(train_images.shape[1:])

#lest see labels
print(train_labels[:10])

#computation fast garauna data lai normalize garekai ramro hunxa
train_images = train_images/255.0
test_images = test_images/255.0


#lets see images that are saved as arrays
dummy = [[1,2,3,3,4,4,3,2],[1,2,3,4,5,6,7,7]]
plt.figure()
plt.imshow(train_images[1],cmap=plt.cm.binary)
plt.colorbar
plt.grid(False)
plt.show()

'''
#use this part to train..but i have already trained and saved the model so..i will load it only
#Sequential vanya most basic form of NN ho jun ma data sequentially flow hunxa from one layer to the other
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(train_images.shape[1:])),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])
model.summary()
model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images,test_labels, verbose = 2)
print("Tested Acc", test_acc)
model.save('4th:NN.h5')

'''
model = keras.models.load_model('4th:NN.h5')

test_loss, test_acc = model.evaluate(test_images,test_labels, verbose = 2)
print("Tested Acc", test_acc)

predict_for = test_images[10].reshape(1,28,28)
prediction = model.predict(predict_for)
print(f"prediction is {class_names[np.argmax(prediction)]} with chances of {np.max(prediction)*100}%")
print(f"But actually its a {class_names[test_labels[10]]} ")
