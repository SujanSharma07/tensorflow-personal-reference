import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

'''
There are three built-in RNN layers in Keras:

keras.layers.SimpleRNN, a fully-connected RNN where the output from previous timestep is to be fed to next timestep.

keras.layers.GRU, first proposed in Cho et al., 2014.

keras.layers.LSTM, first proposed in Hochreiter & Schmidhuber,

'''


if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")



data = keras.datasets.imdb

#yesma 10000 le indicate garni vanya minimum 10,000 times use vayo word matra leko , kam use vayo tati matter gardaina
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words= 10000)

print(train_data[0])

word_index = data.get_word_index()

#yo 3 jodya chai pading/start/unknown/unused ko lagespace banauna ho
word_index = {k:(v+3) for k, v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3



#hamro model ma ta input ko sape dina parxa..tara comments ko length different huda
#kura milena so., aauta fix length banauni ,, badi xa vaye cut exceeding ra kam vaye add padding
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index['<PAD>'], padding='post', maxlen= 255)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index['<PAD>'], padding='post', maxlen= 255)


#hamro data ma word haru hunxa jasle integer lai indicate garxa but to make integer indicate words
reverse_word_index = dict([(value, key) for (key,value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(test_data[0]))

#To train the model, i have trained and saved the model as NLP.h5

'''
#Model Definition
model = keras.Sequential()

#word haru different vaye ni meaning aautai dina sakxa..numericallly different hunxa but meaning aautai dinxa
#numerically matra herda ta hamro model le similar world lai ni extremly different sochxa
#so aaha ko 10,000 word ko lagi 10,000 embedded vector create garca
#image like vectors ani sab ko angles haru change garxa training ma ra similar words bich ko angle minimize ra differet bich ko angle maximize garni
#16 le chai 16 diminsion ma compare garxa
model.add(keras.layers.Embedding(10000,16))

#16 dimension chai abdi hunxa computation garna so yo layer le chai dimension minimize garxa
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1,activation='sigmoid'))


model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

fitModel = model.fit(x_train, y_train, epochs=25, batch_size=512, validation_data=(x_val, y_val), verbose=1)

loss, acc = model.evaluate(test_data, test_labels)

print(f"Accuracy is {acc}")

model.save('NLP.h5')
'''

#loading pretrained saved model
model = keras.models.load_model('NLP.h5')
loss, acc = model.evaluate(test_data, test_labels)

print(f"Accuracy is {acc}")


def review_encode(s):
    #1st ma start tag add garna, 1 indicate start tag
    encoded = [1]

    for word in s:
        if word in word_index:
            encoded.append(word_index[word.lower()])
        else:
            #unknown tag if word chindaina vaye
            encoded.append(2)
    return encoded        

#check for personal review
test_text = "(bad bad bad bad bad bad bad bad bad bab bad bad bad bad bad.)"
for i in range(500):
    test_text+=' good'
test_text = test_text.replace(",","").replace(".","").replace("(","").replace(")","").replace(":","")
encode = review_encode(test_text)
encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index['<PAD>'], padding='post', maxlen= 255)
predict = model.predict(encode)
print(encode[0])
print(str(predict[0][0]*100) + "%" )
print(test_text)


#for .txt file
'''
with open('test.txt', encoding="utf-8") as f:
    for line in f.readlines():
        nline = line.replace(",","").replace(".","").replace("(","").replace(")","").replace(":","")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index['<PAD>'], padding='post', maxlen= 255)
        predict = model.predict(encode)
        print(line)
        print(encode[0])
        print(predict[0])
'''        