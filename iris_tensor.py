import tensorflow as tf
import pandas as pd
import numpy as np

train_data = pd.read_csv('data/iris.data',header=0)
train_data = train_data.sample(frac=1)
data = train_data.copy
y = train_data.pop('class_name')
y = pd.get_dummies(y)
#yesma irror aauxa cause hamro data set ma labels numeric xaina...tensorflow le afai dini data xa tesma class ko thau ma numeric value xa tyo use garda chai milxa


print(train_data.head)
print(y.head)
def input_fn(features,labels,training=True,batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features),labels))

    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)    


my_feature_columns = data.columns
print(my_feature_columns)

output_columns = ['Iris-setosa','Iris-versicolor','Iris-virginica']


classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units= [30,10],
    n_classes=3,
    activation_fn=tf.nn.softmax

)

classifier.train(
input_fn=lambda: input_fn(train_data,y,training= True)
,steps=5000

)
