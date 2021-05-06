import tensorflow as tf
import pandas as pd


#Model Parameters    
w = tf.random.uniform((4,500), minval=0, maxval=1, dtype=tf.dtypes.float32, seed=None)
b = tf.random.uniform((500,1), minval=0, maxval=1, dtype=tf.dtypes.float32, seed=None)

#input and output
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
train_data = pd.read_csv('/home/sujan/Downloads/data/titanic/train.csv')
y_in = train_data["Survived"][:500]
test_data = pd.read_csv('/home/sujan/Downloads/data/titanic/test.csv')

features = ["Pclass", "Sex", "SibSp", "Parch"]



#Can use this get dummies function from pandas to change string data to numeric data
x_in = pd.get_dummies(train_data[features])[:500]
X_test = pd.get_dummies(test_data[features])

linar_model = tf.add(tf.matmul(w,x), b)


#loss
squared_delta = tf.square(tf.math.subtract(linar_model,y))
loss = tf.reduce_sum(squared_delta)

#optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

#Yo chai sab varaiables lai initialize garna
init = tf.global_variables_initializer()


sess = tf.Session()
sess.run(init)
for i in range(1000):
    sess.run(train,{x:x_in,y:y_in})


print(sess.run([w,b]))    

