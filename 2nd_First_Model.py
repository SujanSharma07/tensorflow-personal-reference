import tensorflow as tf

#Model Parameters    
w = tf.Variable([.3],tf.float32)
b = tf.Variable([-.3],tf.float32)



#input and output
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)


linar_model = w * x + b

#loss
squared_delta = tf.square(linar_model-y)
loss = tf.reduce_sum(squared_delta)

#optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

#Yo chai sab varaiables lai initialize garna
init = tf.global_variables_initializer()


sess = tf.Session()
sess.run(init)
for i in range(1000):
    sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})


print(sess.run([w,b]))    

