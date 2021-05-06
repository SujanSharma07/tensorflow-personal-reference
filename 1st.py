import tensorflow as tf
import numpy as np

print(tf.__version__)
#tensorflow ma grap create hunxa kunai method banauda
#run garna chai session nai banauna parxa else execute hunna
#example
node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0)

print(node1,node2)
#Gives tf.Tensor(3.0, shape=(), dtype=float32) tf.Tensor(4.0, shape=(), dtype=float32)
#yo aauta abstract tensor matra ho execution hoina...execute garna session create garna parxa
#session le chai grap operations laii chai device ma execute garxa 


#sess = tf.Session() reinstall or kai garda yesle kam garna sakxa..aaile chai error aayo.session lai chineyna
with tf.compat.v1.Session() as sess:
  node1 = tf.constant(3.0,tf.float32)
  node2 = tf.constant(4.0)
  output = sess.run([node1,node2])
  print(output)  #[3.0, 4.0]
'''
The tensorflow core r2.0 have enabled eager execution by default so doesn't need to write tf.compat.v1.Session() and use .run() function
If we want to use tf.compat.v1.Session() then we need to do thi
tf.compat.v1.disable_eager_execution() in the starting of algorithm. Now we can use tf.compat.v1.Session() and .run() function.  
'''



#yesare we can create any valiable with multiple dimension also 
#tf.string jastai tf.float32,int32 ani auru auru ni xa
string1 = tf.Variable(["This is String"],tf.string)
string2 = tf.Variable([["This is String",'String2'],['Dummy1',"dummy2"]],tf.string)
string3 = tf.Variable([["This is String",'String2'],['Dummy1',"dummy2"],['Dummy1',"dummy2"]],tf.string)

#jati dimension badyo tati nai rank dekhauxa..only one dimension vanya 0 rank
print(tf.rank(string1))
print(tf.rank(string2))
print(tf.rank(string3))


#one main list with 2 list inside with 3 elements on each
tensor1 = tf.ones([1,2,3])

tensor2 = tf.reshape(tensor1,[2,3,1]) #reshape exixting data to shape [2,3,1]
#yo reshape garda total elemts of tensor1 i.e 6 ra tensor2 ko total elements ko number same huna parxa

tensor3 = tf.reshape(tensor1,[3,-1])# yesma -1 indicate remaining elemts cover garna jati elements rakhnu parxa rakhda hunxa
#i.e 3 ta list with each 2 ta elements in it.. yo 2 chai -1 le afai indicate garxa



#AAUta kura yaad rakhna parxa
# suppose x_train ma pd.read_csv garya data x avaye
# yo case ma x_train ma ta o/p ni xa
#teslai tya bata hatayera y_train m arakhna y_train = x_train.pop('o/p column name') rakhda hunxa
#yesle direct tyo column lai aauta bata pop garera arko ma tore garxa


with tf.compat.v1.Session() as sess:
  node1 = tf.constant(3.0,tf.float32)
  node2 = tf.constant(4.0)
  output = sess.run([node1,node2])
  File_Writer = tf.summary.FileWriter('TensorBoardFirst',sess.graph)
  #To see this stats we can run in terminal
  # tensorboard --logdir='PATH TO THAT FILEWRITER WRITTEN DIRECTORY'

  



  #PLACEHOLDER
  #Placeholder vanya its like a promish yesma kai value aauxa
  #External input haru dini jasto cases ma placeholder use garinxa.
  #kinaki constants haru chnage garna mildaina paxi ani variable haru ni pailai dina parxa incase we can to provide input tyo case ma palxeholder comes to play role
  a = tf.placeholder(tf.float32)
  b = tf.placeholder(tf.float32)

  adder_node = a + b
  sess = tf.Session()
  print(sess.run(adder_node,{a:[1,4],b:[1,7]}))
  sess.close()
  

