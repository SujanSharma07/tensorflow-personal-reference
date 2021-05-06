#about input function
#hamro data lai epoch ra required batches ma convert garna ko lagi

import tensorflow as tf
import tensorflow.compat.v2.fetaure_column as fc
import pandas as pd

dftrain = pd.read_csv('train.csv')
dfeval = pd.read_csv('test.csv')

y_train = dftrain['Survived']
y_eval = dfeval.pop['Survived']





def make_input_fn(data_df, num_epochs = 10, suffle = True, batch_size = 32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices(dict(data_df), label_df)
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs) 
        return ds
    return input_function



train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs, suffle=False)


