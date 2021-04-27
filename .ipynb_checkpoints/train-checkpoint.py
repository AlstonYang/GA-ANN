#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Metric
from tensorflow.python.ops import math_ops

# In[10]:


gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
tf.compat.v1.keras.backend.set_session(sess)


# In[11]:


# Helper: Early stopping.
early_stopper = EarlyStopping(monitor='DINKLE_Accuracy', patience=10, mode='max')


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))


# In[12]:


# Customized performance indicator - DINKLE_Accuracy
class DINKLE_Accuracy(Metric):

    def __init__(self, name="DINKLE_Accuracy", **kwargs):
        super(Metric, self).__init__(name=name, **kwargs)
#         self.total_count = self.add_weight(name = "total_count", initializer=init_ops.zeros_initializer)
#         self.match_count = self.add_weight(name = "match_count", initializer=init_ops.zeros_initializer)
        self.matches_rate = self.add_weight(name = "matches_rate", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        
#         y_true = tf.convert_to_tensor(sc.inverse_transform(y_true))
#         y_pred = tf.convert_to_tensor(sc.inverse_transform(y_pred))
            
        match_count = tf.reduce_sum(tf.cast(tf.less_equal(tf.abs(y_true- y_pred), 0.02), dtype = tf.float32))
        total_count = y_true.shape[0]
        self.matches_rate = math_ops.div_no_nan(match_count, total_count)

         
    def result(self):
        return  self.matches_rate
    
    def reset_state(self):
        self.matches_rate = tf.zeros(shape=(1, 1))


# In[3]:


# def get_cifar10():
#     """Retrieve the CIFAR dataset and process the data."""
#     # Set defaults.
#     nb_classes = 10
#     batch_size = 64
#     input_shape = (3072,)

#     # Get the data.
#     (x_train, y_train), (x_test, y_test) = cifar10.load_data()
#     x_train = x_train.reshape(50000, 3072)
#     x_test = x_test.reshape(10000, 3072)
#     x_train = x_train.astype('float32')
#     x_test = x_test.astype('float32')
#     x_train /= 255
#     x_test /= 255

#     # convert class vectors to binary class matrices
#     y_train = to_categorical(y_train, nb_classes)
#     y_test = to_categorical(y_test, nb_classes)

#     return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)


# In[4]:


# def get_mnist():
#     """Retrieve the MNIST dataset and process the data."""
#     # Set defaults.
#     nb_classes = 10
#     batch_size = 128
#     input_shape = (784,)

#     # Get the data.
#     (x_train, y_train), (x_test, y_test) = mnist.load_data()
#     x_train = x_train.reshape(60000, 784)
#     x_test = x_test.reshape(10000, 784)
#     x_train = x_train.astype('float32')
#     x_test = x_test.astype('float32')
#     x_train /= 255
#     x_test /= 255

#     # convert class vectors to binary class matrices
#     y_train = to_categorical(y_train, nb_classes)
#     y_test = to_categorical(y_test, nb_classes)

#     return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)


# In[13]:


def read(path):
    return pd.read_csv(path)


# In[14]:


def buildTrain(train, pastWeek=4, futureWeek=1, defaultWeek=1):
    X_train, Y_train = [], []
    for i in range(train.shape[0]-futureWeek-pastWeek):
        X = np.array(train.iloc[i:i+defaultWeek])
        X = np.append(X,train["CCSP"].iloc[i+defaultWeek:i+pastWeek])
        X_train.append(X.reshape(X.size))
        Y_train.append(np.array(train.iloc[i+pastWeek:i+pastWeek+futureWeek]["CCSP"]))
    return np.array(X_train), np.array(Y_train)


# In[16]:


def get_data():
    
    ## Read weekly copper price data
    path = "WeeklyFinalData.csv"
    data = read(path)
    
    date = data["Date"]
    data.drop("Date", axis=1, inplace=True)
    
    ## Add time lag (pastWeek=4, futureWeek=1)
    x_data, y_data = buildTrain(data)
    
    ## Data split
    x_train = x_data[0:int(x_data.shape[0]*0.8)]
    x_test = x_data[int(x_data.shape[0]*0.8):]
    
    y_train = y_data[0:int(y_data.shape[0]*0.8)]
    y_test = y_data[int(y_data.shape[0]*0.8):]
    
    ## Normalize
    x_train_scaled = sc.fit_transform(x_train)
    x_test_scaled = sc.transform(x_test)
    
    y_train_scaled = sc.fit_transform(y_train)
    y_test_scaled = sc.transform(y_test)
    
    ## Other information
    nb_output = 1
    batch_size = 10
    input_shape = (x_train_scaled.shape[1],)
    
    return (nb_output, batch_size, input_shape, x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled)


# In[5]:


def compile_model(network, nb_output, input_shape):
    """Compile a sequential model.

    Args:
        network (dict): the parameters of the network

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0:
            model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))
        else:
            model.add(Dense(nb_neurons, activation=activation))

        model.add(Dropout(0.2))  # hard-coded dropout

    # Output layer.
    model.add(Dense(nb_output))

    model.compile(loss='mean_squared_error', 
                  optimizer=optimizer,
                  metrics=[DINKLE_Accuracy()]
                 )

    return model


# In[17]:


def train_and_score(network):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """
#     if dataset == 'cifar10':
#         nb_classes, batch_size, input_shape, x_train, \
#             x_test, y_train, y_test = get_cifar10()
#     elif dataset == 'mnist':
#         nb_classes, batch_size, input_shape, x_train, \
#             x_test, y_train, y_test = get_mnist()

    nb_output, batch_size, input_shape, x_train, x_test, y_train, y_test = get_data()
    
    model = compile_model(network, nb_output, input_shape)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=500,  # using early stopping, so no real limit
              verbose=1
#               , validation_data=(x_test, y_test)
              ,callbacks=[early_stopper]
             )

    score = model.evaluate(x_test, y_test, batch_size = len(y_test),verbose=0)

    return score[1]  # 1 is accuracy. 0 is loss.

