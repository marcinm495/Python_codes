
# coding: utf-8

# ## This is a construction by hand of a deep neural network for the tytanic classification problem.

# ## Import packages and prepare the data:

# In[55]:


#import libraries and modules:
from __future__ import print_function
import numpy as np
import tensorflow as tf
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[56]:


# This is a path to the python version used by this session. 
# In case of import error install lacking packages in the path below from a terminal using pip install Package_name
import sys
print(sys.executable)


# In[57]:


#size of the test set:
test_size_num = 0.3

#Import data:
data_file = pd.read_csv("train.csv")

#preparation of feature columns:

data_file['age_num']=data_file['Age'].map(lambda x: data_file['Age'].median() if math.isnan(x) else x)
data_file['sex_num']=data_file['Sex'].map(lambda x: 1 if x == 'male' else 0)
data_file['emb_num']=data_file['Embarked'].map(lambda x: 1 if x=="C" else 2 if x=="S" else 3)

def input_data(test_size_num, *feature_names):
    return train_test_split(data_file[list(feature_names)].values, data_file['Survived'].values, 
                            test_size = test_size_num, random_state = 0)
features_train, features_test, labels_train, labels_test =input_data(test_size_num, 'age_num', 
                                                                     'sex_num', 'emb_num', 'Parch', 'SibSp', 'Fare') 
#normalize the data:
scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)
#label columns have to be reshaped into an array, with 0 -> [1,0] and 1 -> [0,1]:

labels_train = (np.arange(2)==labels_train[:,None]).astype(np.float32)
labels_test = (np.arange(2)==labels_test[:,None]).astype(np.float32)


# ## N-layer DNN with Gradient Descent optimization:

# In[58]:


# Construction of a singl-layer DNN, using Gradient Descent:
num_labels = 2
h_layer = 12
n_Hlayers = 2
f_size = features_train.shape[1]

#Regularization parameters:


def lin_op(x, w, b):
    return tf.matmul(x,w)+b

def dnn_h1(x, w_list, b_list):
    l=len(w_list)
    y_in = lin_op(x, w_list[0], b_list[0])
    y_out = y_in
    for k in range(1,l):
        y_mid = tf.nn.relu(y_out)
        y_out = lin_op(y_mid, w_list[k], b_list[k])
    return y_out


graph = tf.Graph()
with graph.as_default():
  #the data are put into constant tensors:
  tf_train_dataset = tf.constant(features_train, dtype = tf.float32)
  tf_train_labels = tf.constant(labels_train, dtype = tf.float32)
  tf_test_dataset = tf.constant(features_test, dtype = tf.float32)
  reg_param_w1 = tf.constant(0.01, dtype = tf.float32)
  reg_param_w2 = tf.constant(0.01, dtype = tf.float32)
    
  #initialization of weight matrices for DNN with random normal inputs:
  #ALL VALUES ARE float64 TYPE:
    
  w_list = []
  b_list = []
  if (n_Hlayers == 0):
    w_list.append(tf.Variable(tf.truncated_normal([f_size, num_labels])))
    b_list.append(tf.Variable(tf.zeros([num_labels])))
  elif (n_Hlayers == 1):
    w_list.append(tf.Variable(tf.truncated_normal([f_size, h_layer])))
    b_list.append(tf.Variable(tf.zeros([h_layer])))
    w_list.append(tf.Variable(tf.truncated_normal([h_layer, num_labels])))
    b_list.append(tf.Variable(tf.zeros([num_labels])))
  else:
    w_list.append(tf.Variable(tf.truncated_normal([f_size, h_layer])))
    b_list.append(tf.Variable(tf.zeros([h_layer])))
    for k in range(1,n_Hlayers):
        w_list.append(tf.Variable(tf.truncated_normal([h_layer, h_layer])))
        b_list.append(tf.Variable(tf.zeros([h_layer])))
    w_list.append(tf.Variable(tf.truncated_normal([h_layer, num_labels])))
    b_list.append(tf.Variable(tf.zeros([num_labels])))
     
  
  # Training computation.
  logits = dnn_h1(tf_train_dataset, w_list, b_list)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)) + (reg_param_w1 * tf.nn.l2_loss(w_list[0]))+ (reg_param_w2 * tf.nn.l2_loss(w_list[1]))
  
  
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss, var_list = (w_list, b_list))


  test_prediction = tf.nn.softmax(dnn_h1(tf_test_dataset, w_list, b_list))
  
  # Predictions for the training and test data, based on optimized weight and bias matrices:
  train_prediction = tf.nn.softmax(logits)
  test_prediction = tf.nn.softmax(dnn_h1(tf_test_dataset, w_list, b_list))


# In[59]:


num_steps = 200

# accuracy calculated with the 'one-hot' encoding, namely output probabilities are mapped as  follows:
#[0.2, 0.6, 0.2]->[0,1,0] etc.

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.Session(graph=graph) as session:
  #Initialization of all global variables:
  tf.global_variables_initializer().run()
  
  for step in range(num_steps):
    _, l, predictions = session.run([optimizer, loss, train_prediction])
    if (step % 10 == 0):
      print('Loss at step %d: %f' % (step, l))
      print('Training accuracy: %.1f%%' % accuracy(predictions, labels_train))
      print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), labels_test))

