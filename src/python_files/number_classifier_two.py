#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python 3.7
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from keras import datasets, layers, models


# In[2]:


(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()


# In[3]:


X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255


# In[4]:


cnn = models.Sequential([
    layers.Conv2D(30, (3,3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2,2)),
 
    layers.Flatten(),
    layers.Dense(100, activation='relu'),
    layers.Dense(10, activation='sigmoid')
])


# In[5]:


cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[6]:


cnn.fit(X_train, y_train, epochs=10)


# In[7]:


cnn.evaluate(X_test,y_test)

