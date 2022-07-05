#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python 3.7
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from keras import datasets, layers, models


# In[2]:


# Load the data
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# In[3]:


classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
y_train = y_train.reshape(-1,)

def plot_sample(X, y, index):
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])
    plt.show()


# In[4]:


X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255


# In[5]:


ann = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(3000, activation='relu'),
    layers.Dense(1000, activation='relu'),
    layers.Dense(10, activation='sigmoid')
])

ann.compile(optimizer='SGD',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

ann.fit(X_train, y_train, epochs=5)


# In[6]:


from sklearn.metrics import classification_report, confusion_matrix
y_pred = ann.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print(classification_report(y_test, y_pred_classes))


# In[7]:


cnn = models.Sequential([
    #cnn
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPool2D((2,2)),
    # Dense
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# In[8]:


cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[9]:


cnn.fit(X_train, y_train, epochs=10)


# In[10]:


cnn.evaluate(X_test, y_test)


# In[11]:


y_pred = cnn.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]
print(classification_report(y_test, y_pred_classes))

