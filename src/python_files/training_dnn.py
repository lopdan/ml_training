#!/usr/bin/env python
# coding: utf-8

# Exercises from _Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow_

# **1. Is it okay to initialize all the weights to the same value as long as that value is
# selected randomly using He initialization?**
# 
# Since we want to avoid symmetries by breaking them, initializing all values to the same will make 
# all the weights to be the same, thus, making impossible to break the symmetry.
# 
# **2. Is it okay to initialize the bias terms to 0?**
# 
# Yes, it is okay, it doest not make much difference.
# 
# **3. Name three advantages of the SELU activation function over ReLU.**
# 
# It can take negative values, alliviating the vanishing gradients problem.
# It has a nonzero derivative, avoiding dying units.
# It is smooth everywhere, since ReLU jumps from 0 to 1 at given point.
# 
# **4. In which cases would you want to use each of the following activation functions:
# ELU, leaky ReLU (and its variants), ReLU, tanh, logistic, and softmax?**
# 
# ELU, leaky ReLU: If you need the neural network to be as fast as possible.
# 
# ReLU: Autoencoders.
# 
# tanh: In a output layer if a number between -1 and 1 is needed.
# 
# logistic: In the output layer to estimate a probability.
# 
# softmax: In the output layer for probabilities that are mutually exclusive classes.
# 
# **5. What may happen if you set the momentum hyperparameter too close to 1 (e.g.,
# 0.99999) when using an SGD optimizer?**
# 
# The algorithm will pick up a lot of speed and it will shoot right past the minimum. It will
# make it several times before converging, resulting in a slower training time.
# 
# **6. Name three ways you can produce a sparse model.**
# 
# Zero out tiny weights. Also apply l1 regularization during training, making it more sparse. At last combining
# l1 regularization with dual averaging.
# 
# **7. Does dropout slow down training? Does it slow down inference (i.e., making
# predictions on new instances)? What are about MC dropout?**
# 
# Yes, dropout does slow down training, in general roughly by a factor of two.
# However, it has no impact on inference since it is only turned on during training.

# In[1]:


import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
assert tf.__version__ >= "2.0"


# In[126]:


keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)


# Build a DNN with five hidden layers of 100 neurons each, He initialization, and the ELU activation function.
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
for _ in range(20):
    model.add(keras.layers.Dense(100,
                                 activation="elu",
                                 kernel_initializer="he_normal"))


# In[127]:


# Using Adam optimization and early stopping, try training it on MNIST but
# only on digits 0 to 4, as we will use transfer learning for digits 5 to 9 in the
# next exercise. You will need a softmax output layer with five neurons, and as
# always make sure to save checkpoints at regular intervals and save the final
# model so you can reuse it later.


# In[27]:


# Loading the dataset
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

X_train = X_train_full[5000:]
y_train = y_train_full[5000:]
X_valid = X_train_full[:5000]
y_valid = y_train_full[:5000]


# In[129]:


# Output layer
model.add(keras.layers.Dense(10, activation="softmax"))

# Optimizer
optimizer = keras.optimizers.Adam(learning_rate=3e-5)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])


# In[130]:


# Early stopping and checkpoint
early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)
model_checkpoint_cb = keras.callbacks.ModelCheckpoint("my_model.h5", save_best_only=True)

run_index = 1 
run_logdir = os.path.join(os.curdir, "my_logs", "run_{:03d}".format(run_index))
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
callbacks = [early_stopping_cb, model_checkpoint_cb, tensorboard_cb]


# In[131]:


get_ipython().run_line_magic('tensorboard', '--logdir=./my_logs --port=6006')


# In[132]:


model.fit(X_train, y_train, epochs=100,
          validation_data=(X_valid, y_valid),
          callbacks=callbacks)


# In[133]:


model = keras.models.load_model("my_model.h5")
model.evaluate(X_valid, y_valid)


# In[135]:


# Now try adding Batch Normalization and compare the learning curves: is it
# converging faster than before? Does it produce a better model?
keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.BatchNormalization())
for _ in range(20):
    model.add(keras.layers.Dense(100, kernel_initializer="he_normal"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("elu"))
model.add(keras.layers.Dense(10, activation="softmax"))

optimizer = keras.optimizers.Nadam(learning_rate=5e-4)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])

early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)
model_checkpoint_cb = keras.callbacks.ModelCheckpoint("my_bn_model.h5", save_best_only=True)
run_index = 1
run_logdir = os.path.join(os.curdir, "my_logs", "run_bn_{:03d}".format(run_index))
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
callbacks = [early_stopping_cb, model_checkpoint_cb, tensorboard_cb]

model.fit(X_train, y_train, epochs=100,
          validation_data=(X_valid, y_valid),
          callbacks=callbacks)

model = keras.models.load_model("my_bn_model.h5")
model.evaluate(X_valid, y_valid)


# In[ ]:


# Is the model overfitting the training set? Try adding dropout to every layer
# and try again. Does it help?
session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))


# In[137]:


tf.debugging.set_log_device_placement(True)


# In[140]:


# Try replacing Batch Normalization with SELU, and make the necessary 
# adjustements to ensure the network self-normalizes 
keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
for _ in range(20):
    model.add(keras.layers.Dense(100,
                                 kernel_initializer="lecun_normal",
                                 activation="selu"))
model.add(keras.layers.Dense(10, activation="softmax"))

optimizer = keras.optimizers.Nadam(learning_rate=7e-4)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])

early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)
model_checkpoint_cb = keras.callbacks.ModelCheckpoint("my_selu_model.h5", save_best_only=True)
run_index = 1 # increment every time you train the model
run_logdir = os.path.join(os.curdir, "my_logs", "run_selu_{:03d}".format(run_index))
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
callbacks = [early_stopping_cb, model_checkpoint_cb, tensorboard_cb]


model.fit(X_train, y_train, epochs=100,
          validation_data=(X_valid, y_valid),
          callbacks=callbacks)

model = keras.models.load_model("my_selu_model.h5")
model.evaluate(X_valid, y_valid)


# In[141]:


# Is the model overfitting the training set? Try adding dropout to every layer
# and try again. Does it help?

keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
for _ in range(20):
    model.add(keras.layers.Dense(100,
                                 kernel_initializer="lecun_normal",
                                 activation="selu"))

model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(10, activation="softmax"))

optimizer = keras.optimizers.Adam(learning_rate=5e-4)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])

early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)
model_checkpoint_cb = keras.callbacks.ModelCheckpoint("my_alpha_dropout_model.h5", save_best_only=True)
run_index = 1 # increment every time you train the model
run_logdir = os.path.join(os.curdir, "my_logs", "run_alpha_dropout_{:03d}".format(run_index))
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
callbacks = [early_stopping_cb, model_checkpoint_cb, tensorboard_cb]

model.fit(X_train, y_train, epochs=100,
          validation_data=(X_valid, y_valid),
          callbacks=callbacks)

model = keras.models.load_model("my_alpha_dropout_model.h5")
model.evaluate(X_valid, y_valid)


# In[142]:


class MCAlphaDropout(keras.layers.AlphaDropout):
    def call(self, inputs):
        return super().call(inputs, training=True)


# In[143]:


mc_model = keras.models.Sequential([
    MCAlphaDropout(layer.rate) if isinstance(layer, keras.layers.AlphaDropout) else layer
    for layer in model.layers
])


# In[144]:


def mc_dropout_predict_probas(mc_model, X, n_samples=10):
    Y_probas = [mc_model.predict(X) for sample in range(n_samples)]
    return np.mean(Y_probas, axis=0)

def mc_dropout_predict_classes(mc_model, X, n_samples=10):
    Y_probas = mc_dropout_predict_probas(mc_model, X, n_samples)
    return np.argmax(Y_probas, axis=1)


# In[147]:


keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

y_pred = mc_dropout_predict_classes(mc_model, X_valid)
accuracy = np.mean(y_pred == y_valid)
accuracy


# In[148]:


model.summary()


# In[20]:


preload_model = keras.models.load_model("my_model.h5")

# Take all layers except the last one
new_model = keras.models.Sequential()
for layer in preload_model.layers[:-1]:
    new_model.add(layer)


# In[21]:


# Freeze all layers
for layer in new_model.layers:
    layer.trainable = False


# In[24]:


# Add the new output layers
new_model.add(keras.layers.Dense(10, activation="softmax", name='main_output'))


# In[30]:


early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)
model_checkpoint_cb = keras.callbacks.ModelCheckpoint("my_transfer_model.h5", save_best_only=True)
run_index = 1
run_logdir = os.path.join(os.curdir, "my_logs", "run_bn_{:03d}".format(run_index))
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
callbacks = [early_stopping_cb, model_checkpoint_cb, tensorboard_cb]

new_model.compile(loss="sparse_categorical_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"])

new_model.fit(X_train, y_train, epochs=100,
                validation_data=(X_valid, y_valid),
                callbacks=callbacks)

new_model = keras.models.load_model("my_transfer_model.h5")
new_model.evaluate(X_valid, y_valid)

