#!/usr/bin/env python
# coding: utf-8

# # **Forest Fire Detection Using Convolutional Neural Network**
# 
# ---
# 
# 

# In this notebook let's see how we can differentiate between an image that shows forest with fire from an image of forrest without fire. To do this I've used CNN.

# link to dataset: https://www.kaggle.com/datasets/brsdincer/wildfire-detection-image-data

# Importing necessary libraries

# In[ ]:


import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


# Making saperate datasets for training and testing

# In[ ]:


train = ImageDataGenerator(rescale=1/255)
test = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory("../input/wildfire-detection-image-data/forest_fire/Training and Validation/",
                                          target_size=(150,150),
                                          batch_size = 32,
                                          class_mode = 'binary')
                                         
test_dataset = test.flow_from_directory("../input/wildfire-detection-image-data/forest_fire/Testing/",
                                          target_size=(150,150),
                                          batch_size =32,
                                          class_mode = 'binary')


# In[ ]:


test_dataset.class_indices


# Model Building

# In[ ]:


model = keras.Sequential()
model.add(keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512,activation='relu'))
model.add(keras.layers.Dense(1,activation='sigmoid'))


# Compiling the model

# In[ ]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# Fitting the model

# In[ ]:


r = model.fit(train_dataset,
         epochs = 10,
         validation_data = test_dataset)


# Predicting on Test Dataset

# In[ ]:


predictions = model.predict(test_dataset)
predictions = np.round(predictions)


# In[ ]:


predictions


# In[ ]:


print(len(predictions))


# Plotting loss per iteration

# In[ ]:


import matplotlib.pyplot as plt
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()


# Plotting accuracy per iteration

# In[ ]:


plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()


# Making a function to see any image from dataset with predicted label

# In[ ]:


def predictImage(filename):
    img1 = image.load_img(filename,target_size=(150,150))
    plt.imshow(img1)
    Y = image.img_to_array(img1)
    X = np.expand_dims(Y,axis=0)
    val = model.predict(X)
    print(val)
    if val == 1:
        plt.xlabel("No Fire",fontsize=30)
    elif val == 0:
        plt.xlabel("Fire",fontsize=30)


# In[ ]:


predictImage("../input/wildfire-detection-image-data/forest_fire/Testing/fire/abc182.jpg")


# In[ ]:


predictImage('../input/wildfire-detection-image-data/forest_fire/Testing/fire/abc190.jpg')


# In[ ]:


predictImage('../input/wildfire-detection-image-data/forest_fire/Testing/nofire/abc346.jpg')


# In[ ]:


predictImage('../input/wildfire-detection-image-data/forest_fire/Testing/nofire/abc361.jpg')


# In[ ]:


predictImage('../input/wildfire-detection-image-data/forest_fire/Training and Validation/fire/abc011.jpg')


# In[ ]:


predictImage('../input/wildfire-detection-image-data/forest_fire/Testing/fire/abc172.jpg')


# In[ ]:


predictImage('../input/wildfire-detection-image-data/forest_fire/Testing/nofire/abc341.jpg')


# 
# 
# -----
# 
# 

# # Final Thoughts

# 
# 
# 1.   Model is well performing in testing.
# 2.   The model can be improved further more as the graphs showing accuracy and loss are bit messy.
# 3.   Transfer Learning can be used to reduce the learning/training time significantly.
# 
# 
# 
# 

# 
# 
# ---
# 
# 
