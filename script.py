''' 
This project uses chest X-rays to identify the COVID patients if they have infection using Artificial Intelligence. The data is broken into training and test datasets using two categories
i.e. Normal and Pneumonia. 

This is a beginner project in which I am going to create models using PyTorch and Keras and check their accuracy
'''

# Others
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2 
import os

# Keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

# sklearn
from sklearn.metrics import classification_report, confusion_matrix

# tensorflow
import tensorflow as tf


''' 
            Step 2 : Loading the data 
'''

labels = ['NORMAL', 'PNEUMONIA']
img_size = 224 # Standard pixel size

# Defining a function to get the data from directory
def get_data(data_dir):
    data = []
    for l in labels:
        path = os.path.join(data_dir, l)
        class_num = labels.index(l)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)
        
# Getting the training and test data sets
train = get_data('C:/Users/hukha/Desktop/MS - Data Science/Projects/covid19-xray/xray/train')
val = get_data('C:/Users/hukha/Desktop/MS - Data Science/Projects/covid19-xray/xray/test')



''' 
        Step 3 : Visualizing the data
'''
l = []
for i in train:
    if(i[1] == 0):
        l.append('NORMAL')
    else:
        l.append('PNEUMONIA')
sns.set_style('darkgrid')
sns.countplot(l)

# Now let's visualize random pictures for normal and pneumonia from both training
plt.figure(figsize = (5,5))
plt.imshow(train[1][0])
plt.title(labels[train[0][1]])

# pneumonia
plt.figure(figsize = (5,5))
plt.imshow(train[100][0])
plt.title(labels[train[100][1]])



'''
        Step 4 - Data Preprocessing and Data Augmentation
        
'''

x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train:
  x_train.append(feature)
  y_train.append(label)

for feature, label in val:
  x_val.append(feature)
  y_val.append(label)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)
    

# Data Augmentation

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)



''' 
        Step 5 - Define the Model using CNN

'''
model = Sequential()
model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(224,224,3)))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(2, activation="softmax"))

model.summary()


# Compiling the model using Adam
opt = Adam(lr= 0.000001)
model.compile(optimizer = opt, loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics= ['accuracy'])


# Now let's train our model for 500 epochs since our data is small
history = model.fit(x_train, y_train, epochs=500, validation_data = (x_val, y_val))



'''
        Step 6 - Evaluating the Result
        
'''

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(500)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()



# Precision and Recall
predictions = model.predict_classes(x_val)
predictions = predictions.reshape(1,-1)[0]
print(classification_report(y_val, predictions, target_names = ['Normal (Class 0)','Pneumonia (Class 1)']))











