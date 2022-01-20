from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import numpy as np
import random
import cv2
import os
import glob

epochs=100
lr=1e-3
batch_size=64
batch_size=64
img_dims=(96,96,3)

data = []
labels=[]

#load image files from dataset
image_files = [f for f  in glob.glob(r'D:\_study\6th\SDP  Project\gender_dataset_face' + "/**/*" , recursive=True) if not os.path.isdir(f)]
random.shuffle(image_files)

#converting images to arrays and labelling categories
for img in image_files:
    image = cv2.imread(img)

    image = cv2.resize(image,(img_dims[0],img_dims[1]))
    image = img_to_array(image)
    data.append(image)

    label = img.split(os.path.sep)[-2] # path of img folder which gives man or woman
    if label == "man" :
        label=0
    else:
        label=1
    
    labels.append(label) #append label to labels list

#convert data nd labels into arrays
data = np.array(data, dtype="float")/255.0 #every img have pixel value 1 to 255 so when we deal with large value computation power can be high
labels = np.array(labels)

#split dataset for training nd validation
(trainX,testX, trainY,testY) = train_test_split(data, labels , test_size=0.2,random_state=42) 

trainY = to_categorical(trainY, num_classes=2) # [[1, 0], [0, 1], [0, 1], ...]
testY = to_categorical(testY, num_classes=2)

# augmenting datset  , datagenerator use when u have list of images in ur dataset , create lot of images with single image by changing feature
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

#define model class
def build(width, height, depth, classes):
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if K.image_data_format() == "channels_first": #Returns a string, either 'channels_first' or 'channels_last'
        inputShape = (depth, height, width)
        chanDim = 1
    
    # The axis that should be normalized, after a Conv2D layer with data_format="channels_first", 
    # set axis=1 in BatchNormalization.

    #add bunch of conclusion and match pulling layer
    model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3,3))) #remove noise
    model.add(Dropout(0.25))#means 25% nurons deactivated -> not cause overfitting (just class knowledge)

    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten()) #conver 2D to 1D
    model.add(Dense(1024))
    model.add(Activation("relu"))#input layer
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes))
    model.add(Activation("sigmoid"))#output layer

    return model


# build model
model = build(width=img_dims[0], height=img_dims[1], depth=img_dims[2],
                            classes=2)

# compile the model
opt = Adam(lr=lr, decay=lr/epochs)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the model
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size),
                        validation_data=(testX,testY),
                        steps_per_epoch=len(trainX) // batch_size,
                        epochs=epochs, verbose=1)