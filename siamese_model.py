import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Input, BatchNormalization, Activation, Dropout
import cv2
import os
import numpy as np
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras import backend as K
from keras.layers.core import Lambda
from keras.callbacks import ModelCheckpoint, TensorBoard
from itertools import combinations

#Load data

filenames = os.listdir("../Datasets/CUHK01")
labels = np.array([filename[:4] for filename in filenames])
labels = to_categorical(labels)

def create_input_pairs(X, labels):
    input = zip(X, labels)
    input1, input2 = [], []
    labels = []
    combs = list(combinations(input, 2))
    input1 += [comb[0][0] for comb in combs]
    input2 += [comb[1][0] for comb in combs]
    labels += [comb[0][1] == comb[1][1] for comb in combs]
    input1 = np.array(input1)
    input2 = np.array(input2)
    labels = np.array(labels)
    return input1, input2, labels


inp_1,inp_2,labels = create_input_pairs(filenames,labels)

# Create custom data genetaor

class My_Custom_Generator(keras.utils.Sequence) :
    def __init__(self, image_filenames_1, image_filenames_2, labels, batch_size) :
        self.image_filenames_1 = image_filenames_1
        self.image_filenames_2 = image_filenames_2
        self.labels = labels
        self.batch_size = batch_size
        
    def __len__(self) :
        return (np.ceil(len(self.image_filenames_1) / float(self.batch_size))).astype(np.int)
    
    def __getitem__(self, idx) :
        batch_x_1 = self.image_filenames_1[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_x_2 = self.image_filenames_2[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
        
        return ([np.array([cv2.imread('../Datasets/CUHK01/' + str(file_name)) for file_name in batch_x_1]),np.array([cv2.imread('../Datasets/CUHK01/' + str(file_name)) for file_name in batch_x_2])], np.array(batch_y))


batch_size =5

batch_generator = My_Custom_Generator(inp_1,inp_2,labels,batch_size)


# create feature extraction model
 
def feature_model():
	inp = Input(shape = (160,60,3))
	conv1 = Conv2D(64,(3,3),padding = "same")(inp)
	bn1 = BatchNormalization(axis = 1, momentum = 0.99, epsilon = 1e-3)(conv1)
	act1 = Activation("relu")(bn1)
	conv2 = Conv2D(64,(3,3),padding = "same")(act1)
	bn2 = BatchNormalization(axis = 1, momentum = 0.99, epsilon = 1e-3)(conv2)
	act2 = Activation("relu")(bn2)
	mp1 = MaxPool2D(pool_size = (2,2), strides = (2,2))(act2)
	conv3 = Conv2D(128,(3,3),padding = "same")(mp1)
	bn3 = BatchNormalization(axis = 1, momentum = 0.99, epsilon = 1e-3)(conv3)
	act3 = Activation("relu")(bn3)
	conv4 = Conv2D(128,(3,3),padding = "same")(act3)
	bn4 = BatchNormalization(axis = 1, momentum = 0.99, epsilon = 1e-3)(conv4)
	act4 = Activation("relu")(bn4)
	mp2 = MaxPool2D(pool_size = (2,2), strides = (2,2))(act4)
	conv5 = Conv2D(256,(3,3),padding = "same")(mp2)
	bn5 = BatchNormalization(axis = 1, momentum = 0.99, epsilon = 1e-3)(conv5)
	act5 = Activation("relu")(bn5)
	conv6 = Conv2D(256,(3,3),padding = "same")(act5)
	bn6 = BatchNormalization(axis = 1, momentum = 0.99, epsilon = 1e-3)(conv6)
	act6 = Activation("relu")(bn6)
	conv7 = Conv2D(256,(3,3),padding = "same")(act6)
	bn7 = BatchNormalization(axis = 1, momentum = 0.99, epsilon = 1e-3)(conv7)
	act7 = Activation("relu")(bn7)
	mp3 = MaxPool2D(pool_size=(2,2), strides = (2,2))(act7)
	conv8 = Conv2D(512,(3,3),padding = "same")(mp3)
	bn8 = BatchNormalization(axis = 1, momentum = 0.99, epsilon = 1e-3)(conv8)
	act8 = Activation("relu")(bn8)
	conv9 = Conv2D(512,(3,3),padding = "same")(act8)
	bn9 = BatchNormalization(axis = 1, momentum = 0.99, epsilon = 1e-3)(conv9)
	act9 = Activation("relu")(bn9)
	conv10 = Conv2D(512,(3,3),padding = "same")(act9)
	bn10 = BatchNormalization(axis = 1, momentum = 0.99, epsilon = 1e-3)(conv10)
	act10 = Activation("relu")(bn10)
	mp4 = MaxPool2D(pool_size = (2,2), strides = (2,2))(act10)
	fc = Flatten()(mp4)
	fc6 = Dense(4096, activation = 'relu')(fc)
	fc7 = Dense(4096, activation = 'relu')(fc6)
	drop = Dropout(0.5)(fc7)
	out = Dense(972, activation = "relu")(drop)
	model = Model(inp,out)
	model.summary()

	return model


# create function for find distance between two images

def distance_(inputs): 
	input_1,input_2  = inputs
	return K.abs(input_1 - input_2)

# create siamese network

def siamese_model():
	feature_model_ = feature_model()
	input_1 = Input(shape = (160,60,3,))
	input_2 = Input(shape = (160,60,3,))
	feature_vec_1 = feature_model_(input_1)
	feature_vec_2 = feature_model_(input_2)
	distance = Lambda(distance_)([feature_vec_1,feature_vec_2])
	output  = Activation("sigmoid")(distance)

	model = Model(inputs = [input_1,input_2], outputs = output)

	return model

model = siamese_model()
model.compile(optimizer='rmsprop', loss='binary_crossentropy')
model.summary()

# Trainig of  model

filepath = './siamese/weights.{epoch:02d}-{loss:.2f}.hdf5'
cpkt1 = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=1)
cpkt2 = TensorBoard(log_dir='./siamese/tensorboard', histogram_freq=0, write_graph=True, write_images=True)

model.fit_generator(generator = batch_generator,
                    steps_per_epoch = (len(filenames)/batch_size),
                    epochs = 500,
                    verbose = 1,
                    callbacks=[cpkt1,cpkt2])


# Testing of model

model.load_weights("../Person-Re-identification-using-Siamese-networks/siamese/weights.____.hdf5")
img_1_path = "../Datasets/CUHK01/0001002.png"
img_2_path = "../Datasets/CUHK01/0001003.png"
img_3_path = "../Datasets/CUHK01/0002002.png"
img_1 = np.array([cv2.imread(img_1_path)])
img_2 = np.array([cv2.imread(img_2_path)])
img_3 = np.array([cv2.imread(img_3_path)])

pred_1 = model.predict([img_1,img_2])
pred_2 = model.predict([img_2,img_3])
