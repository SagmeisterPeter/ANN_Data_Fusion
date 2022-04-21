# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 09:13:16 2021

@author: Peter Sagmeister
Code accompanying the publication "Artificial Neural Networks and Data Fusion Enable Concentration Predictions for Inline Process Analyticsy" 
Authors: Peter Sagmeister, Robin Hierzegger, Jason D. Williams, C. Oliver Kappe and Stefan Kowarik
publication link:   https://
data available at:  https://doi.org/10.5281/zenodo.6066166
"""
#%% import functions
import numpy as np
import matplotlib.pyplot as plt

#%% NN functions
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau
from keras.models import load_model
from tensorflow.keras.layers import Dense, Input, Flatten, LocallyConnected1D,Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler


#%% read in trainings data, shuffel and assign UV and NMR data
path1 = ''                                                            # insert path where the training data and validation data is saved
train_NMR = np.load(path1+"/datafusion_NMR_training.npy")
train_UV = np.load(path1+"/datafusion_UV_training.npy")

combine_NMR_UV_train = np.concatenate((train_NMR, train_UV), axis=1)
np.random.shuffle(combine_NMR_UV_train)                                    #shuffel data

training_NMR = combine_NMR_UV_train[:,:(len(train_NMR[0]))]             #assign training NMR data
training_UV = combine_NMR_UV_train[:,(len(train_NMR[0])):]              #assign training UV data
training_X_NMR = training_NMR[:,3:]                                     #assign NMR specrtra for training
training_Y_NMR = training_NMR[:,:3]                                     #assign concentration tags of NMR for training
training_X_UV =  training_UV[:,5:]                                      #assign UV specrtra for training
training_Y_UV =  training_UV[:,1:5]                                     #assign UV concentration tags of UV for training
#%% normalization of NMR and UVvis spectra

training_X_NMR_norm = np.empty(np.shape(training_X_NMR))                 #normalization of NMR spectra
for i in range(len(training_X_NMR)):
   training_X_NMR_norm[i] = training_X_NMR[i] / 1000  
   
training_X_UV_norm = np.empty(np.shape(training_X_UV))                   #normalization of UV spectra
for i in range(len(training_X_UV)):
    training_X_UV_norm[i] = training_X_UV[i] / 0.6   

#%%normallization of concentrations

training_Y_UV_norm = np.empty(np.shape(training_Y_UV))                   #normalization of UV concentrations
for i in range(len(training_Y_UV)):
        training_Y_UV_norm[i,0] = training_Y_UV[i,0] / 0.1
        training_Y_UV_norm[i,1] = training_Y_UV[i,1] / 0.28
        training_Y_UV_norm[i,2] = training_Y_UV[i,2] / 0.1
        training_Y_UV_norm[i,3] = training_Y_UV[i,3] / 0.28      

training_Y_NMR_norm = np.empty(np.shape(training_Y_NMR))                 #normalization of NMR concentrations
for i in range(len(training_Y_NMR)):
        training_Y_NMR_norm[i,0] = training_Y_NMR[i,0] / 0.28
        training_Y_NMR_norm[i,1] = training_Y_NMR[i,1] / 0.1
        training_Y_NMR_norm[i,2] = training_Y_NMR[i,2] / 0.28

  
#%% prepare for cnn
training_X_UV_norm = training_X_UV_norm.reshape((training_X_UV_norm.shape[0],training_X_UV_norm.shape[1],1))            #reshape UV data
training_X_NMR_norm = training_X_NMR_norm.reshape((training_X_NMR_norm.shape[0],training_X_NMR_norm.shape[1],1))        #reshape NMR data

#%% define the model
path4 = '/'                                     # insert path where the model should be saved
model_path_name = 'model_1.hdf5'                # insert path where the model should be saved

# import NMR
visible_NMR = Input(shape=(600,1))         #input NMR layer (spectrum size)
# import UV/vis
visible_UV = Input(shape=(187,1)           #input UV layer (spectrum size)
                   
#architecture for the ANN for NMR
conv1_NMR = LocallyConnected1D(filters=16, kernel_size=9, strides=9, activation ='elu')(visible_NMR) #convolutional layer for NMR
flat_NMR = Flatten()(conv1_NMR)                                                           #flatten of the NMR convolutional layer
hidden11_NMR = Dense(27, activation='elu')(flat_NMR)                                      #dense layer 1
hidden12_NMR = Dense(9, activation='elu')(hidden11_NMR)                                   #dense layer 2
output_NMR = Dense(3, activation='relu')(hidden12_NMR)                                    #output NMR layer

#architecture for the ANN for UV/vis
conv1_UV = LocallyConnected1D(filters=10, kernel_size=5, strides=2, activation ='elu')(visible_UV)   #convolutional layer for UV
flat_UV = Flatten()(conv1_UV)                                                                        #flatten of the UV convolutional layer
hidden11_UV = Dense(64, activation='relu')(flat_UV)                                                  #dense layer 1       
output_UV = Dense(32, activation='relu')(hidden11_UV)                                                #dense layer 2 (output of UV)

#architecture for the ANN to merge ANN for NMR and ANN for UV/vis
merge1 = concatenate([output_NMR,output_UV])                                        #combining UV and NMR layers
hidden11 = Dense(99, activation='relu')(merge1)                                     #dense layer 1
hidden12 = Dense(64, activation='relu')(hidden11)                                   #dense layer 2
hidden13 = Dense(16, activation='relu')(hidden11)                                   #dense layer 3
output = Dense(4, activation='relu')(hidden13)                                      #output layer
model = Model(inputs=[visible_NMR , visible_UV], outputs=[output_NMR, output])      #assign inputs and outputs of the model

# summarize layers
print(model.summary())
# compile the ANN model
learning_rate = 0.001
optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=10e-8, decay=0, amsgrad=False)
model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
# define learning and checkpointer
learning = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=0.001)
checkpointer = ModelCheckpoint(filepath = path4 + model_path_name, monitor='val_loss', verbose=1, save_best_only=True)

#%% train the model with validation split
Epochs = 1000
hist = model.fit([training_X_NMR_norm, training_X_UV_norm],[training_Y_NMR_norm, training_Y_UV_norm], epochs=Epochs, batch_size=1000,
                verbose=1, validation_split = 0.2, callbacks=[checkpointer, learning])  

loss = hist.history['loss']
val_loss = hist.history['val_loss']
plt.figure(1)
plt.plot(range(Epochs), loss,'r-', markersize=1, label='loss')
plt.plot(range(Epochs), val_loss,'g-', markersize=1, label='val_loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.title("Evaluation of loss")
plt.ylim((0.000001,1))
plt.yscale("log")




