# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 09:38:05 2022

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
from keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau
from keras.models import load_model
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import Dense, Input, Flatten, LocallyConnected1D,Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
maxnorm = max_norm(max_value=2, axis=0)
#%%import training data and validation data
path1 = ''                              # insert path where the training data and validation data is saved

simulated_set = np.load(path1+'NMR_simulated_set.npy')
experimental_mixtures_set = np.load(path1+'NMR_experimental_mixtures_set.npy')
dynamic_set = np.load(path1+'NMR_dynamic_set.npy')
train_set = np.concatenate((experimental_mixtures_set, simulated_set), axis=0) 

#shuffel data sets
np.random.shuffle(dynamic_set )
np.random.shuffle(train_set)

train_spec = train_set[:,3:]            #assign spectra of the training set to a variable 
train_conc = train_set[:,0:3]           #assign concentrations of the training set to a variable 
val_spec = dynamic_set [:,3:]           #assign spectra of the validation set to a variable 
val_conc = dynamic_set [:,0:3]          #assign concentrations of the validation set to a variable 

#%% normalization of the NMR spectrum of training and validation set
train_spec_norm = np.empty(np.shape(train_spec))                    # normalization of the training spectra 
for i in range(len(train_spec)):
    train_spec_norm[i] = train_spec[i] / 1000

val_spec_norm = np.empty(np.shape(val_spec))                        # normalization of the validation spectra
for i in range(len(val_spec)):
    val_spec_norm[i] = val_spec[i] / 1000

train_spec = train_spec_norm
val_spec = val_spec_norm

#%%  normalization of concentration of the training and validation set
train_conc_norm = np.empty(np.shape(train_conc)) 
for i in range(len(train_conc)):                                # normalization of the training concentrations
        train_conc_norm[i,0] = train_conc[i,0] / 0.28
        train_conc_norm[i,1] = train_conc[i,1] / 0.1
        train_conc_norm[i,2] = train_conc[i,2] / 0.28
    
val_conc_norm  = np.empty(np.shape(val_conc)) 
for i in range(len(val_conc)):                                  # normalization of the validation concentrations
    val_conc_norm[i,0] = val_conc [i,0] / 0.28
    val_conc_norm[i,1] = val_conc [i,1] / 0.1
    val_conc_norm[i,2] = val_conc [i,2] / 0.28
   
train_conc = train_conc_norm     
val_conc = val_conc_norm

#%% reduce the data for the NMR spectra
xp = np.linspace(7, 9, 1148)    
x = np.linspace(7,9 ,600)

train_spec_red = np.zeros((len(train_spec),600))                # reduce the data points of each NMR spectra
for i in range(0,len(train_spec)):  
    interpol = np.interp(x, xp, train_spec[i], left=None, right=None, period=None)
    train_spec_red[i] = interpol

train_spec = train_spec_red

val_spec_red = np.zeros((len(val_spec),600))                   # reduce the data points of each NMR spectra
for i in range(0,len(val_spec)):  
    interpol = np.interp(x, xp, val_spec[i], left=None, right=None, period=None)
    val_spec_red[i] = interpol

val_spec= val_spec_red

#%% define the model and define the architecture of the ANN
path2 = ''                                      # insert path where the model should be saved
model_path_name = 'model_1.hdf5'                # insert model name  

# Architecture of the NMR model
visible = Input(shape=(600,))                       # input layer (spectrum size)
hidden11 = Dense(600,activation='relu')(visible)    # dense layer 1
hidden12 = Dense(243, activation='relu')(hidden11)  # dense layer 2
hidden13 = Dense(81, activation='relu')(hidden12)   # dense layer 3
hidden14 = Dense(9, activation='relu')(hidden13)    # dense layer 4
output = Dense(3, activation='relu')(hidden14)      # dense layer 5
model = Model(inputs=visible, outputs=output)       # output layer
# summarize model
print(model.summary())
# compile the fully dense ANN model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# define learning and checkpointer
learning = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=0.001)
checkpointer = ModelCheckpoint(filepath = path2 + model_path_name, monitor='val_loss', verbose=1, save_best_only=True)


#%% training of the model with validation data
Epochs = 2000
hist = model.fit(train_spec, train_conc, epochs=Epochs, batch_size=500, verbose=1, validation_data=(val_spec, val_conc) , callbacks=[checkpointer, learning])
loss = hist.history['loss']
val_loss = hist.history['val_loss']

plt.figure(1)
plt.plot(range(Epochs), loss,'r-', markersize=1, label='loss')
plt.plot(range(Epochs), val_loss,'g-', markersize=1, label='val_loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.title("Evaluation of loss")
plt.ylim((0.000001,100))
plt.yscale("log")
#load the best model
model = load_model(path2 + model_path_name)

#Evaluate model
scores = model.evaluate(val_spec, val_conc, batch_size = (len(val_conc)))
