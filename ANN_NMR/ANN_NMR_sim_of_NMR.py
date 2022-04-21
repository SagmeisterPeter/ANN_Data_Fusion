# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 16:49:36 2021

@author: Peter Sagmeister
Code accompanying the publication "Artificial Neural Networks and Data Fusion Enable Concentration Predictions for Inline Process Analyticsy" 
Authors: Peter Sagmeister, Robin Hierzegger, Jason D. Williams, C. Oliver Kappe and Stefan Kowarik
publication link:   https://
data available at:  https://doi.org/10.5281/zenodo.6066166
"""

import numpy as np

#%% load in mean NMR spectra
path1 = '/NMR_pure_spectrum_2ClBA.npy'              # insert path for the file
conc_spec1 = np.load(path1)
spec_1 = conc_spec1[3:]                             # assign spectra to a variable
conc1 = conc_spec1[0:3]                             # assign concentration to a variable
path2 = '/NMR_pure_spectrum_3N-2ClBA.npy'           # insert path for the file
conc_spec2 = np.load(path2)
spec_2 = conc_spec2[3:]                             # assign spectra to a variable
conc2 = conc_spec2[0:3]                             # assign concentration to a variable

path3 = '/NMR_pure_spectrum_5N-2ClBA.npy'           # insert path for the file
conc_spec3 = np.load(path3)
spec_3 = conc_spec3[3:]                             # assign spectra to a variable
conc3 = conc_spec3[0:3]                             # assign concentration to a variable

#%% functions for creating the concentration frame

def gen_conc(gc1,gc2,gc3):                      
    c_range = np.linspace(gc1,gc2,gc3)
    c_sim1 = []
    for i in range(0,gc3): 
        for j in range(0,gc3):
            c_sim1.append(c_range[i])
        
    c_sim1 = np.array(c_sim1)
    c_sim1 = c_sim1.reshape(gc3*gc3,1)

    c_sim2 = []
    for k in range(0,gc3):
        c_sim2.append(c_range)
    
    c_sim2 = np.array(c_sim2)
    c_sim2 = c_sim2.reshape(gc3*gc3,1)

    c_sim = np.concatenate((c_sim1,c_sim2),axis=1)

    c_sim3 = []
    for i in range(0,gc3*gc3): 
        for j in range(0,gc3):
            c_sim3.append(c_sim1[i])
    c_sim3 = np.array(c_sim3)
    c_sim3 = c_sim3.reshape(gc3*gc3*gc3,1)

    c_t = []
    for k in range(0,gc3):
        c_t.append(c_sim)
    c_t = np.array(c_t)
    c_t = c_t.reshape(gc3*gc3*gc3,2)

    c_sim_t = np.concatenate((c_sim3,c_t),axis=1) 
    return c_sim_t

#%% create a data frame for different simulated concentration levels and add for each level multiple spectra
sim_test_con = gen_conc(0.001,2,7)     #creating dataframe for levels

conc_set_1 = np.array([0.14, 0 , 0], dtype=np.float32) 
conc_set_2 = np.array([0, 0.05, 0], dtype=np.float32) 
conc_set_3 = np.array([0, 0, 0.14], dtype=np.float32) 

spec_set_1 = spec_1 / (conc1[0] / conc_set_1[0])
spec_set_2 = spec_2 / (conc2[1] / conc_set_2[1])
spec_set_3 = spec_3 / (conc3[2] / conc_set_3[2])

spec_set50_1 = np.empty(([50,np.size(spec_set_1)]))
spec_set50_2 = np.empty(([50,np.size(spec_set_2)]))
spec_set50_3 = np.empty(([50,np.size(spec_set_3)]))
conc_set50_1 = np.empty(([50,np.size(conc_set_1)]))
conc_set50_2 = np.empty(([50,np.size(conc_set_2)]))
conc_set50_3 = np.empty(([50,np.size(conc_set_3)]))

for i in range(0, np.size(conc_set50_1[:,0])):
    conc_set50_1[i,:] = conc_set_1
for i in range(0, np.size(conc_set50_2[:,0])):
    conc_set50_2[i,:] = conc_set_2
for i in range(0, np.size(conc_set50_3[:,0])):
    conc_set50_3[i,:] = conc_set_3
    
for i in range(0, np.size(spec_set50_1[:,0])):
    spec_set50_1[i,:] = spec_set_1 
for i in range(0, np.size(spec_set50_2[:,0])):
    spec_set50_2[i,:] = spec_set_2 
for i in range(0, np.size(spec_set50_3[:,0])):
    spec_set50_3[i,:] = spec_set_3 
    
#%% function for simulated NMR spectra

def specgen(c1,c2,c3):
    c_1 = c1*conc_set50_1
    c_2 = c2*conc_set50_2
    c_3 = c3*conc_set50_3
    c = c_1+c_2+c_3
    spec1 = c1*spec_set50_1
    spec2 = c2*spec_set50_2
    spec3 = c3*spec_set50_3
    mixspec = spec1+spec2+spec3
    noise_mixspec = np.empty(np.shape(mixspec))
    for i in range(0, np.size(mixspec[:,0])):
        noise_mixspec[i,:] = mixspec[i,:] + np.random.normal(loc=1.0, scale=2, size=1148)
    return c,noise_mixspec

#%%simulate the spectra based on the data frame
c_ati = []
spec_ati = []
for i in range(0,sim_test_con.shape[0]):
    cs,specs= specgen(sim_test_con[i,0],sim_test_con[i,1],sim_test_con[i,2])
    spec_ati.append(specs)
    c_ati.append(cs)
spec_ati = np.array(spec_ati)  
c_ati = np.array(c_ati)  
spec_ati = spec_ati.reshape(sim_test_con.shape[0]*50,1148)
c_ati = c_ati.reshape(sim_test_con.shape[0]*50,3)
data_ati = np.concatenate((c_ati,spec_ati),axis=1)


#%% implementation of slight shift to the right and slight shift to the left in the data
spec_ati_shift2 = np.zeros((spec_ati.shape))
spec_ati_shift3 = np.zeros((spec_ati.shape))

for j in range(0,17100,49):                 # shift data to the right
    for i in range(0,50):
        a = spec_ati[i+j,:i] 
        b = spec_ati[i+j,i:] 
        spec_ati_shift2[i+j,:] = np.concatenate((b,a),axis=0)

for j in range(0,17100,49):                 # shift data to the left
    for i in range(0,50):
        a = spec_ati[i+j,-i:] 
        b = spec_ati[i+j,:-i] 
        spec_ati_shift3[i+j,:] = np.concatenate((a,b),axis=0)

NMR_comb_notshift = np.concatenate((c_ati,spec_ati),axis=1)
NMR_comb_shifted_l = np.concatenate((c_ati,spec_ati_shift2),axis=1)
NMR_comb_shifted_r = np.concatenate((c_ati,spec_ati_shift3),axis=1)
NMR_comb = np.concatenate((NMR_comb_notshift,NMR_comb_shifted_l,NMR_comb_shifted_r),axis=0)

#%% save simulated one in a npy
dirname_save = ''                                       # insert path for the file
np.save(dirname_save+"/NMR_simulated_set.npy",NMR_comb)

