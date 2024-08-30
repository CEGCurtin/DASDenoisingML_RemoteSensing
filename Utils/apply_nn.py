# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 10:25:47 2023

@author: 253863J
"""
# import numpy as np
# from tensorflow.keras.models import *
from scipy import signal

from Utils.preproc_for_nn import data_norm, data_pad, data_patching, data_unpatching
from Utils.RP_Proc import fRingdownRemoval

def fDenoiseDAS(D, model, 
                apply_detrend=False, apply_ringDownRemoval=False):
    """
    Apply DAS denoising using input trained neural network.

    Parameters
    ----------
    D : np.array
        Input data array (n_time x n_chan).
    model : keras.engine.functional.Functional
        Trained neural model (needs to be loaded before).
    apply_detrend : bool, optional
        If True, apply detrend on the time axis. The default is False.
    apply_ringDownRemoval : TYPE, optional
        If True, remove median on each channel. The default is False.

    Returns
    -------
    st_array_out : np.array
        Input data array (n_time x n_chan).

    """
    
    #=== Set patch and step sizes from model input shape ====
    #-- Get model input shape
    model_input_shape = model.input_shape[-2:]
    
    #-- Set patch size and step size based on model input
    patch_size = [model_input_shape[0],model_input_shape[1]]
    step_size = [int(patch_size[0]/2), int(patch_size[1]/2)]
    
    #=== Data pre-processing ===
    #-- Optional trend and median removal
    if apply_detrend:
        D = signal.detrend(D,axis=0)
    if apply_ringDownRemoval:
        D = fRingdownRemoval(D)
        
    #-- Data normalization
    st_array = D.reshape(1,D.shape[0],D.shape[1]) # reshape to 3D (n_shot,nt, nchan) for data_norm
    st_array, mean_per_shot_X, norm_factor_per_shot_X = data_norm(st_array)
    
    #-- Data padding        
    st_array_pad = data_pad(st_array, model_input_shape)
    padded_size = st_array_pad.shape #store shpe for unpacthing
    
    #-- Data patching
    model_in = data_patching(st_array_pad, patchsize=patch_size, step=step_size, verbose=True)
    
    #==== Run Model ====
    preds = model.predict(model_in, batch_size = 24, verbose=0) #may want to have batch_size and verbose as input
    
    #==== Post-processing ====
    st_array_out = preds
    
    #-- Reshape back (undo patching)
    outsize = padded_size
    st_array_out = data_unpatching(st_array_out,outsize=outsize, 
                                   patchsize=patch_size, step=step_size, 
                                   output_weights_QC=False)

    #-- Un-normalise data
    st_array_out = st_array_out * norm_factor_per_shot_X[0] + mean_per_shot_X[0]
        
    #-- Cut back to unpadded size
    st_array_out = st_array_out[0,:st_array.shape[1],:st_array.shape[2]]
        
    #-- Optional median removal
    if apply_ringDownRemoval:
        st_array_out = fRingdownRemoval(st_array_out)
        
    return st_array_out   