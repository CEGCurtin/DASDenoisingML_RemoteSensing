# -*- coding: utf-8 -*-
"""
Pre-processing routines for neural network training and application.

Created on Thu May 11 11:37:10 2023

@author: 253863J
"""
import numpy as np
import itertools

# FUNCTION data_norm
def data_norm(D, std_norm=True):
    """
    Data normalization: remove mean and optionally divide by standard deviation.

    Parameters
    ----------
    D : np.array
        Input data array (n_shots x n_time x n_chan).
    std_norm : bool, optional
        Flag for normalisation by standard deviation. If True, divide by standard dev. 
        The default is True.

    Returns
    -------
    D_out : np.array
        Normalized data array (n_shots x n_time x n_chan).
            
   mean_per_shot : np.array
       Mean value for each shot (n_shots x 1).    
   
   norm_factor_per_shot : np.array
       Stdev for each shot (n_shots x 1). 
    """
    D_out = np.zeros(D.shape)
    n_shots = D.shape[0]
    
    mean_per_shot = np.zeros((n_shots,1))
    norm_factor_per_shot = np.zeros((n_shots,1))
    
    for i_shot in range(0,n_shots): #should avoid loop
        # Remove mean
        mean_per_shot[i_shot] = D[i_shot,:,:].mean(axis=None) 
        D_out[i_shot,:,:] = D[i_shot,:,:] - mean_per_shot[i_shot]
        # Normalise data
        norm_factor_per_shot[i_shot] = np.std(D_out[i_shot,:,:], axis=None) # get std of whole signal for each shot
        D_out[i_shot,:,:] = D_out[i_shot,:,:] / norm_factor_per_shot[i_shot] # normalise
        
    return D_out, mean_per_shot, norm_factor_per_shot

# FUNCTION data_pad
def data_pad(D, model_input_shape):
    """
    Pad data before to get correct block input shapes for model.

    Parameters
    ----------
    D : np.array
        Input data array (n_shots x n_time x n_chan).
    model_input_shape : tuple
        Input size for neural network model.

    Returns
    -------
    D_out : np.array
        Padded data array (nshots x n_time_pad x n_chan_pad).

    """
    
    # Get patch sizes
    n_time_patch = model_input_shape[0]
    n_chan_patch = model_input_shape[1]
    
    # Initialise padded data
    n_time = D.shape[1]
    n_chan = D.shape[2]
    n_time_pad = D.shape[1] + n_time_patch - (n_time % n_time_patch)
    n_chan_pad = D.shape[2] + n_chan_patch - (n_chan % n_chan_patch)
    D_out = np.zeros((D.shape[0],n_time_pad,n_chan_pad))
    
    # Pad arrays    
    for i_shot in range(0,D.shape[0]):
        D_out[i_shot,:,:] = np.pad(D[i_shot,:,:], 
                                   ((0, n_time_patch - (n_time % n_time_patch)), 
                                    (0, n_chan_patch - (n_chan % n_chan_patch))), 
                                     mode='reflect')
        
    return D_out

# FUNCTION data_patching
def data_patching(D, patchsize=[64, 64], step=[16, 16], 
                  verbose=True):
    """
    Extract overlapping patches from input matrix. 

    Parameters
    ----------
    D : np.array
        Input matrix (n_shots x n_time x n_chan).
    patchsize : list, optional
        Size of the patches (n_time x n_chan). The default is [64, 64].
    step : list, optional
        Step for overlapping. The default is [16, 16].
    verbose : bool, optional
        Display number of extracted patches. The default is True.

    Returns
    -------
    patches : np.array
        Output patches (n_patches x n_time_patch x n_chan_patch)..
        
    Notes
    -----
    Arrays should be padded prior to running the patching script.

    """

    # Get number of shots
    n_shots = D.shape[0]
    
    # Find starting indices
    x1_start_indices = np.arange(0, D.shape[1] - patchsize[0] + 1, step=step[0])
    x2_start_indices = np.arange(0, D.shape[2] - patchsize[1] + 1, step=step[1])
    starting_indices = list(itertools.product(x1_start_indices, x2_start_indices))

    # Extract patches
    n_patches_per_shot = len(starting_indices)
    if verbose:
        print('Extracting {:.0f} patches'.format(n_patches_per_shot*n_shots))
    
    patches = np.zeros([n_patches_per_shot*n_shots, patchsize[0], patchsize[1]])

    for i_shot in range(0,n_shots):
        for i, pi in enumerate(starting_indices):
            patches[i_shot*n_patches_per_shot+i] = D[i_shot, pi[0]:pi[0]+patchsize[0], pi[1]:pi[1]+patchsize[1]]

    return patches

# FUNCTION data_unpatching
def data_unpatching(patches, outsize,
                    patchsize=[64, 64], step=[16, 16],
                    weighting='Hann', output_weights_QC=False):
    """
    Recreate matrix of original size from overlapping patches. 
    Average is taken where patches are overlapping.

    Parameters
    ----------
    patches : np.array
        Input patches (n_patches x n_time_patch x n_chan_patch).
    outsize : tuple
        Output size (n_shots x n_time_pad x n_chan_pad).
    patchsize : list, optional
        Size of the patches (n_time x n_chan). The default is [64, 64].
    step : list, optional
        Step for overlapping. The default is [16, 16].
    weighting: str, optional
        Type of weighting. The default is 'Hann'.
    output_weights_QC: bool, optional
        Flag to output weighting matrix for QC. The default is False. 
    

    Returns
    -------
    D : np.array
        Output matrix (n_shots x n_time x n_chan).
        
    Notes
    -----
    Arrays should be padded prior to running the patching script.
    Only Hann weighting option is coded; should implement avg. 

    """

    # Get number of shots
    n_shots = outsize[0]
    
    # Find starting indices
    x1_start_indices = np.arange(0, outsize[1] - patchsize[0] + 2, step=step[0])
    x2_start_indices = np.arange(0, outsize[2] - patchsize[1] + 2, step=step[1])
    starting_indices = list(itertools.product(x1_start_indices, x2_start_indices))
    
    # Create weight matrix
#     weights = fWeightHann(fullsize, step, reverse_flg=reverse_flg)
#     print('weights' + str(weights.shape))
#     weights = np.zeros((outsize[1],outsize[2]))
#     for i, pi in enumerate(starting_indices):
#         weights[pi[0]:pi[0]+patchsize[0], pi[1]:pi[1]+patchsize[1]] += 1
    w_Hann, wUp_Hann, wDown_Hann, wLeft_Hann, wRight_Hann,\
    wUpLeft_Hann, wUpRight_Hann, wDownLeft_Hann, wDownRight_Hann = wHann_submatrices(patchsize, reverse_flg=False)

    # Extract patches
    n_patches_per_shot = len(starting_indices)
    D = np.zeros([n_shots, outsize[1], outsize[2]])
    weights_QC = np.zeros([outsize[1], outsize[2]])
    print(D.shape)
    for i_shot in range(0,n_shots):
        for i, pi in enumerate(starting_indices):
            # Define weights depending on patch location
            if pi[0]==0:
                if pi[1]==0:
                    weights = wUpLeft_Hann 
                elif pi[1]==np.max(x2_start_indices):
                    weights = wDownLeft_Hann
                else:
                    weights = wLeft_Hann 
            elif pi[0]==np.max(x1_start_indices):
                if pi[1]==0:
                    weights = wUpRight_Hann 
                elif pi[1]==np.max(x2_start_indices):
                    weights = wDownRight_Hann
                else:
                    weights = wRight_Hann 
            else:
                if pi[1]==0:
                    weights = wUp_Hann 
                elif pi[1]==np.max(x2_start_indices):
                    weights = wDown_Hann
                else:
                    weights = w_Hann
                    
            # Multiply current patch by weights
            D[i_shot, pi[0]:pi[0]+patchsize[0], pi[1]:pi[1]+patchsize[1]] += weights*patches[i_shot*n_patches_per_shot+i]
            
            # Build QC weighting matrix
            if (i_shot==0) and (output_weights_QC):
                weights_QC[pi[0]:pi[0]+patchsize[0], pi[1]:pi[1]+patchsize[1]] += weights
    
    if output_weights_QC:
        return D, weights_QC
    else:
        return D

#%% Utils for Hann windowing (used to define weight for overlapping patches)

# FUNCTION window_Hann
def window_Hann(in_size):
    """
    Compute 1D Hann window of specified size.

    Parameters
    ----------
    in_size : int
        Length of Hann window.

    Returns
    -------
    w : np.array
        1D Hann window of length input_size.

    """
    w = np.zeros(in_size)
    for i in range(0,in_size):
        w[i] = 0.5*(1-np.cos(2*np.pi*(i+1)/(in_size-1)))
    return w
        
# FUNCTION wHann_submatrices
def wHann_submatrices(patchsize, reverse_flg=True):
    """
    Compute 2D Hann windows (including corner and edge windows).

    Parameters
    ----------
    patchsize : list
        List containg width and height of patch size e.g. [width,height].
    reverse_flg : bool, optional
        Flag to reverse input patch size. The default is True.

    Returns
    -------
    w_Hann : np.array
        Center Hann weighting window.
    wUp_Hann : np.array
        Upper edge Hann weighting window.
    wDown_Hann : np.array
        Lower edge Hann weighting window.
    wLeft_Hann : np.array
        Left edge Hann weighting window.
    wRight_Hann : np.array
        Right edge Hann weighting window.
    wUpLeft_Hann : np.array
        Upper left corner Hann weighting window.
    wUpRight_Hann : np.array
        Upper right corner Hann weighting window.
    wDownLeft_Hann : np.array
        Lower left corner Hann weighting window.
    wDownRight_Hann : np.array
        Lower right corner Hann weighting window.

    """
    if reverse_flg==True:
        size = np.flip(patchsize)
    else:
        size = patchsize
    print('wHann_submatrices'+str(size))
    
    w_Hann = np.zeros((size))

    width = size[0]
    height = size[1]

    wI = window_Hann(width)
    wJ = window_Hann(height)
    
    # Weight matrix for middle patch
    for i in range(0,width):
        for j in range(0,height):
            w_Hann[i,j] = wI[i]*wJ[j]
            
    # Generate edge and corner matrices
    wUp_Hann = np.zeros((size))
    wDown_Hann = np.zeros((size))
    wLeft_Hann = np.zeros((size))
    wRight_Hann = np.zeros((size))
    wUpLeft_Hann = np.zeros((size))
    wUpRight_Hann = np.zeros((size))
    wDownLeft_Hann = np.zeros((size))
    wDownRight_Hann = np.zeros((size))
    # -- Edge matrices
    #wUpHann
    for i in range(0,width):
        for j in range(0, height):
            if j<height/2 - 1:
                wUp_Hann[i,j] = wI[i]
            else:
                wUp_Hann[i,j] = wI[i]*wJ[j]
    #wDownHann
    for i in range(0,width):
        for j in range(0, height):
            if j>height/2 - 1:
                wDown_Hann[i,j] = wI[i]
            else:
                wDown_Hann[i,j] = wI[i]*wJ[j]
    #wLeftHann
    for i in range(0,width):
        for j in range(0,height):
            if i<width/2 - 1:
                wLeft_Hann[i,j] = wJ[j]
            else:
                wLeft_Hann[i,j] = wI[i]*wJ[j] 
    #wRightHann
    for i in range(0,width):
        for j in range(0,height):
            if i>width/2 - 1:
                wRight_Hann[i,j] = wJ[j]
            else:
                wRight_Hann[i,j] = wI[i]*wJ[j]
    # -- Corner matrices
    #wUpLeft
    for i in range(0,width):
        for j in range(0, height):
            if (i<=width/2 - 1) & (j<=height/2-1):
                wUpLeft_Hann[i,j] = 1
            elif (i>width/2 - 1) & (j<height/2-1):
                wUpLeft_Hann[i,j] = wI[i]
            elif (i<width/2 - 1) & (j>height/2-1):
                wUpLeft_Hann[i,j] = wJ[j]
            else:
                wUpLeft_Hann[i,j] = wI[i]*wJ[j]    
    half_width = int(width/2)
    half_height = int(height/2)
    #wUpRight
    wUpRight_Hann[0:half_width,0:half_height] = wUp_Hann[0:half_width,0:half_height]
    wUpRight_Hann[half_width:width,0:half_height] = 1
    wUpRight_Hann[0:half_width,half_height:height] = w_Hann[0:half_width,half_height:height]
    wUpRight_Hann[half_width:width,half_height:height] = wRight_Hann[half_width:width,half_height:height]
    #wDownLeft
    wDownLeft_Hann[0:half_width,0:half_height] = wLeft_Hann[0:half_width,0:half_height]
    wDownLeft_Hann[half_width:width,0:half_height] = w_Hann[half_width:width,0:half_height]
    wDownLeft_Hann[0:half_width,half_height:height] = 1
    wDownLeft_Hann[half_width:width,half_height:height] = wDown_Hann[half_width:width,half_height:height]
    #wDownRight
    wDownRight_Hann[0:half_width,0:half_height] = w_Hann[0:half_width,0:half_height]
    wDownRight_Hann[half_width:width,0:half_height] = wRight_Hann[half_width:width,0:half_height]
    wDownRight_Hann[0:half_width,half_height:height] = wDown_Hann[0:half_width,half_height:height]
    wDownRight_Hann[half_width:width,half_height:height] = 1
    
    return w_Hann, wUp_Hann, wDown_Hann, wLeft_Hann, wRight_Hann,\
           wUpLeft_Hann, wUpRight_Hann, wDownLeft_Hann, wDownRight_Hann

        
    

