## calc_AccordDisc : calculate Accordance and Discordance as in Meskaldji et al 2016 (NeuroImage: Clinical)
#
# Based on:
#    D.-E. Meskaldji, M. G. Preti, T. Bolton, M.-L. Montandon, C. K. Rodriguez, S. Morgenthaler, P. Giannakopoulos, S. Haller, D. Van De Ville.
#    Prediction of Long-Term Memory Scores in MCI Based on Resting-State fMRI   
#    Neuroimage: Clinical, 2016, 12, 785-795
#
# Input is:
#   1) a Numpy Array where each column is a time series and each row is a time point
#   2) an optional threshold used to binarie the time series (this is optional;
#      in the absence of this threshold, the deafult value of 0.8 is used.
# Output is two square Numpy Arrays:
#   the first contains the Accordance between all time series
#   the second contains the Discordance between all time series

import numpy as np

def calc_AccordDisc(ts, quantileThreshold = 0.8, verbose=True):
    ## Median-center and normalize the time-series as in Meskaldji et al 2016
    # Median-centering
    ts = np.subtract(ts, np.median(ts, axis=0))
    # Normalize by dividing by the median absolute deviation (MAD)
    mad = np.median(np.abs(ts - np.median(ts, axis=0)), axis=0)
    ts = np.divide(ts, mad)
    
    if verbose: print("Calculating Accordance and Discordance")
    
    numTimePoints, numTS = ts.shape
    if verbose: print(f"Input: number of time series = {numTS}")
    if verbose: print(f"Input: number of time points = {numTimePoints}")
    if verbose: print(f"Using quantile threshold = {quantileThreshold}")

    ## binarize the time-series
    # get upper and lower limits
    ul = np.quantile(ts, quantileThreshold, axis=0)
    ll = np.quantile(ts, 1 - quantileThreshold, axis=0)
    # apply upper and lower limits
    ul_ts = (ts > ul[np.newaxis, :]).astype(int)
    ll_ts = (ts < ll[np.newaxis, :]).astype(int) * -1
    
    ### Calculate Accordance and Discordance
    # containers
    accordance_all = np.empty((numTS, numTS))
    discordance_all = np.empty((numTS, numTS))
    
    ## Prepare parts of the denominators
    # scalar products of each column by itself in ul_ts and ll_ts
    # (obs: the scalar product of each column by itself is the same as
    #       the sum of element-wise square)
    scalar_product_ul_ts = np.sum(np.square(ul_ts), axis=0)
    scalar_product_ll_ts = np.sum(np.square(ll_ts), axis=0)
    # square root of the sum of scalar products of ul and ll
    sigma_all = np.sqrt(scalar_product_ul_ts + scalar_product_ll_ts)
    
    ## Prepare all possible denominators
    # Denominator of time series pair i and j = sigma_all[i] * sigma_all[j]
    # So the outer product of sigma_all by itself calculates all denominators
    denominators = np.outer(sigma_all, sigma_all)
    # Avoid division by zero
    denominators[denominators == 0] = np.finfo(np.float32).eps
    
    ## loop through all possible pairs of regions 
    for i in range(numTS):
        for j in range(numTS):
            denominator = denominators_all[i, j]
            accordance = ( (ul_ts[:,i] @ ul_ts[:,j]) + (ll_ts[:,i] @ ll_ts[:,j]) ) / denominator
            discordance = ( (ul_ts[:,i] @ ll_ts[:,j]) + (ll_ts[:,i] @ ul_ts[:,j]) ) / denominator
            accordance_all[i, j] = accordance
            discordance_all[i, j] = discordance
    if verbose: print('Done calculating Accordance and Discordance')
    return accordance_all, discordance_all
