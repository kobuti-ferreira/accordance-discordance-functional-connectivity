## calc_AccordDisc : calculate Accordance and Discordance as in Meskaldji et al 2016 (NeuroImage: Clinical)
# Input is:
#   1) a Numpy Array where each column is a time series and each row is a time point
#   2) an optional threshold used to binarie the time series (this is optional;
#      in the absence of this threshold, the deafult value of 0.8 is used.
# Output is two square Numpy Arrays:
#   the first contains the Accordance between all time series
#   the second contains the Discordance between all time series

import numpy as np

def calc_AccordDisc(ts, quantileThreshold = 0.8, verbose=True):    
    if verbose: print("Calculating Accordance and Discordance")
    
    numTimePoints, numTS = ts.shape
    if verbose: print( "Input: number of time series = " + str(numTS) )
    if verbose: print( "Input: number of time points = " + str(numTimePoints) )
    if verbose: print("Using quantile threshold = " + str(quantileThreshold))

    ## binarize the time-series
    # get upper and lower limits
    ul = np.quantile(ts, quantileThreshold, axis=0)
    ll = np.quantile(ts, 1 - quantileThreshold, axis=0)
    
    # repeat the limits to match the shape of ts
    ul = np.tile(ul, (numTimePoints, 1))
    ll = np.tile(ll, (numTimePoints, 1))
    
    # binarize the time-series
    ul_ts = np.where(ts > ul, 1, 0)
    ll_ts = np.where(ts < ll, -1, 0)
    
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
    # sum of scalar products of ul and ll
    sum_scalar_product_ul_ll = scalar_product_ul_ts + scalar_product_ll_ts
    # square root of the sum of scalar products of ul and ll
    sigma_all = np.sqrt(sum_scalar_product_ul_ll)
    
    ## Prepare all possible denominators
    # Denominator of time series pair i and j = all_sigmas[i] * all_sigmas[j]
    #     while all_sigmas[i] = sqrt_sum_scalar_product_ul_ll[i]
    # So to get all possible denominators, calculate the outer product of
    #      sqrt_sum_scalar_product_ul_ll by itself
    denominators_all = np.outer(sigma_all, sigma_all)    
    
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