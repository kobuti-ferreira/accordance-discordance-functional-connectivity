## import libraries
import pandas as pd
import numpy as np


###########################################################################################################


## calc_AccordDisc : calculate Accordance and Discordance as in Meskaldji et al 2016 
# Input is:
# 1) a Pandas DataFrame where each column is a time series and each row is a time point
# 2) an optional threshold used to binarie the time series (this is optional;
# in the absence of this threshold, the deafult value of 0.8 is used.
# Output is two square Pandas DataFrame
# the first DataFrame contains the Accordance between all time series
# the second DataFrame contains the Discordance between all time series

def calc_AccordDisc(ts, quantileThreshold = 0.8, verbose=True):
    if verbose: print("Calculating Accordance and Discordance")
    numTimePoints, numTS = ts.shape
    if verbose: print( "Input: number of time series = " + str(numTS) )
    if verbose: print( "Input: number of time points = " + str(numTimePoints) )
    if verbose: print("Using quantile threshold = " + str(quantileThreshold))
    ## binarize the time-series
    # get upper and lower limits
    ul = ts.quantile(quantileThreshold)
    ll = ts.quantile(1 - quantileThreshold)
    # repeat the limits to match the shape of ts
    ul = np.tile(ul.values, (numTimePoints,1))
    ll = np.tile(ll.values, (numTimePoints,1))
    # binarize the time-series
    ul_ts = np.where(ts > ul, 1, 0)
    ll_ts = np.where(ts < ll, -1, 0)
    
    ### Calculate Accordance and Discordance
    # containers where the names of the columns ad rows are the same as in the input
    accordanceDF = pd.DataFrame(columns = ts.columns, index = ts.columns, dtype=float)
    discordanceDF = pd.DataFrame(columns = ts.columns, index = ts.columns, dtype=float)
    
    ## Prepare parts of the denominators
    # scalar products of each column by itself in ul_ts and ll_ts
    # (obs: the scalar product of each column by itself is the same as
    #       the sum of element-wise square)
    scalar_product_ul_ts = np.sum(np.square(ul_ts), axis=0)
    scalar_product_ll_ts = np.sum(np.square(ll_ts), axis=0)
    # sum of scalar products of ul and ll
    sum_scalar_product_ul_ll = scalar_product_ul_ts + scalar_product_ll_ts
    # square root of the sum of scalar products of ul and ll
    sqrt_sum_scalar_product_ul_ll = np.sqrt(sum_scalar_product_ul_ll)
    
    ## loop through all possible pairs of regions 
    for i in range(numTS):
        denominator_i = sqrt_sum_scalar_product_ul_ll[i]
        for j in range(numTS):
            denominator_j = sqrt_sum_scalar_product_ul_ll[j]
            denominator = denominator_i * denominator_j
            accordance = ( (ul_ts[:,i] @ ul_ts[:,j]) + (ll_ts[:,i] @ ll_ts[:,j]) ) / denominator
            discordance = ( (ul_ts[:,i] @ ll_ts[:,j]) + (ll_ts[:,i] @ ul_ts[:,j]) ) / denominator
            accordanceDF.iloc[i, j] = accordance
            discordanceDF.iloc[i, j] = discordance
    return accordanceDF, discordanceDF
    if verbose: print('Done calculating Accordance and Discordance')