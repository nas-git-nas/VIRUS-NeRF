import numpy as np

def linInterpolateArray(
    x1:np.array,
    y1:np.array,
    x2:np.array,
    border_condition:str="nan",
):
    """
    Find corresponding y2 values for x2 values by linear interpolation,
    where x1 and y1 are two correlated arrays and used for interpolation.
    Apply original order of x2 to y2.
    Args:
        x1: input x values; np.array of shape (N,)
        y1: input y values; np.array of shape (N,)
        x2: output x values; np.array of shape (M,)
        border_condition: how to handle x2 values that are outside of the range of x1; str
            "nan": return nan for those values
            "nearest": return nearest y1 value for those values
    Returns:
        y2: output y values; np.array of shape (M,)
    """
    x1 = np.copy(x1)
    y1 = np.copy(y1)
    x2 = np.copy(x2)

    # check input
    if x1.shape != y1.shape:
        print(f"ERROR: data_fcts.linInterpolateArray: x1.shape != y1.shape")
    if border_condition not in ["nan", "nearest"]:
        print(f"ERROR: data_fcts.linInterpolateArray: border_condition not in ['nan', 'nearest']")
    
    if np.min(x2)<np.min(x1):
        print(f"Warning: data_fcts.linInterpolateArray: np.min(x2)={np.min(x2)} < np.min(x1)={np.min(x1)}")
        if border_condition == "nan":
            print(f"Warning: data_fcts.linInterpolateArray: returning nan for values outside of x1 range")
        else:
            print(f"Warning: data_fcts.linInterpolateArray: returning nearest y1 value for values outside of x1 range")
    if np.max(x2)>np.max(x1):
        print(f"Warning: data_fcts.linInterpolateArray: np.max(x2)={np.max(x2)} > np.max(x1)={np.max(x1)}")
        if border_condition == "nan":
            print(f"Warning: data_fcts.linInterpolateArray: returning nan for values outside of x1 range")
        else:
            print(f"Warning: data_fcts.linInterpolateArray: returning nearest y1 value for values outside of x1 range")

    # sort x1 and y1 by x1
    idxs_sort1 = np.argsort(x1)
    x1 = x1[idxs_sort1]
    y1 = y1[idxs_sort1]

    # sort x2
    idxs_sort2 = np.argsort(x2)
    x2 = x2[idxs_sort2]

    # find corresponding y2 values for x2 values by linear interpolation
    if border_condition == "nan":
        y2 = np.interp(x2, x1, y1, left=np.nan, right=np.nan)
    else:
        y2 = np.interp(x2, x1, y1, left=y1[0], right=y1[-1])

    # return y1 in original order of x2
    return y2[idxs_sort2]

def linInterpolateNans(
    arr:np.array,
):
    """
    Replace nan values in array by linear interpolation of closest valid values.
    Args:
        arr: input array; np.array of shape (N,)
    Returns:
        arr: array with replaced nan values; np.array of shape (N,)
    """
    arr = np.copy(arr)
    N = arr.shape[0]
    n = np.sum(~np.isnan(arr))

    if n == 0:
        print(f"ERROR: data_fcts.convolveIgnorNan: all values are nan")
        return arr
    
    if n == N:
        return arr

    # find next value above nan values
    arr_val_idxs = np.arange(N)[~np.isnan(arr)] # [0, N-1], (n,)
    cumsum = np.cumsum(~np.isnan(arr)) # (N,)
    next_val_idx_above = arr_val_idxs[np.clip(cumsum, 0, n-1)] # (N,)
    next_val_above = arr[next_val_idx_above] # (N,) 

    arr_val_idxs_inv = np.arange(N)[~np.isnan(np.flip(arr))] # [0, N-1], (n,)
    cumsum_inv = np.cumsum(~np.isnan(np.flip(arr))) # (N,)
    next_val_idx_below = arr_val_idxs_inv[np.clip(cumsum_inv, 0, n-1)] # (N,)
    next_val_idx_below = N - 1 - np.flip(next_val_idx_below)
    next_val_below = arr[next_val_idx_below] # (N,)
      
    # calculate weights for linear interpolation
    next_val_below_dist = (np.arange(N) - next_val_idx_below).astype(np.int64) # (N,)
    next_val_above_dist = (next_val_idx_above - np.arange(N)).astype(np.int64) # (N,)
    next_val_below_dist = np.where(
        next_val_below_dist<=0, 
        np.iinfo(np.int64).max,
        next_val_below_dist,
    )
    next_val_above_dist = np.where(
        next_val_above_dist<=0,
        np.iinfo(np.int64).max,
        next_val_above_dist,
    )
    weigts_below = 1 / next_val_below_dist # (N,)
    weigts_above = 1 / next_val_above_dist # (N,)
    weights_sum = weigts_below + weigts_above # (N,)
    weigts_below = weigts_below / weights_sum # (N,)
    weigts_above = weigts_above / weights_sum # (N,)
    
    # linear interpolation of nan values
    arr_inter = weigts_below * next_val_below + weigts_above * next_val_above # linear interpolation
    arr[np.isnan(arr)] = arr_inter[np.isnan(arr)] # replace nan values by linear interpolation
    return arr

def convolveIgnorNans(
    arr:np.array,
    kernel:np.array,
):
    """
    Convolve array while ignoring nan values e.g. replace nan values by linear interpolation.
    Args:
        arr: input array; np.array of shape (N,)
        kernel: kernel for convolution; np.array of shape (M,)
    Returns:
        arr_conv: convolved array; np.array of shape (N,)
    """
    arr = np.copy(arr)
    kernel = np.copy(kernel)

    # linear interpolate nan values
    arr = linInterpolateNans(arr)

    # convolve array
    return np.convolve(arr, kernel, mode="same")

def dataConverged(
    arr:np.array,
    threshold:float,
    data_increasing:bool
):
    """
    Verify at which index the data has converged.
    Args:
        arr: input array; np.array of shape (N,)
        threshold: threshold for convergence; float
        data_increasing: whether the data is increasing or decreasing; bool
    Returns:
        idx_converged: index at which the data has converged; int
                        return -1 if data has not converged
    """
    arr = np.copy(arr)

    arr_binary = np.where(
        arr > threshold, 
        1 if data_increasing else 0, 
        0 if data_increasing else 1,
    )
    arr_binary = np.cumprod(arr_binary[::-1])[::-1]

    if not np.any(arr_binary):
        return -1 # data has not converged
    return np.argmax(arr_binary)





