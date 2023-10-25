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






def test_linInterpolateArray():
    x1 = np.array([1, 2, 3, 4, 5])
    y1 = 2*x1
    x2 = np.array([2.5, 4.5, 3.5, 4.5])

    y2 = linInterpolateArray(x1, y1, x2)
    print(f"y1: {y1}")
    print(f"y2: {y2}")

if __name__ == "__main__":
    test_linInterpolateArray()