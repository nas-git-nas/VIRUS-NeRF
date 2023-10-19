import numpy as np
from alive_progress import alive_bar
from contextlib import nullcontext



def findNearestNeighbour(array1, array2, batch_size=None, progress_bar=False, ignore_nan=False):
    """
    Find the closest points in array2 for each point in array1
    and return the indices of array2 for each point in array1.
    Args:
        array1: array of float (N, 2/3)
        array2: array of float (M, 2/3)
        batch_size: batchify array1 to avoid memory error, if None, batch_size = N; int/None
        progress_bar: show progress bar; bool
        ignore_nan: ignore nan values in array2; bool
    Returns:
        nn_idxs: indices of nearest neighbours from array2 with respect to array1; array of int (N,)
        nn_dists: distances of nearest neighbours from array2 with respect to array1; array of float (N,)
    """
    # downsample arrays
    array1 = np.copy(array1.astype(np.float32))
    array2 = np.copy(array2.astype(np.float32))

    # remove nan values
    if ignore_nan:
        array2 = array2[~np.isnan(array2).any(axis=1)]

    # define batch size
    if batch_size is None:
        batch_size = array1.shape[0]
    else:
        while array1.shape[0]%batch_size != 0:
            batch_size -= 1

    # determine nearest neighbour indices
    nn_idxs = np.zeros(array1.shape[0], dtype=np.int32) # (N,)
    with alive_bar(array1.shape[0]//batch_size, bar = 'bubbles', receipt=False) if progress_bar else nullcontext() as bar:

        # split calculation in batches to avoid memory error
        for i in range(0, array1.shape[0], batch_size):
            dist = np.linalg.norm(array2[:, np.newaxis] - array1[i:i+batch_size], axis=2) # (M, batch_size)
            nn_idxs[i:i+batch_size] = np.argmin(dist, axis=0)

            if progress_bar:
                bar()

    # determine nearest neighbour distances
    nn_dists = np.linalg.norm(array2[nn_idxs] - array1, axis=1) # (N,)
    
    return nn_idxs, nn_dists