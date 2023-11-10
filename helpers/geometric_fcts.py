import numpy as np
import torch
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

def createScanRays(
        self,
        rays_o:torch.Tensor,
        res_angular:int,
        h_tol_c:float,
        num_avg_heights:int,
):
    """
    Create scan rays for gievn image indices.
    Args:
        rays_o: ray origins; array of shape (N, 3)
        res_angular: number of angular samples (M); int
        h_tol_c: height tolerance in cube coordinates (meters); float
        num_avg_heights: number of heights to average over (A); int
    Returns:
        rays_o: ray origins; array of shape (N*M*A, 3)
        rays_d: ray directions; array of shape (N*M*A, 3)
    """
    rays_o = rays_o.detach().clone() # (N, 3)
    N = rays_o.shape[0] # number of points

    # duplicate rays for different angles
    rays_o = torch.repeat_interleave(rays_o, res_angular, dim=0) # (N*M, 3)

    # create directions
    rays_d = torch.linspace(-np.pi, np.pi-2*np.pi/res_angular, res_angular, 
                            dtype=torch.float32, device=self.args.device) # (M,)
    rays_d = torch.stack((torch.cos(rays_d), torch.sin(rays_d), torch.zeros_like(rays_d)), axis=1) # (M, 3)
    rays_d = rays_d.repeat(N, 1) # (N*M, 3)

    if num_avg_heights == 1:
        return rays_o, rays_d

    # get rays for different heights
    rays_o_avg = torch.zeros(N*res_angular, num_avg_heights, 3).to(self.args.device) # (N*M, A, 3)
    rays_d_avg = torch.zeros(N*res_angular, num_avg_heights, 3).to(self.args.device) # (N*M, A, 3)   
    for i, h in enumerate(np.linspace(-h_tol_c, h_tol_c, num_avg_heights)):
        h_tensor = torch.tensor([0.0, 0.0, h], dtype=torch.float32, device=self.args.device)
        rays_o_avg[:,i,:] = rays_o + h_tensor
        rays_d_avg[:,i,:] = rays_d

    return rays_o_avg.reshape(-1, 3), rays_d_avg.reshape(-1, 3) # (N*M*A, 3), (N*M*A, 3)