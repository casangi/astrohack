import scipy
import scipy.signal as scisig
import numba

import numpy as np

def _apply_mask(data, scaling=0.5):
    """ Applies a cropping mask to the input data according to the scale factor

    Args:
        data (numpy,ndarray): numpy array containing the aperture grid.
        scaling (float, optional): Scale factor which is used to determine the amount of the data to crop, ex. scale=0.5 
                                   means to crop the data by 50%. Defaults to 0.5.

    Returns:
        numpy.ndarray: cropped aperture grid data
    """

    x, y = data.shape
    assert scaling > 0, console.error("Scaling must be > 0")

    mask = int(x // (1 // scaling))

    assert mask > 0, console.error(
        "Scaling values too small. Minimum values is:{}, though search may still fail due to lack of poitns.".format(
            1 / x
        )
    )

    start = int(x // 2 - mask // 2)
    return data[start : (start + mask), start : (start + mask)]

def _calc_coords(image_size, cell_size):
    """Calculate the center pixel of the image given a cell and image size

    Args:
        image_size (float): image size
        cell_size (float): cell size

    Returns:
        float, float: center pixel location in coordinates x, y
    """
    image_center = image_size // 2

    x = np.arange(-image_center[0], image_size[0] - image_center[0]) * cell_size[0]
    y = np.arange(-image_center[1], image_size[1] - image_center[1]) * cell_size[1]
    
    return x, y


def _find_nearest(array, value):
    """ Find the nearest entry in array to that of value.

    Args:
        array (numpy.array): _description_
        value (float): _description_

    Returns:
        int, float: index, array value
    """
    
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()

    return idx, array[idx]


#@njit(cache=False, nogil=True)
def _chunked_average(data, weight, avg_map, avg_freq):

    avg_chan_index = np.arange(avg_freq.shape[0])

    data_avg_shape = list(data.shape)
    n_time, n_chan, n_pol = data_avg_shape

    n_avg_chan = avg_freq.shape[0]
    
    # Update new chan dim.
    data_avg_shape[1] = n_avg_chan  

    data_avg = np.zeros(data_avg_shape, dtype=np.complex)
    weight_sum = np.zeros(data_avg_shape, dtype=np.float)

    index = 0

    for avg_index in avg_chan_index:

        while (index < n_chan) and (avg_map[index] == avg_index):

            # Most probably will have to unravel assigment
            data_avg[:, avg_index, :] = (data_avg[:, avg_index, :] + weight[:, index, :] * data[:, index, :])
            weight_sum[:, avg_index, :] = weight_sum[:, avg_index, :] + weight[:, index, :]
            
            index = index + 1

        for time_index in range(n_time):
            for pol_index in range(n_pol):
                if weight_sum[time_index, avg_index, pol_index] == 0:
                    data_avg[time_index, avg_index, pol_index] = 0.0

                else:
                    data_avg[time_index, avg_index, pol_index] = (data_avg[time_index, avg_index, pol_index] / weight_sum[time_index, avg_index, pol_index])

    return data_avg, weight_sum

def _calculate_euclidean_distance(x, y, center):
    """ Calculates the euclidean distance between a pair of pair of input points.

    Args:
        x (float): x-coordinate
        y (float): y-coordinate
        center (tuple (float)): float tuple containing the coordinates to the center pixel

    Returns:
        float: euclidean distance of points from center pixel
    """

    return np.sqrt(np.power(x - center[0], 2) + np.power(y - center[1], 2))


def _find_peak_beam_value(data, height=0.5, scaling=0.5):
    """ Search algorithm to determine the maximal signal peak in the beam pattern.

    Args:
        data (numpy.ndarray): beam data grid
        height (float, optional): Peak threshold. Looks for the maixmimum peak in data and uses a percentage of this 
                                  peak to determine a threhold for other peaks. Defaults to 0.5.
        scaling (float, optional): scaling factor for beam data cropping. Defaults to 0.5.

    Returns:
        float: peak maximum value
    """
    masked_data = _apply_mask(data, scaling=scaling)

    array = masked_data.flatten()
    cutoff = np.abs(array).max() * height

    index, _ = scisig.find_peaks(np.abs(array), height=cutoff)
    x, y = np.unravel_index(index, masked_data.shape)

    center = (masked_data.shape[0] // 2, masked_data.shape[1] // 2)

    distances = _calculate_euclidean_distance(x, y, center)
    index = distances.argmin()

    return masked_data[x[index], y[index]]

def _gauss_elimination_numpy(system, vector):
    """
    Gauss elimination solving of a system using numpy
    Args:
        system: System matrix to be solved
        vector: Vector that represents the right hand side of the system

    Returns:
    The solved system
    """
    inverse = np.linalg.inv(system)
    return np.dot(inverse, vector)

def _least_squares_fit(system, vector):
    """
    Least squares fitting of a system of linear equations
    The variances are simplified as the diagonal of the covariances
    Args:
        system: System matrix to be solved
        vector: Vector that represents the right hand side of the system

    Returns:
    The solved system, the variances of the system solution and the sum of the residuals
    """
    if len(system.shape) != 2:
        raise Exception('System must have 2 dimensions')
    if system.shape[0] < system.shape[1]:
        raise Exception('System must have at least the same number of rows as it has of columns')
    
    fit = np.linalg.lstsq(system, vector, rcond=None)
    
    result = fit[0]
    residuals = fit[1]
    covar = np.matrix(np.dot(system.T, system)).I
    variances = np.diagonal(covar)
    
    return result, variances, residuals

def _least_squares_fit_block(system, vector):
    """
    Least squares fitting of a system of linear equations
    The variances are simplified as the diagonal of the covariances
    Args:
        system: System matrix to be solved
        vector: Vector that represents the right hand side of the system

    Returns:
    The solved system and the variances of the system solution
    """
    if len(system.shape) < 2:
        raise Exception('System block must have at least 2 dimensions')
    if system.shape[-2] < system.shape[-1]:
        raise Exception('Systems must have at least the same number of rows as they have of columns')
    shape = system.shape
    results = np.zeros_like(vector)
    variances = np.zeros_like(vector)
    if len(shape) > 2:
        for it0 in range(shape[0]):
            results[it0], variances[it0] = _least_squares_fit_block(system[it0], vector[it0])
    else:
        results, variances, _ = _least_squares_fit(system, vector)
    return results, variances

def _average_repeated_pointings(vis_map_dict, weight_map_dict, flagged_mapping_antennas,time_vis,pnt_map_dict):
    
    for ant_id in vis_map_dict.keys():
        diff = np.diff(pnt_map_dict['ant_'+str(ant_id)],axis=0)
        r_diff = np.sqrt(np.abs(diff[:,0]**2 + diff[:,1]**2))
    
        max_dis = np.max(r_diff)/100
        
        print('max_dis',max_dis)
        n_avg = np.sum([r_diff > max_dis]) + 1
    
        vis_map_avg, weight_map_avg, time_vis_avg, pnt_map_avg = _average_repeated_pointings_jit(vis_map_dict[ant_id], weight_map_dict[ant_id],time_vis,pnt_map_dict['ant_'+str(ant_id)],n_avg,max_dis,r_diff)
        
        vis_map_dict[ant_id] = vis_map_avg
        weight_map_dict[ant_id] = weight_map_avg
        pnt_map_dict['ant_'+str(ant_id)] = pnt_map_avg
        
        #print('$$$',vis_map_avg.shape,weight_map_avg.shape,pnt_map_avg.shape)
        
    return time_vis_avg
        
 
        
#@numba.njit(cache=False, nogil=True)
def _average_repeated_pointings_jit(vis_map, weight_map,time_vis,pnt_map,n_avg,max_dis,r_diff):

    vis_map_avg = np.zeros((n_avg,)+ vis_map.shape[1:], dtype=vis_map.dtype)
    weight_map_avg = np.zeros((n_avg,)+ weight_map.shape[1:], dtype=weight_map.dtype)
    time_vis_avg = np.zeros((n_avg,), dtype=time_vis.dtype)
    pnt_map_avg = np.zeros((n_avg,)+ pnt_map.shape[1:], dtype=pnt_map.dtype)
    
    
    k = 0
    n_samples = 1
    
    vis_map_avg[0,:,:] = vis_map_avg[k,:,:] + weight_map[0,:,:]*vis_map[0,:,:]
    weight_map_avg[0,:,:] = weight_map_avg[k,:,:] + weight_map[0,:,:]
    time_vis_avg[0] = time_vis_avg[k] + time_vis[0]
    pnt_map_avg[0,:] = pnt_map_avg[k,:] + pnt_map[0,:]

    for i in range(vis_map.shape[0]-1):
        
        point_dis = r_diff[i]
        
        if point_dis < max_dis:
            n_samples = n_samples + 1
        else:
            #vis_map_avg[k,:,:] = vis_map_avg[k,:,:]/weight_map_avg[k,:,:]
            vis_map_avg[k,:,:] = np.divide(vis_map_avg[k,:,:],weight_map_avg[k,:,:],out=np.zeros_like(vis_map_avg[k,:,:]),where=weight_map_avg[k,:,:]!=0)
            weight_map_avg[k,:,:] = weight_map_avg[k,:,:]/n_samples
            
            time_vis_avg[k] = time_vis_avg[k]/n_samples
            pnt_map_avg[k,:] = pnt_map_avg[k,:]/n_samples
        
            k=k+1
            n_samples = 1
            
        vis_map_avg[k,:,:] = vis_map_avg[k,:,:] + weight_map[i+1,:,:]*vis_map[i+1,:,:]
        weight_map_avg[k,:,:] = weight_map_avg[k,:,:] + weight_map[i+1,:,:]
        time_vis_avg[k] = time_vis_avg[k] + time_vis[i+1]
        pnt_map_avg[k,:] = pnt_map_avg[k,:] + pnt_map[i+1,:]
        
    vis_map_avg[-1,:,:] = vis_map_avg[1,:,:]/weight_map_avg[-1,:,:]
    weight_map_avg[-1,:,:] = weight_map_avg[-1,:,:]/n_samples
    time_vis_avg[-1] = time_vis_avg[-1]/n_samples
    pnt_map_avg[-1,:] = pnt_map_avg[-1,:]/n_samples

    return vis_map_avg, weight_map_avg, time_vis_avg, pnt_map_avg
