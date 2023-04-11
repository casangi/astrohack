import math
import scipy

import numpy as np

import astropy.units as u
import astropy.coordinates as coord

from skimage.draw import disk

from astrohack._utils._algorithms import _calc_coords

from memory_profiler import profile

from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger

def _parallactic_derotation(data, parallactic_angle_dict):
    """ Uses samples of parallactic angle (PA) values to correct differences in PA between scans. The reference PA is selected 
        to be the first scans median parallactic angle. All values are rotated to this PA value using scypi.ndimage.rotate(...)

    Args:
        data (numpy.ndarray): beam data grid (scan, chan, pol, l, m)
        parallactic_angle_dict (dict): dictionary containing antenna selected xds from which the aprallactic angle samples 
                                       are retrieved ==> [scan](xds), here the scan referres to the scan values not the scan index.

    Returns:
        numpy.ndarray: rotation adjusted beam data grid
    """
    # Find the middle index of the array. This is calcualted because there might be a desire to change 
    # the array length at some point and I don't want to hard code the middle value.
    #
    # It is assumed, and should be true, that the parallacitc angle array size is consistent over scan.
    scans = list(parallactic_angle_dict.keys())

    # Get the median index for the first scan (this should be the same for every scan).
    median_index = len(parallactic_angle_dict[scans[0]].parallactic_samples)//2
    
    # This is the angle we will rotated the scans to.
    median_angular_reference = parallactic_angle_dict[scans[0]].parallactic_samples[median_index]
    
    for scan, scan_value in enumerate(scans):
        print(scan,scan_value)
        #median_angular_offset = median_angular_reference - parallactic_angle_dict[scan_value].parallactic_samples[median_index]
        #median_angular_offset *= 180/np.pi
        
        #parallactic_angle = 360 - parallactic_angle_dict[scan_value].parallactic_samples[median_index]*180/np.pi
        
        data[scan] = scipy.ndimage.rotate(input=data[scan, ...], angle=90, axes=(3, 2), reshape=False)
        
    return data

def _mask_circular_disk(center, radius, array, mask_value=np.nan):
    """ Create a mask to trim an image

    Args:
        center (tuple): tuple describing the center of the image
        radius (int): disk radius
        array (numpy.ndarray): data array to mask
        mask_value (int, optional): Value to set masked value to. Defaults to 1.
        make_nan (bool, optional): Set maked values to NaN. Defaults to True.

    Returns:
        _type_: _description_
    """
    shape = np.array(array.shape[-2:])

    if center == None:
        center = shape//2

    r, c = disk(center, radius, shape=shape)
    mask = np.zeros(shape, dtype=array.dtype)   
    mask[r, c] = 1
    
    mask = np.tile(mask, reps=(array.shape[:-2] + (1, 1)))
    
    mask[mask==0] = mask_value
    
    return mask

def _calculate_aperture_pattern(grid, delta, padding_factor=50):
    """ Calcualtes the aperture illumination pattern from the beam data.

    Args:
        grid (numpy.ndarray): gridded beam data
        frequency (float): channel frequency
        delta (float): incremental spacing between lm values, ie. delta_l = l_(n+1) - l_(n)
        padding_factor (int, optional): Padding to apply to beam data grid before FFT. Padding is applied on outer edged of 
                                        each beam data grid and not between layers. Defaults to 20.

    Returns:
        numpy.ndarray, numpy.ndarray, numpy.ndarray: aperture grid, u-coordinate array, v-coordinate array
    """
    logger = _get_astrohack_logger()
    logger.info("Calculating aperture illumination pattern ...")

    assert grid.shape[-1] == grid.shape[-2] ###To do: why is this expected that l.shape == m.shape
    initial_dimension = grid.shape[-1]

    # Calculate padding as the nearest power of 2
    # k log (2) = log(N) => k = log(N)/log(2)
    # New shape => K = math.ceil(k) => shape = (K, K)

    k = np.log(initial_dimension * padding_factor) / np.log(2)
    K = math.ceil(k)

    padding = (np.power(2, K) - padding_factor * initial_dimension) // 2

    padded_grid = np.pad(
        array=grid,
        pad_width=[(0, 0), (0, 0), (0, 0), (padding, padding), (padding, padding)],
        mode="constant",
    )

    import scipy.fftpack
    shifted = scipy.fftpack.ifftshift(padded_grid)

    grid_fft = scipy.fftpack.fft2(shifted)

    aperture_grid = scipy.fftpack.fftshift(grid_fft)

    u_size = aperture_grid.shape[-2]
    v_size = aperture_grid.shape[-1]

    image_size = np.array([u_size, v_size])

    cell_size = 1 / (image_size * delta)

    u, v = _calc_coords(image_size, cell_size)

    return aperture_grid, u, v, cell_size

def _calculate_parallactic_angle_chunk(
    time_samples,
    observing_location,
    direction,
    dir_frame="FK5",
    zenith_frame="FK5",
):
    """
    Converts a direction and zenith (frame FK5) to a topocentric Altitude-Azimuth (https://docs.astropy.org/en/stable/api/astropy.coordinates.AltAz.html)
    frame centered at the observing_location (frame ITRF) for a UTC time. The parallactic angles is calculated as the position angle of the Altitude-Azimuth
    direction and zenith.

    Parameters
    ----------
    time_samples: str np.array, [n_time], 'YYYY-MM-DDTHH:MM:SS.SSS'
        UTC time series. Example '2019-10-03T19:00:00.000'.
    observing_location: int np.array, [3], [x,y,z], meters
        ITRF geocentric coordinates.
    direction: float np.array, [n_time,2], [time,ra,dec], radians
        The pointing direction.
    Returns
    -------
    parallactic_angles: float np.array, [n_time], radians
        An array of parallactic angles.
    """

    observing_location = coord.EarthLocation.from_geocentric(
        x=observing_location[0] * u.m,
        y=observing_location[1] * u.m,
        z=observing_location[2] * u.m,
    )

    direction = coord.SkyCoord(
        ra=direction[:, 0] * u.rad, dec=direction[:, 1] * u.rad, frame=dir_frame.lower()
    )
    zenith = coord.SkyCoord(0, 90, unit=u.deg, frame=zenith_frame.lower())

    altaz_frame = coord.AltAz(location=observing_location, obstime=time_samples)
    zenith_altaz = zenith.transform_to(altaz_frame)
    direction_altaz = direction.transform_to(altaz_frame)

    return direction_altaz.position_angle(zenith_altaz).value
