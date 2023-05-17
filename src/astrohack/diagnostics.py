from astrohack._utils._diagnostics import _the_plotly_inspection_function
from astrohack._utils._diagnostics import _the_matplotlib_inspection_function

def inspect_calibration_plots(data, delta=0.01, pol='RR', width=1000, height=450, type='matplotlib'):
  """ Wrapper function around backend plotting functions for calibration inspection.

  Args:
      data (dict): extracted holography data file, .holog.zarr
      delta (float, optional): Fraction of the cell size to use as filter for of l, m data to be included in plot. For example, for delta=0.01 all l,m values with r = sqrt(2)*cell_size*delta will be included in calibration plots. This is meant to be an estimate of the (l, m)=0 points where teh peaks will apprear for properly calibrated data. Defaults to 0.01.
      pol (str, optional): Polarization type. Defaults to 'RR'.
      width (int, optional): Plot width in pixels. Defaults to 1000.
      height (int, optional): Plot height in pixels. Defaults to 450.
      type (str, optional): Plot package to use. Defaults to matplotlib.
  """

  if type == 'plotly':
    return _the_plotly_inspection_function(data, delta=0.01, pol='RR', width=width, height=height)
  
  else:
    return _the_matplotlib_inspection_function(data, delta=0.01, pol='RR', width=width, height=height)
  
  

