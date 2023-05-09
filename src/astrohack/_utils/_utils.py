import os
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _well_positioned_colorbar(ax, fig, image, label, location='right', size='5%', pad=0.05):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(location, size=size, pad=pad)
    return fig.colorbar(image, label=label, cax=cax)


def _remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string


def _add_prefix(input_string, prefix):
    wrds = input_string.split('/')
    wrds[-1] = prefix+'_'+wrds[-1]
    return '/'.join(wrds)


def _parm_to_list(parm, path):
    if parm == 'all':
        oulist = os.listdir(path)
    elif isinstance(parm, str):
        oulist = [parm]
    else:
        oulist = parm
    return oulist
