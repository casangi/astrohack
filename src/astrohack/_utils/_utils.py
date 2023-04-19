import os


def _remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string


def _parm_to_list(parm, path):
    if parm == 'all':
        oulist = os.listdir(path)
    elif isinstance(parm, str):
        oulist = [parm]
    else:
        oulist = parm
    return oulist
