import os
import re
import gdown
import shutil
import json

from astrohack._utils._tools import _remove_suffix

from prettytable import PrettyTable

gdown_ids = {
    'ea25_cal_small_before_fixed.split.ms':'1oydlR7kA7F4n0i9KF9HgRc2jq1ziUslt',
    'ea25_cal_small_after_fixed.split.ms':'1TATMxKTFYIEO-l9L3jdYj62lZ8TZex4T',
    'J1924-2914.ms.calibrated.split.SPW3': '1OSDjWM1IskPOlC0w1wVBqsTp8JAbNGzL',
    'extract_holog_verification.json':'1Wd79KCl-wxlUwBRxYFUnofG8mN0Xfzga',
    'holog_numerical_verification.json':'16kl_DMHWVb0TwxuHq1dRr1TbIor_IU-a',
    'vla.after.split.holog.zarr':'1AnUg1_n97h54FBEvckjQWNHcwBLxRQ_Z',
    'vla.after.split.image.zarr':'1xIyRzypc84ck_K9d5hQt1EA8mQYTcSam',
    'vla.after.split.panel.zarr':'1xJzCG1Kfnct6c0QG0JiqFOAe4LgNs_fE',
    'vla.after.split.point.zarr':'1Fzw5oI9AuEJEGV3cz0zQb037oiavATqf',
    'vla.before.split.holog.zarr':'1DmTl1Zhqj2TJtGggnv_ydXz9FJFnwpNk',
    'vla.before.split.image.zarr':'10cAMNXPUpHIWI2rhAvE4EKRsS5FtH8CN',
    'vla.before.split.panel.zarr':'1wNoDol7K4SEfq1PTXyiur6mT9QhvCDND',
    'vla.before.split.point.zarr':'1xfMIs41cL7wEjC3WeN1iZY4XtH837_Vn',
    'alma.split.holog.zarr':'1VJSvqsaozz-6XjLsTsY-ZOKQ7xe9tEar',
    'alma.split.image.zarr':'1HHjFgByPiOnAYRHbZQ0aP0j4gJXvy0U4',
    'alma.split.panel.zarr':'1Xb6XHhTHF37Yc4_D3qSw1gaRi_RCxjBA',
    'alma.split.point.zarr':'1htWk4wD-gHUeM1BxnQblv6mHn_NKhIRl'
}

def check_download(name, folder, id):
    fullname = os.path.join(folder,name)
    if not os.path.exists(fullname):
        url = 'https://drive.google.com/u/0/uc?id='+id+'&export=download'
        gdown.download(url, fullname+'.zip')
        shutil.unpack_archive(filename=fullname+'.zip', extract_dir=folder)

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def build_folder_structure(dataname, resultname):
    create_folder(dataname)
    create_folder(resultname)

def download(file, folder='.', unpack=False):
    """ Allows access to stakeholder and unit testing data and configuration files via gdown.

    :param file: File to download for gdirve storage. A list of the available measurement sets can be accessed via `astrohack.gdown_utils.list_datasets()`. 
    :type file: str
    :param folder: Destination folder if not the current directory, defaults to '.'
    :type folder: str, optional
    :param unpack: Unzip file, defaults to False
    :type unpack: bool, optional
    """

    if file == 'vla-test': 
        matched = [(key, value) for key, value in gdown_ids.items() if re.search(r"^vla.+(before|after).split.+(holog|image|panel|point).*zarr$", key)]
        files = files = list(dict(matched).keys())

    elif file == 'alma-test':
        matched = [(key, value) for key, value in gdown_ids.items() if re.search(r"^alma.split.+(holog|image|panel|point).*zarr$", key)]
        files = list(dict(matched).keys())

    else:
        files = [file]

    for file in files:
        assert file in gdown_ids, "File {file} not available. Available files are:".format(file=file) + str(gdown_ids.keys())

        id = gdown_ids[file]
        create_folder(folder)

        fullname = os.path.join(folder, file)

        if os.path.exists(fullname) or os.path.exists(fullname + '.zip'):
            continue   

        if unpack:
            fullname = fullname + '.zip'

        url = 'https://drive.google.com/u/0/uc?id=' + id + '&export=download'
        gdown.download(url, fullname)

        # Unpack results
        if unpack: 
            shutil.unpack_archive(filename=fullname, extract_dir=folder)

            # Let's clean up after ourselves
            os.remove(fullname)
        

def gdown_data(ms_name, download_folder='.'):
    assert ms_name in gdown_ids, "Measurement set not available. Available measurement sets are:" + str(gdown_ids.keys())
    
    id = gdown_ids[ms_name]
    create_folder(download_folder)
    check_download(ms_name, download_folder, id)

def list_datasets():
    table = PrettyTable()
    table.field_names = ["Measurement Table", "Description"]
    table.align = "l"

    for key, _ in gdown_ids.items():
        basename = key.split('.')[0]
        file = ''.join((basename, '.json'))
        path = os.path.dirname(__file__)       

        with open('{path}/data/.file_meta_data/{file}'.format(path=path, file=file)) as file:
            ms_info = json.load(file)
        
        description_string = f"""
        Observer: {ms_info['observer']}
        Project:{ms_info['project']}
        Elapsed Time: {ms_info['elapsed time']}
        Observed: {ms_info['observed']}
        SPW-ID: {ms_info['spwID']}
        Name: {ms_info['name']}
        Channels: {ms_info['channels']}
        Frame: {ms_info['frame']}
        Channel0: {ms_info['chan0']} MHz
        Channel Width: {ms_info['chan-width']} kHz
        Total Bandwidth: {ms_info['total-bandwidth']} kHz
        Center Frequency: {ms_info['center-frequency']} MHz
        Correlations: {ms_info['corrs']}
        RA: {ms_info['ra']}
        DEC: {ms_info['dec']}
        EPOCH: {ms_info['epoch']}
        Notes: {ms_info['notes']}
        """

        table.add_row([str(key), description_string])
        
    print(table)
