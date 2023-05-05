import os
import gdown
import shutil
import json

from prettytable import PrettyTable
from prettytable import DOUBLE_BORDER

gdown_ids = {
    'ea25_cal_small_spw1_4_60_ea04_before.ms': '1-v1foZ4G-kHTOS2idylx-3S4snKgRHmM',
    'ea25_cal_small_spw1_4_60_ea04_after.ms': '1PmWvPA0rUtAfegVu9wOb4AGJXiQIp3Cp',
    'ea25_cal_small_spw1_4_60_ea04_before_fixed.ms': '1dV4e8FiVRdAw2NLNzSuPVvjwyqXPNXTf',
    'ea25_cal_small_spw1_4_60_ea04_after_fixed.ms': '1Gn5GEJ3V43bl2vXxznHk6oXF5LtWOMzj',
    'ea25_cal_small_before_fixed.split.ms':'1oydlR7kA7F4n0i9KF9HgRc2jq1ziUslt',
    'ea25_cal_small_after_fixed.split.ms':'1TATMxKTFYIEO-l9L3jdYj62lZ8TZex4T'
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