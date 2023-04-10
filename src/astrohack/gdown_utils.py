import os
import gdown
import shutil

gdown_ids = {'ea25_cal_small_spw1_4_60_ea04_before.ms':'1-v1foZ4G-kHTOS2idylx-3S4snKgRHmM',
             'ea25_cal_small_spw1_4_60_ea04_after.ms':'1PmWvPA0rUtAfegVu9wOb4AGJXiQIp3Cp'}

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

def gdown_data(ms_name,download_folder='.'):
    assert ms_name in gdown_ids, "Measurement set not available. Available measurement sets are:" + str(gdown_ids.keys())
    
    id = gdown_ids[ms_name]
    create_folder(download_folder)
    check_download(ms_name, download_folder, id)
