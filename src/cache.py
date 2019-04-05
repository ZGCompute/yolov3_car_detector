import os
import requests
import cv2
import subprocess

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
CACHE_DIR = os.path.join(SCRIPT_PATH, 'data', 'cache')
INDEX_FILE = 'index.txt'
INDEX_URL = 'https://path/to/video/index.txt' 
TS_URL = 'https:/yourserver.com/video/{}.ts'

def get_cache_dir():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    return CACHE_DIR


def get_index_file():
    cache = get_cache_dir()
    filename = os.path.join(cache, INDEX_FILE)

    if not os.path.exists(filename):
        download_file(INDEX_URL, filename)

    return [line.strip() for line in open(filename)]

def get_frame_jpg(filename):
    '''
    takes a downloaded .ts video file, extracts the first frame 
    and saves it to .jpg in local cache

    filename is the name of the .ts file downloaded to local cache
    '''
    outfile = filename.replace('.ts','.mp4')  
    ret = subprocess.call('ffmpeg -y -hide_banner -loglevel panic -i %s %s' %(filename, outfile), shell=True)
    if not ret:
        #print("Error reading .ts video. Check ffmpeg results.")
        None
    
    vidcap = cv2.VideoCapture(outfile)  
    success,image = vidcap.read()

    if success:
        out_jpg = outfile.replace('.mp4', '.jpg')
        cv2.imwrite(out_jpg, image)
    else:
        #print("Error downloading file. Check that your filename exists.")
        None
        
def get_image(timestamp):
    '''
    downloads a ts file.

    timestamp is an integer (seconds since unix epoch)
    '''
    cache = get_cache_dir()
    ts_file = timestamp + '.ts'
    filename = os.path.join(cache, ts_file)
    ts_url = TS_URL.replace('{}', timestamp)

    if not os.path.exists(filename):
        download_file(ts_url, filename)

    get_frame_jpg(filename)
    
def download_file(url, filename):
    '''
    downloads the contents of the provided url to a local file
    '''
    contents = requests.get(url).content
    with open(filename, 'wb') as f:
        f.write(contents)
