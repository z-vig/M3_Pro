'''
Script for downloading large batch files from NASA's planetary data system
'''

import requests
import threading
import asyncio
import os
from tkinter.filedialog import askdirectory as askdir
from tkinter.filedialog import askopenfilename as askfile
from typing import List
import time
import datetime

def find_all(s,c):
    return [n for n,i in enumerate(s) if i==c]

def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
    return wrapped

@background
def download(url:str,savePath:str)->None:
    r = requests.get(url)
    ind = find_all(url,'/')
    name = url[ind[-1]:].lower()
    with open(f'{savePath}/{name}', "wb") as f:
        f.write(r.content)

if __name__ == "__main__":
    start = time.time()
    print ('Choose path to save files to:')
    savePath = askdir()
    print (f'{savePath} selected.')
    print ('Choose .txt file of URLs:')
    urlTextFile = askfile()
    print (f'{urlTextFile} selected.')

    with open(urlTextFile) as file:
        lines = file.readlines()

    urls:List[str]=[]
    for line in lines:
        urls.append(line[:-1])

    prog = 1
    already_downloaded = os.listdir(f'{savePath}') ##Checks for files already in download folder
    for url in urls:
        ind = find_all(url,'/')
        if f'{savePath}/{url[ind[-1]:].lower()}' in already_downloaded:
            print (f'{savePath}/{url[ind[-1]:].lower()} already downloaded! ({prog/len(urls):.0%})')
            prog+=1
            continue
        elif f'{savePath}/{url[ind[-1]:].lower()}' not in already_downloaded:
            download(url,savePath)
            print (f'{prog} of {len(urls)} files downloaded to {savePath} ({prog/len(urls):.0%})')
            prog+=1
    
    folderLength = len(os.listdir(f'{savePath}'))
    total_imgfiles = len([i for i in urls if i[-4:].find('IMG')>-1])
    while folderLength < len(urls):
        folderLength = len(os.listdir(f'{savePath}'))
        imgfiles_current = len([i for i in os.listdir(f'{savePath}') if (i[-4:].find('img')>-1) and (os.path)])
        print (f'\r{folderLength} of {len(urls)} files processed ({folderLength/len(urls):.2%}) ({imgfiles_current/total_imgfiles:.2%} of IMG files)',end='\r')

    end = time.time()
    runtime = end-start
    print (f'Downloads finished at {datetime.datetime.now()}')
    if runtime < 1:
        print(f'Program Executed in {runtime*10**3:.3f} milliseconds')
    elif runtime < 60 and runtime > 1:
        print(f'Program Executed in {runtime:.3f} seconds')
    else:
        print(f'Program Executed in {runtime/60:.0f} minutes and {runtime%60:.3f} seconds')