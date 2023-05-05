'''
Script for downloading large batch files from NASA's planetary data system
'''

import requests
import threading
import asyncio
import os
from tkinter.filedialog import askdirectory as askdir

print ('Choose path to save files to:')
folderPath = askdir()

def find_all(s,c):
    return [n for n,i in enumerate(s) if i==c]

def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
    return wrapped

@background
def download(url):
    r = requests.get(url)
    ind = find_all(url,'/')
    name = url[ind[-1]:].lower()
    with open(f'{folderPath}/{name}', "wb") as f:
        f.write(r.content)

if __name__ == "__main__":
    with open('D:/Data/ODECartFiles_L2.txt') as file:
        lines = file.readlines()

    urls=[]
    for line in lines:
        urls.append(line[:-1])

    prog = 1
    already_downloaded = os.listdir(f'{folderPath}')
    for url in urls:
        download(url)
        print (f'{prog} of {len(urls)} files downloaded to {folderPath} ({prog/len(urls):.0%})')
        prog+=1
    
    folderLength = len(os.listdir('D:/Data/L2_Data_allSP_test2/'))
    total_imgfiles = len([i for i in urls if i[-4:].find('IMG')>-1])
    while folderLength < len(urls):
        folderLength = len(os.listdir('D:/Data/L2_Data_allSP_test2/'))
        imgfiles_current = len([i for i in os.listdir('D:/Data/L2_Data_allSP_test2/') if i[-4:].find('img')>-1])
        print (f'\r{folderLength} of {len(urls)} files processed ({folderLength/len(urls):.2%}) ({imgfiles_current/total_imgfiles:.2%} of IMG files)',end='\r')
    