#from multiprocessing.pool import ThreadPool

import requests
import threading
import asyncio
import os

folder = 'L2_Data_allSP_test2'

def find_all(s,c):
    return [n for n,i in enumerate(s) if i==c]

MAX_THREADS = 50  # <--- tweak this

with open('D:/Data/ODECartFiles_L2.txt') as file:
    lines = file.readlines()

urls=[]
for line in lines:
    urls.append(line[:-1])

def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
    return wrapped

@background
def download(url):
    r = requests.get(url)
    ind = find_all(url,'/')
    name = url[ind[-1]:].lower()
    with open(f'D:/Data/{folder}/{name}', "wb") as f:
        f.write(r.content)

if __name__ == "__main__":
    prog = 1
    already_downloaded = os.listdir(f'D:/Data/{folder}')
    for url in urls:
        download(url)
        print (f'{prog} of {len(urls)} files downloaded to {folder} ({prog/len(urls):.0%})')
        prog+=1