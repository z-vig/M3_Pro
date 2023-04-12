import numpy as np


def get_nmax(arr,nmax):
    part = arr.flatten()[np.argpartition(arr.flatten(),-nmax)][-nmax:]
    where = [(np.where(arr==i)[0],np.where(arr==i)[1]) for i in part]
    where_formatted = []
    for i in where:
        for x,y in zip(i[0],i[1]):
            if (x,y) not in where_formatted:
                where_formatted.append((x,y))
    where_formatted = np.array(where_formatted)

    return where_formatted[:,0],where_formatted[:,1]

def get_nmin(arr,nmin,**kwargs):
    defaultKwargs = {'excludeZero':True}
    kwargs = {**defaultKwargs,**kwargs}

    if kwargs.get('excludeZero') == True:
        arr_noZeros = arr[np.where(arr!=0)]
        part = arr_noZeros[np.argpartition(arr_noZeros,nmin-1)][0:nmin]
    else:
        part = arr.flatten()[np.argpartition(arr.flatten(),nmin-1)][0:nmin]

    where = [(np.where(arr==i)[0],np.where(arr==i)[1]) for i in part]
    where_formatted = []
    for i in where:
        for x,y in zip(i[0],i[1]):
            if (x,y) not in where_formatted:
                where_formatted.append((x,y))
    where_formatted = np.array(where_formatted)

    return where_formatted[:,0],where_formatted[:,1]

if __name__ == "__main__":
    arr = np.random.choice(range(0,100000),(100,100))
    print(arr,'\n')
    maxRows,maxCols = get_nmax(arr,5)
    minRows,minCols = get_nmin(arr,5)

    print (f'Maxima Are: {arr[maxRows,maxCols]}')
    print (f'Maxima Located at: {[(x,y) for x,y in zip(maxRows,maxCols)]}')
    print (f'Minima Are: {arr[minRows,minCols]}')
    print (f'Minima Located at: {[(x,y) for x,y in zip(minRows,minCols)]}')

