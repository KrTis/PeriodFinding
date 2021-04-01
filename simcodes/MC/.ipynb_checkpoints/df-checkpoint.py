import pandas as pd
import numpy as np
import tqdm
import os
from .fileop import *
def chunking(dataset,chunksize, path,desc='chunking', client = None):
    ind = pd.unique(dataset.index)   
    l   = np.arange(0,np.size(ind)+1,chunksize)
    locations = {}
    def chunker(cut):
        loc = path+f'chunk_{cut}.csv'
        dataset.loc[ind[l[cut-1]]:ind[l[cut]-1]].to_csv(loc)
        return loc
    if client is None:
        for cut in tqdm.notebook.tqdm(range(1,len(l)),desc=desc):
            locations[cut] = chunker(cut)
            #dataset.loc[ind[l[cut-1]]:ind[l[cut]-1]].to_csv(path+f'chunk_{cut}.csv')
            #locations[cut] = path+f'chunk_{cut}.csv'
    return locations

class Chunk:
    def __init__(self,dataset,chunksize, path, counter=True):
        if isinstance(dataset,pd.DataFrame):
            self.locations = chunking(dataset,chunksize, make_file(path))
            self.N = len(self.locations.keys())
        else:
            self.load(dataset)
        self.counter = counter
        self.chunksize= chunksize
    def load(self,locations):
        self.locations = locations
        self.N = len(self.locations.keys())
    def __str__(self):
        return f'<Chunk object divided in {self.N} chunks of size {self.chunksize} at {hex(id(self))}>'
    def Files(self,desc=''):
        if self.counter:
            yield from tqdm.notebook.tqdm(self.locations.items(), total=self.N,desc=desc)
        else:
            yield from self.locations.items()
    def join(self, *args,**kwargs):
        for key, path in self.Files('join'):
            pd.read_csv(path,index_col=0)\
                .join(*args,**kwargs)\
                .to_csv(path)
        return self
    def to_csv(self, path_new,*args,**kwargs):
        locations = {}
        path_new = make_file(path_new)
        for cut, path in self.Files('to_csv'):
            loc = path_new+f'chunk_{cut}.csv'
            pd.read_csv(path,index_col=0)\
                .to_csv(loc,*args,**kwargs)
            locations[cut] = loc
        return Chunk(locations,self.chunksize,path_new,self.counter)
    def set_index(self, *args,**kwargs):
        for key, path in self.Files('set_index'):
            pd.read_csv(path,index_col=0)\
                .set_index(*args,**kwargs)\
                .to_csv(path)
        return self
    def reset_index(self, *args,**kwargs):
        for key, path in self.Files('reset_index'):
            pd.read_csv(path,index_col=0)\
                .reset_index(*args,**kwargs)\
                .to_csv(path)
        return self
    def drop_duplicates(self, *args,**kwargs):
        for key, path in self.Files('drop_duplicates'):
            pd.read_csv(path,index_col=0)\
                .drop_duplicates(*args,**kwargs)\
                .to_csv(path)
        return self
    def applyfunc(self, func,*args,desc='apply',**kwargs):
        for key, path in self.Files(desc):
            func(pd.read_csv(path,index_col=0),*args,**kwargs)\
                .to_csv(path)
        return self
    def iterchunks(self):
        for key, path in self.Files('iterating'):
            yield key,path, pd.read_csv(path,index_col=0)
    
        
            
            