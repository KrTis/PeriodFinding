import pandas as pd
import matplotlib.pyplot as plt
import astropy
from astropy.coordinates import SkyCoord
import numpy as np
import matplotlib.pyplot as plt
from gatspy import datasets, periodic
import astropy.timeseries as astropy_timeseries
from astropy.table import Table, vstack
import warnings
from scipy.stats import sem

class correctedNaiveMultiband:
    def __init__(self,*args,**kwargs):
        
        self.args = args
        self.kwargs = kwargs
    def fit(self, T, M, ME, F):
        self.filters = pd.unique(F)
        self.models = {}
        for _filter in self.filters:
            cut = F==_filter
            self.models[_filter]=periodic.LombScargleFast(*self.args,**self.kwargs)
            self.models[_filter].optimizer.period_range=(0.1, 10)
            self.models[_filter].fit(T[cut],M[cut],ME[cut])
    def predict(self, T,filts):
        predictions = []
        for i,_filter in enumerate(pd.unique(filts[:,0])):
           
            predictions.append(self.models[_filter].predict(T[:,i]))
        return np.array(predictions)
    @property
    def best_period(self):
        return np.array([self.models[_filter].best_period for _filter in self.filters])
    def periodogram_auto(self):
        PA = [self.models[_filter].periodogram_auto() for _filter in self.filters]
        return np.array([p[0] for p in PA]), np.array([p[1] for p in PA])
    def periodogram(self,*args,**kwargs):
        return np.array([self.models[_filter].periodogram(*args,**kwargs) for _filter in self.filters])

def testing(Number_in_simulation, P0, original_file, TYPE='fast', doMC=True, Periodogram_auto=False):

        o = np.linspace(1/100,24,10000)
        def process(df, filters=list('IV')):

            if TYPE == 'fast':
                model = periodic.LombScargleMultibandFast(fit_period=True, Nterms=1,optimizer_kwds=dict(quiet=True))
                
                model.optimizer.period_range=(0.1, 10)
            if TYPE == 'slow':
                model = periodic.LombScargleMultiband(fit_period=True,optimizer_kwds=dict(quiet=True))
                
                model.optimizer.period_range=(0.1, 10)
            if TYPE == 'naive':
                #model = periodic.NaiveMultiband(fit_period=True)
                model = correctedNaiveMultiband(fit_period=True,optimizer_kwds=dict(quiet=True))
            
            model.fit(df['t'], df['mag'], df['magerr'], df['filt'])
            

            tfit = np.linspace(0, model.best_period, 1000)
            filtsfit = np.array(filters)[:, np.newaxis]
            magfit = model.predict(tfit, filts=filtsfit)
            
            if np.size(model.best_period)>1:
                phase = (df['t'][np.newaxis,:] / model.best_period[:,np.newaxis]) % 1
                phasefit = (tfit.T / model.best_period[:,np.newaxis])
            else:
                phase = (df['t'] / model.best_period) % 1
                phasefit = (tfit / model.best_period)
           

            if Periodogram_auto:
                return np.mean(model.best_period), tfit, filtsfit, magfit, phasefit,model.periodogram_auto()
            return np.mean(model.best_period), tfit, filtsfit, magfit, phasefit,[o,model.periodogram(o)]
        if isinstance(original_file,str):
            df = pd.read_csv(original_file)
        else:
            df = original_file
        print(df.head())
        print(pd.unique(df['filt']))
        
        if doMC:
            DF = pd.DataFrame()
            for _filter in pd.unique(df['filt']):
                cut = df[df['filt']==_filter]
                print(Number_in_simulation,cut.shape[0])
                DF = DF.append(cut.iloc[np.random.randint(0,cut.shape[0],min(Number_in_simulation,cut.shape[0])),:])
            df = DF.reset_index()
        
        with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                K = process(df, np.unique(df["filt"]))
       
        return  Number_in_simulation,K[0],\
                K[-1][0],\
                K[-1][1],\
                K[2].flatten(), K[3], K[4],TYPE   
def unpack(X):
    N = len(X)
    L = len(X[0])
    outs = [np.array([X[i][j] for i in range(L)]) for j in range(L)]
    return outs