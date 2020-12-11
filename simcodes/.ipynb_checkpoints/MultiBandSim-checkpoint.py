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
from dask.distributed import Client, SSHCluster
import dask.array as da
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
def testing(Number_in_simulation,P0,original_file,TYPE='fast',doMC=True,Periodogram_auto=False):

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
        df = pd.read_csv(original_file)
        if doMC:
            DF = pd.DataFrame()
            for _filter in pd.unique(df['filt']):
                cut = df[df['filt']==_filter]
                DF = DF.append(cut.iloc[np.random.randint(0,cut.shape[0],min(Number_in_simulation,cut.shape[0])),:])
            df = DF.reset_index()
        with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                K = process(df, np.unique(df["filt"]))
       
        return  Number_in_simulation,K[0],\
                K[-1][0],\
                K[-1][1],\
                K[2].flatten(), K[3], K[4],TYPE    

class MCSimulation:
    def __init__(self, data_, P0, initial_lightcurve):
        self.P0                 = P0
        self.data_              = data_
        self.initial_lightcurve = initial_lightcurve
        self.best_fitting       = testing(None, P0,initial_lightcurve,doMC=False,TYPE="naive",Periodogram_auto=True)
        self.simulations  = {}
        self.lightcurve_p = {}
        self.lightcurve = {}
        self.simulated_periods = {}
    def run_simulation(self,method, Sizes, Nreps, output='outputs/',cluster=None):
        if cluster is None:
            self.simulations[method]\
                    = np.vectorize(testing,signature="(),(),(),(),(),()->(),(),(a),(a),(b),(b,c),(c),()")(
                np.sort(np.tile( Sizes, Nreps)),
                self.P0,self.initial_lightcurve,method, True,False)
        else:
            with Client(cluster) as client:
                fun = lambda i:np.array(testing(i,self.P0,self.initial_lightcurve,method, True,False))[0]
                #TO_IMPLEMENT
                self.simulations[method]=da.map_blocks(lambda x:fun(x),np.sort(np.tile( Sizes, Nreps))).compute()
        self.lightcurve_p[method],\
        self.lightcurve[method],\
        self.simulated_periods[method] = self.decompose(self.simulations[method],method, output)
    def decompose(self,VectorizedInput,method,output_):
        MCN,MCPeriods,\
        MCPeriodogram_p, MCPeriodogram_A,\
        MCFilter, MCMag,  MCPhase,MCType = VectorizedInput
        simulated_periodogram = pd.DataFrame({"N":np.tile(MCN,MCPeriodogram_p.shape[1]).reshape(-1,MCPeriods.size).T.flatten(),
                                             "P":np.tile(MCN,MCPeriodogram_p.shape[1]).reshape(-1,MCPeriods.size).T.flatten(),
                                             "o":MCPeriodogram_p.flatten(),
                                             'A':MCPeriodogram_A.flatten()
                                             })
        self.simulated_periods[method]= pd.DataFrame({"N":MCN,"P":MCPeriods})
        lightcurve = pd.DataFrame()
        for _filter in pd.unique(MCFilter.flatten()):

            cut = MCFilter==_filter
            phase =  MCPhase
            for i,N in enumerate(pd.unique(MCN)):

                lightcurve=lightcurve.append(pd.DataFrame({"N":MCPhase.shape[-1]*[N], 
                                                           "Filter":MCPhase.shape[-1]*[_filter], 
                                                           "phase":MCPhase[i], 
                                                           "mag":MCMag[cut][i],"kind":MCType[i]}),ignore_index=True)
        lightcurve['phase bin'] = pd.cut(lightcurve.phase,20).apply(lambda x: x.mid)
        self.lightcurve[method] = lightcurve
        self.lightcurve_p[method] = lightcurve.groupby(["kind",'N','Filter','phase bin',]).agg({'mag': [
                                                                                lambda x: np.percentile(x,16),
                                                                                lambda x: np.percentile(x,50),
                                                                                lambda x: np.percentile(x,84),
                                                                                lambda x: np.max(x),
                                                                                lambda x: np.min(x),
                                                                                lambda x: sem(x)]})
        self.lightcurve_p[method].columns = ['p_16', 'p_50', 'p_84','max','min','sem']
        self.lightcurve[method].to_csv(output_+'lightcurve.csv')
        self.lightcurve_p[method].to_csv(output_+'lightcurve_p.csv')
        self.simulated_periods[method].to_csv(output_+'periods.csv')
        return self.lightcurve_p[method], self.lightcurve[method],self.simulated_periods[method]
        
   