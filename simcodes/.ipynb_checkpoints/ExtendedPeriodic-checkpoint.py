import gatspy.periodic
import numpy as np
import pandas as pd
import scipy.stats
from dask.distributed import Client, SSHCluster
class ExtendedLS(gatspy.periodic.LombScargleMultiband):
    def __init__(self,*args,**kwargs):
        gatspy.periodic.LombScargleMultiband.__init__(self,*args,**kwargs)
        self.t     = None
        self.dy    = None
        self.filts = None
        
    def copy_parameters(self,model:gatspy.periodic.LombScargleMultiband):
        self._best_period   = model._best_period
        self.unique_filts_  = model.unique_filts_
        self.ymean_by_filt_ = model.ymean_by_filt_
        self.omega          = 2 * np.pi / model._best_period
        self.theta          = model._best_params(self.omega)
        
    @staticmethod
    def get_parameters(model:gatspy.periodic.LombScargleMultiband)->dict:
        return dict(
                _best_period   = model._best_period,
                unique_filts_  = model.unique_filts_,
                ymean_by_filt_ = model.ymean_by_filt_,
                omega          = 2 * np.pi / model._best_period,
                theta          = model._best_params(2 * np.pi / model._best_period))
    
    @staticmethod
    def set_parameters(model:gatspy.periodic.LombScargleMultiband,parameters):
        model._best_period   = parameters['_best_period']
        model.unique_filts_  = parameters['unique_filts_']
        model.ymean_by_filt_ = parameters['ymean_by_filt_']

    def import_parameters(self,parameters:dict):
        self._best_period   = parameters['_best_period']
        self.unique_filts_  = parameters['unique_filts_']
        self.ymean_by_filt_ = parameters['ymean_by_filt_']
        self.omega          = parameters['omega']
        self.theta          = parameters['theta']
        
    def export_parameters(self)->dict:
        return dict(
                _best_period   = self._best_period,
                unique_filts_  = self.unique_filts_,
                ymean_by_filt_ = self.ymean_by_filt_,
                omega          = self.omega,
                theta          = self.theta)
        
    def predict(self, t:np.array,filts:np.array)->np.array:
        # need to make sure all unique filters are represented
        u, i = np.unique(np.concatenate([filts, self.unique_filts_]),
                         return_inverse=True)
        ymeans = self.ymean_by_filt_[i[:-len(self.unique_filts_)]]

        
        X = self._construct_X(self.omega, weighted=False, t=t, filts=filts)

        if self.center_data:
            return ymeans + np.dot(X, self.theta)
        else:
            return np.dot(X, self.theta)
def concatenating(g1:pd.DataFrame,folder:str,distribution_relative_error:scipy.stats.rv_continuous):
    Library = []

    for i, row in g1.iterrows():
        filename = folder+f"{row['source_id']}.csv"
        D =  pd.read_csv(filename)
        if np.sum(D['catflags']==0)>0 and np.sum(D[D['catflags']==0].columns.isin(['t','phase','mag','filt','magerr']))==5:
               
                D = D[D['catflags']==0][['t','phase','mag','filt','magerr']]
                D['t'] = D.t-D.t.min()
                D['row'] = i
                D['relative_error']=D.magerr/D.mag
                Library.append(D)
    lib= pd.concat(Library)
    fit = distribution_relative_error.fit(lib.relative_error)
    return lib, fit, distribution_relative_error(*fit)

def produce_samples(data:pd.DataFrame,Ns:np.array,distribution_relative_error:scipy.stats.rv_continuous):
    dfs = []
    num = 0
    for i,datum in data.iterrows():
        for N in Ns:
            m = ExtendedLS(fit_period=True,optimizer_kwds=dict(quiet=True),Nterms_base=1)
            m.import_parameters(datum)
            phase = np.random.uniform(0,10,N)
            t     = np.tile(phase*datum['_best_period'],datum['unique_filts_'].size)
            t     = t-t.min()
            filt  = np.repeat(datum['unique_filts_'],phase.size)
            clean_prediction = m.predict(t,filt)
            relative_errs  = distribution_relative_error.rvs(size=t.size)
            errs = clean_prediction*relative_errs
            value = clean_prediction+np.random.normal(0,errs)
            dfs.append(pd.DataFrame({'t':t,'mag':value,'magerr':errs,'filt':filt,'data index':i,'simulation index':num,'_best_period':datum['_best_period'],
                                    'Size':N}))
            num+=1
    return pd.concat(dfs).reset_index(drop=True)
def produce_samples_dask(client,data:pd.DataFrame,Ns:np.array,distribution_relative_error:scipy.stats.rv_continuous):
    def Wrapper_samples(V:list):
        i,datum = V
        ls=[]
        for j,N in enumerate(Ns):
            num=Ns.size*i+j
            m = ExtendedLS(fit_period=True,optimizer_kwds=dict(quiet=True),Nterms_base=1)
            m.import_parameters(datum)
            phase = np.random.uniform(0,10,N)
            t     = np.tile(phase*datum['_best_period'],datum['unique_filts_'].size)
            t     = t-t.min()
            filt  = np.repeat(datum['unique_filts_'],phase.size)
            clean_prediction = m.predict(t,filt)
            relative_errs  = distribution_relative_error.rvs(size=t.size)
            errs = clean_prediction*relative_errs
            value = clean_prediction+np.random.normal(0,errs)
            ls.append( pd.DataFrame({'t':t,'mag':value,'magerr':errs,'filt':filt,'data index':i,'simulation index':num,'_best_period':datum['_best_period'],
                                    'Size':N}))
        return ls
                #inputs = client.scatter(self.dfs)
                
    futures= client.map(Wrapper_samples,list(data.iterrows()))
    dfs = client.gather(futures)

    return pd.concat([_ for df in dfs for _ in df]).reset_index(drop=True)

        

def LibraryCreation(g1:pd.DataFrame,library_name:str,folder:str,FourierComponents:list):
    Library = pd.DataFrame({})

    for i, row in g1.iterrows():
        for Nterms in FourierComponents:
            filename = folder+f"{row['source_id']}.csv"
            D =  pd.read_csv(filename)
            if np.sum(D['catflags']==0)>0 and np.sum(D[D['catflags']==0].columns.isin(['t','phase','mag','filt','magerr']))==5:
               
                D = D[D['catflags']==0][['t','phase','mag','filt','magerr']]
                D['t'] = D.t-D.t.min()
            

                model = gatspy.periodic.LombScargleMultiband(fit_period=True,optimizer_kwds=dict(quiet=True),Nterms_base=Nterms)
                
                model.optimizer.period_range=(max(D.t.diff().min(),0.01*row['pf']), min(100*row['pf'],D.t.max()))
                model.fit(D.t,D.mag,D.magerr, D.filt)        
                bestpers = model.find_best_periods(10,True)
                m = ExtendedLS(fit_period=True,optimizer_kwds=dict(quiet=True),Nterms_base=Nterms)
                m.import_parameters(m.get_parameters(model))
                m.copy_parameters(model)

                D['prediction'] = m.predict(D.t,D.filt)
                D['phase'] = (D.t/row['pf'])%1
                D['fit phase'] = (D.t/m._best_period)%1
                
                D.sort_values('phase',inplace=True)
                
                ϕ = np.linspace(0,1)
                t = ϕ*m._best_period
                filts = pd.unique(D.filt)
                predictions = pd.DataFrame({'t fit':np.tile( ϕ*m._best_period,filts.size),
                                            't Gaia':np.tile( ϕ*row['pf'],filts.size),
                                            'phase':np.tile(ϕ,filts.size),
                                           'filt':np.repeat(filts, ϕ.size)})
                
                    
                
                
                predictions['mag fit']  = m.predict(predictions['t fit'],predictions['filt'])
                predictions['mag Gaia'] = m.predict(predictions['t Gaia'],predictions['filt'])
                
                    
                params= m.export_parameters()
                params['source_id'] = row['source_id']
                params['Expected'] = row['pf']
                params['E-C'] = params['Expected']-params['_best_period']
                params['dof'] = D['prediction'].size-1
                params['χ2'] = np.sum((D['prediction']-D['mag'])**2/D['magerr']**2)
                params['χ2 dof'] = np.sum((D['prediction']-D['mag'])**2/D['magerr']**2)/params['dof']
                params['Type'] = row['Type']
                params['Subtype'] = row['Subtype']
                params['Nterms'] = Nterms
                
                for _i,(_p,score) in enumerate(zip(*bestpers)):
                    D[f'fit phase {_i}'] = (D.t/_p)%1
                    D[f'prediction {_i}'] =  model._predict(D.t,D.filt,_p)
                    
                    predictions[f't fit {_i}'] =np.tile( ϕ*_p,filts.size)
                    predictions[f'mag fit {_i}']  = model._predict(predictions[f't fit {_i}'],predictions['filt'],_p)
                    params[f"LS period {_i}"] = _p
                    params[f"LS period {_i} score"] = score
                    params[f'χ2 {_i}'] = np.sum((D[f'prediction {_i}']-D['mag'])**2/D['magerr']**2)
                    params[f'χ2 {_i}/dof'] = np.sum((D[f'prediction {_i}']-D['mag'])**2/D['magerr']**2)/(D.mag.size-1)
                Library = Library.append([params])

                #fig.savefig(f"plots/ztf/{i}_Nterms_{Nterms}_{row['source_id']}.png" ,bbox_extra_artists=[fig.suptitle(f"E-C={D['E-C']}")], bbox_inches='tight')
                #plt.close(fig)
                Library.reset_index(drop=True)
                Library.to_csv(library_name)
                o = np.logspace(*np.log10(model.optimizer.period_range),num=10000)
                
                yield Library, params,D, predictions,[o,model.periodogram(o)],bestpers#,#model.periodogram_auto()