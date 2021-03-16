import gatspy.periodic
import numpy as np
import pandas as pd

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
def LibraryCreation(g1:pd.DataFrame,library_name:str,folder:str,FourierComponents:list):
    Library = pd.DataFrame({})

    for i, row in g1.iterrows():
        for Nterms in FourierComponents:
            filename = folder+f"{row['source_id']}.csv"
            D =  pd.read_csv(filename)
            D = D.loc[D['catflags']==0,['t','phase','mag','filt','magerr']]
            D['t'] = D.t-D.t.min()
            if D.shape[0]>0:

                model = gatspy.periodic.LombScargleMultiband(fit_period=True,optimizer_kwds=dict(quiet=True),Nterms_base=Nterms)
                
                model.optimizer.period_range=(max(D.t.diff().min(),0.1*row['pf']), min(10*row['pf'],D.t.max()))
                model.fit(D.t,D.mag,D.magerr, D.filt)        
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
                params['χ2'] = np.sum((D['prediction']-D['mag'])**2/D['magerr']**2)
                
                params['Nterms'] = Nterms
                Library = Library.append([params])

                #fig.savefig(f"plots/ztf/{i}_Nterms_{Nterms}_{row['source_id']}.png" ,bbox_extra_artists=[fig.suptitle(f"E-C={D['E-C']}")], bbox_inches='tight')
                #plt.close(fig)
                Library.reset_index(drop=True)
                Library.to_csv(library_name)
                yield Library, params,D, predictions