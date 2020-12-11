import pandas as pd
import matplotlib.pyplot as plt
import astropy
from astropy.coordinates import SkyCoord
import numpy as np
import matplotlib.pyplot as plt
from gatspy import datasets, periodic
import astropy.timeseries as astropy_timeseries
from astropy.table import Table, vstack
def decompose(VectorizedInput,output_):
    MCN,MCPeriods,\
    MCPeriodogram_p, MCPeriodogram_A,\
    MCFilter, MCMag,  MCPhase,MCType = VectorizedInput
    simulated_periodogram = pd.DataFrame({"N":np.tile(MCN,MCPeriodogram_p.shape[1]).reshape(-1,MCPeriods.size).T.flatten(),
                                         "P":np.tile(MCN,MCPeriodogram_p.shape[1]).reshape(-1,MCPeriods.size).T.flatten(),
                                         "o":MCPeriodogram_p.flatten(),
                                         'A':MCPeriodogram_A.flatten()
                                         })
    simulated_periods= pd.DataFrame({"N":MCN,"P":MCPeriods})
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

    lightcurve_p = lightcurve.groupby(["kind",'N','Filter','phase bin',]).agg({'mag': [
                                                                            lambda x: np.percentile(x,16),
                                                                            lambda x: np.percentile(x,50),
                                                                            lambda x: np.percentile(x,84)]})
    lightcurve_p.columns = ['p_16', 'p_50', 'p_84']
    lightcurve.to_csv(output_+'lightcurve.csv')
    lightcurve_p.to_csv(output_+'lightcurve_p.csv')
    simulated_periods.to_csv(output+'periods.csv')
    return lightcurve_p, lightcurve,simulated_periods