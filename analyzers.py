'''
Define custom analyzers for HPVsim for GHlab analyses
'''

import numpy as np
import pandas as pd
import sciris as sc
import hpvsim as hpv



class econ_analyzer(hpv.Analyzer):
    '''
    Analyzer for feeding into costing/health economic analysis.
    
    Produces a dataframe by year storing:
        
        - Resource use: number of vaccines, screens, lesions treated, cancers treated
        - Cases/deaths: number of new cancer cases and cancer deaths
        - Average age of new cases, average age of deaths, average age of noncancer death
    '''

    def __init__(self, start=2020, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start=start
        return


    def initialize(self, sim):
        super().initialize(sim)
        columns = ['new_hpv_screens', 'new_poc_hpv_screens', 'new_ave_screens', 'new_via_screens',
                   'new_thermal_ablations', 'new_leeps', 'new_cancer_treatments',
                   'new_cancers', 'new_cancer_deaths', 'new_other_deaths',
                   'av_age_cancers', 'av_age_cancer_deaths', 'av_age_other_deaths']
        self.si = sc.findinds(sim.res_yearvec,self.start)[0]
        self.df = pd.DataFrame(0.0, index=pd.Index(sim.res_yearvec[self.si:], name='year'), columns=columns)
        return


    def apply(self, sim):
        if sim.yearvec[sim.t]>=self.start:
            ppl = sim.people
            def count(arr): return ppl.scale_flows(hpv.true(arr))
            def av_age(arr):
                if len(hpv.true(arr)): return np.mean(sim.people.age[hpv.true(arr)])
                else: return np.nan
            li = np.floor(sim.yearvec[sim.t])
            lt = sim.t-1

            # Pull out characteristics of sim to decide what resources we need
            simvals = sim.meta.vals
            primary_screen = simvals.primary
            triage_screen = simvals.triage
            scen_label = simvals.scen_label

            if primary_screen:

                self.df.loc[li].new_hpv_

            # Resources
            self.df.loc[li].new_screens += count(ppl.date_screened[:] == lt)
            self.df.loc[li].new_cin_treatments += count(ppl.date_cin_treated[:] == lt)
            self.df.loc[li].new_cancer_treatments += count(ppl.date_cancer_treated[:] == lt)

            # Age outputs
            self.df.loc[li].av_age_other_deaths = av_age(ppl.date_dead_other == lt)
            self.df.loc[li].av_age_cancer_deaths = av_age(ppl.date_dead_cancer == lt)
            self.df.loc[li].av_age_cancers = av_age(ppl.date_cancerous == lt)
        return


    def finalize(self, sim):
        # Add in results that are already generated (NB, these have all been scaled already)
        self.df['new_cancers'] = sim.results['total_cancers'][self.si:]
        self.df['new_cancer_deaths'] = sim.results['total_cancer_deaths'][self.si:]
        self.df['new_other_deaths'] = sim.results['other_deaths'][self.si:]
        return
