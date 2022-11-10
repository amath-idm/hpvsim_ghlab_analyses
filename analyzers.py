'''
Define custom analyzers for HPVsim
'''

import numpy as np
import pandas as pd
import sciris as sc
import hpvsim as hpv


class ScenarioAnalyzer(hpv.Analyzer):
    '''
    For saving cancers by single age bucket, year, and country.
    Primarily used for generating outputs to feed into the IPM tool.
    '''

    age_bins = np.arange(0,136)

    def __init__(self, *args, location, index_type='simple', **kwargs):
        super().__init__(*args, **kwargs)
        self.location = location
        self.index_type = index_type # Set to 'simple' to just use years as the index, or 'multi' to use location + scenario name as well
        return


    def initialize(self, sim):
        super().initialize(sim)
        self.df = pd.DataFrame(0.0, index=pd.Index(sim.yearvec, name='Year'), columns=self.age_bins[:-1])
        return


    def apply(self, sim):
        for g in range(sim['n_genotypes']):
            if g == 0:
                cancerous_today = hpv.true(sim.people.date_cancerous[g,:] == sim.t)
            else:
                cancerous_today = np.append(cancerous_today, hpv.true(sim.people.date_cancerous[g,:] == sim.t))
        self.df.loc[sim.yearvec[sim.t]] = np.histogram(sim.people.age[cancerous_today], self.age_bins)[0]
        return


    def finalize(self, sim):
        self.df = self.df.multiply(sim.rescale_vec[0]*np.ones_like(sim.tvec), axis=0)  # Adjust outputs by rescale vector - currently assumes rescale factor doesn't change
        self.df = self.df.groupby(int).sum()   # Group into annual years

        if self.index_type == 'multi':
            # Prepend simulation/scenario label and location to dataframe
            idx = self.df.index.to_frame()
            idx.insert(0, 'Location', self.location)
            self.df.index = pd.MultiIndex.from_frame(idx)
        return

    @staticmethod
    def reduce(analyzers, quantile=None):
        ''' Create an averaged dataframe from a list of scenario analyzers '''

        # Process inputs for statistical calculations
        if quantile is None:
            quantile = 0.5

        # Check that a list of analyzers has been provided
        if not isinstance(analyzers, list):
            errormsg = 'ScenarioAnalyzer.reduce() expects a list of ScenarioAnalyzer instances'
            raise TypeError(errormsg)

        # Initialize the reduced analyzer
        df = pd.concat([analyzer.df for analyzer in analyzers])
        by_row_index = df.groupby(df.index)
        df = by_row_index.quantile(q=quantile)

        return df


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
        columns = ['new_screens', 'new_cin_treatments', 'new_cancer_treatments',
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

            # Age outputs
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
        self.df['new_txvx_doses'] = sim.results['new_txvx_doses'][self.si:]
        self.df['new_cancers'] = sim.results['total_cancers'][self.si:]
        self.df['new_cancer_deaths'] = sim.results['total_cancer_deaths'][self.si:]
        self.df['new_other_deaths'] = sim.results['other_deaths'][self.si:]
        return
