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
            def av_age(arr):
                if len(hpv.true(arr)): return np.mean(sim.people.age[hpv.true(arr)])
                else: return np.nan
            li = np.floor(sim.yearvec[sim.t])
            ltt = np.int((sim.t-1)*sim['dt'])
            lt = (sim.t-1)

            # Pull out characteristics of sim to decide what resources we need
            simvals = sim.meta.vals
            scen_label = simvals.scen_label
            if scen_label != 'No screening':
                primary_screen = simvals.primary
                if simvals.get('triage'):
                    triage_screen = simvals.triage
                else:
                    triage_screen = 'no_triage'

                resource_dict = {
                    'hpv': 'new_hpv_screens',
                    'poc_hpv': 'new_poc_hpv_screens',
                    'ave': 'new_ave_screens',
                    'via': 'new_via_screens'
                }
                # Resources
                if primary_screen in ['hpv', 'poc_hpv', 'via', 'ave']:
                    self.df.loc[li][resource_dict[primary_screen]] += sim.get_intervention('screening').n_products_used.values[ltt]

                if triage_screen in ['via', 'ave']:
                    self.df.loc[li][resource_dict[triage_screen]] += sim.get_intervention('triage').n_products_used.values[ltt]

                self.df.loc[li].new_thermal_ablations += sim.get_intervention('ablation').n_products_used.values[ltt]
                self.df.loc[li].new_leeps += sim.get_intervention('excision').n_products_used.values[ltt]
                self.df.loc[li].new_cancer_treatments += sim.get_intervention('radiation').n_products_used.values[ltt]

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


class test_characteristics_analyzer(hpv.Analyzer):
    '''
    Analyzer for computing test characteristics.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.primary_df = None
        self.triage_df = None
        self.primary_screen = None
        self.triage_screen = None
        return

    def initialize(self, sim):
        super().initialize(sim)

        # Pull out characteristics of sim to decide what we need
        simvals = sim.meta.vals
        scen_label = simvals.scen_label
        if scen_label != 'No screening':
            columns = ['test_positive', 'test_negative']
            self.primary_screen = 'screening'
            self.primary_df = pd.DataFrame(0.0, index=pd.Index(['disease_positive', 'disease_negative'],
                                                               name='health state'), columns=columns)
            if simvals.get('triage'):
                self.triage_screen = 'triage'
                self.triage_df = pd.DataFrame(0.0, index=pd.Index(['disease_positive', 'disease_negative'],
                                                                  name='health state'), columns=columns)
            else:
                self.triage_screen = None

        return

    def apply(self, sim):
        if self.primary_screen is not None:
            disease_pos_states = ['cin2', 'cin3', 'cancerous']
            if sim.yearvec[sim.t] >= sim.get_intervention(self.primary_screen).start_year:
                intv = sim.get_intervention(self.primary_screen).outcomes
                test_positives = intv['positive']
                test_negatives = intv['negative']

                test_pos_disease_pos = []
                test_neg_disease_pos = []
                for disease_pos_state in disease_pos_states:
                    test_pos_disease_pos_inds = sim.people[disease_pos_state][:,test_positives].sum(axis=0).nonzero()[0]
                    if len(test_pos_disease_pos_inds):
                        test_pos_disease_pos = np.append(test_pos_disease_pos, test_pos_disease_pos_inds)
                    test_neg_disease_pos_inds = sim.people[disease_pos_state][:,test_negatives].sum(axis=0).nonzero()[0]
                    if len(test_neg_disease_pos_inds):
                        test_neg_disease_pos = np.append(test_neg_disease_pos, test_neg_disease_pos_inds)

                test_pos_disease_pos = list(set(test_pos_disease_pos))
                test_neg_disease_pos = list(set(test_neg_disease_pos))
                test_pos_disease_neg = np.setdiff1d(test_positives, test_pos_disease_pos)
                test_neg_disease_neg = np.setdiff1d(test_negatives, test_neg_disease_pos)

                if len(test_pos_disease_pos):
                    self.primary_df.loc['disease_positive'].test_positive += sim.people.scale_flows(test_pos_disease_pos)
                if len(test_pos_disease_neg):
                    self.primary_df.loc['disease_negative'].test_positive += sim.people.scale_flows(test_pos_disease_neg)
                if len(test_neg_disease_pos):
                    self.primary_df.loc['disease_positive'].test_negative += sim.people.scale_flows(test_neg_disease_pos)
                if len(test_neg_disease_neg):
                    self.primary_df.loc['disease_negative'].test_negative += sim.people.scale_flows(test_neg_disease_neg)

                if self.triage_screen is not None:
                    intv = sim.get_intervention(self.triage_screen).outcomes
                    test_positives = intv['positive']
                    test_negatives = intv['negative']

                    test_pos_disease_pos = []
                    test_neg_disease_pos = []
                    for disease_pos_state in disease_pos_states:
                        test_pos_disease_pos_inds = sim.people[disease_pos_state][:, test_positives].sum(axis=0).nonzero()[0]
                        if len(test_pos_disease_pos_inds):
                            test_pos_disease_pos = np.append(test_pos_disease_pos, test_pos_disease_pos_inds)
                        test_neg_disease_pos_inds = sim.people[disease_pos_state][:, test_negatives].sum(axis=0).nonzero()[0]
                        if len(test_neg_disease_pos_inds):
                            test_neg_disease_pos = np.append(test_neg_disease_pos, test_neg_disease_pos_inds)

                    test_pos_disease_pos = list(set(test_pos_disease_pos))
                    test_neg_disease_pos = list(set(test_neg_disease_pos))
                    test_pos_disease_neg = np.setdiff1d(test_positives, test_pos_disease_pos)
                    test_neg_disease_neg = np.setdiff1d(test_negatives, test_neg_disease_pos)

                    self.triage_df.loc['disease_positive'].test_positive += len(test_pos_disease_pos)
                    self.triage_df.loc['disease_negative'].test_positive += len(test_pos_disease_neg)
                    self.triage_df.loc['disease_positive'].test_negative += len(test_neg_disease_pos)
                    self.triage_df.loc['disease_negative'].test_negative += len(test_neg_disease_neg)

        return



