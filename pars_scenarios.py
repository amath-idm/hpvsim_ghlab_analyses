'''
Define parameters used in scenarios
'''

import numpy as np
import pandas as pd
import sciris as sc
import hpvsim as hpv
import pars_data as dp


def get_screen_intvs(location, primary=None, triage=None, ltfu=None, precin=None, cin1=None, cin2=None, cin3=None,
                     cancerous=None, start_year=2020, end_year=2040, sim_end_year=2060):
    ''' Make interventions for screening scenarios '''

    # Return empty list if nothing is defined
    if primary is None: return []

    # Create AVE products
    if primary=='ave' or triage=='ave':
        if precin is None or cin1 is None or cin2 is None or cin3 is None or cancerous is None:
            raise ValueError('Must provide test positivity values if using AVE test.')
        else:
            ave = make_AVE(precin, cin1, cin2, cin3, cancerous)
        if primary == 'ave': primary = ave
        else: triage=ave

    # Define gradual scale-up of screening
    screen_ramp = np.arange(start_year, end_year, dtype=int) # Ramp-up years
    screen_prob_final = 0.7
    treat_prob = 0.9
    screen_coverage = list(np.linspace(start=dp.screening_coverage[location], stop=screen_prob_final, num=len(screen_ramp)))
    screen_coverage += [screen_prob_final] * (sim_end_year - end_year + 1)

    # Routine screening
    screen_eligible = lambda sim: np.isnan(sim.people.date_screened) | \
                                  (sim.t > (sim.people.date_screened + 5 / sim['dt']))
    screening = hpv.routine_screening(
        product=primary,
        prob=screen_coverage,
        eligibility=screen_eligible,
        age_range=[30, 50],
        start_year=start_year,
        label='screening'
    )

    if triage is not None:
        # Triage screening
        screen_positive = lambda sim: sim.get_intervention('screening').outcomes['positive']
        triage_screening = hpv.routine_triage(
            start_year=start_year,
            prob=1 - ltfu,
            annual_prob=False,
            product=triage,
            eligibility=screen_positive,
            label='triage'
        )
        triage_positive = lambda sim: sim.get_intervention('triage').outcomes['positive']
        assign_treatment = hpv.routine_triage(
            start_year=start_year,
            prob=1.0,
            annual_prob=False,
            product='tx_assigner',
            eligibility=triage_positive,
            label='tx assigner'
        )
    else:
        # Assign treatment
        screen_positive = lambda sim: sim.get_intervention('screening').outcomes['positive']
        triage_screening = hpv.routine_triage(
            start_year=start_year,
            prob=1.0,
            annual_prob=False,
            product='tx_assigner',
            eligibility=screen_positive,
            label='tx assigner'
        )

    ablation_eligible = lambda sim: sim.get_intervention('tx assigner').outcomes['ablation']
    ablation = hpv.treat_num(
        prob=treat_prob,
        annual_prob=False,
        product='ablation',
        eligibility=ablation_eligible,
        label='ablation'
    )

    excision_eligible = lambda sim: list(set(sim.get_intervention('tx assigner').outcomes['excision'].tolist() +
                                             sim.get_intervention('ablation').outcomes['unsuccessful'].tolist()))
    excision = hpv.treat_num(
        prob=treat_prob,
        annual_prob=False,
        product='excision',
        eligibility=excision_eligible,
        label='excision'
    )

    radiation_eligible = lambda sim: sim.get_intervention('tx assigner').outcomes['radiation']
    radiation = hpv.treat_num(
        prob=treat_prob,
        annual_prob=False,
        product=hpv.radiation(),
        eligibility=radiation_eligible,
        label='radiation'
    )

    if triage is not None:
        st_intvs = [screening, triage_screening, assign_treatment, ablation, excision, radiation]
    else:
        st_intvs = [screening, triage_screening, ablation, excision, radiation]

    return st_intvs


def make_AVE(precin=0.25, cin1=0.3, cin2=0.45, cin3=0.45, cancerous=0.6):
    '''
    Make AVE product using P(T+| health state) for health states HPV, CIN1, CIN2, CIN3, and cancer
    '''

    basedf = pd.read_csv('dx_pars.csv')
    not_changing_states = ['susceptible', 'latent']
    not_changing = basedf.loc[basedf.state.isin(not_changing_states)].copy()

    new_states = sc.autolist()
    for state, posval in zip(['precin', 'cin1', 'cin2', 'cin3', 'cancerous'],
                             [precin, cin1, cin2, cin3, cancerous]):
        new_pos_vals = basedf.loc[(basedf.state == state) & (basedf.result == 'positive')].copy()
        new_pos_vals.probability = posval
        new_neg_vals = basedf.loc[(basedf.state == state) & (basedf.result == 'negative')].copy()
        new_neg_vals.probability = 1-posval
        new_states += new_pos_vals
        new_states += new_neg_vals
    new_states_df = pd.concat(new_states)

    # Make the ave product
    ave = hpv.dx(pd.concat([not_changing, new_states_df]), hierarchy=['positive', 'negative'])
    return ave