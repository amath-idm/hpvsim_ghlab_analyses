'''
Define parameters used in scenarios
'''

import numpy as np
import pandas as pd
import sciris as sc
import hpvsim as hpv
import pars_data as dp


def get_screen_intvs(location, screen_scen=None, screen_prod=None,
                     start_year=2020, end_year=2040, ltfu=None):
    ''' Make interventions for screening scenarios '''

    # Create inputs
    primary_screen = screen_prod[0] # Primary screening product
    triage_screen = screen_prod[1]
    screen_ramp = np.arange(start_year, end_year, dtype=int) # Ramp-up years
    ltfu = ltfu if ltfu else 0.3

    # Only difference between the two screening scenarios is coverage over 2020-2040
    if screen_scen == '0sc_10tx':
        screen_prob_final = 0.0
        treat_prob = 0.1
    elif screen_scen == '10sc_10tx':
        screen_prob_final = 0.1
        treat_prob = 0.1
    elif screen_scen == '30sc_30tx':
        screen_prob_final = 0.3
        treat_prob = 0.3
    elif screen_scen == '50sc_50tx':
        screen_prob_final = 0.5
        treat_prob = 0.5
    elif screen_scen == '70sc_90tx':
        screen_prob_final = 0.7
        treat_prob = 0.9
    else:
        errormsg = f'Screening scenario {screen_scen} not recognized'
        raise NotImplementedError(errormsg)

    # Gradual scale-up of screening
    screen_coverage = list(np.linspace(start=dp.screening_coverage[location], stop=screen_prob_final, num=len(screen_ramp)))
    screen_coverage += [screen_prob_final] * (end_year - start_year + 1)

    # Routine screening
    screen_eligible = lambda sim: np.isnan(sim.people.date_screened) | \
                                  (sim.t > (sim.people.date_screened + 10 / sim['dt']))
    screening = hpv.routine_screening(
        product=primary_screen,
        prob=screen_coverage,
        eligibility=screen_eligible,
        age_range=[30, 50],
        start_year=start_year,
        label='screening'
    )

    if triage_screen is not None:
        # Triage screening
        screen_positive = lambda sim: sim.get_intervention('screening').outcomes['positive']
        triage_screening = hpv.routine_triage(
            start_year=start_year,
            prob=1 - ltfu,
            annual_prob=False,
            product=triage_screen,
            eligibility=screen_positive,
            label='triage'
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

    triage_positive = lambda sim: sim.get_intervention('triage').outcomes['positive']
    assign_treatment = hpv.routine_triage(
        start_year=start_year,
        prob=1.0,
        annual_prob=False,
        product='tx_assigner',
        eligibility=triage_positive,
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

    st_intvs = [screening, triage_screening, assign_treatment, ablation, excision, radiation]

    return st_intvs


def make_AVE(sens=None, spec=None):
    ''' Get AVE parameters '''

    basedf = pd.read_csv('dx_pars.csv')
    lo_grade = ['precin', 'latent', 'cin1']  # No lesions or low grade lesions (a specificity test will count these as negative)
    hi_grade = ['cin2', 'cin3', 'cancerous']  # High grade lesions (a sensitive test will count these as positive)

    # Randomly perturb the test positivity values (lower-bounded by characteristics of VIA)
    if sens is None: sens = np.random.uniform(0.4, 1)
    if spec is None: spec = np.random.uniform(0.7, 1)

    ave_spec = basedf.loc[(basedf.state.isin(lo_grade)) & (basedf.result == 'negative')].copy()
    ave_sens = basedf.loc[(basedf.state.isin(hi_grade)) & (basedf.result == 'positive')].copy()
    ave_fpr = basedf.loc[(basedf.state.isin(lo_grade)) & (basedf.result == 'positive')].copy()
    ave_fnr = basedf.loc[(basedf.state.isin(hi_grade)) & (basedf.result == 'negative')].copy()
    ave_spec.probability = spec
    ave_sens.probability = sens
    ave_fpr.probability = 1-spec
    ave_fnr.probability = 1-sens

    # Make the ave product
    ave = hpv.dx(pd.concat([ave_spec, ave_fpr, ave_sens, ave_fnr]), hierarchy=['positive', 'negative'])
    eff_vals = sc.objdict(
        sens=sens,
        spec=spec
    )
    return ave, eff_vals