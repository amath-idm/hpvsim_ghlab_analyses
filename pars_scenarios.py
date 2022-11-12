'''
Define parameters used in scenarios
'''

import numpy as np
import pandas as pd
import sciris as sc
import hpvsim as hpv
import pars_data as dp


def get_screen_intvs(location, screen_scen=None, product=None, start_year=2020, end_year=2040):
    ''' Make interventions for screening scenarios '''

    # Create inputs
    primary_screen = product if product else 'hpv' # Screening product
    screen_ramp = np.arange(start_year, end_year, dtype=int) # Ramp-up years

    # Only difference between the two screening scenarios is coverage over 2020-2040
    if screen_scen == '0sc_10tx':
        screen_prob_final = dp.screening_coverage[location]
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

    # Assign treatment
    confirmed_positive = lambda sim: sim.get_intervention('screening').outcomes['positive']
    assign_treatment = hpv.routine_triage(
        start_year=start_year,
        prob=1.0,
        annual_prob=False,
        product='tx_assigner',
        eligibility=confirmed_positive,
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

    st_intvs = [screening, assign_treatment, ablation, excision, radiation]

    return st_intvs


def get_mv_intvs(dose1=None, dose2=None, campaign_coverage=None, routine_coverage=None, # Things that must be supplied
                 campaign_years=None, campaign_age=None, dose2_uptake=0.8, intro_year=2030, routine_age=None): # Can be omitted
    ''' Get mass txvx interventions'''

    # Handle inputs
    if campaign_years is None: campaign_years = [intro_year]
    if campaign_age is None: campaign_age = [25, 50]
    if routine_age is None: routine_age = [25, 26]

    # Eligibility
    first_dose_eligible = lambda sim: (sim.people.txvx_doses == 0)
    second_dose_eligible = lambda sim: (sim.people.txvx_doses == 1) & (
            sim.t > (sim.people.date_tx_vaccinated + 0.5 / sim['dt']))


     # Campaign txvx
    campaign_txvx_dose1 = hpv.campaign_txvx(
        prob=campaign_coverage,
        years=campaign_years,
        age_range=campaign_age,
        product=dose1,
        eligibility=first_dose_eligible,
        label='campaign txvx'
    )

    campaign_txvx_dose2 = hpv.campaign_txvx(
        prob=dose2_uptake,
        years=campaign_years,
        age_range=campaign_age,
        product=dose2,
        eligibility=second_dose_eligible,
        label='campaign txvx 2nd dose'
    )

    routine_txvx_dose1 = hpv.routine_txvx(
        prob=routine_coverage,
        start_year=intro_year,
        age_range=routine_age,
        eligibility=first_dose_eligible,
        product=dose1,
        label='routine txvx'
    )

    routine_txvx_dose2 = hpv.routine_txvx(
        prob=dose2_uptake,
        start_year=intro_year,
        age_range=routine_age,
        product=dose2,
        eligibility=second_dose_eligible,
        label='routine txvx 2nd dose'
    )

    mv_intvs = [campaign_txvx_dose1, campaign_txvx_dose2, routine_txvx_dose1, routine_txvx_dose2]

    return mv_intvs


def get_tnv_intvs(dose1=None, dose2=None, campaign_coverage=None, routine_coverage=None, # Things that must be supplied
                  test_product=None, campaign_years=None, campaign_age=None, dose2_uptake=0.8, intro_year=2030, routine_age=None):  # Can be omitted
    ''' Get test & txvx interventions'''

    # Handle inputs
    if campaign_years is None: campaign_years = [intro_year]
    if campaign_age   is None: campaign_age   = [25, 50]
    if routine_age    is None: routine_age    = [25, 26]
    if test_product   is None: test_product   = 'hpv'

    # Run a one-time campaign to test & vaccinate everyone aged 25-50
    test_eligible = lambda sim: (sim.people.txvx_doses == 0)
    txvx_campaign_testing = hpv.campaign_screening(
        product=test_product,
        prob=campaign_coverage,
        eligibility=test_eligible,
        age_range=campaign_age,
        years=campaign_years,
        label='txvx_campaign_testing'
    )

    # In addition, run routine vaccination of everyone aged 25
    test_eligible = lambda sim: (sim.people.txvx_doses == 0)
    txvx_routine_testing = hpv.routine_screening(
        product=test_product,
        prob=routine_coverage,
        eligibility=test_eligible,
        age_range=routine_age,
        start_year=intro_year,
        label='txvx_routine_testing'
    )

    screened_pos = lambda sim: list(set(sim.get_intervention('txvx_routine_testing').outcomes['positive'].tolist()
                                        + sim.get_intervention('txvx_campaign_testing').outcomes['positive'].tolist()))
    deliver_txvx = hpv.linked_txvx(
        prob=1.0,
        product=dose1,
        eligibility=screened_pos,
        label='routine txvx'
    )

    second_dose_eligible = lambda sim: (sim.people.txvx_doses == 1) & (sim.t > (sim.people.date_tx_vaccinated + 0.5 / sim['dt']))
    txvx_dose2 = hpv.linked_txvx(
        prob=dose2_uptake,
        annual_prob=False,
        product=dose2,
        eligibility=second_dose_eligible,
        label='routine txvx 2nd dose'
    )

    tnv_intvs = [txvx_campaign_testing, txvx_routine_testing, deliver_txvx, txvx_dose2]

    return tnv_intvs


def get_txvx_intvs(use_case=None, indication=None, txvx_prods=None, intro_year=2030):
    ''' Get txvx interventions'''

    if indication is not None:
        txvx_prods = make_txvx_indication(indication)
        dose1 = txvx_prods[0]
        dose2 = txvx_prods[1]
    elif txvx_prods is not None:
        dose1 = txvx_prods[0]
        dose2 = txvx_prods[1]
    else:
        dose1 = 'txvx1'
        dose2 = 'txvx2'

    # Values taken from IPM analysis, no longer used
    # campaign_coverage   = {'india': 0.765, 'nigeria':0.5695, 'tanzania':0.765}
    # routine_coverage    = {'india': 0.765, 'nigeria':0.5695, 'tanzania':0.765}
    common_args = dict(dose1=dose1, dose2=dose2, campaign_age=[25, 50], dose2_uptake=0.8,
                       campaign_coverage=0.7, routine_coverage=0.7,
                       intro_year=2030, routine_age=[25, 26])
    if use_case == 'mass_vaccination':
        common_args = sc.mergedicts(common_args, {'routine_age':[25,26]})
        intvs = get_mv_intvs(**common_args)
    elif use_case == 'test_and_vaccinate':
        common_args = sc.mergedicts(common_args, {'campaign_coverage':0.7})
        intvs = get_tnv_intvs(**common_args)

    return intvs


def make_txvx_prods(lo_eff=None, hi_eff=None):
    ''' Get random txvx indication parameters '''

    basedf = pd.read_csv('txvx_pars.csv')
    lo_grade = ['precin', 'cin1']   # No lesions or low grade lesions
    hi_grade = ['cin2', 'cin3']     # High grade lesions
    no_grade = ['latent', 'cancerous'] # All other states - txvx not effective

    # Randomly perturb the efficacy values
    if lo_eff is None: lo_eff = np.random.uniform(0, 1)
    if hi_eff is None: hi_eff = np.random.uniform(0, 1)
    txv1_lo = basedf.loc[(basedf.state.isin(lo_grade)) & (basedf.name == 'txvx1')].copy()
    txv1_hi = basedf.loc[(basedf.state.isin(hi_grade)) & (basedf.name == 'txvx1')].copy()
    txv1_no = basedf.loc[(basedf.state.isin(no_grade)) & (basedf.name == 'txvx1')].copy()
    txv1_lo.efficacy = lo_eff
    txv1_hi.efficacy = hi_eff

    # Make the assumption that the dose efficacy is independent
    lo_eff2 = 1-(1-lo_eff)**2
    hi_eff2 = 1-(1-hi_eff)**2
    txv2_lo = basedf.loc[(basedf.state.isin(lo_grade)) & (basedf.name == 'txvx2')].copy()
    txv2_hi = basedf.loc[(basedf.state.isin(hi_grade)) & (basedf.name == 'txvx2')].copy()
    txv2_no = basedf.loc[(basedf.state.isin(no_grade)) & (basedf.name == 'txvx2')].copy()
    txv2_lo.efficacy = lo_eff2
    txv2_hi.efficacy = hi_eff2

    # Make the products
    txvx1 = hpv.tx(pd.concat([txv1_lo,txv1_hi,txv1_no]))
    txvx2 = hpv.tx(pd.concat([txv2_lo,txv2_hi,txv2_no]))

    txvx_prods = [txvx1 ,txvx2]
    eff_vals = sc.objdict(
        lo_eff = lo_eff,
        hi_eff = hi_eff
    )

    return txvx_prods, eff_vals


def make_txvx_indication(indication='lesion_regression'):
    ''' Get txvx indication parameters '''

    basedf = pd.read_csv('txvx_pars.csv')
    lo_grade = ['precin', 'cin1']   # No lesions or low grade lesions
    hi_grade = ['cin2', 'cin3']     # High grade lesions
    no_grade = ['latent', 'cancerous'] # All other states - txvx not effective

    # Randomly perturb the efficacy values

    txv1_lo = basedf.loc[(basedf.state.isin(lo_grade)) & (basedf.name == 'txvx1')].copy()
    txv1_hi = basedf.loc[(basedf.state.isin(hi_grade)) & (basedf.name == 'txvx1')].copy()
    txv1_no = basedf.loc[(basedf.state.isin(no_grade)) & (basedf.name == 'txvx1')].copy()
    txv1_lo.efficacy = 0.01
    txv1_hi.efficacy = 0.01

    txv2_lo = basedf.loc[(basedf.state.isin(lo_grade)) & (basedf.name == 'txvx2')].copy()
    txv2_hi = basedf.loc[(basedf.state.isin(hi_grade)) & (basedf.name == 'txvx2')].copy()
    txv2_no = basedf.loc[(basedf.state.isin(no_grade)) & (basedf.name == 'txvx2')].copy()

    if indication == 'lesion_regression':
        txv2_lo.efficacy = 0.5
        txv2_hi.efficacy = 0.9
    elif indication == 'virologic_clearance':
        txv2_lo.efficacy = 0.9
        txv2_hi.efficacy = 0.5
        
    # Make the products
    txvx1 = hpv.tx(pd.concat([txv1_lo,txv1_hi,txv1_no]))
    txvx2 = hpv.tx(pd.concat([txv2_lo,txv2_hi,txv2_no]))

    txvx_prods = [txvx1 ,txvx2]

    return txvx_prods


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

    return [ave]