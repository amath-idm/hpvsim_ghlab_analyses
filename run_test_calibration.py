'''
Run HPVsim scenarios for each location.

Note: requires an HPC to run with debug=False; with debug=True, should take 5-15 min
to run.
'''


#%% General settings

# Standard imports
import numpy as np
import pandas as pd
import sciris as sc
import hpvsim as hpv

# Imports from this repository
import run_sim as rs
import utils as ut
import pars_scenarios as sp
import analyzers as an



# Comment out locations to not run
locations = [
    'india',    # 0
    'nigeria',  # 1
    'tanzania', # 2
]


debug = 0
n_seeds = [5, 1][debug] # How many seeds to use for stochasticity in projections
n_draws = [240, 1][debug]

#%% Functions

def run_screen_test(location=None, n_draws=1, sens_vals=None, spec_vals=None, # Input data
                    screen_scens=None, # Input data
                    debug=0, verbose=-1# Sim settings
              ):
    '''
    Run all screening/triage product scenarios for a given location
    '''

    # All scenarios assume scaled-up prophylactic vaccination coverage
    try:
        vaccination_coverage = sc.loadobj(f'{ut.datafolder}/vaccination_coverage.obj')
    except:
        errormsg = 'Vaccination coverage not provided and cannot be loaded from disk. Try either providing it directly or running read_coverage to generate them.'
        raise ValueError(errormsg)

    # Set up iteration arguments
    ikw = []
    count = 0
    n_sims = len(screen_scens) * n_draws * n_draws

    for i_sc, scen_label, screen_scen_pars in screen_scens.enumitems():
        for i_d1 in range(n_draws):
            for i_d2 in range(n_draws):
                sens = sens_vals[i_d1]
                spec = spec_vals[i_d2]
                screen_scen_pars['sens'] = sens
                screen_scen_pars['spec'] = spec
                screen_intvs = sp.get_screen_intvs(location=location, **screen_scen_pars)
                count += 1
                meta = sc.objdict()
                meta.count = count
                meta.n_sims = n_sims
                meta.inds = [i_sc, i_d1, i_d2]
                meta.vals = sc.objdict(sc.mergedicts(screen_scen_pars, dict(scen_label=scen_label)))
                ikw.append(sc.objdict(screen_intvs=screen_intvs, label=scen_label))
                ikw[-1].meta = meta

    # Actually run
    sc.heading(f'Running {len(ikw)} scenario sims...')
    kwargs = dict(use_calib_pars=True, verbose=verbose, vaccination_coverage=vaccination_coverage,
                  sens_analyzer=True, debug=debug, location=location)
    all_sims = sc.parallelize(rs.run_sim, iterkwargs=ikw, kwargs=kwargs)

    # Rearrange sims
    sims = np.empty((len(screen_scens), n_draws, n_draws), dtype=object)
    sensdfs = sc.autolist()

    for sim in all_sims:  # Unflatten array
        i_dx, i_d1, i_d2 = sim.meta.inds
        sims[i_dx, i_d1, i_d2] = sim

        sensdf_primary = sim.get_analyzer(an.test_characteristics_analyzer).primary_df
        sensdf_triage = sim.get_analyzer(an.test_characteristics_analyzer).triage_df
        sensdf = pd.DataFrame()
        sensdf['primary_sens'] = [sensdf_primary.loc['disease_positive'].test_positive / np.sum(sensdf_primary.loc['disease_positive'])]
        sensdf['primary_spec'] = [sensdf_primary.loc['disease_negative'].test_negative / np.sum(sensdf_primary.loc['disease_negative'])]

        if sensdf_triage is not None:
            sensdf['triage_sens'] = [
                sensdf_triage.loc['disease_positive'].test_positive / np.sum(sensdf_triage.loc['disease_positive'])]
            sensdf['triage_spec'] = [
                sensdf_triage.loc['disease_negative'].test_negative / np.sum(sensdf_triage.loc['disease_negative'])]
        else:
            sensdf['triage_sens'] = np.nan
            sensdf['triage_spec'] = np.nan
        sensdf['location'] = location
        for var in ['primary', 'triage', 'sens', 'spec', 'ltfu']:
            if sim.meta.vals.get(var):
                sensdf[var] = sim.meta.vals.get(var)
            else:
                sensdf[var] = np.nan

        # Store label - this will be used for plotting
        sensdf['scen_label'] = sim.meta.vals.scen_label
        sensdfs += sensdf

        sim['analyzers'] = []  # Remove the analyzer so we don't need to reduce it

    allsensdf = pd.concat(sensdfs)
    allsensdf.to_csv(f'{ut.resfolder}/{location}_sens_calibration.csv')

    return allsensdf


#%% Run as a script
if __name__ == '__main__':

    T = sc.timer()

    #################################################################
    # RUN TEST CHARACTERISTICS CALIBRATION
    #################################################################
    # Run scenarios (usually on VMs, runs n_seeds in parallel over M scenarios)

    filestem = 'screen_test_results'
    alldfs = sc.autolist()
    for location in locations:

        sens_vals = np.random.uniform(0, 1, n_draws)
        spec_vals = np.random.uniform(0, 1, n_draws)

        screen_scens = sc.objdict({
            'AVE': dict(primary='ave'),
            'HPV+AVE': dict(primary='hpv', triage='ave', ltfu=0.3)
        })
        df = run_screen_test(screen_scens=screen_scens, n_draws=n_draws,
                                       sens_vals=sens_vals, spec_vals=spec_vals,
                                       location=location, debug=debug)
