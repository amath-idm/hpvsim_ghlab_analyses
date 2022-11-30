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

# Comment out to not run
to_run = [
    'run_scenarios',
    # 'plot_scenarios',
]

# Comment out locations to not run
locations = [
    'india',    # 0
    'nigeria',  # 1
    'tanzania', # 2
]

debug = False
n_seeds = [5, 1][debug] # How many seeds to use for stochasticity in projections


#%% Functions

def make_msims(sims, use_mean=True, save_msims=False):
    '''
    Utility to take a slice of sims and turn it into a multisim
    '''

    msim = hpv.MultiSim(sims)
    msim.reduce(use_mean=use_mean)
    i_vx, i_sc, i_dx, i_s = sims[0].meta.inds
    for s,sim in enumerate(sims): # Check that everything except seed matches
        assert i_vx   == sim.meta.inds[0]
        assert i_sc   == sim.meta.inds[1]
        assert i_dx   == sim.meta.inds[2]
        assert (s==0) or i_s != sim.meta.inds[3]
    msim.meta = sc.objdict()
    msim.meta.inds = [i_vx, i_sc, i_dx]
    msim.meta.vals = sc.dcp(sims[0].meta.vals)
    msim.meta.vals.pop('seed')

    print(f'Processing multisim {msim.meta.vals.values()}...')
    if save_msims: # Warning, generates a lot of files!
        id_str = '_'.join([str(i) for i in msim.meta.inds])
        msimfile = f'{ut.resfolder}/final_msim{id_str}.msim'
        msim.save(msimfile)

    return msim


def run_scens(location=None, vaccination_coverage=None, ltfu=None, # Input data
              vx_scens=None, screen_scens=None, screen_prods=None,
              multiscale=True, debug=0, n_seeds=2, verbose=-1, do_shrink=True# Sim settings
              ):
    '''
    Run all scenarios for a given location
    '''

    # Get vaccination coverage assumptions to use in scenarios
    if vx_scens is not None:
        if vaccination_coverage is None:
            try:
                vaccination_coverage = sc.loadobj(f'{ut.datafolder}/vaccination_coverage.obj')
            except:
                errormsg = 'Vaccination coverage not provided and cannot be loaded from disk. Try either providing it directly or running read_coverage to generate them.'
                raise ValueError(errormsg)

    # Set up iteration arguments
    ikw = []
    count = 0
    n_sims = len(vx_scens) * len(screen_scens) * len(screen_prods) * n_seeds

    for i_vx, vx_scen in enumerate(vx_scens): # 2 prophylactic scenrios: baseline or scaleup
        for i_sc, screen_scen in enumerate(screen_scens):  # 2 screening scenarios: baseline or 70/90/70
            for i_dx, screen_prod in enumerate(screen_prods):  # options for screening
                for i_s in range(n_seeds):  # n seeds
                    count += 1
                    meta = sc.objdict()
                    meta.count = count
                    meta.n_sims = n_sims
                    meta.inds = [i_vx, i_sc, i_dx, i_s]
                    meta.vals = sc.objdict(vx_scen=vx_scen, screen_scen=screen_scen, screen_prod=screen_prod if isinstance(screen_prod, str) else screen_prod[0], seed=i_s)
                    ikw.append(sc.dcp(meta.vals))
                    ikw[-1].meta = meta

    # Actually run
    sc.heading(f'Running {len(ikw)} scenario sims...')
    kwargs = dict(use_calib_pars=True, verbose=verbose, vaccination_coverage=vaccination_coverage, ltfu=ltfu, debug=debug,
                  location=location, multiscale=multiscale, do_shrink=do_shrink)
    all_sims = sc.parallelize(rs.run_sim, iterkwargs=ikw, kwargs=kwargs)

    # Rearrange sims
    sims = np.empty((len(vx_scens), len(screen_scens), len(screen_prods), n_seeds), dtype=object)

    for sim in all_sims:  # Unflatten array
        i_vx, i_sc, i_dx, i_s = sim.meta.inds
        sims[i_vx, i_sc, i_dx, i_s] = sim

    # Prepare to convert sims to msims
    all_sims_for_multi = []
    for i_vx, vx_scen in enumerate(vx_scens):
        for i_sc, screen_scen in enumerate(screen_scens):
            for i_dx, dx_prod in enumerate(screen_prods):
                sim_seeds = sims[i_vx, i_sc, i_dx, :].tolist()
                all_sims_for_multi.append(sim_seeds)

    # Convert sims to msims
    msims = np.empty((len(vx_scens), len(screen_scens), len(screen_prods)), dtype=object)
    all_msims = sc.parallelize(make_msims, iterarg=all_sims_for_multi)

    # Now strip out all the results and place them in a dataframe
    dfs = sc.autolist()
    for msim in all_msims:
        i_vx, i_sc, i_dx = msim.meta.inds
        msims[i_vx, i_sc, i_dx] = msim
        df = pd.DataFrame()
        df['year']                     = msim.results['year']
        df['total_cancers']            = msim.results['total_cancers'][:] # TODO: process in a loop
        df['total_cancers_low']        = msim.results['total_cancers'].low
        df['total_cancers_high']       = msim.results['total_cancers'].high
        df['asr_cancer']               = msim.results['asr_cancer'][:]
        df['asr_cancer_low']           = msim.results['asr_cancer'].low
        df['asr_cancer_high']          = msim.results['asr_cancer'].high
        df['total_cancer_deaths']      = msim.results['total_cancer_deaths'][:]
        df['total_cancer_deaths_low']  = msim.results['total_cancer_deaths'].low
        df['total_cancer_deaths_high'] = msim.results['total_cancer_deaths'].high
        df['n_screened']               = msim.results['n_screened'][:]
        df['n_screened_low']           = msim.results['n_screened'].low
        df['n_screened_high']          = msim.results['n_screened'].high
        df['n_cin_treated']            = msim.results['n_cin_treated'][:]
        df['n_cin_treated_low']        = msim.results['n_cin_treated'].low
        df['n_cin_treated_high']       = msim.results['n_cin_treated'].high
        df['n_vaccinated']             = msim.results['n_vaccinated'][:]
        df['n_vaccinated_low']         = msim.results['n_vaccinated'].low
        df['n_vaccinated_high']        = msim.results['n_vaccinated'].high
        df['location'] = location
        vx_scen_label = 'no_vx' if vx_scens[i_vx] is None else vx_scens[i_vx]
        df['vx_scen'] = vx_scen_label
        df['screen_scen'] = screen_scens[i_sc]
        df['ltfu'] = ltfu
        df['primary_screen'] = 'hpv'
        if isinstance(screen_prods[i_dx][1], str):
            df['triage_screen'] = screen_prods[i_dx][1]
            df['sens'] = np.nan
            df['spec'] = np.nan
        elif screen_prods[i_dx] is None:
            df['triage_screen'] = 'no_screen'
            df['sens'] = np.nan
            df['spec'] = np.nan
        else:
            df['triage_screen'] = 'ave'
            df['sens'] = screen_prods[i_dx][1][1]['sens']
            df['spec'] = screen_prods[i_dx][1][1]['spec']
        dfs += df

    alldf = pd.concat(dfs)

    return alldf, msims


#%% Run as a script
if __name__ == '__main__':

    T = sc.timer()

    #################################################################
    # RUN AND PLOT SCENARIOS
    #################################################################
    # Run scenarios -- usually on VMs, runs n_seeds in parallel over M scenarios
    if 'run_scenarios' in to_run:
        filestem = 'triage_results'
        alldfs = sc.autolist()
        for location in locations:
            vx_scens = ['90vx_9to14']
            screen_scens = ['70sc_90tx']
            ave_prods = sc.autolist()
            for sens, spec in zip([.95, .9], [.55, .7]):
                ave_prods.append(['hpv', sp.make_AVE(sens=sens, spec=spec)])
            screen_prods = ['hpv', 'via'] + ave_prods
            alldf, msims = run_scens(screen_scens=screen_scens, vx_scens=vx_scens, screen_prods=screen_prods, ltfu=0.3,
                                     n_seeds=n_seeds, location=location, debug=debug)
            alldfs += alldf
            sc.saveobj(f'{ut.resfolder}/{location}_{filestem}.obj', alldf)
        bigdf = pd.concat(alldfs)
        sc.saveobj(f'{ut.resfolder}/{filestem}.obj', bigdf)

    # Plot results of scenarios
    if 'plot_scenarios' in to_run:
        ut.plot_residual_burden(
            locations=['india', 'nigeria', 'tanzania'],
            background_scens={
                'VIA': {
                    'screen_prod': 'via'
                },
                 'AVE, 90%/83%': {
                     'screen_prod': 'ave',
                     'sens': .9,
                     'spec': .83
                },
                'AVE, 82%/86%': {
                    'screen_prod': 'ave',
                     'sens': .82,
                     'spec': .86
                },
                'AVE, 62%/86%': {
                    'screen_prod': 'ave',
                     'sens': .62,
                     'spec': .86
                },
            }
        )