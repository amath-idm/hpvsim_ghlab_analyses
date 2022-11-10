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

# Comment out to not run
to_run = [
    'run_scenarios',
    'plot_scenarios',
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
    i_vx, i_sc, i_txs, i_inds, i_iys, i_cc, i_s = sims[0].meta.inds
    for s,sim in enumerate(sims): # Check that everything except seed matches
        assert i_vx   == sim.meta.inds[0]
        assert i_sc   == sim.meta.inds[1]
        assert i_txs  == sim.meta.inds[2]
        assert i_inds == sim.meta.inds[3]
        assert i_iys  == sim.meta.inds[4]
        assert i_cc   == sim.meta.inds[5]
        assert (s==0) or i_s != sim.meta.inds[6]
    msim.meta = sc.objdict()
    msim.meta.inds = [i_vx, i_sc, i_txs, i_inds, i_iys, i_cc]
    msim.meta.vals = sc.dcp(sims[0].meta.vals)
    msim.meta.vals.pop('seed')
    
    print(f'Processing multisim {msim.meta.vals.values()}...')
    if save_msims: # Warning, generates a lot of files!
        id_str = '_'.join([str(i) for i in msim.meta.inds])
        msimfile = f'{ut.resfolder}/final_msim{id_str}.msim'
        msim.save(msimfile)

    return msim


def run_scens(location=None, vaccination_coverage=None, # Input data
              vx_scens=None, screen_scens=None, tx_vx_scens=None, indications=None, intro_years=None, ccuts=None, # Scenarios
              multiscale=True, debug=0, n_seeds=2, verbose=-1, do_shrink=True, save_econ=False# Sim settings
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
    n_sims = len(vx_scens) * len(screen_scens) * len(tx_vx_scens) * len(indications) * len(intro_years) * len(ccuts) * n_seeds

    for i_vx, vx_scen in enumerate(vx_scens): # 2 prophylactic scenrios: baseline or scaleup
        for i_sc, screen_scen in enumerate(screen_scens):  # 2 screening scenarios: baseline or 70/90/70
            for i_txs, tx_vx_scen in enumerate(tx_vx_scens):  # 3 use cases
                for i_inds, indication in enumerate(indications): # 2 indications
                    for i_iys, intro_year in enumerate(intro_years): # introduction years
                        for i_cc, ccut in enumerate(ccuts): # 2 options for lesion growth
                            for i_s in range(n_seeds): # n seeds
                                count += 1
                                meta = sc.objdict()
                                meta.count = count
                                meta.n_sims = n_sims
                                meta.inds = [i_vx, i_sc, i_txs, i_inds, i_iys, i_cc, i_s]
                                meta.vals = sc.objdict(vx_scen=vx_scen, screen_scen=screen_scen, tx_vx_scen=tx_vx_scen,
                                                       indication=indication, intro_year=intro_year, ccut=ccut, seed=i_s)
                                ikw.append(sc.dcp(meta.vals))
                                ikw[-1].meta = meta

    # Actually run
    sc.heading(f'Running {len(ikw)} scenario sims...')
    kwargs = dict(use_calib_pars=True, verbose=verbose, vaccination_coverage=vaccination_coverage, debug=debug,
                  location=location, multiscale=multiscale, do_shrink=do_shrink, save_econ=save_econ)
    all_sims = sc.parallelize(rs.run_sim, iterkwargs=ikw, kwargs=kwargs)

    # Rearrange sims
    sims = np.empty((len(vx_scens), len(screen_scens), len(tx_vx_scens), len(indications),
                     len(intro_years), len(ccuts), n_seeds), dtype=object)
    econdfs = sc.autolist()

    for sim in all_sims:  # Unflatten array
        i_vx, i_sc, i_txs, i_inds, i_iys, i_cc, i_s = sim.meta.inds

        if save_econ: # Save econ analyzer dataframe if requested
            econdf = sim.get_analyzer().df
            econdf['location'] = location
            vx_scen_label = 'no_vx' if vx_scens[i_vx] is None else vx_scens[i_vx]
            econdf['vx_scen'] = vx_scen_label
            econdf['screen_scen'] = screen_scens[i_sc]
            econdf['progression'] = 'linear' if ccuts[i_cc] else 'fast'
            tx_vx_scen_label = 'no_txvx' if tx_vx_scens[i_txs] is None else tx_vx_scens[i_txs]
            econdf['tx_vx_scen'] = tx_vx_scen_label
            econdf['intro_year'] = intro_years[i_iys]
            econdf['indication'] = indications[i_inds]
            econdf['seed'] = i_s
            econdfs += econdf
            sim['analyzers'] = [] # Remove the analyzer so we don't need to reduce it

        sims[i_vx, i_sc, i_txs, i_inds, i_iys, i_cc, i_s] = sim

    if save_econ:
        allecondf = pd.concat(econdfs)
        allecondf.to_csv(f'{ut.resfolder}/{location}_econ.csv')

    # Prepare to convert sims to msims
    all_sims_for_multi = []
    for i_vx, vx_scen in enumerate(vx_scens):
        for i_sc, screen_scen in enumerate(screen_scens):
            for i_txs, tx_vx_scen in enumerate(tx_vx_scens):
                for i_inds, indication in enumerate(indications):
                    for i_iys, intro_year in enumerate(intro_years):
                        for i_cc, ccut in enumerate(ccuts):
                            sim_seeds = sims[i_vx, i_sc, i_txs, i_inds, i_iys, i_cc, :].tolist()
                            all_sims_for_multi.append(sim_seeds)

    # Convert sims to msims
    msims = np.empty((len(vx_scens), len(screen_scens), len(tx_vx_scens), len(indications), len(intro_years), len(ccuts)), dtype=object)
    all_msims = sc.parallelize(make_msims, iterarg=all_sims_for_multi)

    # Now strip out all the results and place them in a dataframe
    dfs = sc.autolist()
    for msim in all_msims:
        i_vx, i_sc, i_txs, i_inds, i_iys, i_cc = msim.meta.inds
        msims[i_vx, i_sc, i_txs, i_inds, i_iys, i_cc] = msim

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
        df['n_tx_vaccinated']          = msim.results['n_tx_vaccinated'][:]
        df['n_tx_vaccinated_low']      = msim.results['n_tx_vaccinated'].low
        df['n_tx_vaccinated_high']     = msim.results['n_tx_vaccinated'].high

        df['location'] = location
        vx_scen_label = 'no_vx' if vx_scens[i_vx] is None else vx_scens[i_vx]
        df['vx_scen'] = vx_scen_label
        df['screen_scen'] = screen_scens[i_sc]
        df['progression'] = 'linear' if ccuts[i_cc] else 'fast'
        tx_vx_scen_label = 'no_txvx' if tx_vx_scens[i_txs] is None else tx_vx_scens[i_txs]
        df['tx_vx_scen'] = tx_vx_scen_label
        df['intro_year'] = intro_years[i_iys]
        df['indication'] = indications[i_inds]
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

        alldfs = sc.autolist()
        for location in locations:
            filestem = 'scenario_results'

            vx_scens = [None, '90vx_9to14']
            screen_scens = ['0sc_10tx', '70sc_90tx']
            tx_vx_scens = [None, 'mass_vaccination', 'test_and_vaccinate']
            progressions = [False, True]
            indications = ['virologic_clearance', 'lesion_regression']
            intro_years=[2030]
            alldf, msims = run_scens(screen_scens=screen_scens, vx_scens=vx_scens,
                                     ccuts=progressions, tx_vx_scens=tx_vx_scens, intro_years=intro_years,
                                     indications=indications, n_seeds=n_seeds, location=location, debug=debug,
                                     save_econ=True)

            alldfs += alldf
            sc.saveobj(f'{ut.resfolder}/{location}_{filestem}.obj', alldf)
        bigdf = pd.concat(alldfs)
        sc.saveobj(f'{ut.resfolder}/{filestem}.obj', bigdf)

    # Plot results of scenarios
    if 'plot_scenarios' in to_run:
        for prog in ['linear', 'fast']:
            ut.plot_scens(
                locations=['india', 'nigeria', 'tanzania'],
                background_scens={
                    'No scale-up': {
                        'vx_scen' : 'no_vx',
                        'screen_scen': '0sc_10tx'
                    },
                    '90% vaccine scale-up': {
                        'vx_scen': '90vx_9to14',
                        'screen_scen': '0sc_10tx'
                     },
                    '90% vaccine, 70% screening, 90% treatment': {
                        'vx_scen': '90vx_9to14',
                        'screen_scen': '70sc_90tx'
                    },
                },
                progression=prog,
                indication='virologic_clearance',
                tx_vx_scens=['no_txvx', 'mass_vaccination', 'test_and_vaccinate'],
                debug=debug,
            )

            ut.plot_tx_vx_scens(
                locations=['india', 'nigeria', 'tanzania'],
                background_scens={
                    'No scale-up': {
                        'vx_scen': 'no_vx',
                        'screen_scen': '0sc_10tx'
                    },
                    '90% vaccine scale-up': {
                        'vx_scen': '90vx_9to14',
                        'screen_scen': '0sc_10tx'
                    },
                    '90% vaccine, 70% screening, 90% treatment': {
                        'vx_scen': '90vx_9to14',
                        'screen_scen': '70sc_90tx'
                    },
                },
                progression=prog,
                indication='virologic_clearance',
                tx_vx_scens=['mass_vaccination', 'test_and_vaccinate']
            )

        ut.plot_residual_burden(
            locations=['india', 'nigeria', 'tanzania'],
            background_scens={
                'No scale-up': {
                    'vx_scen': 'no_vx',
                    'screen_scen': '0sc_10tx'
                },
                '90% vaccine scale-up': {
                    'vx_scen': '90vx_9to14',
                    'screen_scen': '0sc_10tx'
                },
                '90% vaccine, 70% screening, 90% treatment': {
                    'vx_scen': '90vx_9to14',
                    'screen_scen': '70sc_90tx'
                },
            },
            compare_models=False
        )