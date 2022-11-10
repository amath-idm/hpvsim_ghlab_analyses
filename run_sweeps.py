'''
Run parameter sweeps.

With debug=False, takes 20-40 min to run; with debug=True, should take 2-4 min
to run on a laptop.
'''


#%% General settings

# Standard imports
import numpy as np
import pandas as pd
import sciris as sc

# Imports from this repository
import run_sim as rs
import pars_scenarios as sp
import utils as ut


# Comment out to not run
to_run = [
    'run_sweeps',
    'plot_sweeps',
]

# Comment out locations to not run
locations = [
    'india',    # 0
    'nigeria',  # 1
    'tanzania', # 2
]

debug = False
n_draws = [240, 1][debug]     # How many seeds to use for stochasticity in projections


#%% Sweep function
def run_sweeps(location=None, n_draws=1, lo_effs=None, hi_effs=None, # Input data
               tx_vx_scen=None, vx_scen=None, screen_scen=None, ccuts=None, # Scenarios (for now, fix txvx, vx and screening, vary ccut)
               debug=0, verbose=-1, multiscale=True, do_shrink=True, # Sim settings
               ):
    '''
    Run parameter sweeps
    '''

    vaccination_coverage = sc.loadobj(f'{ut.datafolder}/vaccination_coverage.obj')
    n_sims = len(ccuts) * n_draws

    # Get baseline
    kwargs = dict(vaccination_coverage=vaccination_coverage,
                  vx_scen=vx_scen, screen_scen=screen_scen,
                  multiscale=multiscale, debug=debug, location=location)

    args = rs.make_sim_parts(**kwargs)
    args[0]['end'] = 2060
    sim = rs.make_sim(*args)
    file = f'{ut.resfolder}/{location}_pars.obj'
    try:
        calib_pars = sc.loadobj(file)
    except:
        errormsg = 'Calibration parameters cannot be loaded from disk. Try running load_calib to generate them.'
        raise ValueError(errormsg)
    sim.initialize()  # Important to do this here, otherwise the genotype pars get overwritten
    sim.update_pars(calib_pars)
    sim.run()
    ys = sc.findinds(sim.results['year'],2023)[0]
    ye = sc.findinds(sim.results['year'],2060)[0]
    base_cancers = sim.results['total_cancers'][ys:ye].sum()

    # Set up iteration arguments
    ikw = []
    count = 0
    for i_ccs, ccut in enumerate(ccuts):
        for draw in range(n_draws):
            txvx_prods, eff_vals = sp.make_txvx_prods(lo_eff=lo_effs[draw], hi_eff=hi_effs[draw]) # This takes a random sample of efficacy values
            print(f'Creating arguments for sim {count} of {n_sims}...')
            count += 1
            meta = sc.objdict()
            meta.count = count
            meta.n_sims = n_sims
            meta.inds = [i_ccs, draw]
            meta.eff_vals = eff_vals
            meta.vals = sc.objdict(tx_vx_scen=tx_vx_scen, vx_scen=vx_scen, screen_scen=screen_scen,
                                   txvx_prods=txvx_prods, ccut=ccut, use_calib_pars=True)
            ikw.append(sc.dcp(meta.vals))
            ikw[-1].meta = meta

    sc.heading(f'Running {len(ikw)} sweep sims...')
    kwargs = sc.mergedicts(kwargs, dict(verbose=verbose, do_shrink=do_shrink))
    all_sims = sc.parallelize(rs.run_sim, iterkwargs=ikw, kwargs=kwargs)

    # Now strip out all the results and place them in a dataframe
    dfs = sc.autolist()
    sims = np.empty((len(ccuts), n_draws), dtype=object) # Rearrange sims
    for sim in all_sims:  # Unflatten array
        i_ccs, draw = sim.meta.inds
        sims[i_ccs, draw] = sim

        df = pd.DataFrame()
        df['cancers_averted'] = np.array([base_cancers - sim.results['total_cancers'][ys:ye].sum()])
        df['tx_vaccinated'] = np.array(sim.results['n_tx_vaccinated'])[ys:ye].sum()
        df['location'] = [location]
        df['progression'] = 'linear' if ccuts[i_ccs] else 'fast'
        df['tx_vx_scen'] = [tx_vx_scen]
        df['vx_scen'] = [vx_scen]
        df['screen_scen'] = [screen_scen]
        df['lo_eff'] = [sim.meta.eff_vals['lo_eff']]
        df['hi_eff'] = [sim.meta.eff_vals['hi_eff']]
        dfs += df

    alldf = pd.concat(dfs)

    return alldf, sims



#%% Run as a script
if __name__ == '__main__':

    T = sc.timer()

    # Run sweeps
    if 'run_sweeps' in to_run:
        for tx_vx_scen in ['mass_vaccination', 'test_and_vaccinate']:
            alldfs = sc.autolist()
            lo_effs = np.random.uniform(0,1,n_draws)
            hi_effs = np.random.uniform(0,1,n_draws)
            for location in locations:
                vx_scens = '90vx_9to14' # Not varying prophylactic vax
                screen_scens = '0sc_10tx' # Not varying S&T
                tx_vx_scens = tx_vx_scen # Not varying txvx use case
                progs = [False, True] # Only varying progression assumptions
    
                alldf, msims = run_sweeps(tx_vx_scen=tx_vx_scens, ccuts=progs, vx_scen=vx_scens, screen_scen=screen_scens,
                                          n_draws=n_draws, lo_effs=lo_effs, hi_effs=hi_effs,
                                          location=location, debug=debug)
    
                alldfs += alldf
                sc.saveobj(f'{ut.resfolder}/{location}_{tx_vx_scen}_sweep_results.obj', alldf)
    
            bigdf = pd.concat(alldfs)
            sc.saveobj(f'{ut.resfolder}/{tx_vx_scen}_sweep_results.obj', bigdf)

    # Plot results of sweeps
    if 'plot_sweeps' in to_run:
        for prog in ['fast', 'linear']:
            ut.plot_sweeps(locations=locations, progression=prog)
    
    T.toc('Done')