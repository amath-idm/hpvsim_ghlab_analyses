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

# Options for sens/spec for AVE as primary - comment out any not to run
ave_primary_ss = [
    [0.90,0.83],
    [0.82,0.86],
    [0.62,0.86],
]

# Options for sens/spec for AVE as triag - comment out any not to run
ave_triage_ss = [
    [0.95,0.55],
    [0.90,0.70],
]

debug = 0
n_seeds = [5, 1][debug] # How many seeds to use for stochasticity in projections


#%% Functions

def run_scens(location=None, screen_scens=None, # Input data
              debug=0, n_seeds=2, verbose=-1# Sim settings
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
    n_sims = len(screen_scens) * n_seeds

    for i_sc, scen_label, screen_scen_pars in screen_scens.enumitems():
        screen_intvs = sp.get_screen_intvs(location=location, **screen_scen_pars)
        for i_s in range(n_seeds):  # n seeds
            count += 1
            meta = sc.objdict()
            meta.count = count
            meta.n_sims = n_sims
            meta.inds = [i_sc, i_s]
            meta.vals = sc.objdict(sc.mergedicts(screen_scen_pars, dict(seed=i_s, scen_label=scen_label)))
            ikw.append(sc.objdict(screen_intvs=screen_intvs, seed=i_s, label=scen_label))
            ikw[-1].meta = meta

    # Actually run
    sc.heading(f'Running {len(ikw)} scenario sims...')
    kwargs = dict(use_calib_pars=True, verbose=verbose, vaccination_coverage=vaccination_coverage, debug=debug, location=location)
    all_sims = sc.parallelize(rs.run_sim, iterkwargs=ikw, kwargs=kwargs)

    # Rearrange sims
    sims = np.empty((len(screen_scens), n_seeds), dtype=object)
    econdfs = sc.autolist()

    for sim in all_sims:  # Unflatten array
        i_dx, i_s = sim.meta.inds
        sims[i_dx, i_s] = sim
        econdf = sim.get_analyzer(an.econ_analyzer).df
        econdf['location'] = location
        econdf['seed'] = i_s
        for var in ['primary', 'triage', 'sens', 'spec', 'ltfu']:
            if sim.meta.vals.get(var):
                econdf[var] = sim.meta.vals.get(var)
            else:
                econdf[var] = np.nan

        # Store label - this will be used for plotting
        econdf['scen_label'] = sim.meta.vals.scen_label
        econdfs += econdf
        sim['analyzers'] = []  # Remove the analyzer so we don't need to reduce it

    allecondf = pd.concat(econdfs)
    allecondf.to_csv(f'{ut.resfolder}/{location}_econ.csv')

    # Prepare to convert sims to msims
    all_sims_for_multi = []
    for i_dx, dx_prod in enumerate(screen_scens):
        sim_seeds = sims[i_dx, :].tolist()
        all_sims_for_multi.append(sim_seeds)

    # Convert sims to msims
    msims = np.empty((len(screen_scens)), dtype=object)
    all_msims = sc.parallelize(ut.make_msims, iterarg=all_sims_for_multi)

    # Now strip out all the results and place them in a dataframe
    dfs = sc.autolist()
    for msim in all_msims:
        i_dx = msim.meta.inds
        msims[i_dx] = msim
        df = pd.DataFrame()
        df['year']                     = msim.results['year']
        df['total_cancers']            = msim.results['total_cancers'][:] # TODO: process in a loop
        df['total_cancers_low']        = msim.results['total_cancers'].low
        df['total_cancers_high']       = msim.results['total_cancers'].high
        df['total_cancer_incidence']   = msim.results['total_cancer_incidence'][:]
        df['total_cancer_incidence_high']   = msim.results['total_cancer_incidence'].high
        df['total_cancer_incidence_low']   = msim.results['total_cancer_incidence'].low
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

        # Store metadata about run
        for var in ['primary','triage','sens','spec','ltfu']:
            if msim.meta.vals.get(var):
                df[var] = msim.meta.vals.get(var)
            else:
                df[var] = np.nan

        # Store label - this will be used for plotting
        df['scen_label'] = msim.meta.vals.scen_label

        dfs += df

    alldf = pd.concat(dfs)

    return alldf, msims


#%% Run as a script
if __name__ == '__main__':

    T = sc.timer()

    #################################################################
    # RUN AND PLOT SCENARIOS
    #################################################################
    # Run scenarios (usually on VMs, runs n_seeds in parallel over M scenarios)

    if 'run_scenarios' in to_run:
        filestem = 'screening_results'
        alldfs = sc.autolist()
        poc_ltfus = {'POC-HPV': 0.05, 'HPV':0.3}
        for location in locations:

            # Construct the scenarios
            # Scenario list: 12 total
            # Baseline scenarios    : No screening, HPV, VIA, HPV+VIA, POCHPV+VIA (5x options)
            # AVE as primary screen : AVE (1 x 3 sens/spec combos)
            # AVE as triage         : HPV+AVE, POCHPV+AVE (2x 2 sens/spec combos)
            screen_scens = sc.objdict({
                'No screening': dict(),
                'HPV': dict(primary='hpv'),
                'VIA': dict(primary='via')
            })
            for sens, spec in ave_primary_ss:
                screen_scens[f'AVE, {int(sens*100)}%/{int(spec*100)}%'] = dict(primary='ave', sens=sens, spec=spec)
            for sens, spec in ave_triage_ss:
                for poc,ltfu in poc_ltfus.items():
                    screen_scens[f'{poc}+AVE, {int(sens*100)}%/{int(spec*100)}%'] = dict(primary='hpv', triage='ave', sens=sens, spec=spec, ltfu=ltfu)
            for poc, ltfu in poc_ltfus.items():
                screen_scens[f'{poc}+VIA'] = dict(primary='hpv', triage='via', ltfu=ltfu)
            alldf, msims = run_scens(screen_scens=screen_scens, n_seeds=n_seeds, location=location, debug=debug)
            alldfs += alldf
            sc.saveobj(f'{ut.resfolder}/{location}_{filestem}.obj', alldf)

        bigdf = pd.concat(alldfs)
        sc.saveobj(f'{ut.resfolder}/{filestem}.obj', bigdf)

    # Plot results of scenarios
    if 'plot_scenarios' in to_run:
        for location in ['india', 'nigeria', 'tanzania']:
            # First plot: comparing AVE as a primary screen to existing primary screen options
            ut.plot_residual_burden(
                filestem='screening_results',
                locations=[location],
                scens=['No screening', 'HPV', 'VIA', 'AVE, 90%/83%', 'AVE, 82%/86%', 'AVE, 62%/86%'],
                fig_filestem=f'ave_primary_{location}'
            )

            # Second plot: comparing AVE as a triage screen against existing triage options
            ut.plot_residual_burden(
                filestem='screening_results',
                locations=[location],
                scens=['No screening', 'HPV+VIA', 'HPV+AVE, 95%/55%', 'HPV+AVE, 90%/70%'],
                fig_filestem=f'ave_triage_{location}'
            )

            # Third plot: Evaluating impact of POC
            ut.plot_residual_burden(
                filestem='screening_results',
                locations=[location],
                scens=['No screening', 'HPV+VIA', 'HPV+AVE, 95%/55%', 'POC-HPV+VIA', 'POC-HPV+AVE, 95%/55%'],
                fig_filestem=f'poc_effect_{location}'
            )


        # ut.plot_ICER(
        #     filestem='screening_results',
        #     locations=['india', 'nigeria', 'tanzania'],
        #     scens=[
        #         'HPV',
        #         'VIA',
        #         'AVE, 90%/83%',
        #         'AVE, 82%/86%',
        #         'AVE, 62%/86%',
        #         'HPV+VIA',
        #         'HPV+AVE, 95%/55%',
        #         'HPV+AVE, 90%/70%',
        #         'POC HPV+VIA',
        #         'POC HPV+AVE, 95%/55%',
        #         'POC HPV+AVE, 90%/70%',
        #     ],
        #     fig_filestem='icer'
        # )