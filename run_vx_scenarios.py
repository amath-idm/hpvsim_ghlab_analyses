'''
Run HPVsim vaccination scenarios
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

# Only running Nigeria for now
locations = [
    'nigeria',  # 1
]

# Options for vaccination coverage
vax_coverage_levels = np.linspace(0,1,11)

#%% Functions

def run_vx_scens(location=None, vax_coverage_levels=None, debug=0, verbose=-1):
    '''
    Run all vaccination coverage levels
    '''

    # Set up iteration arguments
    ikw = []
    count = 0
    n_sims = len(vax_coverage_levels)

    for i_vx, vx_coverage in enumerate(vax_coverage_levels):

        # Make vaccine interventions - campaign for 2025, routine starting in 2025
        routine_vx = hpv.routine_vx(
            prob=vx_coverage,
            start_year=2025,
            product='bivalent',
            age_range=(9, 10),
            label=f'Routine {vx_coverage:.2f}'
        )
        campaign_vx = hpv.campaign_vx(
            prob=vx_coverage,
            years=2025,
            product='bivalent',
            age_range=(9, 14),
            label=f'Campaign {vx_coverage:.2f}'
        )

        meta = sc.objdict()
        meta.count = count
        meta.n_sims = n_sims
        meta.vals = sc.objdict(vx_coverage=vx_coverage)
        ikw.append(sc.objdict(routine_vx=routine_vx, campaign_vx=campaign_vx, label=routine_vx.label))
        ikw[-1].meta = meta
        count += 1

    # Actually run
    sc.heading(f'Running {len(ikw)} scenario sims...')
    kwargs = dict(use_calib_pars=True, verbose=verbose, debug=debug, location=location)
    sims = sc.parallelize(rs.run_sim, iterkwargs=ikw, kwargs=kwargs)

    # SAve results
    dfs = sc.autolist()
    for sim in sims:
        df = pd.DataFrame()
        df['year']                  = sim.results['year']
        df['cancers']               = sim.results['cancers'][:]
        df['cancer_incidence']      = sim.results['cancer_incidence'][:]
        df['asr_cancer_incidence']  = sim.results['asr_cancer_incidence'][:]
        df['cancer_deaths']         = sim.results['cancer_deaths'][:]
        df['n_vaccinated']          = sim.results['n_vaccinated'][:]
        df['n_vaccinated']          = sim.results['n_vaccinated'][:]
        df['hpv_incidence']         = sim.results['hpv_incidence'][:]
        df['location'] = location
        dfs += df

    alldf = pd.concat(dfs)

    return alldf, sims


#%% Run as a script
if __name__ == '__main__':

    T = sc.timer()

    #################################################################
    # RUN AND PLOT SCENARIOS
    #################################################################
    # Run scenarios (usually on VMs, runs n_seeds in parallel over M scenarios)
    filestem = 'vx_results'

    if 'run_scenarios' in to_run:
        alldfs = sc.autolist()
        for location in locations:
            alldf, sims = run_vx_scens(vax_coverage_levels=vax_coverage_levels, location=location)
            alldfs += alldf
            sc.saveobj(f'{ut.resfolder}/{location}_{filestem}.obj', alldf)

        bigdf = pd.concat(alldfs)
        sc.saveobj(f'{ut.resfolder}/{filestem}.obj', bigdf)

    # Plot results of scenarios
    if 'plot_scenarios' in to_run:
        for location in locations:

            df = sc.loadobj(f'{ut.resfolder}/{location}_{filestem}.obj')

            fig, ax = pl.subplots(1, 3, figsize=(16, 10))
            for i_vx, vx_coverage in enumerate(vax_coverage_levels):

                years = np.array(df.index)[50:106]
                best = np.array(df[res])[50:106]
                low = np.array(df[f'{res}_low'])[50:106]
                high = np.array(df[f'{res}_high'])[50:106]

                ax.plot(years, best, color=colors[cn], label=scen_label)
                ax.fill_between(years, low, high, color=colors[cn], alpha=0.3)

            if res == 'asr_cancer_incidence' or res == 'cancer_incidence':
                ax.plot(years, np.full(len(years), fill_value=4), linestyle='dashed', label='Elimination target')
                ax.set_ylim([0, 1.1 * max(high)])
            ax.legend(bbox_to_anchor=(1.05, 0.8), fancybox=True)
            sc.SIticks(ax)
            ax.set_ylabel(f'{reslabel}')
            # ax.set_title(f'{reslabel} in {location.capitalize()}')
            fig.tight_layout()
            fig_name = f'{figfolder}/{res}_{fig_filestem}.png'
            sc.savefig(fig_name, dpi=100)
