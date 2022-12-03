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
import pylab as pl

# Imports from this repository
import run_sim as rs
import utils as ut
import pars_scenarios as sp
import analyzers as an

# Comment out to not run
to_run = [
    # 'run_scenarios',
    'plot_scenarios',
]

# Comment out locations to not run
locations = [
    'india',    # 0
    'nigeria',  # 1
    'tanzania', # 2
]

resfolder = 'results'
figfolder = 'figures'
datafolder = 'data'

debug = 0
n_draws = [20, 1][debug]

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


def plot_sweeps(df=None, locations=None, screen_scens=None):

    for location in locations:
        for screen_scen in screen_scens:
            df_to_plot = df[(df.scen_label==screen_scen) & (df.location == location)].groupby(['sens', 'spec']).mean().reset_index()
            if screen_scen == 'AVE':
                to_plot = 'primary'
            else:
                to_plot = 'triage'

            fig = pl.figure(figsize=(12, 10))
            gs = fig.add_gridspec(1, 3, width_ratios=[20, .1, 1])
            pl.subplots_adjust(hspace=0.25, wspace=0.1, left=0.1, right=0.9, top=0.95, bottom=0.1)
            x = np.array(df_to_plot['sens'])
            y = np.array(df_to_plot['spec'])
            z = np.array(df_to_plot[f'{to_plot}_sens'])

            z_min = 0
            z_max = round(max(z), 1)
            npts = 100
            scale = 0.08
            xi = np.linspace(np.min(x), np.max(x), npts)
            yi = np.linspace(np.min(y), np.max(y), npts)
            xx, yy = np.meshgrid(xi, yi)
            zz = sc.gauss2d(x, y, z, xi, yi, scale=scale, xscale=1, yscale=1, grid=True)
            scolors = sc.vectocolor(z, cmap='plasma', minval=z_min, maxval=z_max)

            # Plot heatmap
            axa = fig.add_subplot(gs[0, 0])
            ima = axa.contourf(xx, yy, zz, cmap='plasma', levels=np.linspace(z_min, z_max, 100))
            axa.scatter(x, y, marker='o', c=scolors, edgecolor=[0.3] * 3, s=50, linewidth=0.1, alpha=0.5)
            axa.contour(xx, yy, zz, levels=7, linewidths=0.5, colors='k')
            axa.set_xlabel('Input sensitivity of AVE')
            axa.set_ylabel('Input specificity of AVE')
            axa.set_xlim([np.min(x), np.max(x)])
            axa.set_ylim([np.min(y), np.max(y)])
            axa.set_title(f'Model sensitivity of AVE, {to_plot}', fontsize=28)

            # Colorbar
            axc = fig.add_subplot(gs[0, 2])
            pl.colorbar(ima, ticks=np.linspace(z_min, z_max, 6), cax=axc)

            fig_name = f'{figfolder}/{location}_AVE_sens_{to_plot}.png'
            sc.savefig(fig_name, dpi=100)

            fig = pl.figure(figsize=(12, 10))
            gs = fig.add_gridspec(1, 3, width_ratios=[20, .1, 1])
            pl.subplots_adjust(hspace=0.25, wspace=0.1, left=0.1, right=0.9, top=0.95, bottom=0.1)
            x = np.array(df_to_plot['sens'])
            y = np.array(df_to_plot['spec'])
            z = np.array(df_to_plot[f'{to_plot}_spec'])

            z_min = 0
            z_max = round(max(z), 1)
            npts = 100
            scale = 0.08
            xi = np.linspace(np.min(x), np.max(x), npts)
            yi = np.linspace(np.min(y), np.max(y), npts)
            xx, yy = np.meshgrid(xi, yi)
            zz = sc.gauss2d(x, y, z, xi, yi, scale=scale, xscale=1, yscale=1, grid=True)
            scolors = sc.vectocolor(z, cmap='plasma', minval=z_min, maxval=z_max)

            # Plot heatmap
            axa = fig.add_subplot(gs[0, 0])
            ima = axa.contourf(xx, yy, zz, cmap='plasma', levels=np.linspace(z_min, z_max, 100))
            axa.scatter(x, y, marker='o', c=scolors, edgecolor=[0.3] * 3, s=50, linewidth=0.1, alpha=0.5)
            axa.contour(xx, yy, zz, levels=7, linewidths=0.5, colors='k')
            axa.set_xlabel('Input sensitivity of AVE')
            axa.set_ylabel('Input specificity of AVE')
            axa.set_xlim([np.min(x), np.max(x)])
            axa.set_ylim([np.min(y), np.max(y)])
            axa.set_title(f'Model specificity of AVE, {to_plot}', fontsize=28)

            # Colorbar
            axc = fig.add_subplot(gs[0, 2])
            pl.colorbar(ima, ticks=np.linspace(z_min, z_max, 6), cax=axc)

            fig_name = f'{figfolder}/AVE_spec_{to_plot}_{location}.png'
            sc.savefig(fig_name, dpi=100)

#%% Run as a script
if __name__ == '__main__':

    T = sc.timer()

    #################################################################
    # RUN TEST CHARACTERISTICS CALIBRATION
    #################################################################
    # Run scenarios (usually on VMs, runs n_seeds in parallel over M scenarios)

    filestem = 'sens_calibration'
    if 'run_scenarios' in to_run:
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

    if 'plot_scenarios' in to_run:
        sensdfs = sc.autolist()
        for location in locations:
            sensdfs += pd.read_csv(f'results/{location}_{filestem}.csv')
        sens_res = pd.concat(sensdfs)
        screen_scens = ['AVE', 'HPV+AVE']
        plot_sweeps(sens_res, locations, screen_scens)
        print('done')
