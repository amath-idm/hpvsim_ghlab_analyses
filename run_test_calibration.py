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
import hpvsim as hpv

# Imports from this repository
import run_sim as rs
import utils as ut
import pars_scenarios as sp
import analyzers as an

# Comment out to not run
to_run = [
    'run_test_calib',
    # 'analyze_test_calib',
    # 'plot_test_calib',
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
n_draws = [2000, 5][debug]

#%% Functions

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


def run_screen_test(location=None, n_draws=1, test_pos_vals=None, # Input data
                    screen_scens=None, # Input data
                    debug=0, verbose=-1, end=2030# Sim settings
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
    n_sims = len(screen_scens) * n_draws

    for i_sc, scen_label, screen_scen_pars in screen_scens.enumitems():
        if end < 2060:
            screen_scen_pars['end_year'] = end
            screen_scen_pars['sim_end_year'] = end
        for i_d in range(n_draws):
            test_pos_val = dict()
            for key,val in test_pos_vals.items():
                test_pos_val[key] = val[i_d]
            AVE = make_AVE(**test_pos_val)
            if screen_scen_pars['primary']=='ave':
                screen_scen_pars['primary'] = AVE
            elif 'triage' in screen_scen_pars.keys():
                screen_scen_pars['triage'] = AVE
            screen_intvs = sp.get_screen_intvs(location=location, **screen_scen_pars)
            if not screen_scen_pars.get('triage'):
                screen_scen_pars['primary'] = 'ave'
            count += 1
            meta = sc.objdict()
            meta.count = count
            meta.n_sims = n_sims
            meta.inds = [i_sc, i_d]
            meta.vals = sc.objdict(sc.mergedicts(screen_scen_pars, dict(scen_label=scen_label), test_pos_val))
            ikw.append(sc.objdict(screen_intvs=screen_intvs, label=scen_label))
            ikw[-1].meta = meta

    # Actually run
    sc.heading(f'Running {len(ikw)} scenario sims...')
    kwargs = dict(use_calib_pars=True, verbose=verbose, vaccination_coverage=vaccination_coverage,
                  sens_analyzer=True, debug=debug, location=location, end=end)
    all_sims = sc.parallelize(rs.run_sim, iterkwargs=ikw, kwargs=kwargs)

    # Rearrange sims
    sims = np.empty((len(screen_scens), n_draws), dtype=object)
    sensdfs = sc.autolist()

    for sim in all_sims:  # Unflatten array
        i_dx, i_d = sim.meta.inds
        sims[i_dx, i_d] = sim

        sensdf_primary = sim.get_analyzer(an.test_characteristics_analyzer).primary_df
        sensdf_triage = sim.get_analyzer(an.test_characteristics_analyzer).triage_df
        sensdf = pd.DataFrame()
        sensdf['primary_sens'] = [sensdf_primary.loc['disease_positive'].test_positive / np.sum(sensdf_primary.loc['disease_positive'])]
        sensdf['primary_spec'] = [sensdf_primary.loc['disease_negative'].test_negative / np.sum(sensdf_primary.loc['disease_negative'])]

        if sensdf_triage is not None:
            sensdf['triage_sens'] = [sensdf_triage.loc['disease_positive'].test_positive / np.sum(sensdf_triage.loc['disease_positive'])]
            sensdf['triage_spec'] = [sensdf_triage.loc['disease_negative'].test_negative / np.sum(sensdf_triage.loc['disease_negative'])]
        else:
            sensdf['triage_sens'] = np.nan
            sensdf['triage_spec'] = np.nan
        sensdf['location'] = location
        for var in ['primary', 'triage', 'precin', 'cin1', 'cin2', 'cin3', 'cancerous', 'ltfu']:
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


def plot_sweeps(df=None, location=None, screen_scens=None):
    for screen_scen in screen_scens:
        df_to_plot = df[df.scen_label == screen_scen].groupby(['sens', 'spec']).mean().reset_index()
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

        fig_name = f'{figfolder}/AVE_sens_{to_plot}_{location}.png'
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
    if 'run_test_calib' in to_run:
        alldfs = sc.autolist()
        for location in locations:
            precin_vals = np.random.uniform(0, .2, n_draws)
            cin1_vals = np.random.uniform(0, .3, n_draws)
            cin2_vals = np.random.uniform(0.5, 1, n_draws)
            cin3_vals = np.random.uniform(0.6, 1, n_draws)
            cancerous_vals = np.random.uniform(0.7, 1, n_draws)

            test_pos_vals ={
                'precin': precin_vals,
                'cin1': cin1_vals,
                'cin2': cin2_vals,
                'cin3': cin3_vals,
                'cancerous': cancerous_vals
            }

            screen_scens = sc.objdict({
                'AVE': dict(primary='ave'),
                # 'HPV+AVE': dict(primary='hpv', triage='ave', ltfu=0.3)
            })
            df = run_screen_test(screen_scens=screen_scens, n_draws=n_draws,
                                 test_pos_vals=test_pos_vals, location=location,
                                 debug=debug)

    if 'plot_test_calib' in to_run:
        screen_scens = ['AVE', 'HPV+AVE']
        for location in locations:
            sensdf = pd.read_csv(f'results/{location}_{filestem}.csv')
            plot_sweeps(sensdf, location, screen_scens)

    if 'analyze_test_calib' in to_run:
        # Options for sens/spec for AVE as primary
        ave_primary_ss = [
            [0.90, 0.83],
            [0.82, 0.86],
            [0.62, 0.86],
        ]
        # Options for sens/spec for AVE as triage
        ave_triage_ss = [
            [0.95, 0.55],
            [0.90, 0.70],
        ]
        screen_scens = {
            'AVE': ave_primary_ss,
            'HPV+AVE': ave_triage_ss
        }

        for location in locations:
            sensdf = pd.read_csv(f'results/{location}_{filestem}.csv')
            scendf_list = sc.autolist()
            for screen_scen, scen_vals in screen_scens.items():
                if screen_scen == 'AVE':
                    test_type = 'primary'
                else:
                    test_type = 'triage'
                df = sensdf[sensdf.scen_label == screen_scen]
                scenvaldf_list = sc.autolist()
                for vals_to_fit in scen_vals:
                    df_to_compare = df.copy()
                    model_output = np.array([df_to_compare[f'{test_type}_sens'], df_to_compare[f'{test_type}_spec']]).transpose()
                    mismatch = sc.autolist()
                    for vals in model_output:
                        gofs = hpv.compute_gof(vals_to_fit, vals)
                        mismatch += gofs.sum()
                    df_to_compare['fit'] = mismatch
                    df_to_compare = df_to_compare.sort_values('fit').reset_index()
                    df_to_compare = df_to_compare.drop(columns=['index'])
                    df_to_compare = pd.DataFrame(df_to_compare[df_to_compare.index==0])
                    df_to_compare['primary_sens_to_fit'] = vals_to_fit[0] if test_type == 'primary' else np.nan
                    df_to_compare['primary_spec_to_fit'] = vals_to_fit[1] if test_type == 'primary' else np.nan
                    df_to_compare['triage_sens_to_fit'] = vals_to_fit[0] if test_type == 'triage' else np.nan
                    df_to_compare['triage_spec_to_fit'] = vals_to_fit[1] if test_type == 'triage' else np.nan
                    scenvaldf_list += df_to_compare
                scenvaldf = pd.concat(scenvaldf_list)
                scendf_list += scenvaldf
            scendf = pd.concat(scendf_list)
            scendf.to_csv(f'{ut.resfolder}/{location}_sens_calibration_results.csv')

    print('done')
