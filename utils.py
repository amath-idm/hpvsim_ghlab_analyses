'''
Utilities for HPVsim therapeutic vaccine analyses, mostly related to plotting
'''

import sciris as sc
import numpy as np
import pandas as pd

import pylab as pl
import seaborn as sns
import hpvsim.plotting as hppl

import hpvsim as hpv


resfolder = 'results'
figfolder = 'figures'
datafolder = 'data'


########################################################################
#%% Plotting utils
########################################################################

def set_font(size=None, font='Libertinus Sans'):
    ''' Set a custom font '''
    sc.fonts(add=sc.thisdir(aspath=True) / 'assets' / 'LibertinusSans-Regular.otf')
    sc.options(font=font, fontsize=size)
    return


def process_country_files(locations, top_results=100, do_save=True):
    '''
    Read in all country files and create a master dataframe
    '''

    dfs = []
    for i, location in enumerate(locations):
        file = f'{resfolder}/{location}_calib.obj'
        calib = sc.loadobj(file)
        thisdf = calib.df.sort_values(by=['mismatch'])[:top_results]
        thisdf['location'] = f'{location.capitalize()}'
        dfs.append(thisdf)

    all_calib_pars = pd.concat(dfs)
    if do_save:
        sc.save(f'{resfolder}/all_calib_pars.obj', all_calib_pars)

    return all_calib_pars


def plot_trend(calib, best_thresh=2):
    '''
    Plot the trend in best mismatch over time.
    '''
    mismatch = sc.dcp(calib.df['mismatch'].values)
    best_mismatch = np.zeros(len(mismatch))
    for i in range(len(mismatch)):
        best_mismatch[i] = mismatch[:i+1].min()
    smoothed_mismatch = sc.smooth(mismatch)
    fig = pl.figure(figsize=(16,12), dpi=120)

    ax1 = pl.subplot(2,1,1)
    pl.plot(mismatch, alpha=0.2, label='Original')
    pl.plot(smoothed_mismatch, lw=3, label='Smoothed')
    pl.plot(best_mismatch, lw=3, label='Best')

    ax2 = pl.subplot(2,1,2)
    max_mismatch = mismatch.min()*best_thresh
    inds = sc.findinds(mismatch<=max_mismatch)
    pl.plot(best_mismatch, lw=3, label='Best')
    pl.scatter(inds, mismatch[inds], c=mismatch[inds], label='Usable indices')
    for ax in [ax1, ax2]:
        pl.sca(ax)
        pl.grid(True)
        pl.legend()
        sc.setylim()
        sc.setxlim()
        pl.xlabel('Trial number')
        pl.ylabel('Mismatch')
    return hppl.handle_show_return(fig=fig)


def pairplotpars(data, inds=None, color_column=None, bounds=None, cmap='parula', bins=None, edgecolor='w',
                 facecolor='#F8A493', figsize=(20, 16)):
    '''
    Plot scatterplots, histograms, and kernel densities for calibration results
    '''

    data = sc.odict(sc.dcp(data))

    # Create the dataframe
    df = pd.DataFrame.from_dict(data)
    if inds is not None:
        df = df.iloc[inds, :].copy()

    # Choose the colors
    if color_column:
        colors = sc.vectocolor(df[color_column].values, cmap=cmap)
    else:
        colors = [facecolor for i in range(len(df))]
    df['color_column'] = [sc.rgb2hex(rgba[:-1]) for rgba in colors]

    # Make the plot
    grid = sns.PairGrid(df)
    grid = grid.map_lower(pl.scatter, **{'facecolors': df['color_column']})
    grid = grid.map_diag(pl.hist, bins=bins, edgecolor=edgecolor, facecolor=facecolor)
    grid = grid.map_upper(sns.kdeplot)
    grid.fig.set_size_inches(figsize)
    grid.fig.tight_layout()

    # Set bounds
    if bounds:
        for ax in grid.axes.flatten():
            xlabel = ax.get_xlabel()
            ylabel = ax.get_ylabel()
            if xlabel in bounds:
                ax.set_xlim(bounds[xlabel])
            if ylabel in bounds:
                ax.set_ylim(bounds[ylabel])

    return grid


def plot_best(calib, best_thresh=2): # pragma: no cover
    ''' Plot only the points with lowest mismatch. '''
    max_mismatch = calib.df['mismatch'].min()*best_thresh
    inds = sc.findinds(calib.df['mismatch'].values <= max_mismatch)
    g = pairplotpars(calib.data, inds=inds, color_column='mismatch', bounds=calib.par_bounds)
    return g


def plot_calib_pars(locations=None, to_plot=None, do_save=True):
    '''
    Plot posterior distributions of calibration parameters for specified locations
    Args:
         locations (list): locations to plot
         to_plot (list): parameters to plot
         do_save (bool): whether to save the figure
    '''

    all_calib_pars = process_country_files(locations=locations, do_save=True)

    if to_plot is None:
        to_plot = [
            'hpv16_dur_dysp_par1',
            'hpv18_dur_dysp_par1',
            'hrhpv_dur_dysp_par1',
            'hpv16_dysp_rate',
            'hpv18_dysp_rate',
            'hrhpv_dysp_rate',
            'hpv16_prog_rate',
            'hpv18_prog_rate',
            'hrhpv_prog_rate',
            'hrhpv_rel_beta',
        ]

    # Figure settings
    set_font(size=26)
    colors = sc.gridcolors(10)

    # Create figure
    n_rows, n_cols = sc.get_rows_cols(len(to_plot), ratio=0.5)
    fig, axes = pl.subplots(n_rows, n_cols, figsize=(24, 12))
    for ax in axes.flat[len(to_plot):]:
        ax.set_visible(False)
    axes = axes.flatten()

    for pn,par in enumerate(to_plot):
        ax = axes[pn]
        legendon = True if pn==1 else False
        for li, location in enumerate(locations):
            label = f'{location.capitalize()}'
            location_label = f'{location.capitalize()}'
            data_to_plot = all_calib_pars[all_calib_pars['location'] == location_label]
            data_to_plot = data_to_plot[par].values
            if ~np.isnan(data_to_plot).any():
                count, bins_count = np.histogram(data_to_plot, bins=10)
                pdf = count / sum(count)
                if legendon:
                    ax.plot(bins_count[1:], pdf, color=colors[li], label=label)
                else:
                    ax.plot(bins_count[1:], pdf, color=colors[li])

        if pn==1:
            ax.legend(frameon=False).set_title('Location')
            # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
            #           fancybox=True, shadow=True, ncol=5)
        if par != 'hpv_reactivation':
            ax.set_title(par[:5], y=1.05)

        if 'dur_dysp' in par:
            ax.set_xlabel("Duration of dysplasia")
        elif 'rel_beta' in par:
            ax.set_xlabel("Transmissibility relative to HPV16")
        elif 'prog_rate' in par:
            ax.set_xlabel('Rate of dysplasia progression')
        elif 'dysp_rate' in par:
            ax.set_xlabel('Rate of HPV progression')
        elif 'hpv_reactivation' in par:
            ax.set_xlabel('Annual probability of HPV reactivation')
        ax.set_ylabel("")
    fig.suptitle('Comparison of calibration parameters across locations')
    fig.tight_layout()
    if do_save:
        sc.savefig(f'{figfolder}/parplot.png', dpi=100)

    return


def plot_residual_burden(locations=None, background_scens=None, triage=False):
    '''
    Plot the residual burden of HPV
    '''
    
    set_font(size=24)

    location_legend = [i.capitalize() for i in locations]
    if triage:
        results_filestem='triage_results'
        fig_filestem='triage'
    else:
        results_filestem = 'general_screening_results'
        fig_filestem = 'general_screening'

    try:
        bigdf = sc.loadobj(f'{resfolder}/{results_filestem}.obj')
    except:
        print('bigdf not available, trying to load for each location and generate it')
        alldfs = sc.autolist()
        for location in locations:
            alldf = sc.loadobj(f'{resfolder}/{location}_{results_filestem}.obj')
            alldfs += alldf
        bigdf = pd.concat(alldfs)
    colors = sc.gridcolors(10)

    fig, axes = pl.subplots(2, 1, figsize=(10, 10), sharex=True)
    for ir, (res, reslabel) in enumerate({'total_cancers': 'Annual cases of cervical cancer', 'total_cancer_deaths': 'Annual deaths from cervical cancer'}.items()):
        ax = axes[ir]
        for cn, (background_scen_label, background_scen) in enumerate(background_scens.items()):
            screen_prod = background_scen['screen_prod']
            if triage:
                assert len(screen_prod) > 1
                screen_prod = screen_prod[1]
                col_name = 'triage_screen'
            else:
                col_name = 'primary_screen'
            if screen_prod == 'via':
                df = bigdf[(bigdf[col_name] == screen_prod)].groupby('year')[[f'{res}', f'{res}_low', f'{res}_high']].sum()
            else:
                sens = background_scen['sens']
                spec = background_scen['spec']
                df = bigdf[(bigdf[col_name] == screen_prod) & (bigdf.sens == sens)
                           & (bigdf.spec == spec)].groupby('year')[[f'{res}', f'{res}_low', f'{res}_high']].sum()

            years = np.array(df.index)[50:106]
            best = np.array(df[res])[50:106]
            low = np.array(df[f'{res}_low'])[50:106]
            high = np.array(df[f'{res}_high'])[50:106]

            ax.plot(years, best, color=colors[cn], label=background_scen_label)
            ax.fill_between(years, low, high, color=colors[cn], alpha=0.3)

        if ir:
            ax.legend(loc='upper left')
        sc.SIticks(ax)
        ax.set_title(f'{reslabel}')
    fig.tight_layout()
    fig_name = f'{figfolder}/{fig_filestem}_health_impact.png'
    sc.savefig(fig_name, dpi=100)

    cancers_averted = dict()
    cancer_deaths_averted = dict()
    cin_treatments = dict()
    cancers_averted_low = dict()
    cancer_deaths_averted_low = dict()
    cin_treatments_low = dict()
    cancers_averted_high = dict()
    cancer_deaths_averted_high = dict()
    cin_treatments_high = dict()

    df = bigdf[(bigdf.primary_screen == 'via')].groupby('year')[
        ['total_cancers', f'total_cancers_low', f'total_cancers_high',
         'total_cancer_deaths', f'total_cancer_deaths_low', f'total_cancer_deaths_high',
         'n_cin_treated', f'n_cin_treated_low', f'n_cin_treated_high',
         ]].sum()
    base_cancers = np.array(df['total_cancers'])[50:106].sum()
    base_cancers_low = np.array(df['total_cancers_low'])[50:106].sum()
    base_cancers_high = np.array(df['total_cancers_high'])[50:106].sum()

    base_cancer_deaths = np.array(df['total_cancer_deaths'])[50:106].sum()
    base_cancer_deaths_low = np.array(df['total_cancer_deaths_low'])[50:106].sum()
    base_cancer_deaths_high = np.array(df['total_cancer_deaths_high'])[50:106].sum()

    base_cin_treated = np.array(df['n_cin_treated'])[50:106].sum()
    base_cin_treated_low = np.array(df['n_cin_treated_low'])[50:106].sum()
    base_cin_treated_high = np.array(df['n_cin_treated_high'])[50:106].sum()
    for cn, (background_scen_label, background_scen) in enumerate(background_scens.items()):
        screen_prod = background_scen['screen_prod']
        if screen_prod != 'via':
            sens = background_scen['sens']
            spec = background_scen['spec']
            df = bigdf[(bigdf.primary_screen == screen_prod) & (bigdf.sens == sens)
                       & (bigdf.spec == spec)].groupby('year')[['total_cancers', f'total_cancers_low', f'total_cancers_high',
                 'total_cancer_deaths', f'total_cancer_deaths_low', f'total_cancer_deaths_high',
                 'n_cin_treated', f'n_cin_treated_low', f'n_cin_treated_high',
                 ]].sum()
            cancers_averted[background_scen_label] = base_cancers - np.array(df['total_cancers'])[50:106].sum()
            cancer_deaths_averted[background_scen_label] = base_cancer_deaths - np.array(df['total_cancer_deaths'])[50:106].sum()
            cin_treatments[background_scen_label] = np.array(df['n_cin_treated'])[50:106].sum() - base_cin_treated

            cancers_averted_low[background_scen_label] = base_cancers_low - np.array(df['total_cancers_low'])[50:106].sum()
            cancer_deaths_averted_low[background_scen_label] = base_cancer_deaths_low - np.array(df['total_cancer_deaths_low'])[50:106].sum()
            cin_treatments_low[background_scen_label] = np.array(df['n_cin_treated_low'])[50:106].sum() - base_cin_treated_low

            cancers_averted_high[background_scen_label] = base_cancers_high - np.array(df['total_cancers_high'])[50:106].sum()
            cancer_deaths_averted_high[background_scen_label] = base_cancer_deaths_high - np.array(df['total_cancer_deaths_high'])[50:106].sum()
            cin_treatments_high[background_scen_label] = np.array(df['n_cin_treated_high'])[50:106].sum() - base_cin_treated_high

    nnt_cases = np.array(list(cin_treatments.values())) / np.array(list(cancers_averted.values()))
    nnt_deaths = np.array(list(cin_treatments.values())) / np.array(list(cancer_deaths_averted.values()))
    nnt_cases_low = np.array(list(cin_treatments_low.values())) / np.array(list(cancers_averted_low.values()))
    nnt_cases_high = np.array(list(cin_treatments_high.values())) / np.array(list(cancers_averted_high.values()))
    nnt_deaths_low = np.array(list(cin_treatments_low.values())) / np.array(list(cancer_deaths_averted_low.values()))
    nnt_deaths_high = np.array(list(cin_treatments_high.values())) / np.array(list(cancer_deaths_averted_high.values()))

    fig, axes = pl.subplots(2, 1, figsize=(10, 10), sharex=True)
    scen_labels = cancers_averted.keys()
    axes[0].bar(scen_labels, nnt_cases, yerr = np.absolute(nnt_cases_low - nnt_cases_high))
    axes[0].set_title('Ablations per case averted (relative to VIA)')
    axes[1].bar(scen_labels, nnt_deaths, yerr = np.absolute(nnt_deaths_low - nnt_deaths_high))
    axes[1].set_title('Ablations per death averted (relative to VIA)')
    sc.SIticks(axes[0])
    sc.SIticks(axes[1])
    fig.tight_layout()
    fig_name = f'{figfolder}/{fig_filestem}_NNT.png'
    sc.savefig(fig_name, dpi=100)
    return


def plot_sweeps(fulldf=None, location='india', ltfu=None, scale=1e6): # TODO: set this up to plot cancers averted and/or NNT
    '''
    Plot parameter sweeps
    '''

    # Initialize figure
    fig = pl.figure(figsize=(12, 10))
    gs = fig.add_gridspec(1, 3, width_ratios=[20, .1, 1])
    pl.subplots_adjust(hspace=0.25, wspace=0.1, left=0.1, right=0.9, top=0.95, bottom=0.1)

    df = fulldf[fulldf.ltfu==ltfu].groupby(['sens','spec']).sum().reset_index()
    x = np.array(df['sens'])
    y = np.array(df['spec'])
    z = np.array(df['cancers_averted'])/scale
    z_min = 0
    z_max = round(max(z),1)
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
    axa.scatter(x, y, marker='o', c=scolors, edgecolor=[0.3]*3, s=50, linewidth=0.1, alpha=0.5)
    axa.contour(xx, yy, zz, levels=7, linewidths=0.5, colors='k')
    axa.set_xlabel('Sensitivity of AVE')
    axa.set_ylabel('Specificity of AVE')
    axa.set_xlim([np.min(x), np.max(x)])
    axa.set_ylim([np.min(y), np.max(y)])
    axa.set_title('Cancers averted, relative to VIA (millions)', fontsize=28)

    # Colorbar
    axc = fig.add_subplot(gs[0, 2])
    pl.colorbar(ima, ticks=np.linspace(z_min, z_max, 6), cax=axc)

    fig_name = f'{figfolder}/{location}_AVE_impact_sweeps_{ltfu}ltfu.png'
    sc.savefig(fig_name, dpi=100)

    # Initialize figure for treatments
    fig = pl.figure(figsize=(12, 10))
    gs = fig.add_gridspec(1, 3, width_ratios=[20, .1, 1])
    pl.subplots_adjust(hspace=0.25, wspace=0.1, left=0.1, right=0.9, top=0.95, bottom=0.1)

    scale=1e6
    z = np.array(df['new_cin_treatments'])/scale
    z_min = round(min(z),1)
    z_max = round(max(z),1)
    npts = 100
    scale = 0.08
    xi = np.linspace(np.min(x), np.max(x), npts)
    yi = np.linspace(np.min(y), np.max(y), npts)
    xx, yy = np.meshgrid(xi, yi)

    zz = sc.gauss2d(x, y, z, xi, yi, scale=scale, xscale=1, yscale=1, grid=True)
    scolors = sc.vectocolor(z, cmap='viridis', minval=z_min, maxval=z_max)

    # Plot heatmap
    axa = fig.add_subplot(gs[0, 0])
    ima = axa.contourf(xx, yy, zz, cmap='viridis', levels=np.linspace(z_min, z_max, 100))
    axa.scatter(x, y, marker='o', c=scolors, edgecolor=[0.3]*3, s=50, linewidth=0.1, alpha=0.5)
    axa.contour(xx, yy, zz, levels=7, linewidths=0.5, colors='k')
    axa.set_xlabel('Sensitivity of AVE')
    axa.set_ylabel('Specificity of AVE')
    axa.set_xlim([np.min(x), np.max(x)])
    axa.set_ylim([np.min(y), np.max(y)])
    axa.set_title('Number CIN treatments, relative to VIA (millions)', fontsize=28)

    # Colorbar
    axc = fig.add_subplot(gs[0, 2])
    pl.colorbar(ima, ticks=np.linspace(z_min, z_max, 6), cax=axc)

    fig_name = f'{figfolder}/{location}_AVE_treatments_sweeps_{ltfu}ltfu.png'
    sc.savefig(fig_name, dpi=100)

    # Initialize figure for NNT
    fig = pl.figure(figsize=(12, 10))
    gs = fig.add_gridspec(1, 3, width_ratios=[20, .1, 1])
    pl.subplots_adjust(hspace=0.25, wspace=0.1, left=0.1, right=0.9, top=0.95, bottom=0.1)

    z = np.array(df['new_cin_treatments'])/np.array(df['cancers_averted'])
    z_min = 0
    z_max = 70
    npts = 100
    scale = 0.08
    xi = np.linspace(np.min(x), np.max(x), npts)
    yi = np.linspace(np.min(y), np.max(y), npts)
    xx, yy = np.meshgrid(xi, yi)

    zz = sc.gauss2d(x, y, z, xi, yi, scale=scale, xscale=1, yscale=1, grid=True)
    scolors = sc.vectocolor(z, cmap='magma', minval=z_min, maxval=z_max)

    # Plot heatmap
    axa = fig.add_subplot(gs[0, 0])
    ima = axa.contourf(xx, yy, zz, cmap='magma', levels=np.linspace(z_min, z_max, 100))
    axa.scatter(x, y, marker='o', c=scolors, edgecolor=[0.3]*3, s=50, linewidth=0.1, alpha=0.5)
    axa.contour(xx, yy, zz, levels=7, linewidths=0.5, colors='k')
    axa.set_xlabel('Sensitivity of AVE')
    axa.set_ylabel('Specificity of AVE')
    axa.set_xlim([np.min(x), np.max(x)])
    axa.set_ylim([np.min(y), np.max(y)])
    axa.set_title('Number CIN treatments to avert a case, relative to VIA', fontsize=28)

    # Colorbar
    axc = fig.add_subplot(gs[0, 2])
    pl.colorbar(ima, ticks=np.linspace(z_min, z_max, 6), cax=axc)

    fig_name = f'{figfolder}/{location}_AVE_NNT_sweeps_{ltfu}ltfu.png'
    sc.savefig(fig_name, dpi=100)