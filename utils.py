'''
Utilities for HPVsim GHlabs collaboration analyses, mostly related to plotting
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


def plot_residual_burden(locations=None, background_scens=None):
    '''
    Plot the residual burden of HPV
    '''
    
    set_font(size=24)

    results_filestem=f'{filestem}_results'
    fig_filestem=filestem

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
            primary_screen = background_scen['primary_screen']
            triage_screen = background_scen['triage_screen']
            if primary_screen == 'via' or triage_screen == 'via':
                df = bigdf[(bigdf.primary_screen == primary_screen) & (bigdf.triage_screen == triage_screen)].groupby('year')[[f'{res}', f'{res}_low', f'{res}_high']].sum()
            else:
                sens = background_scen['sens']
                spec = background_scen['spec']
                df = bigdf[(bigdf.primary_screen == primary_screen) & (bigdf.triage_screen == triage_screen) & (bigdf.sens == sens)
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

    return


def plot_ICER(locations=None, background_scens=None, filestem='screening'):
    '''
    Plot the residual burden of HPV
    '''

    set_font(size=24)
    results_filestem=f'{filestem}_results'
    fig_filestem=filestem

    try:
        bigdf = sc.loadobj(f'{resfolder}/{results_filestem}.obj')
    except:
        print('bigdf not available, trying to load for each location and generate it')
        alldfs = sc.autolist()
        for location in locations:
            alldf = sc.loadobj(f'{resfolder}/{location}_{results_filestem}.obj')
            alldfs += alldf
        bigdf = pd.concat(alldfs)

    cancers = dict()
    cancer_deaths = dict()
    cin_treatments = dict()

    for cn, (background_scen_label, background_scen) in enumerate(background_scens.items()):
        primary_screen = background_scen['primary_screen']
        triage_screen = background_scen['triage_screen']
        if primary_screen == 'via' or triage_screen == 'via':
            df = bigdf[(bigdf.primary_screen == primary_screen) & (bigdf.triage_screen == triage_screen)].groupby('year')[
                ['total_cancers', 'total_cancer_deaths', 'n_cin_treated']].sum()
        else:
            sens = background_scen['sens']
            spec = background_scen['spec']
            df = bigdf[(bigdf.primary_screen == primary_screen) & (bigdf.triage_screen == triage_screen) & (bigdf.sens == sens)
                & (bigdf.spec == spec)].groupby('year')[['total_cancers', 'total_cancer_deaths', 'n_cin_treated']].sum()

        cancers[background_scen_label] = np.array(df['total_cancers'])[50:106].sum()
        cancer_deaths[background_scen_label] = np.array(df['total_cancer_deaths'])[50:106].sum()
        cin_treatments[background_scen_label] = np.array(df['n_cin_treated'])[50:106].sum()

    data_for_plot = pd.DataFrame()
    data_for_plot['scen'] = np.array(list(cancers.keys()))
    data_for_plot['cases'] = np.array(list(cancers.values()))
    data_for_plot['deaths'] = np.array(list(cancer_deaths.values()))
    data_for_plot['cin_txs'] = np.array(list(cin_treatments.values()))

    colors = sc.gridcolors(len(data_for_plot))
    fig, axes = pl.subplots(1, 2, figsize=(16, 8), sharey=True)
    grouped = data_for_plot.groupby('scen')
    for i, (key, group) in enumerate(grouped):
        group.plot(ax=axes[0], kind='scatter', x='cases', y='cin_txs', label=key, color=colors[i], s=100)
        group.plot(ax=axes[1], kind='scatter', x='deaths', y='cin_txs', label=key, color=colors[i], s=100)

    axes[0].set_xlabel('Cancer cases')
    axes[1].set_xlabel('Cancer deaths')
    axes[0].set_ylabel('CIN treatments')
    axes[0].get_legend().remove()
    axes[1].legend(loc='upper center', bbox_to_anchor=(1.65, 0.95), fancybox=True, title='Screening method')
    sc.SIticks(axes[0])
    sc.SIticks(axes[1])
    fig.tight_layout()
    fig_name = f'{figfolder}/{fig_filestem}_ICER.png'
    sc.savefig(fig_name, dpi=100)
    return


def plot_sweeps(fulldf=None, location='india', ltfu=None, scale=1e6): # TODO: set this up to plot cancers averted and/or NNT
    '''
    Plot sweeps across different sensitivity and specificity values
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