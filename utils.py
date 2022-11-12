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


def plot_residual_burden(locations=None, background_scens=None, tx_vx_scen='no_txvx', 
                         progression='fast', indication='virologic_clearance', compare_models=False):
    '''
    Plot the residual burden of HPV
    '''
    
    set_font(size=24)

    location_legend = [i.capitalize() for i in locations]
    try:
        bigdf = sc.loadobj(f'{resfolder}/scenario_results.obj')
    except:
        print('bigdf not available, trying to load for each location and generate it')
        alldfs = sc.autolist()
        for location in locations:
            alldf = sc.loadobj(f'{resfolder}/{location}_scenario_results.obj')
            alldfs += alldf
        bigdf = pd.concat(alldfs)
    colors = sc.gridcolors(10)
    if compare_models:
        try:
            harvard_burden = pd.read_csv('results/burden_estimates_Harvard-Sweet_202110gavi-3_hpv-routine-default.csv')
            LSHTM_burden = pd.read_csv('results/burden_estimates_LSHTM-Jit_202110gavi-3_hpv-routine-default.csv')
            harvard_burden = harvard_burden[harvard_burden['country_name'].isin(location_legend)]
            LSHTM_burden = LSHTM_burden[LSHTM_burden['country_name'].isin(location_legend)]
            harvard_burden = harvard_burden.groupby(['year']).sum()[['cases', 'deaths']]
            LSHTM_burden = LSHTM_burden.groupby(['year']).sum()[['cases', 'deaths']]
        except FileNotFoundError as E:
            errormsg = 'File(s) not found: data files are large and need to be downloaded separately'
            raise FileNotFoundError(errormsg) from E

    for res, reslabel in {'total_cancers': 'Annual cases of cervical cancer', 'total_cancer_deaths': 'Annual deaths from cervical cancer'}.items():
        fig, ax = pl.subplots(figsize=(16, 8))

        for cn, (background_scen_label, background_scen) in enumerate(background_scens.items()):
            vx_scen = background_scen['vx_scen']
            screen_scen = background_scen['screen_scen']
            df = bigdf[(bigdf.vx_scen == vx_scen) & (bigdf.screen_scen == screen_scen) & (bigdf.progression == progression) & (
                        bigdf.tx_vx_scen == tx_vx_scen) & (bigdf.indication == indication)].groupby('year')[
                [f'{res}', f'{res}_low', f'{res}_high']].sum()
            years = np.array(df.index)[50:106]
            best = np.array(df[res])[50:106]
            low = np.array(df[f'{res}_low'])[50:106]
            high = np.array(df[f'{res}_high'])[50:106]

            ax.plot(years, best, color=colors[cn], label=background_scen_label)
            ax.fill_between(years, low, high, color=colors[cn], alpha=0.3)

        if compare_models:
            ax.plot(years, harvard_burden['cases'][0:56], color=colors[cn+1], label='Harvard model')
            ax.plot(years, LSHTM_burden['cases'][0:56], color=colors[cn+2], label='LSHTM model')

        ax.legend(loc='upper left')
        sc.SIticks(ax)
        ax.set_ylabel(f'{reslabel}')
        fig.tight_layout()
        if compare_models:
            fig_name = f'{figfolder}/residual_{res}_model_comparison.png'
        else:
            fig_name = f'{figfolder}/residual_{res}.png'
        sc.savefig(fig_name, dpi=100)

    return


def plot_tx_vx_scens(locations=None, background_scens=None, tx_vx_scens=None, progression=None, indication=None):
    '''
    Plot therapeutic vaccine scenarios
    '''
    
    set_font(size=20)

    try:
        bigdf = sc.loadobj(f'{resfolder}/scenario_results.obj')
    except:
        print('bigdf not available, trying to load for each location and generate it')
        alldfs = sc.autolist()
        for location in locations:
            alldf = sc.loadobj(f'{resfolder}/{location}_scenario_results.obj')
            alldfs += alldf
        bigdf = pd.concat(alldfs)

    if progression is None: progression = 'fast'
    if indication is None: indication = 'virologic_clearance'

    tx_vx_scen_labels = ['Routine vaccination', 'Test and vaccinate']
    
    # Start creating plot
    table_data_cases  = pd.DataFrame(columns=tx_vx_scens, index=list(background_scens.keys())) # TODO: see if these can be pulled out to avoid duplication
    table_data_deaths = pd.DataFrame(columns=tx_vx_scens, index=list(background_scens.keys()))
    table_nnv_cases   = pd.DataFrame(columns=tx_vx_scens, index=list(background_scens.keys()))
    table_nnv_deaths  = pd.DataFrame(columns=tx_vx_scens, index=list(background_scens.keys()))

    for cn, (background_scen_label, background_scen) in enumerate(background_scens.items()):
        vx_scen = background_scen['vx_scen']
        screen_scen = background_scen['screen_scen']

        base_df = bigdf[(bigdf.vx_scen == vx_scen) & (bigdf.screen_scen == screen_scen) & (
                bigdf.progression == progression) & (bigdf.tx_vx_scen == 'no_txvx') &
                        (bigdf.indication == indication)].groupby('year')[
            ['total_cancers', 'total_cancer_deaths', 'n_tx_vaccinated']].sum()
        base_cancers = np.array(base_df['total_cancers'])[50:106].sum()
        base_cancer_deaths = np.array(base_df['total_cancer_deaths'])[50:106].sum()
        for tx_vx_scen in tx_vx_scens:
            df = bigdf[(bigdf.vx_scen == vx_scen) & (bigdf.screen_scen == screen_scen) & (
                    bigdf.progression == progression) & (bigdf.tx_vx_scen == tx_vx_scen) &
                       (bigdf.indication == indication)].groupby('year')[
                ['total_cancers', 'total_cancer_deaths', 'n_tx_vaccinated']].sum()
            cancers_scen = np.array(df['total_cancers'])[50:106].sum()
            cancer_deaths_scen = np.array(df['total_cancer_deaths'])[50:106].sum()
            vaccinated = np.array(df['n_tx_vaccinated'])[50:106].sum()

            cancers_averted = base_cancers - cancers_scen
            cancer_deaths_averted = base_cancer_deaths - cancer_deaths_scen
            NNV_cases = vaccinated / cancers_averted #/ 1000
            NNV_deaths = vaccinated / cancer_deaths_averted  # / 1000
            table_data_cases.loc[background_scen_label, tx_vx_scen] = cancers_averted
            table_data_deaths.loc[background_scen_label, tx_vx_scen] = cancer_deaths_averted
            table_nnv_cases.loc[background_scen_label, tx_vx_scen] = NNV_cases
            table_nnv_deaths.loc[background_scen_label, tx_vx_scen] = NNV_deaths

    table_nnv_deaths[table_nnv_deaths < 0] = np.nan
    table_nnv_cases[table_nnv_cases < 0] = np.nan
    fig, ax = pl.subplots(1,2, figsize=(16,8))
    colors=sc.gridcolors(10)
    table_nnv_cases.transpose().plot(kind='bar', ax=ax[0], color=colors)
    sc.SIticks(ax[0])
    ax[0].set_title('NNV to avert a cervical cancer case')
    ax[0].set_xticklabels(tx_vx_scen_labels, rotation=0)
    table_nnv_deaths.transpose().plot(kind='bar', ax=ax[1], color=colors)
    sc.SIticks(ax[1])
    ax[1].set_title('NNV to avert a cervical cancer death')
    ax[1].set_xticklabels(tx_vx_scen_labels, rotation = 0)
    ax[0].get_legend().remove()
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Background 3-pillars', fontsize=16)

    fig.tight_layout()
    fig_name = f'{figfolder}/txvx_nnv_{progression}_prog_{indication}.png'
    sc.savefig(fig_name, dpi=100)

    fig, ax = pl.subplots(1, 2, figsize=(16, 8))
    colors = sc.gridcolors(10)
    table_data_cases.transpose().plot(kind='bar', ax=ax[0], color=colors)
    sc.SIticks(ax[0])
    ax[0].set_title('Cervical cancer cases averted')
    ax[0].set_xticklabels(tx_vx_scen_labels, rotation=0)
    table_data_deaths.transpose().plot(kind='bar', ax=ax[1], color=colors)
    sc.SIticks(ax[1])
    ax[1].set_title('Cervical cancer deaths averted')
    ax[1].set_xticklabels(tx_vx_scen_labels, rotation=0)
    ax[0].get_legend().remove()
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Background 3-pillars', fontsize=16)

    fig.tight_layout()
    fig_name = f'{figfolder}/txvx_impact_{progression}_prog_{indication}.png'
    sc.savefig(fig_name, dpi=100)

    return


def plot_tx_vx_scens_sens(locations=None, tx_vx_scens=None, progression=None, sens_backgrounds=None, intro_year=2030):
    '''
    Plot sensitivity for therapeutic vaccine scenarios
    '''
    
    set_font(size=20)

    alldfs = sc.autolist()
    for location in locations:
        alldf = sc.loadobj(f'{resfolder}/{location}_sensitivity_results.obj')
        alldfs += alldf
    bigdf = pd.concat(alldfs)

    if progression is None: progression = 'fast'

    # Start creating plot
    table_data_cases = pd.DataFrame(columns=tx_vx_scens, index=list(sens_backgrounds.keys()))
    table_data_deaths = pd.DataFrame(columns=tx_vx_scens, index=list(sens_backgrounds.keys()))
    table_nnv_cases = pd.DataFrame(columns=tx_vx_scens, index=list(sens_backgrounds.keys()))
    table_nnv_deaths = pd.DataFrame(columns=tx_vx_scens, index=list(sens_backgrounds.keys()))

    for cn, (background_scen_label, background_scen) in enumerate(sens_backgrounds.items()):
        vx_scen = background_scen['vx_scen']
        screen_scen = background_scen['screen_scen']

        base_df = bigdf[(bigdf.vx_scen == vx_scen) & (bigdf.screen_scen == screen_scen) & (
                bigdf.progression == progression) & (bigdf.tx_vx_scen == 'no_txvx') & (bigdf.intro_year == intro_year)].groupby('year')[
            ['total_cancers', 'total_cancer_deaths', 'n_tx_vaccinated']].sum()
        base_cancers = np.array(base_df['total_cancers'])[50:106].sum()
        base_cancer_deaths = np.array(base_df['total_cancer_deaths'])[50:106].sum()
        for tx_vx_scen in tx_vx_scens:
            df = bigdf[(bigdf.vx_scen == vx_scen) & (bigdf.screen_scen == screen_scen) & (
                    bigdf.progression == progression) & (bigdf.tx_vx_scen == tx_vx_scen) & (bigdf.intro_year == intro_year)].groupby('year')[
                ['total_cancers', 'total_cancer_deaths', 'n_tx_vaccinated']].sum()
            cancers_scen = np.array(df['total_cancers'])[50:106].sum()
            cancer_deaths_scen = np.array(df['total_cancer_deaths'])[50:106].sum()
            vaccinated = np.array(df['n_tx_vaccinated'])[50:106].sum()

            cancers_averted = base_cancers - cancers_scen
            cancer_deaths_averted = base_cancer_deaths - cancer_deaths_scen
            NNV_cases = vaccinated / cancers_averted #/ 1000
            NNV_deaths = vaccinated / cancer_deaths_averted  # / 1000
            table_data_cases.loc[background_scen_label, tx_vx_scen] = cancers_averted
            table_data_deaths.loc[background_scen_label, tx_vx_scen] = cancer_deaths_averted
            table_nnv_cases.loc[background_scen_label, tx_vx_scen] = NNV_cases
            table_nnv_deaths.loc[background_scen_label, tx_vx_scen] = NNV_deaths

    table_nnv_deaths[table_nnv_deaths < 0] = np.nan
    table_nnv_cases[table_nnv_cases < 0] = np.nan
    fig, ax = pl.subplots(1, 2, figsize=(16, 8))
    colors = sc.gridcolors(10)[4:]
    table_nnv_cases.plot(kind='bar', ax=ax[0], color=colors)
    sc.SIticks(ax[0])
    ax[0].set_title('NNV to avert a cervical cancer case')
    ax[0].set_xticklabels(sens_backgrounds.keys(), rotation=0)
    ax[0].set_xlabel('Screen coverage by 2040 (%)')
    table_nnv_deaths.plot(kind='bar', ax=ax[1], color=colors)
    sc.SIticks(ax[1])
    ax[1].set_title('NNV to avert a cervical cancer death')
    ax[1].set_xticklabels(sens_backgrounds.keys(), rotation=0)
    ax[1].set_xlabel('Screen coverage by 2040 (%)')
    ax[0].get_legend().remove()
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), title='TxVx Delivery', fontsize=16)

    fig.tight_layout()
    fig_name = f'{figfolder}/txvx_nnv_{progression}_prog_sensitivity.png'
    sc.savefig(fig_name, dpi=100)

    fig, ax = pl.subplots(1, 2, figsize=(16, 8))
    table_data_cases.plot(kind='bar', ax=ax[0], color=colors)
    sc.SIticks(ax[0])
    ax[0].set_title('Cervical cancer cases averted')
    ax[0].set_xticklabels(sens_backgrounds.keys(), rotation=0)
    ax[0].set_xlabel('Screen coverage by 2040 (%)')
    table_data_deaths.plot(kind='bar', ax=ax[1], color=colors)
    sc.SIticks(ax[1])
    ax[1].set_title('Cervical cancer deaths averted')
    ax[1].set_xticklabels(sens_backgrounds.keys(), rotation=0)
    ax[1].set_xlabel('Screen coverage by 2040 (%)')
    ax[0].get_legend().remove()
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), title='TxVx Delivery', fontsize=16)

    fig.tight_layout()
    fig_name = f'{figfolder}/txvx_impact_{progression}_prog_sensitivity.png'
    sc.savefig(fig_name, dpi=100)
    return


def plot_tx_vx_sens_cov_intro(locations=None, tx_vx_scen=None, progression=None, 
                              sens_backgrounds=None, intro_years=None):
    '''
    Plot sensitivity in coverage for therapeutic vaccine scenarios
    '''

    set_font(size=20)

    alldfs = sc.autolist()
    for location in locations:
        alldf = sc.loadobj(f'{resfolder}/{location}_sensitivity_results.obj')
        alldfs += alldf
    bigdf = pd.concat(alldfs)

    if progression is None: progression = 'fast'

    table_data_cases = pd.DataFrame(columns=intro_years, index=list(sens_backgrounds.keys()))
    table_data_deaths = pd.DataFrame(columns=intro_years, index=list(sens_backgrounds.keys()))
    table_nnv_cases = pd.DataFrame(columns=intro_years, index=list(sens_backgrounds.keys()))
    table_nnv_deaths = pd.DataFrame(columns=intro_years, index=list(sens_backgrounds.keys()))

    for background_scen_label, background_scen in sens_backgrounds.items():
        vx_scen = background_scen['vx_scen']
        screen_scen = background_scen['screen_scen']

        base_df = bigdf[(bigdf.vx_scen == vx_scen) & (bigdf.screen_scen == screen_scen) & (
                bigdf.progression == progression) & (bigdf.tx_vx_scen == 'no_txvx') & (
                                    bigdf.intro_year == 2030)].groupby('year')[
            ['total_cancers', 'total_cancer_deaths', 'n_tx_vaccinated']].sum()
        base_cancers = np.array(base_df['total_cancers'])[50:106].sum()
        base_cancer_deaths = np.array(base_df['total_cancer_deaths'])[50:106].sum()
        for intro_year in intro_years:
            df = bigdf[(bigdf.vx_scen == vx_scen) & (bigdf.screen_scen == screen_scen) & (
                    bigdf.progression == progression) & (bigdf.tx_vx_scen == tx_vx_scen)
                       & (bigdf.intro_year == intro_year)].groupby('year')[
                ['total_cancers', 'total_cancer_deaths', 'n_tx_vaccinated']].sum()
            cancers_scen = np.array(df['total_cancers'])[50:106].sum()
            cancer_deaths_scen = np.array(df['total_cancer_deaths'])[50:106].sum()
            vaccinated = np.array(df['n_tx_vaccinated'])[50:106].sum()
            cancers_averted = base_cancers - cancers_scen
            cancer_deaths_averted = base_cancer_deaths - cancer_deaths_scen
            NNV_cases = vaccinated / cancers_averted
            NNV_deaths = vaccinated / cancer_deaths_averted
            table_data_cases.loc[background_scen_label, intro_year] = cancers_averted
            table_data_deaths.loc[background_scen_label, intro_year] = cancer_deaths_averted
            table_nnv_cases.loc[background_scen_label, intro_year] = NNV_cases
            table_nnv_deaths.loc[background_scen_label, intro_year] = NNV_deaths

    table_nnv_deaths[table_nnv_deaths < 0] = np.nan
    table_nnv_cases[table_nnv_cases < 0] = np.nan
    fig, ax = pl.subplots(2, 2, figsize=(16, 12))
    colors = sc.gridcolors(10)[6:]
    table_data_cases.plot(kind='bar', ax=ax[0,0], color=colors)
    sc.SIticks(ax[0,0])
    ax[0,0].set_title('Cervical cancer cases averted')
    ax[0,0].set_xticklabels(sens_backgrounds.keys(), rotation=0)
    ax[0,0].set_xlabel('Screen coverage by 2040 (%)')
    table_data_deaths.plot(kind='bar', ax=ax[0,1], color=colors)
    sc.SIticks(ax[0,1])
    ax[0,1].set_title('Cervical cancer deaths averted')
    ax[0,1].set_xticklabels(sens_backgrounds.keys(), rotation=0)
    ax[0,1].set_xlabel('Screen coverage by 2040 (%)')
    ax[0,0].get_legend().remove()
    ax[0,1].legend(loc='center left', bbox_to_anchor=(1, 0.5), title='TxVx Intro Year', fontsize=16)

    table_nnv_cases.plot(kind='bar', ax=ax[1, 0], color=colors)
    sc.SIticks(ax[1, 0])
    ax[1, 0].set_title('NNV to avert a cervical cancer case')
    ax[1, 0].set_xticklabels(sens_backgrounds.keys(), rotation=0)
    ax[1, 0].set_xlabel('Screen coverage by 2040 (%)')
    table_nnv_deaths.plot(kind='bar', ax=ax[1, 1], color=colors)
    sc.SIticks(ax[1, 1])
    ax[1, 1].set_title('NNV to avert a cervical cancer death')
    ax[1, 1].set_xticklabels(sens_backgrounds.keys(), rotation=0)
    ax[1, 1].set_xlabel('Screen coverage by 2040 (%)')
    ax[1, 0].get_legend().remove()
    ax[1, 1].get_legend().remove()

    fig.suptitle(f'Health impact and efficiency of {tx_vx_scen} TxVx\nby screen coverage and introduction year')
    fig.tight_layout()
    fig_name = f'{figfolder}/txvx_impact_2waysens_{tx_vx_scen}_{progression}_prog.png'
    sc.savefig(fig_name, dpi=100)
    return


def plot_scens(locations=None, background_scens=None, tx_vx_scens=None, progression=None, indication=None, debug=False):
    '''
    Plot overall scenarios
    '''
    
    set_font(size=24)

    background_scen_tags = {
        'No scale-up': 'baseline',
        '90% vaccine scale-up': '90vx_0sc_10tx',
        '90% vaccine, 70% screening, 90% treatment': '90vx_70sc_90tx'
    }

    try:
        bigdf = sc.loadobj(f'{resfolder}/scenario_results.obj')
    except:
        print('bigdf not available, trying to load for each location and generate it')
        alldfs = sc.autolist()
        for location in locations:
            alldf = sc.loadobj(f'{resfolder}/{location}_scenario_results.obj')
            alldfs += alldf
        bigdf = pd.concat(alldfs)

    if progression is None: progression = 'fast'
    if indication is None: indication = 'virologic_clearance'

    whattoplot = {'total_cancers':'Annual cases of cervical cancer', 'asr_cancer':'Age-standardized incidence of cervical cancer', 'n_tx_vaccinated': 'Total tx vaccinated'}
    tx_vx_scen_labels = {
        'no_txvx': 'None',
        'mass_vaccination': 'Routine/campaign delivery',
        'test_and_vaccinate': 'Test and vaccinate'
    }
    # Start creating plot
    timepoints = np.arange(50, 106) if not debug else np.arange(20,76) # Define a subset of timepoints to plot # TODO: remove hardcoding
    for res,reslabel in whattoplot.items():
        colors = sc.gridcolors(10)
        for cn, (background_scen_label, background_scen) in enumerate(background_scens.items()):
            fig, ax = pl.subplots(figsize=(16, 8))
            ci = 0
            sc.SIticks(ax)
            ax.set_title(f'{background_scen_label}')
            ax.set_ylabel(f'{reslabel}')
            vx_scen = background_scen['vx_scen']
            screen_scen = background_scen['screen_scen']

            for tx_vx_scen in tx_vx_scens:
                df = bigdf[
                    (bigdf.vx_scen == vx_scen) & (bigdf.screen_scen == screen_scen) & (bigdf.progression == progression) & (
                                bigdf.tx_vx_scen == tx_vx_scen) & (bigdf.indication == indication)].groupby('year')[
                    [f'{res}', f'{res}_low', f'{res}_high']].sum()
                years = np.array(df.index)[timepoints]
                best  = np.array(df[res])[timepoints]
                low   = np.array(df[f'{res}_low'])[timepoints]
                high  = np.array(df[f'{res}_high'])[timepoints]

                ax.plot(years, best, color=colors[ci+3], label=tx_vx_scen_labels[tx_vx_scen])
                ax.fill_between(years, low, high, color=colors[ci+3], alpha=0.3)
                ci += 1

            ax.legend(loc='best', title='Therapeutic vaccine')
            fig.tight_layout()
            fig_name = f'{figfolder}/residual_{res}_{background_scen_tags[background_scen_label]}_{progression}_prog_{indication}.png'
            sc.savefig(fig_name, dpi=100)

    return

def plot_sweeps(fulldf=None, location='india', progression='fast', scale=1e6): # TODO: set this up to plot cancers averted and/or NNT
    '''
    Plot parameter sweeps
    '''

    # Initialize figure
    fig = pl.figure(figsize=(12, 10))
    gs = fig.add_gridspec(1, 3, width_ratios=[20, .1, 1])
    pl.subplots_adjust(hspace=0.25, wspace=0.1, left=0.1, right=0.9, top=0.95, bottom=0.1)

    df = fulldf[(fulldf.progression==progression)].groupby(['sens','spec']).sum().reset_index()
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
    axa.set_title('Cancers averted 2023-60 (million)', fontsize=28)

    # Colorbar
    axc = fig.add_subplot(gs[0, 2])
    pl.colorbar(ima, ticks=np.linspace(z_min, z_max, 6), cax=axc)

    fig_name = f'{figfolder}/{progression}_{location}_AVE_impact_sweeps.png'
    sc.savefig(fig_name, dpi=100)

    # Initialize figure
    fig = pl.figure(figsize=(12, 10))
    gs = fig.add_gridspec(1, 3, width_ratios=[20, .1, 1])
    pl.subplots_adjust(hspace=0.25, wspace=0.1, left=0.1, right=0.9, top=0.95, bottom=0.1)

    z = np.array(df['new_cin_treatments'])/np.array(df['cancers_averted'])
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
    axa.set_title('Number needed to treat to avert a cancer', fontsize=28)

    # Colorbar
    axc = fig.add_subplot(gs[0, 2])
    pl.colorbar(ima, ticks=np.linspace(z_min, z_max, 6), cax=axc)

    fig_name = f'{figfolder}/{progression}_{location}_AVE_NNT_sweeps.png'
    sc.savefig(fig_name, dpi=100)