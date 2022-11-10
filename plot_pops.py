'''
Create plots of the population and networks for Tanzania, Nigeria, and India.
Supplementary material for therapeutic vaccine analyses.

Before running this, first run run_sim.py.
''' 

#%% General settings
import numpy as np
import pandas as pd
import sciris as sc
import pylab as pl
import matplotlib as mpl
import seaborn as sns
import utils as ut

resfolder = 'results'
figfolder = 'figures'
locations = ['tanzania', 'nigeria', 'india']




#%% Run as a script
if __name__ == '__main__':
    
    T = sc.timer()
    ut.set_font(size=26)


    ################################
    # Plot sexual mixing
    ################################
    fig, axes = pl.subplots(nrows=2, ncols=3, figsize=(24, 12))
    layer_keys = ['Marital', 'Casual']

    for c,location in enumerate(locations):
        try:
            filename = f'{resfolder}/{location}.sim'
            sim = sc.loadobj(filename)
        except FileNotFoundError as E:
            errormsg = f'File {filename} not found: have you run run_analyses.py?'
            raise FileNotFoundError(errormsg) from E
        a = sim.get_analyzer()
        people = a.snapshots[0]

        for r,lkey in enumerate(['m','c']):
            ax = axes[r, c]
            fc = people.contacts[lkey]['age_f']
            mc = people.contacts[lkey]['age_m']
            h = ax.hist2d(fc, mc, bins=np.linspace(0, 75, 16), density=True, norm=mpl.colors.LogNorm())
            ax.set_xlabel('Age of female partner')
            ax.set_ylabel('Age of male partner')
            fig.colorbar(h[3], ax=ax)
            ax.set_title(f'{layer_keys[r]} mixing - {location.capitalize()}')

    fig.tight_layout()
    sc.savefig(f'{figfolder}/networks.png', dpi=100)


    ################################
    # Plot age pyramids
    ################################
    fig, axes = pl.subplots(nrows=2, ncols=3, figsize=(24, 12))
    years = [1990, 2020]
    percentages=True
    m_color = '#4682b4'
    f_color = '#ee7989'
    xlabel = 'Share of population by sex' if percentages else 'Population by sex'

    for c,location in enumerate(locations):
        sim = sc.loadobj(f'{resfolder}/{location}.sim')
        a = sim.get_analyzer(1)
        pyramidsdict = sc.odict(a.age_pyramids)
        labels = list(reversed(a.age_labels))

        for r, date, pyramid in pyramidsdict.enumitems():
            bins = pyramid['bins']
            ax = axes[r, c]

            # Prepare data
            pydf = pd.DataFrame(pyramid)
            if percentages:
                pydf['m'] = pydf['m'] / sum(pydf['m'])
                pydf['f'] = pydf['f'] / sum(pydf['f'])
            pydf['f'] = -pydf['f']  # Reverse values for females to get on same axis

            # Start making plot
            sns.barplot(x='m', y='bins', data=pydf, order=np.flip(bins), orient='h', ax=ax, color=m_color)
            sns.barplot(x='f', y='bins', data=pydf, order=np.flip(bins), orient='h', ax=ax, color=f_color)

            datadf = a.data[a.data.year == float(date)]
            datadf.columns = datadf.columns.str[0]
            datadf.columns = datadf.columns.str.lower()
            if percentages:
                datadf = datadf.assign(m=datadf['m'] / sum(datadf['m']), f=datadf['f'] / sum(datadf['f']))
            datadf = datadf.assign(f=-datadf['f'])
            sns.pointplot(x='m', y='a', data=datadf, order=np.flip(bins), orient='h', ax=ax, color='k', linestyles='')
            sns.pointplot(x='f', y='a', data=datadf, order=np.flip(bins), orient='h', ax=ax, color='k', linestyles='')

            ax.set_xlabel(xlabel)
            ax.set_ylabel('Age group')
            ax.set_yticklabels(labels[1:])
            ax.set_xlim([-0.4,0.4])
            xticks = ax.get_xticks()
            if percentages:
                xlabels = [f'{abs(i):.2f}' for i in xticks]
            else:
                xlabels = [f'{sc.sigfig(abs(i), sigfigs=2, SI=True)}' for i in xticks]
            ax.set_xticks(xticks, xlabels)
            ax.set_title(f'{location.capitalize()} - {years[r]}')

    fig.tight_layout()
    sc.savefig(f'{figfolder}/pyramids.png', dpi=100)
    
    T.toc('Done')
