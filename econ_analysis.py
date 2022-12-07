'''
Run economic analysis.

'''


#%% General settings

# Standard imports
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
import sciris as sc
import pylab as pl
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

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

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

locations = [
    # 'india',
    'nigeria',
    # 'tanzania'
]

location_dict = {
    'india': 'India',
    'nigeria': 'Nigeria',
    'tanzania': 'United Republic of Tanzania'
}
ex = sc.loadobj('data/ex.obj')
life_expectancies = dict()
for location in locations:
    life_expectancies[location] = ex[location_dict[location]][(ex[location_dict[location]]['Time']==2030) &
                                                            (ex[location_dict[location]]['Sex'] == 'Female')]
dfs = sc.autolist()
sensdfs = sc.autolist()
for location in locations:
    dfs += pd.read_csv(f'results/{location}_econ.csv')
    sensdfs += pd.read_csv(f'results/{location}_sens.csv')
model_res = pd.concat(dfs)
sens = pd.concat(sensdfs)

n_seeds = len(np.unique(model_res['seed']))
cost_params = pd.DataFrame()
cost_params['location'] = np.array(['india', 'nigeria', 'tanzania'])
cost_params['HPV'] = np.array([14.8*258.811/240.007, 36, 9.1])
cost_params['HPV_sd'] = (2*1.96)*3.8/14.8
cost_params['VIA'] = np.array([5.2*258.811/240.007, 13, 2.89])
cost_params['VIA_sd'] = (2*1.96)*1.3/5.2

cost_params['POC_HPV'] = np.array([2*258.811/240.007, 2, 2])
cost_params['POC_HPV_sd'] = (2*1.96)*2/2
cost_params['AVE'] = np.array([5.2*258.811/240.007, 13, 2.89])
cost_params['AVE_sd'] = (2*1.96)*4/5.2
cost_params['TA'] = np.array([60*258.811/240.007, 3.5, 3.57])
cost_params['TA_sd'] = (2*1.96)*4.2/16
cost_params['LEEP'] = np.array([90.3*258.811/240.007, 107, 69])
cost_params['LEEP_sd'] = (2*1.96)*23/90
cost_params['cancer'] = np.array([450*258.811/240.007, np.mean(np.array([44.73, 64.13, 281.5, 768, 212])), np.mean(np.array([94, 574, 974, 21]))])
cost_params['cancer_sd'] = (2*1.96)*(33+75+159+104+12+90.3+8.6+5+4.8+241)/450

# Nigeria costs (ref 1)
# HPV DNA testing: financial cost of US$ 36 per service.
# VIA: financial cost of US$ 13 per service.
# Thermal ablation: financial cost of US$ 3.50 per service
# LEEP: financial cost of US$ 107 per service.
# CaTx (take mean of this): $44.73, $64.13, $281.50, $768.88, $212.06

# India costs: mean (SD) (ref 3)
# HPV DNA testing: US$ 14.8 ($3.8 SD).
# VIA: $5.2 ($1.3).
# Cryotherapy: $60.4 ($15)
# LEEP: $90.3 ($23).
# CaTx: $291 - 617

# Tanzania costs (ref 2)
# HPV DNA testing: financial cost of US$ 9.10 per service.
# VIA: financial cost of US$ 2.89 per service.
# Thermal ablation: financial cost of US$ 3.57 per service
# LEEP: financial cost of US$ 69.24 per service.
# CaTx (take mean of this): $94.76, $574.52, $974, $21


# References
# 1. (2020 USD) World Health Organization. (2020). Costing the National Strategic Plan on Prevention and Control of Cervical Cancer: Nigeria, 2017 –2021November 2020.
# 2. (2020 USD) World Health Organization. (2020). Costing the National Strategic Plan on Prevention and Control of Cervical Cancer: Tanzania, 2020 –2024November 2020.
# 3. (2016 USD) Chauhan, A. S., Prinja, S., Srinivasan, R., Rai, B., Malliga, J. S., Jyani, G., Gupta, N., &#38; Ghoshal, S. (2020). Cost effectiveness of strategies for cervical cancer prevention in India. <i>PLoS ONE</i>, <i>15</i>(9 September). https://doi.org/10.1371/journal.pone.0238291</div>
# 4 (2020 USD) Singh et a., Cost of Treatment for Cervical Cancer in India, Asian Pacific Journal of Cancer Prevention https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7779435/

lower_clip = 0.
upper_clip = np.inf

dfs = sc.autolist()
for location in locations:
    simulated_costs = pd.DataFrame()
    costs = cost_params[cost_params['location'] == location]
    simulated_costs['HPV'] = truncnorm.rvs((lower_clip - costs['HPV']) / costs['HPV_sd'],
                                           (upper_clip - costs['HPV']) / costs['HPV_sd'],
                                           loc=costs['HPV'], scale=costs['HPV_sd'], size=n_seeds)
    simulated_costs['POC_HPV'] = truncnorm.rvs((lower_clip - costs['POC_HPV']) / costs['POC_HPV_sd'],
                                           (upper_clip - costs['POC_HPV']) / costs['POC_HPV_sd'],
                                           loc=costs['POC_HPV'], scale=costs['POC_HPV_sd'], size=n_seeds)
    simulated_costs['VIA'] = truncnorm.rvs((lower_clip - costs['VIA']) / costs['VIA_sd'],
                                           (upper_clip - costs['VIA']) / costs['VIA_sd'],
                                           loc=costs['VIA'], scale=costs['VIA_sd'], size=n_seeds)
    simulated_costs['AVE'] = truncnorm.rvs((lower_clip - costs['AVE']) / costs['AVE_sd'],
                                           (upper_clip - costs['AVE']) / costs['AVE_sd'],
                                           loc=costs['AVE'], scale=costs['AVE_sd'], size=n_seeds)
    simulated_costs['TA'] = truncnorm.rvs((lower_clip - costs['TA']) / costs['TA_sd'],
                                           (upper_clip - costs['TA']) / costs['TA_sd'],
                                           loc=costs['TA'], scale=costs['TA_sd'], size=n_seeds)
    simulated_costs['LEEP'] = truncnorm.rvs((lower_clip - costs['LEEP']) / costs['LEEP_sd'],
                                           (upper_clip - costs['LEEP']) / costs['LEEP_sd'],
                                           loc=costs['LEEP'], scale=costs['LEEP_sd'], size=n_seeds)
    simulated_costs['cancer'] = truncnorm.rvs((lower_clip - costs['cancer']) / costs['cancer_sd'],
                                              (upper_clip - costs['cancer']) / costs['cancer_sd'],
                                              loc=costs['cancer'], scale=costs['cancer_sd'], size=n_seeds)
    simulated_costs['location'] = location
    dfs += simulated_costs

simulated_cost_df = pd.concat(dfs)

scenarios = pd.unique(model_res['scen_label'])

dfs = sc.autolist()
for location in locations:
    costs = simulated_cost_df[simulated_cost_df['location'] == location]
    life_expectancy = life_expectancies[location].reset_index()
    for scenario in scenarios:
        df = pd.DataFrame()
        model_output = model_res[(model_res['location'] == location) & (model_res['scen_label'] == scenario)]
        ylls = []
        ylds = []
        dalys = []
        total_costs = []
        total_cin_treatments = []
        for name, group in model_output.groupby('seed'):
            discounted_cancers = np.array([i/1.03**t for t,i in enumerate(group['new_cancers'].values)])
            discounted_cancer_deaths = np.array([i/1.03**t for t,i in enumerate(group['new_cancer_deaths'].values)])
            avg_age_ca_death = np.mean(group['av_age_cancer_deaths'])
            avg_age_ca = np.mean(group['av_age_cancers'])
            ca_years = avg_age_ca_death - avg_age_ca
            yld = np.sum(np.sum([0.54*.1, 0.049*.5, 0.451*.3, 0.288*.1]) * ca_years * discounted_cancers)
            ylds += [yld]
            ind = sc.findnearest(life_expectancy['AgeGrpStart'], avg_age_ca_death)
            yll = np.sum(life_expectancy['ex'][ind] * discounted_cancer_deaths)
            ylls += [yll]
            daly = yll + yld
            dalys += [daly]
            total_cost = (group['new_hpv_screens'].values * costs['HPV'].values[name]) + \
                            (group['new_via_screens'].values * costs['VIA'].values[name]) + \
                            (group['new_poc_hpv_screens'].values * costs['POC_HPV'].values[name]) + \
                            (group['new_ave_screens'].values * costs['AVE'].values[name]) + \
                            (group['new_thermal_ablations'].values * costs['TA'].values[name]) + \
                            (group['new_leeps'].values * costs['LEEP'].values[name]) + \
                            (group['new_cancer_treatments'].values * costs['cancer'].values[name])
            discounted_cost = np.sum([i/1.03**t for t,i in enumerate(total_cost)])
            total_costs += [discounted_cost]
            total_cin_treatments += [np.sum(group['new_thermal_ablations'].values + group['new_leeps'].values)]

        df['ylls'] = ylls
        df['ylds'] = ylds
        df['DALYs'] = dalys
        df['total_cin_treatments'] = total_cin_treatments
        if scenario == 'No screening':
            base_DALYs = dalys
        df['DALYs_averted'] = base_DALYs - df['DALYs']
        df['total_costs'] = total_costs
        df['location'] = location
        df['scen_label'] = scenario
        dfs += df

alldfs = pd.concat(dfs)

## Now generate a range of costs for AVEs
AVE_scenarios = [x for x in scenarios if 'AVE' in x]
AVE_cost_range = np.linspace(2,15, 10)
dfs = sc.autolist()
location = 'nigeria'
for cost in AVE_cost_range:
    costs = simulated_cost_df[simulated_cost_df['location'] == location]
    life_expectancy = life_expectancies[location].reset_index()
    for scenario in AVE_scenarios:
        df = pd.DataFrame()
        model_output = model_res[(model_res['location'] == location) & (model_res['scen_label'] == scenario)]
        ylls = []
        ylds = []
        dalys = []
        total_costs = []
        total_cin_treatments = []
        for name, group in model_output.groupby('seed'):
            discounted_cancers = np.array([i/1.03**t for t,i in enumerate(group['new_cancers'].values)])
            discounted_cancer_deaths = np.array([i/1.03**t for t,i in enumerate(group['new_cancer_deaths'].values)])
            avg_age_ca_death = np.mean(group['av_age_cancer_deaths'])
            avg_age_ca = np.mean(group['av_age_cancers'])
            ca_years = avg_age_ca_death - avg_age_ca
            yld = np.sum(np.sum([0.54*.1, 0.049*.5, 0.451*.3, 0.288*.1]) * ca_years * discounted_cancers)
            ylds += [yld]
            ind = sc.findnearest(life_expectancy['AgeGrpStart'], avg_age_ca_death)
            yll = np.sum(life_expectancy['ex'][ind] * discounted_cancer_deaths)
            ylls += [yll]
            daly = yll + yld
            dalys += [daly]
            total_cost = (group['new_hpv_screens'].values * costs['HPV'].values[name]) + \
                            (group['new_via_screens'].values * costs['VIA'].values[name]) + \
                            (group['new_poc_hpv_screens'].values * costs['POC_HPV'].values[name]) + \
                            (group['new_ave_screens'].values * cost) + \
                            (group['new_thermal_ablations'].values * costs['TA'].values[name]) + \
                            (group['new_leeps'].values * costs['LEEP'].values[name]) + \
                            (group['new_cancer_treatments'].values * costs['cancer'].values[name])
            discounted_cost = np.sum([i/1.03**t for t,i in enumerate(total_cost)])
            total_costs += [discounted_cost]
            total_cin_treatments += [np.sum(group['new_thermal_ablations'].values + group['new_leeps'].values)]

        df['ylls'] = ylls
        df['ylds'] = ylds
        df['DALYs'] = dalys
        df['total_cin_treatments'] = total_cin_treatments
        if scenario == 'No screening':
            base_DALYs = dalys
        df['DALYs_averted'] = base_DALYs - df['DALYs']
        df['total_costs'] = total_costs
        df['location'] = location
        df['scen_label'] = scenario
        dfs += df

alldfs_ave_cost = pd.concat(dfs)

set_font(size=20)
markers = ['.', 'v', '<', '1', 's', 'p', 'P', '*', '+', 'D', '^', 'x']
colors = sc.gridcolors(len(scenarios))
grouped_means = alldfs.groupby(['scen_label','location']).mean().reset_index()

scen_colors = dict()
for scen in scenarios:
    if scen == 'HPV' or scen == 'VIA':
        scen_colors[scen] = 0
    elif scen == 'HPV+VIA' or scen == 'POC-HPV+VIA':
        scen_colors[scen] = 1
    elif scen[:3] == 'AVE':
        scen_colors[scen] = 2
    elif 'HPV+AVE' in scen:
        scen_colors[scen] = 3
    elif scen == 'No screening':
        scen_colors[scen] = 4

scenarios_to_plot = scenarios[1:]

for location in locations:
    data_to_plot = grouped_means[grouped_means['location'] == location]
    ave_cost_data_to_plot = alldfs_ave_cost[alldfs_ave_cost['location']==location]
    # ymin = np.min(data_to_plot[data_to_plot['scen_label'] != 'No screening']['total_costs'])
    # ymax = np.max(data_to_plot[data_to_plot['scen_label'] != 'No screening']['total_costs'])
    #
    # xmin = np.min(data_to_plot[data_to_plot['scen_label'] != 'No screening']['DALYs_averted'])
    # xmax = np.max(data_to_plot[data_to_plot['scen_label'] != 'No screening']['DALYs_averted'])
    # f, axes = pl.subplots(2, 2, figsize=(16, 10), gridspec_kw={'height_ratios': [10, 1], 'width_ratios': [1,10]})
    # ax1 = axes[0,0]
    # ax2 = axes[0,1]
    # ax3 = axes[1,0]
    # ax4 = axes[1,1]
    #
    # for i, scen in enumerate(scenarios):
    #     group = data_to_plot[data_to_plot['scen_label'] == scen]
    #     if scen != 'No screening':
    #         ellipse_group = alldfs[(alldfs.location == location) & (alldfs.scen_label == scen)]
    #         x, y = ellipse_group['DALYs_averted'].values, ellipse_group['total_costs'].values
    #         confidence_ellipse(x, y, ax=ax1, edgecolor=colors[scen_colors[scen]])
    #         confidence_ellipse(x, y, ax=ax2, edgecolor=colors[scen_colors[scen]])
    #         confidence_ellipse(x, y, ax=ax3, edgecolor=colors[scen_colors[scen]])
    #         confidence_ellipse(x, y, ax=ax4, edgecolor=colors[scen_colors[scen]])
    #     group.plot(ax=ax1, kind='scatter', x='DALYs_averted', y='total_costs', color=colors[scen_colors[scen]], marker=markers[i], s=200)
    #     group.plot(ax=ax2, kind='scatter', x='DALYs_averted', y='total_costs', label=scen, marker=markers[i], color=colors[scen_colors[scen]], s=200)
    #     group.plot(ax=ax3, kind='scatter', x='DALYs_averted', y='total_costs', marker=markers[i], color=colors[scen_colors[scen]], s=200)
    #     group.plot(ax=ax4, kind='scatter', x='DALYs_averted', y='total_costs', marker=markers[i], color=colors[scen_colors[scen]], s=200)
    #
    # ax1.set_ylim(ymin*0.6, ymax*1.15)
    # ax2.set_ylim(ymin*0.6, ymax*1.15)
    # ax3.set_ylim(-0.5,1)
    # ax4.set_ylim(-0.5,1)
    #
    # ax1.set_xlim(-0.5,1)
    # ax2.set_xlim(xmin*0.8,xmax*1.1)
    # ax3.set_xlim(-0.5,1)
    # ax4.set_xlim(xmin*0.8, xmax*1.1)
    #
    # # hide the spines between ax and ax2
    # ax1.spines['bottom'].set_visible(False)
    # ax1.spines['right'].set_visible(False)
    # ax2.spines['bottom'].set_visible(False)
    # ax2.spines['left'].set_visible(False)
    # ax3.spines['top'].set_visible(False)
    # ax3.spines['right'].set_visible(False)
    # ax4.spines['top'].set_visible(False)
    # ax4.spines['left'].set_visible(False)
    # ax3.xaxis.tick_bottom()
    # ax4.xaxis.tick_bottom()
    #
    # d = .015  # how big to make the diagonal lines in axes coordinates
    # # arguments to pass to plot, just so we don't keep repeating them
    # # Start with top axes (ax1, ax2)
    # kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    # ax1.plot((-d/.2, +d/.2), (-d, +d), **kwargs)        # top-left diagonal
    # kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    # ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    #
    # # Now bottom axes (ax3, ax4)
    # d = .1  # how big to make the diagonal lines in axes coordinates
    # kwargs.update(transform=ax3.transAxes)  # switch to the bottom axes
    # ax3.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    # d = .015  # how big to make the diagonal lines in axes coordinates
    # kwargs.update(transform=ax4.transAxes)  # switch to the bottom axes
    # ax4.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    #
    # d = .015 # how big to make the diagonal lines in axes coordinates
    # # arguments to pass plot, just so we don't keep repeating them
    # kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    # ax1.plot((1-d,1+d), (1-d,1+d), **kwargs)
    # kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    # ax2.plot((-d/3,+d/3),(1-d,1+d), **kwargs)
    #
    # d = .1 # how big to make the diagonal lines in axes coordinates
    # kwargs.update(transform=ax3.transAxes)  # switch to the bottom axes
    # ax3.plot((1-d,1+d), (-d,+d), **kwargs)
    #
    # d = .15 # how big to make the diagonal lines in axes coordinates
    # kwargs.update(transform=ax4.transAxes)  # switch to the bottom axes
    # ax4.plot((-d/10,+d/10), (-d,+d), **kwargs)
    #
    # ax2.set_ylabel('')
    # ax3.set_ylabel('')
    # ax4.set_ylabel('')
    # ax2.get_yaxis().set_visible(False)
    #
    # ax4.get_yaxis().set_visible(False)
    #
    # ax1.set_xlabel('')
    # ax2.set_xlabel('')
    # ax3.set_xlabel('')
    # ax1.get_xaxis().set_visible(False)
    # ax2.get_xaxis().set_visible(False)
    #
    # ax4.set_xlabel('DALYs averted, 2020-2060')
    # ax1.set_ylabel('Total costs, $USD 2020-2060')
    # ax2.legend(bbox_to_anchor=(1.05, 0.8), fancybox=True, title='Screening method')
    # f.suptitle(f'ICER plot, {location.capitalize()}')
    # sc.SIticks(ax1)
    # sc.SIticks(ax2)
    # sc.SIticks(ax3)
    # sc.SIticks(ax4)
    # f.tight_layout()
    # fig_name = f'{figfolder}/ICER_{location}.png'
    # sc.savefig(fig_name, dpi=100)

    data_to_plot = data_to_plot[data_to_plot['scen_label'] != 'No screening']
    efficiency_data = data_to_plot.copy().sort_values('total_costs').reset_index(drop=True)
    efficient_scenarios = efficiency_data['scen_label'].values
    num_scens = len(efficient_scenarios)-1
    icers = sc.autolist()
    icers += 0
    i = 1
    while i <= num_scens:
        inc_DALYs = efficiency_data.iloc[i]['DALYs_averted'] - efficiency_data.iloc[i - 1]['DALYs_averted']
        inc_cost = efficiency_data.iloc[i]['total_costs'] - efficiency_data.iloc[i - 1]['total_costs']
        if inc_DALYs < 0: # if it averts negative DALYs it is dominated by definition
            efficiency_data = efficiency_data.drop(i).reset_index(drop=True)
            efficient_scenarios = np.delete(efficient_scenarios, i)
            num_scens -=1
        else:
            icer = inc_cost / inc_DALYs
            extended_dominance_check = True
            while extended_dominance_check:
                if icer < icers[i - 1]:
                    efficiency_data = efficiency_data.drop(i - 1).reset_index(drop=True)
                    efficient_scenarios = np.delete(efficient_scenarios, i - 1)
                    num_scens -= 1
                    icers = np.delete(icers, i - 1)
                    i -= 1
                    inc_DALYs = efficiency_data.iloc[i]['DALYs_averted'] - efficiency_data.iloc[i - 1]['DALYs_averted']
                    inc_cost = efficiency_data.iloc[i]['total_costs'] - efficiency_data.iloc[i - 1]['total_costs']
                    if inc_DALYs < 0:
                        efficiency_data = efficiency_data.drop(i).reset_index(drop=True)
                        efficient_scenarios = np.delete(efficient_scenarios, i)
                        num_scens -= 1
                        i -= 1
                        extended_dominance_check = False
                    else:
                        icer = inc_cost / inc_DALYs
                        icers = np.append(icers,icer)
                else:
                    icers = np.append(icers,icer)
                    i += 1
                    extended_dominance_check = False

    f, ax = pl.subplots(figsize=(16, 10))
    efficiency_data.plot(ax=ax, kind='line', x='DALYs_averted', y='total_costs', color='black', label='Efficiency frontier')

    for i, scen in enumerate(scenarios_to_plot):
        group = data_to_plot[data_to_plot['scen_label'] == scen]
        if scen in AVE_scenarios:
            group_sens = ave_cost_data_to_plot[ave_cost_data_to_plot['scen_label']==scen]
            ymax = group_sens['total_costs'].max()
            ymin = group_sens['total_costs'].min()
            xval = group['DALYs_averted'].values
            pl.vlines(x=xval, ymin=ymin, ymax=ymax, color=colors[scen_colors[scen]])
        if scen != 'No screening':
            ellipse_group = alldfs[(alldfs.location == location) & (alldfs.scen_label == scen)]
            x, y = ellipse_group['DALYs_averted'].values, ellipse_group['total_costs'].values
            confidence_ellipse(x, y, ax=ax, edgecolor=colors[scen_colors[scen]])

        group.plot(ax=ax, kind='scatter', x='DALYs_averted', y='total_costs', label=scen, color=colors[scen_colors[scen]], marker=markers[i], s=200)

    ax.set_xlabel('DALYs averted, 2020-2060')
    ax.set_ylabel('Total costs, $USD 2020-2060')
    ax.legend(bbox_to_anchor=(1.05, 0.8), fancybox=True)#, title='Screening method')
    # f.suptitle(f'ICER plot, {location.capitalize()}')
    sc.SIticks(ax)
    f.tight_layout()
    fig_name = f'{figfolder}/ICER_{location}.png'
    sc.savefig(fig_name, dpi=100)

    f, ax = pl.subplots(figsize=(16, 10))
    efficiency_data = data_to_plot.copy().sort_values('total_cin_treatments').reset_index(drop=True)
    efficient_scenarios = efficiency_data['scen_label'].values
    num_scens = len(efficient_scenarios) - 1
    icers = sc.autolist()
    icers += 0
    i = 1
    while i <= num_scens:
        inc_DALYs = efficiency_data.iloc[i]['DALYs_averted'] - efficiency_data.iloc[i - 1]['DALYs_averted']
        inc_cost = efficiency_data.iloc[i]['total_cin_treatments'] - efficiency_data.iloc[i - 1]['total_cin_treatments']
        if inc_DALYs < 0:  # if it averts negative DALYs it is dominated by definition
            efficiency_data = efficiency_data.drop(i).reset_index(drop=True)
            efficient_scenarios = np.delete(efficient_scenarios, i)
            num_scens -= 1
        else:
            icer = inc_cost / inc_DALYs
            extended_dominance_check = True
            while extended_dominance_check:
                if icer < icers[i - 1]:
                    efficiency_data = efficiency_data.drop(i - 1).reset_index(drop=True)
                    efficient_scenarios = np.delete(efficient_scenarios, i - 1)
                    num_scens -= 1
                    icers = np.delete(icers, i - 1)
                    i -= 1
                    inc_DALYs = efficiency_data.iloc[i]['DALYs_averted'] - efficiency_data.iloc[i - 1]['DALYs_averted']
                    inc_cost = efficiency_data.iloc[i]['total_cin_treatments'] - efficiency_data.iloc[i - 1]['total_cin_treatments']
                    if inc_DALYs < 0:
                        efficiency_data = efficiency_data.drop(i).reset_index(drop=True)
                        efficient_scenarios = np.delete(efficient_scenarios, i)
                        num_scens -= 1
                        i -= 1
                        extended_dominance_check = False
                    else:
                        icer = inc_cost / inc_DALYs
                        icers = np.append(icers, icer)
                else:
                    if icer not in icers:
                        icers = np.append(icers, icer)
                    i += 1
                    extended_dominance_check = False

    efficiency_data.plot(ax=ax, kind='line', x='DALYs_averted', y='total_cin_treatments', color='black',
                         label='Efficiency frontier')

    for i, scen in enumerate(scenarios_to_plot):
        group = data_to_plot[data_to_plot['scen_label'] == scen]
        if scen != 'No screening':
            ellipse_group = alldfs[(alldfs.location == location) & (alldfs.scen_label == scen)]
            x, y = ellipse_group['DALYs_averted'].values, ellipse_group['total_cin_treatments'].values
            confidence_ellipse(x, y, ax=ax, edgecolor=colors[scen_colors[scen]])

        group.plot(ax=ax, kind='scatter', x='DALYs_averted', y='total_cin_treatments', label=scen,
                   color=colors[scen_colors[scen]], marker=markers[i], s=200)

    ax.set_xlabel('DALYs averted, 2020-2060')
    ax.set_ylabel('Total CIN treatments, 2020-2060')
    ax.legend(bbox_to_anchor=(1.05, 0.8), fancybox=True)#, title='Screening method')
    sc.SIticks(ax)
    f.tight_layout()
    fig_name = f'{figfolder}/CIN_treatment_efficiency_{location}.png'
    sc.savefig(fig_name, dpi=100)

print('done')



