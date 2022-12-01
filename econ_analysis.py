'''
Run economic analysis.

'''


#%% General settings

# Standard imports
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
import sciris as sc
import hpvsim as hpv

# Comment out to not run
to_run = [
    # 'run_scenarios',
    'plot_scenarios',
]

locations = [
    'india',
    # 'nigeria',
    # 'tanzania'
]

dfs = sc.autolist()
for location in locations:
    dfs += pd.read_csv(f'results/{location}_econ.csv')
model_res = pd.concat(dfs)

n_seeds = len(np.unique(model_res['seed']))
cost_params = pd.DataFrame()
cost_params['location'] = np.array(['india', 'nigeria', 'tanzania'])
cost_params['HPV'] = np.array([14.8, 36, 9.1])
cost_params['HPV_sd'] = (2*1.96)*3.8/14.8
cost_params['VIA'] = np.array([5.2, 13, 2.89])
cost_params['VIA_sd'] = (2*1.96)*1.3/5.2
cost_params['CIN'] = np.array([60, 3.5, 3.57])
cost_params['CIN_sd'] = (2*1.96)*4.2/16
cost_params['cancer'] = np.array([450, np.mean(np.array([44.73, 64.13, 281.5, 768, 212])), np.mean(np.array([94, 574, 974, 21]))])
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
# Thermal ablation: $60.4 ($15)
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
    simulated_costs['VIA'] = truncnorm.rvs((lower_clip - costs['VIA']) / costs['VIA_sd'],
                                           (upper_clip - costs['VIA']) / costs['VIA_sd'],
                                           loc=costs['VIA'], scale=costs['VIA_sd'], size=n_seeds)
    simulated_costs['CIN'] = truncnorm.rvs((lower_clip - costs['CIN']) / costs['CIN_sd'],
                                           (upper_clip - costs['CIN']) / costs['CIN_sd'],
                                           loc=costs['CIN'], scale=costs['CIN_sd'], size=n_seeds)
    simulated_costs['cancer'] = truncnorm.rvs((lower_clip - costs['cancer']) / costs['cancer_sd'],
                                              (upper_clip - costs['cancer']) / costs['cancer_sd'],
                                              loc=costs['cancer'], scale=costs['cancer_sd'], size=n_seeds)
    simulated_costs['location'] = location
    dfs += simulated_costs

simulated_cost_df = pd.concat(dfs)

scenarios = np.unique(model_res['scen_label'])

dfs = sc.autolist()
for location in locations:
    costs = simulated_cost_df[simulated_cost_df['location'] == location]
    for scenario in scenarios:
        df = pd.DataFrame()
        model_output = model_res[(model_res['location'] == location) & (model_res['scen_label'] == scenario)]
        means = model_output.groupby('year')[['new_cancers', 'new_cancer_deaths', 'new_hpv_screens',
                                              'new_poc_hpv_screens', 'new_via_screens', 'new_ave_screens',
                                              'new_thermal_ablations', 'new_leeps', 'new_cancer_treatments']].mean()
        df['total_cancers'] = [means['new_cancers'].sum()]
        df['total_cancer_deaths'] = [means['new_cancer_deaths'].sum()]
        df['total_hpv_screens'] = [means['new_hpv_screens'].sum()]
        df['total_poc_hpv_screens'] = [means['new_poc_hpv_screens'].sum()]
        df['total_via_screens'] = [means['new_via_screens'].sum()]
        df['total_ave_screens'] = [means['new_ave_screens'].sum()]
        df['total_thermal_ablations'] = [means['new_thermal_ablations'].sum()]
        df['total_leeps'] = [means['new_leeps'].sum()]
        df['total_cancer_treatments'] = [means['new_cancer_treatments'].sum()]
        df['location'] = location
        df['scen_label'] = scenario
        dfs += df

alldfs = pd.concat(dfs)
print('done')



