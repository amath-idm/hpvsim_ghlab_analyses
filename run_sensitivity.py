'''
Run parameter sensitivity.

Note: requires an HPC to run with debug=False; with debug=True, should take 20-40 min
to run.
'''


#%% General settings

import pandas as pd
import sciris as sc
import run_scenarios_general as rsc
import utils as ut

# Comment out to not run
to_run = [
    'run_sensitivity',
    'plot_sensitivity',
]

# Comment out locations to not run
locations = [
    'india',    # 0
    'nigeria',  # 1
    'tanzania', # 2
]

debug = True
n_seeds = [5, 1][debug] # How many seeds to use for stochasticity in projections


#%% Run as a script
if __name__ == '__main__':

    T = sc.timer()
    
    if 'run_sensitivity' in to_run:
        alldfs = sc.autolist()
        for location in locations:
            filestem = 'sensitivity_results'
            vx_scens = ['90vx_9to14']
            screen_scens = ['0sc_10tx', '10sc_10tx', '30sc_30tx', '50sc_50tx', '70sc_90tx']
            tx_vx_scens = [None, 'mass_vaccination', 'test_and_vaccinate']
            progressions = [False, True]
            intro_years = [2030, 2035, 2040]
            indications = ['virologic_clearance', 'lesion_regression']
            alldf, msims = rsc.run_scens(screen_scens=screen_scens, vx_scens=vx_scens,
                                     ccuts=progressions, tx_vx_scens=tx_vx_scens, 
                                     intro_years=intro_years, indications=indications, 
                                     n_seeds=n_seeds, location=location, debug=debug, save_econ=False)

            alldfs += alldf
            sc.saveobj(f'{ut.resfolder}/{location}_{filestem}.obj', alldf)
        bigdf = pd.concat(alldfs)
        sc.saveobj(f'{ut.resfolder}/{filestem}.obj', bigdf)

    if 'plot_sensitivity' in to_run:
        for prog in ['fast', 'linear']:
            for delivery in ['mass_vaccination', 'test_and_vaccinate']:
                ut.plot_tx_vx_sens_cov_intro(
                    locations=['india', 'nigeria', 'tanzania'],
                    sens_backgrounds={
                        '0': {
                            'vx_scen': '90vx_9to14',
                            'screen_scen': '0sc_10tx'
                        },
                        '10': {
                            'vx_scen': '90vx_9to14',
                            'screen_scen': '10sc_10tx'
                        },
                        '30': {
                            'vx_scen': '90vx_9to14',
                            'screen_scen': '30sc_30tx'
                        },
                        '50': {
                            'vx_scen': '90vx_9to14',
                            'screen_scen': '50sc_50tx'
                        },
                        '70': {
                            'vx_scen': '90vx_9to14',
                            'screen_scen': '70sc_90tx'
                        },
                    },
                    progression=prog,
                    tx_vx_scen=delivery,
                    intro_years=[2030, 2035, 2040]
                )

            ut.plot_tx_vx_scens_sens(
                locations=['india', 'nigeria', 'tanzania'],
                sens_backgrounds={
                    '0': {
                        'vx_scen': '90vx_9to14',
                        'screen_scen': '0sc_10tx'
                    },
                    '10': {
                        'vx_scen': '90vx_9to14',
                        'screen_scen': '10sc_10tx'
                    },
                    '30': {
                        'vx_scen': '90vx_9to14',
                        'screen_scen': '30sc_30tx'
                    },
                    '50': {
                        'vx_scen': '90vx_9to14',
                        'screen_scen': '50sc_50tx'
                    },
                    '70': {
                        'vx_scen': '90vx_9to14',
                        'screen_scen': '70sc_90tx'
                    },
                },
                progression=prog,
                tx_vx_scens=['mass_vaccination', 'test_and_vaccinate'],
            )
    
    T.toc('Done')