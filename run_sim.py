'''
Define the HPVsim simulations for India, Nigeria, and Tanzania that are used as
the basis for the calibration, scenarios, and sweeps.

By default, all locations are run. To not run a location, comment out the line
below. For all three locations, this script should take 1-5 minutes to run.
'''

# Standard imports
import numpy as np
import sciris as sc
import hpvsim as hpv

# Imports from this repository
import pars_data as dp
import pars_scenarios as sp
import analyzers as an
import utils as ut


#%% Settings and filepaths

# Locations -- comment out a line to not run
locations = [
    'india',        # 0
    'nigeria',      # 1
    'tanzania',     # 2
]

# Debug switch
debug = 0 # Run with smaller population sizes and in serial
do_shrink = True # Do not keep people when running sims (saves memory)

# Save settings
do_save = True
save_plots = True


#%% Simulation creation functions
def make_sim_parts(location=None, calib=False, vaccination_coverage=None,
                   vx_scen=None, tx_vx_scen=None, screen_scen=None, indication=None,
                   save_econ=False, intro_year=None, debug=0, txvx_prods=None, multiscale=True,
                   pop_analyzers=False):
    ''' Define parameters, analyzers, and interventions for the simulation -- not the sim itself '''

    is_scen = (vx_scen is not None) or (tx_vx_scen is not None) or (screen_scen is not None)
    end = 2060 if is_scen else 2020

    # Parameters
    pars = dict(
        n_agents       = [50e3,5e3][debug],
        dt             = [0.5,1.0][debug],
        start          = [1950,1980][debug],
        end            = end,
        network        = 'default',
        location       = location,
        debut          = dp.debut[location],
        mixing         = dp.mixing[location],
        layer_probs    = dp.layer_probs[location],
        genotypes      = [16, 18, 'hrhpv'],
        condoms        = dict(m=0.01, c=0.1, o=0.2),
        eff_condoms    = 0.5,
        use_multiscale = multiscale,
        ms_agent_ratio = 100,
        verbose        = 0.0,
    )

    # Analyzers
    analyzers = sc.autolist()

    # Save age results unless it's a calibration
    if not calib:
        edges = np.array([0., 15., 20., 25., 30., 40., 45., 50., 55., 65., 100.])
        az = hpv.age_results(
            result_keys=sc.objdict(
                total_infections=sc.objdict(
                    timepoints=['2019'],
                    edges=edges,
                ),
                total_hpv_prevalence=sc.objdict(
                    timepoints=['2019'],
                    edges=edges,
                ),
                total_cancers=sc.objdict(
                    datafile=f'data/{location}_cancer_cases.csv',
                ),
                total_cancer_deaths=sc.objdict(
                    datafile=f'data/{location}_cancer_deaths.csv',
                )
            )
        )
        analyzers += az

    # Save population analyzers if requested
    if pop_analyzers:
        analyzers += [
            hpv.age_pyramid(
            timepoints=['1990','2020'],
            datafile=f'data/{location}_age_pyramid_reduced.csv',
            edges=np.linspace(0, 80, 9)),
            hpv.snapshot(timepoints=['2020']),
        ]

    # Save costing analyzer if requested
    if save_econ:
        analyzers += an.econ_analyzer()


    # Interventions, all added as part of specific scenarios
    interventions = []

    if vx_scen is not None:

        # Routine vaccination
        routine_years   = vaccination_coverage[location]['routine']['years'][:81] ### ONLY USING 1980-2060, WHICH IS WHY THIS INDEX IS HERE
        routine_values  = vaccination_coverage[location]['routine']['coverage'][:81]

        routine_vx = hpv.routine_vx(
            prob=routine_values,
            years=routine_years,
            product='bivalent',
            age_range=(9, 10),
            label='Routine'
        )

        # These locations have campaign coverage in the datafile
        if location in ['nigeria', 'india']:
            campaign_value = vaccination_coverage[location]['campaign']['coverage'][0]
            campaign_year  = vaccination_coverage[location]['campaign']['years'][0]
        else:
            campaign_value = vaccination_coverage['nigeria']['campaign']['coverage'][0] # Assume same as nigeria
            campaign_year  = vaccination_coverage['nigeria']['campaign']['years'][0]

        if '9to14' in vx_scen:
            campaign_vx = hpv.campaign_vx(
                prob=campaign_value,
                years=campaign_year,
                product='bivalent',
                age_range=(9, 14),
                label='Campaign'
            )

        else:
            campaign_vx = hpv.campaign_vx(
                prob=campaign_value,
                years=campaign_year,
                product='bivalent',
                age_range=(9, 24),
                label='Campaign'
            )

        interventions += [routine_vx, campaign_vx]

    if tx_vx_scen is not None:
        # Add in therapeutic vaccines
        interventions += sp.get_txvx_intvs(location=location, use_case=tx_vx_scen, indication=indication,
                                           txvx_prods=txvx_prods, intro_year=intro_year)

    if screen_scen is not None:
        interventions += sp.get_screen_intvs(location, screen_scen)

    return pars, analyzers, interventions


def make_sim(pars=None, analyzers=None, interventions=None, datafile=None, seed=1):
    ''' Actually create the sim '''
    sim = hpv.Sim(pars=pars, analyzers=analyzers, interventions=interventions, datafile=datafile, rand_seed=seed)
    return sim



#%% Simulation running functions

def run_sim(location=None, ccut=None, use_calib_pars=False,
            do_plot=False, save_plots=False, seed=0, vaccination_coverage=None,
            vx_scen=None, tx_vx_scen=None, screen_scen=None,
            indication=None, txvx_prods=None, lo_eff=None, hi_eff=None, intro_year=None,
            save_econ=None, multiscale=True, debug=0, label=None, meta=None, verbose=0.1, do_shrink=True,
            do_save=True, pop_analyzers=False, die=False):
    ''' Assemble the parts into a complete sim and run it '''

    # Decide what message to print
    if meta is not None:
        msg = f'Making sim {meta.inds} ({meta.count} of {meta.n_sims}) for {location}'
    else:
        msg = f'Making sim for {location}'
    if debug: msg += ' IN DEBUG MODE'
    print(msg)

    # Make arguments
    args = make_sim_parts(location=location, calib=False, vaccination_coverage=vaccination_coverage,
                          vx_scen=vx_scen, tx_vx_scen=tx_vx_scen, screen_scen=screen_scen, intro_year=intro_year,
                          indication=indication, save_econ=save_econ, txvx_prods=txvx_prods, multiscale=multiscale,
                          debug=debug, pop_analyzers=pop_analyzers)
    sim = make_sim(*args, datafile=f'data/{location}_data.csv')

    # Store metadata
    sim.meta = sc.objdict()
    if meta is not None:
        sim.meta = meta # Copy over meta info
    else:
        sim.meta = sc.objdict()
    sim.meta.location = location # Store location in an easy-to-access place
    sim['rand_seed'] = seed # Set seed
    sim.label = location # Set label

    # Make any parameter updates
    if ccut: sim.update_pars({'clinical_cutoffs': {'cin1': 0.8, 'cin2':0.9, 'cin3':0.99}})
    if use_calib_pars:
        file = f'{ut.resfolder}/{location}_pars.obj'
        try:
            calib_pars = sc.loadobj(file)
        except:
            errormsg = 'Calibration parameters cannot be loaded from disk. Try running load_calib to generate them.'
            raise ValueError(errormsg)

        sim.initialize() # Important to do this here, otherwise the genotype pars get overwritten
        sim.update_pars(calib_pars)

    # Run
    sim['verbose'] = verbose
    sim.run()


    # Check plot()
    if do_plot:
        to_plot = {
            'Total Infections': [
                'total_infections',
            ],
            'Total CINs': [
                'total_cin1s',
                'total_cin2s',
                'total_cin3s'
            ],
            'Cervical cancer cases': [
                'total_cancers',
            ],
            'Cervical cancer incidence': [
                'asr_cancer',
                'total_cancer_incidence'
            ],
        }
        if screen_scen is not None:
            to_plot['Screens'] = [
                'resources_screening',
            ]
            to_plot['Tx'] = [
                'resources_ablation',
                'resources_excision',
            ]
        if tx_vx_scen is not None:
            to_plot['TxVx'] = [
                'resources_campaign txvx',
                'resources_routine txvx',
            ]
        sim.plot(do_save=save_plots, do_show=True, to_plot=to_plot, fig_path=f'{ut.figfolder}/{location}_basic_epi.png')
        # az = sim.get_analyzer(hpv.age_results)
        # az.plot(do_save=save_plots, do_show=True, fig_path=f'{figfolder}/{location}_cancer_by_age.png')
    
    if do_shrink: 
        sim.shrink()
        
        # Keep only the part of the population pyramid snapshot needed for plotting
        people = None
        try:
            people = sim.get_analyzer('snapshot').snapshots[0]
        except:
            pass
        try:
            if people is not None:
                for key in list(people.keys()):
                    if key not in ['uid', 'contacts']:
                        delattr(people, key) # Manually shrink the people object to the bare minimum
        except Exception as E:
            errormsg = f'Could not shrink analyzer, OK if analyzer does not exist: {str(E)}'
            print(errormsg)
        
    if do_save: 
        sim.save(f'{ut.resfolder}/{location}.sim')
    
    return sim


def run_sims(locations=None, *args, **kwargs):
    ''' Run multiple simulations in parallel '''
    
    kwargs = dict(use_calib_pars=True, do_plot=False, debug=debug, save_econ=False, pop_analyzers=True, do_save=True)
    simlist = sc.parallelize(run_sim, iterkwargs=dict(location=locations), kwargs=kwargs, serial=debug, die=True)
    sims = sc.objdict({location:sim for location,sim in zip(locations, simlist)}) # Convert from a list to a dict
    
    return sims


#%% Run as a script
if __name__ == '__main__':

    T = sc.timer()
    
    # Run a single sim per location -- usually locally, can be used for sanity checking and debugging
    sims = run_sims(locations)
    
    T.toc('Done')

