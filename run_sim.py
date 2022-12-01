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
def make_sim_parts(location=None, vaccination_coverage=None,
                   debug=0, screen_intvs=None, multiscale=True, econ_analyzer=True):
    ''' Define parameters, analyzers, and interventions for the simulation -- not the sim itself '''

    # Parameters
    pars = dict(
        n_agents       = [50e3,1e3][debug],
        dt             = [0.5,1.0][debug],
        start          = [1950,1980][debug],
        end            = 2060,
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
    if econ_analyzer:
        analyzers += an.econ_analyzer()

    # Interventions, all added as part of specific scenarios
    interventions = sc.autolist()

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

    campaign_vx = hpv.campaign_vx(
        prob=campaign_value,
        years=campaign_year,
        product='bivalent',
        age_range=(9, 14),
        label='Campaign'
    )

    interventions += [routine_vx, campaign_vx]

    # Add screening interventions
    interventions += screen_intvs

    return pars, analyzers, interventions


def make_sim(pars=None, analyzers=None, interventions=None, datafile=None, seed=1):
    ''' Actually create the sim '''
    sim = hpv.Sim(pars=pars, analyzers=analyzers, interventions=interventions, datafile=datafile, rand_seed=seed)
    return sim



#%% Simulation running functions

def run_sim(location=None, use_calib_pars=False, screen_intvs=None,
            debug=0, seed=0, vaccination_coverage=None,
            label=None, meta=None, verbose=0.1,
            do_save=False, die=False):
    ''' Assemble the parts into a complete sim and run it '''

    # Decide what message to print
    if meta is not None:
        msg = f'Making sim {meta.inds} ({meta.count} of {meta.n_sims}) for {location}'
    else:
        msg = f'Making sim for {location}'
    if debug: msg += ' IN DEBUG MODE'
    print(msg)

    # Make arguments
    args = make_sim_parts(location=location, vaccination_coverage=vaccination_coverage,
                          screen_intvs=screen_intvs, debug=debug)
    sim = make_sim(*args, datafile=f'data/{location}_data.csv')

    # Store metadata
    sim.meta = sc.objdict()
    if meta is not None:
        sim.meta = meta # Copy over meta info
    else:
        sim.meta = sc.objdict()
    sim.meta.location = location # Store location in an easy-to-access place
    sim['rand_seed'] = seed # Set seed
    sim.label = f'{label}--{location}' # Set label

    # Make any parameter updates
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
    sim.shrink()
        
    if do_save:
        sim.save(f'{ut.resfolder}/{location}.sim')
    
    return sim


def run_sims(locations=None, *args, **kwargs):
    ''' Run multiple simulations in parallel '''
    
    kwargs = sc.mergedicts(dict(use_calib_pars=True, debug=debug), kwargs)
    simlist = sc.parallelize(run_sim, iterkwargs=dict(location=locations), kwargs=kwargs, serial=debug, die=True)
    sims = sc.objdict({location:sim for location,sim in zip(locations, simlist)}) # Convert from a list to a dict
    
    return sims


#%% Run as a script
if __name__ == '__main__':

    T = sc.timer()
    
    # Run a single sim per location -- usually locally, can be used for sanity checking and debugging
    sims = run_sims(locations)
    
    T.toc('Done')

