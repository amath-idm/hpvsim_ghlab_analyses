'''
Calibrate HPVsim to high-burden countries and run analyses to produce estimates
of burden of cervical cancer over 2020-2060.

To change whether the calibration is run/plotted, comment out the lines in the
"to_run" list below.

Note that running with debug=False requires an HPC and MySQL to be configured.
With debug=True, should take 5-10 min to run.
'''

# Standard imports
import sciris as sc
import hpvsim as hpv

# Imports from this repository
import run_sim as rs
import pars_genotypes as gp
import utils as ut

# Comment out to not run
to_run = [
    'run_calibration',
    # 'plot_calibration',
]

# Comment out locations to not run
locations = [
    # 'india',    # 0
    'nigeria',  # 1
    # 'tanzania', # 2
]

debug = False # Smaller runs
do_save = True


# Run settings for calibration (dependent on debug)
n_trials    = [1000, 2][debug]  # How many trials to run for calibration
n_workers   = [60, 4][debug]    # How many cores to use
storage     = ["mysql://hpvsim_user@localhost/hpvsim_db", None][debug] # Storage for calibrations


########################################################################
# Run calibration
########################################################################
def run_calib(location=None, calib=True, n_trials=None, n_workers=None,
              do_plot=False, do_save=True):

    pars, analyzers, interventions = rs.make_sim_parts(location=location, calib=calib)
    sim = rs.make_sim(pars, analyzers, interventions, datafile=f'data/{location}_data.csv')

    calib_pars = dict(
        beta=[0.2, 0.1, 0.3],
        dur_transformed=dict(par1=[5, 3, 10]),
    )

    genotype_pars = gp.get_genotype_pars(location)

    if location=='india':
        datafiles = [
            f'data/{location}_cancer_cases.csv',
            f'data/{location}_cin1_types.csv',
            f'data/{location}_cin3_types.csv',
            f'data/{location}_cancer_types.csv',
        ]
    elif location in ['nigeria']:
        datafiles = [
            f'data/{location}_cancer_cases.csv',
            f'data/{location}_cancer_types.csv',
        ]
    else:
        datafiles = [
            f'data/{location}_cancer_cases.csv',
            f'data/{location}_cin3_types.csv',
            f'data/{location}_cancer_types.csv',
        ]
    calib = hpv.Calibration(sim, calib_pars=calib_pars, genotype_pars=genotype_pars,
                            name=f'{location}_calib',
                            datafiles=datafiles,
                            total_trials=n_trials, n_workers=n_workers,
                            storage=storage
                            )
    calib.calibrate()
    filename = f'{location}_calib'
    if do_plot:
        calib.plot(do_save=True, fig_path=f'{ut.figfolder}/{filename}.png')
    if do_save:
        sc.saveobj(f'{ut.resfolder}/{filename}.obj', calib)

    print(f'Best pars are {calib.best_pars}')

    return sim, calib


########################################################################
# Load pre-run calibration
########################################################################
def load_calib(location=None, do_plot=True, which_pars=0, save_pars=True, do_plot_additional=False):

    filename = f'{location}_calib'
    calib = sc.load(f'{ut.resfolder}/{filename}.obj')
    if do_plot:
        sc.fonts(add=sc.thisdir(aspath=True) / 'Libertinus Sans')
        sc.options(font='Libertinus Sans')
        fig = calib.plot(res_to_plot=50, plot_type='sns.boxplot', do_save=True,
                         fig_path=f'{ut.figfolder}/{filename}')
        fig.suptitle(f'Calibration results, {location.capitalize()}')
        fig.tight_layout()
        fig.savefig(f'{ut.figfolder}/{filename}.png')

    if save_pars:
        calib_pars = calib.trial_pars_to_sim_pars(which_pars=which_pars)
        sc.save(f'{ut.resfolder}/{location}_pars.obj', calib_pars)

    if do_plot_additional:
        fig = ut.plot_trend(calib)
        ut.plot_best(calib).fig.show()

    return calib


#%% Run as a script
if __name__ == '__main__':

    T = sc.timer()

    # Run calibration - usually on VMs
    if 'run_calibration' in to_run:
        for location in locations:
            sim, calib = run_calib(location=location, n_trials=n_trials, n_workers=n_workers, do_save=do_save, do_plot=False)

    # Load the calibration, plot it, and save the best parameters -- usually locally
    if 'plot_calibration' in to_run:
        for location in locations:
            calib = load_calib(location=location, do_plot=True, save_pars=True, do_plot_additional=False)

    
    T.toc('Done')