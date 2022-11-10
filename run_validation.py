'''
Run age of causal infection validations with HPVsim.

Takes about 1-2 min to run on a laptop with debug=True.
'''

#%% General settings

# Standard imports
import numpy as np
import pandas as pd
import sciris as sc
import pylab as pl
from scipy.stats import lognorm
import hpvsim as hpv

# Imports from this repository
import pars_data as dp
import utils as ut


locations = [
    'india',        # 0
    'nigeria',      # 1
    'tanzania',     # 2
]
location = locations[0]

do_save = True
save_plots = True
debug = True

if debug:
    n_workers = 2
    n_seeds = 2
else:
    n_workers = 10
    n_seeds = 10


#%% Analyzers

class dwelltime(hpv.Analyzer):
    '''
    Determine time spent in health states for those who do NOT get cancer
    '''

    def __init__(self, start_year=None, **kwargs):
        super().__init__(**kwargs)
        self.start_year = start_year
        self.years = None

    def initialize(self, sim):
        super().initialize(sim)
        self.years = sim.yearvec
        if self.start_year is None:
            self.start_year = sim['start']
        self.dwelltime = dict()
        for state in ['hpv', 'cin1', 'cin2', 'cin3', 'total']:
            self.dwelltime[state] = []

    def apply(self, sim):
        if sim.yearvec[sim.t] >= self.start_year:
            genotypes, inds = (sim.people.date_clearance == sim.t).nonzero()
            if len(inds):
                date_exposed = sim.people.date_exposed[genotypes, inds]
                cin1_inds = hpv.true(~np.isnan(sim.people.date_cin1[genotypes, inds]))
                cin2_inds = hpv.true(~np.isnan(sim.people.date_cin2[genotypes, inds]))
                cin3_inds = hpv.true(~np.isnan(sim.people.date_cin3[genotypes, inds]))
                hpv_time = ((sim.people.date_cin1[genotypes[cin1_inds], inds[cin1_inds]] - date_exposed[cin1_inds]) * sim['dt']).tolist() + \
                           ((sim.people.date_cin1[genotypes[cin2_inds], inds[cin2_inds]] - date_exposed[cin2_inds])*sim['dt']).tolist() + \
                           ((sim.people.date_cin1[genotypes[cin3_inds], inds[cin3_inds]] - date_exposed[cin3_inds])*sim['dt']).tolist()

                cin1_time = ((sim.t - sim.people.date_cin1[genotypes[cin1_inds], inds[cin1_inds]])*sim['dt']).tolist() + \
                            ((sim.people.date_cin2[genotypes[cin2_inds], inds[cin2_inds]] - sim.people.date_cin1[genotypes[cin2_inds], inds[cin2_inds]]) * sim['dt']).tolist() + \
                            ((sim.people.date_cin2[genotypes[cin3_inds], inds[cin3_inds]] - date_exposed[cin3_inds])*sim['dt']).tolist()

                cin2_time = ((sim.t - sim.people.date_cin2[genotypes[cin2_inds], inds[cin2_inds]])*sim['dt']).tolist() + \
                            ((sim.people.date_cin3[genotypes[cin3_inds], inds[cin3_inds]] - sim.people.date_cin2[genotypes[cin3_inds], inds[cin3_inds]]) * sim['dt']).tolist()
                cin3_time = ((sim.t - sim.people.date_cin3[genotypes[cin3_inds], inds[cin3_inds]]) * sim['dt']).tolist()
                total_time = ((sim.t - date_exposed)*sim['dt']).tolist()
                self.dwelltime['hpv'] += hpv_time
                self.dwelltime['cin1'] += cin1_time
                self.dwelltime['cin2'] += cin2_time
                self.dwelltime['cin3'] += cin3_time
                self.dwelltime['total'] += total_time


class rship_count(hpv.Analyzer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_edges = dict()
        self.n_edges_norm = dict()
        for rtype in ['m','c','o']:
            self.n_edges[rtype] = []
            self.n_edges_norm[rtype] = []

    def apply(self, sim):
        for rtype in ['m','c','o']:
            self.n_edges[rtype].append(len(sim.people.contacts[rtype]))
            self.n_edges_norm[rtype].append(len(sim.people.contacts[rtype])/len(hpv.true((sim.people.age>14) & (sim.people.age<65) & (sim.people.alive))))
        return

    def plot(self, sim, do_save=False, filename=None, from_when=1990):
        fig, ax = pl.subplots(2, 3, figsize=(15, 8))
        yi = sc.findinds(sim.yearvec, from_when)[0]
        for rn,rtype in enumerate(['m','c','o']):
            ax[0,rn].plot(sim.yearvec[yi:], self.n_edges[rtype][yi:])
            ax[0,rn].set_title(f'Edges - {rtype}')
            ax[1,rn].plot(sim.yearvec[yi:], self.n_edges_norm[rtype][yi:])
            ax[1,rn].set_title(f'Normalized edges - {rtype}')
        pl.tight_layout()
        if do_save:
            fn = 'networks' or filename
            fig.savefig(f'{fn}.png')
        else:
            pl.show()


#%% Functions for running simulations

def run_sim(location=None, seed=0, debug=0, use_ccut=False):
    '''
    Custom simulation run function; compare with run_sim.py:run_sim()
    '''
    
    # Parameters
    pars = dict(
        n_agents       = [50e3, 5e3][debug],
        start          = [1950, 1980][debug],
        end            = 2050,
        dt             = [0.5, 1.0][debug],
        network        = 'default',
        location       = location,
        genotypes      = [16,18,'hrhpv'],
        debut          = dp.debut[location],
        mixing         = dp.mixing[location],
        layer_probs    = dp.layer_probs[location],
        rand_seed      = seed,
        verbose        = 0.0,
        condoms        = dict(m=0.01, c=0.1, o=0.2),
        eff_condoms    = 0.5,
        use_multiscale = True,
        ms_agent_ratio = 100,
    )

    if use_ccut:
        pars['clinical_cutoffs'] = {'cin1': 0.8, 'cin2': 0.9, 'cin3': 0.99}

    sim = hpv.Sim(pars=pars, analyzers=[dwelltime(start_year=2020), hpv.age_causal_infection(start_year=2020)])

    try:
        file = f'{ut.resfolder}/{location}_pars.obj'
        calib_pars = sc.loadobj(file)
    except:
        errormsg = 'Calibration parameters cannot be loaded from disk. Try running load_calib to generate them.'
        raise ValueError(errormsg)

    sim.initialize()  # Important to do this here, otherwise the genotype pars get overwritten
    sim.update_pars(calib_pars)
    sim.run()

    dt = sim.get_analyzer(dwelltime)
    dt_cancer = sim.get_analyzer()

    return sim, dt, dt_cancer


def run_age_causal_analyzer(locations=locations, debug=0, n_seeds=n_seeds, use_ccut=None, do_load=False, do_save=True):
    sc.heading('run age causal infection analyzer')

    if use_ccut is None:
        use_ccut = [False]*len(locations)

    # Set up arguments
    iterkwargs = sc.autolist()
    for seed in range(n_seeds):
        iterkwargs += dict(
            seed=seed
        )
    age_causal = dict()
    dwelltimes = {'hpv':dict(), 'cin1':dict(), 'cin2':dict(), 'cin3':dict(), 'total':dict()}

    # Actually run
    for i, location in enumerate(locations):
        ccut_tag = ', slower progression' if use_ccut[i] else ''
        location_name = f'{location}{ccut_tag}'
        age_causal[location_name] = []
        for state in ['hpv', 'cin1', 'cin2', 'cin3', 'total']:
            dwelltimes[state][location_name] = []
        kwargs = dict(location=location, debug=debug, use_ccut=use_ccut[i])
        sims = sc.parallelize(run_sim, iterkwargs=iterkwargs, kwargs=kwargs, ncpus=n_workers)

        for sim in sims:
            age_causal[location_name] += sim.get_analyzer(hpv.age_causal_infection).age_causal
            for state in ['hpv', 'cin1', 'cin2', 'cin3', 'total']:
                dwelltimes[state][location_name] += sim.get_analyzer(hpv.age_causal_infection).dwelltime[state]

    if debug or len(np.unique(locations)) == 1:
        fig, ax = pl.subplots(3, 2, figsize=(12, 12))
        ax[0, 0].boxplot([dwelltimes['hpv'][location], dwelltimes['hpv'][f'{location}, slower progression']], showfliers=False)
        ax[0, 1].boxplot([dwelltimes['cin1'][location], dwelltimes['cin1'][f'{location}, slower progression']],
                         showfliers=False)
        ax[1, 0].boxplot([dwelltimes['cin2'][location], dwelltimes['cin2'][f'{location}, slower progression']],
                         showfliers=False)
        ax[1, 1].boxplot([dwelltimes['cin3'][location], dwelltimes['cin3'][f'{location}, slower progression']],
                         showfliers=False)
        ax[2, 0].boxplot([dwelltimes['total'][location], dwelltimes['total'][f'{location}, slower progression']],
                         showfliers=False)
        ax[2, 1].boxplot([age_causal[location], age_causal[f'{location}, slower progression']], showfliers=False)
        ax[0, 0].set_title('HPV dwelltime')
        ax[0, 1].set_title('CIN1 dwelltime')
        ax[1, 0].set_title('CIN2 dwelltime')
        ax[1, 1].set_title('CIN3 dwelltime')
        ax[2, 0].set_title('Total dwelltime')
        ax[2, 1].set_title('Age of causal infection')

        ax[0, 0].set_xticklabels(['Fast progression', 'Slow progression'])
        ax[0, 1].set_xticklabels(['Fast progression', 'Slow progression'])
        ax[1, 0].set_xticklabels(['Fast progression', 'Slow progression'])
        ax[1, 1].set_xticklabels(['Fast progression', 'Slow progression'])
        ax[2, 0].set_xticklabels(['Fast progression', 'Slow progression'])
        ax[2, 1].set_xticklabels(['Fast progression', 'Slow progression'])
        fig.suptitle(f'Underlying natural history, {location}')
        fig.show()
        return

    # fig, ax = pl.subplots(1,1, figsize=(8, 6))
    # for i, location in enumerate(locations):
    #     ccut_tag = ', slower progression' if use_ccut[i] else ''
    #     location_name = f'{location}{ccut_tag}'
    #     for gi, ga in enumerate(age_causal[location_name]):
    #         count, bins_count = np.histogram(ga, bins=10)
    #         pdf = count / sum(count)
    #         cdf = np.cumsum(pdf)
    #         if gi == 0:
    #             ax.plot(bins_count[1:], 100*cdf, color=colors[i], label=location_name)
    #         else:
    #             ax.plot(bins_count[1:], 100*cdf, color=colors[i])
    # ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    # ax.set_title(f'Age of causal HPV infection')
    # ax.legend()
    # ax.set_xlabel('Age')
    # pl.tight_layout()
    # fig.savefig(f'{figfolder}/age_causal.png')

    unique_locations = np.unique(locations)

    def setBoxColors(bp):
        pl.setp(bp['boxes'][0], color='blue')
        pl.setp(bp['caps'][0], color='blue')
        pl.setp(bp['caps'][1], color='blue')
        pl.setp(bp['whiskers'][0], color='blue')
        pl.setp(bp['whiskers'][1], color='blue')
        pl.setp(bp['medians'][0], color='blue')

        pl.setp(bp['boxes'][1], color='red')
        pl.setp(bp['caps'][2], color='red')
        pl.setp(bp['caps'][3], color='red')
        pl.setp(bp['whiskers'][2], color='red')
        pl.setp(bp['whiskers'][3], color='red')
        pl.setp(bp['medians'][1], color='red')

    fig, ax = pl.subplots(3, 2, figsize=(12, 12))
    font_size = 22
    sc.fonts(add=sc.thisdir(aspath=True) / 'Libertinus Sans')
    sc.options(font='Libertinus Sans')
    pl.rcParams.update({'font.size': font_size})
    pl.rcParams['xtick.labelsize'] = 18
    pl.rcParams['xlabel.labelsize'] = 18
    pl.rcParams['ytick.labelsize'] = 18
    pl.rcParams['ylabel.labelsize'] = 18
    # Create second axes, the top-left plot with orange plot
    for i, loc in enumerate(unique_locations):
        y = [dwelltimes['hpv'][loc], dwelltimes['hpv'][f'{loc}, slower progression']]
        xlab = [i*len(unique_locations)+1, i*len(unique_locations)+2]
        bp = ax[0,0].boxplot(y, positions=xlab, widths = 0.6, showfliers=False)
        setBoxColors(bp)
    ax[0,0].set_xticks([1.5, 4.5, 7.5])
    ax[0,0].set_xticklabels(unique_locations)

    ax[0,0].set_title('HPV dwell-time')
    ax[0,0].set_ylabel('Years')

    for i, loc in enumerate(unique_locations):
        y = [dwelltimes['cin1'][loc], dwelltimes['cin1'][f'{loc}, slower progression']]
        xlab = [i*len(unique_locations)+1, i*len(unique_locations)+2]
        bp = ax[0,1].boxplot(y, positions=xlab, widths = 0.6, showfliers=False)
        setBoxColors(bp)
    ax[0,1].set_xticks([1.5, 4.5, 7.5])
    ax[0,1].set_xticklabels(unique_locations)
    ax[0,1].set_title('CIN1 dwell-time')

    for i, loc in enumerate(unique_locations):
        y = [dwelltimes['cin2'][loc], dwelltimes['cin2'][f'{loc}, slower progression']]
        xlab = [i*len(unique_locations)+1, i*len(unique_locations)+2]
        bp = ax[1,0].boxplot(y, positions=xlab, widths = 0.6, showfliers=False)
        setBoxColors(bp)
    ax[1,0].set_xticks([1.5, 4.5, 7.5])
    ax[1,0].set_xticklabels(unique_locations)
    ax[1,0].set_title('CIN2 dwell-time')

    for i, loc in enumerate(unique_locations):
        y = [dwelltimes['cin3'][loc], dwelltimes['cin3'][f'{loc}, slower progression']]
        xlab = [i*len(unique_locations)+1, i*len(unique_locations)+2]
        bp = ax[1,1].boxplot(y, positions=xlab, widths = 0.6, showfliers=False)
        setBoxColors(bp)
    ax[1,1].set_xticks([1.5, 4.5, 7.5])
    ax[1,1].set_xticklabels(unique_locations)
    ax[1,1].set_title('CIN3 dwell-time')

    for i, loc in enumerate(unique_locations):
        y = [dwelltimes['total'][loc], dwelltimes['total'][f'{loc}, slower progression']]
        xlab = [i*len(unique_locations)+1, i*len(unique_locations)+2]
        bp = ax[2,0].boxplot(y, positions=xlab, widths = 0.6, showfliers=False)
        setBoxColors(bp)
    ax[2,0].set_xticks([1.5, 4.5, 7.5])
    ax[2,0].set_xticklabels(unique_locations)
    ax[2,0].set_title('Total dwell-time')

    for i, loc in enumerate(unique_locations):
        y = [age_causal[loc], age_causal[f'{loc}, slower progression']]
        xlab = [i*len(unique_locations)+1, i*len(unique_locations)+2]
        bp = ax[2,1].boxplot(y, positions=xlab, widths = 0.6, showfliers=False)
        setBoxColors(bp)
    ax[2,1].set_xticks([1.5, 4.5, 7.5])
    ax[2,1].set_xticklabels(unique_locations)
    ax[2,1].set_title('Age of causal HPV infection')
    ax[2, 1].boxplot([], showfliers=False)

    # hB, = plot([1, 1], 'b-')
    # hR, = plot([1, 1], 'r-')
    # legend((hB, hR), ('Fast progression', 'Slow progression'))
    # hB.set_visible(False)
    # hR.set_visible(False)

    fig.suptitle('Underling natural history comparison')
    pl.tight_layout()
    fig.savefig(f'{ut.figfolder}/dwelltime.png')

    return

# %% Helper functions
def lognorm_params(par1, par2):
    """
    Given the mean and std. dev. of the log-normal distribution, this function
    returns the shape and scale parameters for scipy's parameterization of the
    distribution.
    """
    mean = np.log(par1 ** 2 / np.sqrt(par2 ** 2 + par1 ** 2))  # Computes the mean of the underlying normal distribution
    sigma = np.sqrt(np.log(par2 ** 2 / par1 ** 2 + 1))  # Computes sigma for the underlying normal distribution

    scale = np.exp(mean)
    shape = sigma
    return shape, scale

def logf1(x, k):
    '''
    The concave part of a logistic function, with point of inflexion at 0,0
    and upper asymptote at 1. Accepts 1 parameter which determines the growth rate.
    '''
    return (2 / (1 + np.exp(-k * x))) - 1

def logf2(x, x_infl, k):
    '''
    Logistic function, constrained to pass through 0,0 and with upper asymptote
    at 1. Accepts 2 parameters: growth rate and point of inlfexion.
    '''
    l_asymp = -1 / (1 + np.exp(k * x_infl))
    return l_asymp + 1 / (1 + np.exp(-k * (x - x_infl)))


def compute_progs(location=location):
    '''
    Compute progression rates
    '''
    
    sim = hpv.Sim(genotypes=[16,18,'hrhpv'])
    try:
        calib_pars = sc.loadobj(f'{ut.resfolder}/{location}_pars.obj')
    except:
        errormsg = 'Calibration parameters cannot be loaded from disk. Try running load_calib to generate them.'
        raise ValueError(errormsg)

    sim.initialize()  # Important to do this here, otherwise the genotype pars get overwritten
    sim.update_pars(calib_pars)

    # Get parameters
    ng = sim['n_genotypes']
    genotype_pars = sim['genotype_pars']
    genotype_map = sim['genotype_map']
    ccut1 = {'cin1': 0.33, 'cin2':0.67, 'cin3':0.99}
    # ccut1 = {'cin1': 0.85, 'cin2':0.95, 'cin3':0.99}

    # Shorten duration names
    dur_precin = [genotype_pars[genotype_map[g]]['dur_precin'] for g in range(ng)]
    dur_dysp = [genotype_pars[genotype_map[g]]['dur_dysp'] for g in range(ng)]
    dysp_rate = [genotype_pars[genotype_map[g]]['dysp_rate'] for g in range(ng)]
    prog_rate = [genotype_pars[genotype_map[g]]['prog_rate'] for g in range(ng)]
    prog_rate_sd = [genotype_pars[genotype_map[g]]['prog_rate_sd'] for g in range(ng)]


    # Figure settings
    font_size = 30
    sc.fonts(add=sc.thisdir(aspath=True) / 'Libertinus Sans')
    sc.options(font='Libertinus Sans')
    pl.rcParams['font.size'] = font_size
    colors = sc.gridcolors(ng)
    x = np.linspace(0.01, 3, 200)

    # %% Preliminary calculations (all to be moved within an analyzer? or sim method?)
    

    ###### Share of women who develop of detectable dysplasia by genotype
    shares = []
    gtypes = []
    for g in range(ng):
        sigma, scale = lognorm_params(dur_precin[g]['par1'], dur_precin[g]['par2'])
        rv = lognorm(sigma, 0, scale)
        aa = np.diff(rv.cdf(x))
        bb = logf1(x, dysp_rate[g])[1:]
        shares.append(np.dot(aa, bb))
        gtypes.append(genotype_map[g].upper())

    ###### Distribution of eventual outcomes for women by genotype
    noneshares, cin1shares, cin2shares, cin3shares, cancershares = [], [], [], [], []
    longx = np.linspace(0.01, 25, 1000)
    for g in range(ng):
        sigma, scale = lognorm_params(dur_dysp[g]['par1'], dur_dysp[g]['par2'])
        rv = lognorm(sigma, 0, scale)
        dd = logf1(longx, prog_rate[g])

        indcin1 = sc.findinds(dd < ccut1['cin1'])[-1]
        if (dd > ccut1['cin1']).any():
            indcin2 = sc.findinds((dd > ccut1['cin1']) & (dd < ccut1['cin2']))[-1]
        else:
            indcin2 = indcin1
        if (dd > ccut1['cin2']).any():
            indcin3 = sc.findinds((dd > ccut1['cin2']) & (dd < ccut1['cin3']))[-1]
        else:
            indcin3 = indcin2
        if (dd > ccut1['cin3']).any():
            indcancer = sc.findinds(dd > ccut1['cin3'])[-1]
        else:
            indcancer = indcin3

        noneshares.append(1 - shares[g])
        cin1shares.append(((rv.cdf(longx[indcin1]) - rv.cdf(longx[0])) * shares[g]))
        cin2shares.append(((rv.cdf(longx[indcin2]) - rv.cdf(longx[indcin1])) * shares[g]))
        cin3shares.append(((rv.cdf(longx[indcin3]) - rv.cdf(longx[indcin2])) * shares[g]))
        cancershares.append(((rv.cdf(longx[indcancer]) - rv.cdf(longx[indcin3])) * shares[g]))

    ######## Outcomes by duration of infection and genotype
    n_samples = 10e3

    # create dataframes
    data = {}
    years = np.arange(1, 26)
    cin1_shares, cin2_shares, cin3_shares, cancer_shares = [], [], [], []
    all_years = []
    all_genotypes = []
    for g in range(ng):
        sigma, scale = lognorm_params(dur_dysp[g]['par1'], dur_dysp[g]['par2'])

        for year in years:
            peaks = logf1(year, hpv.sample(dist='normal', par1=prog_rate[g], par2=prog_rate_sd[g], size=n_samples))
            cin1_shares.append(sum(peaks < ccut1['cin1']) / n_samples)
            cin2_shares.append(sum((peaks > ccut1['cin1']) & (peaks < ccut1['cin2'])) / n_samples)
            cin3_shares.append(sum((peaks > ccut1['cin2']) & (peaks < ccut1['cin3'])) / n_samples)
            cancer_shares.append(sum(peaks > ccut1['cin3']) / n_samples)
            all_years.append(year)
            all_genotypes.append(genotype_map[g].upper())
    data = {'Year': all_years, 'Genotype': all_genotypes, 'CIN1': cin1_shares, 'CIN2': cin2_shares, 'CIN3': cin3_shares,
            'Cancer': cancer_shares}
    sharesdf = pd.DataFrame(data)


    gtypes = ['hpv16', 'hpv18', 'hrhpv']

    ################################################################################
    # BEGIN FIGURE WITH PRECIN DISTRIBUTIONS
    ################################################################################
    fig, ax = pl.subplots(2, 2, figsize=(12, 12))

    pn = 0

    # Output table
    table = ' Type : % no dysp\n'

    for g, gtype in enumerate(gtypes):
        sigma, scale = lognorm_params(genotype_pars[gtype]['dur_precin']['par1'],
                                      genotype_pars[gtype]['dur_precin']['par2'])
        rv = lognorm(sigma, 0, scale)
        ax[0, 0].plot(x, rv.pdf(x), color=colors[pn], lw=2, label=gtype.upper())
        ax[1, 0].plot(x, logf1(x, genotype_pars[gtype]['dysp_rate']), color=colors[pn], lw=2,
                       label=gtype.upper())
        table += f"{gtype.upper().rjust(5)}: {100 - sum(np.diff(rv.cdf(x)) * logf1(x, dysp_rate[g])[1:]) * 100:.0f}\n"
        pn += 1

    ax[0, 0].legend(fontsize=18)
    ax[1, 0].set_xlabel("Duration of infection prior to\ncontrol or dysplasia (months)")
    for row in [0, 1]:
        ax[row, 0].set_ylabel("")
        ax[row, 0].grid(axis='x')
        ax[row, 0].set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        ax[row, 0].set_xticklabels([0, 6, 12, 18, 24, 30, 36])

        # ax[0, ai].get_yaxis().set_ticks([])

    ###### Relationship between duration of dysplasia and clinical severity
    pn = 0
    thisx = np.linspace(0.01, 25, 100)

    cmap = pl.cm.Oranges([0.25, 0.5, 0.75, 1])
    n_samples = 1
    for g, gtype in enumerate(gtypes):
        sigma, scale = lognorm_params(genotype_pars[gtype]['dur_dysp']['par1'],
                                      genotype_pars[gtype]['dur_dysp']['par2'])
        rv = lognorm(sigma, 0, scale)
        ax[0, 1].plot(thisx, rv.pdf(thisx), color=colors[pn], lw=2, label=gtype.upper())
        ax[1, 1].plot(thisx, logf1(thisx, prog_rate[g]), color=colors[pn], lw=2,
                           label=gtype.upper())
        # Plot variation
        for year in range(1, 26):
            peaks = logf1(year, hpv.sample(dist='normal', par1=prog_rate[g],
                                           par2=genotype_pars[gtype]['prog_rate_sd'], size=n_samples))
            ax[1, 1].plot([year] * n_samples, peaks, color=colors[pn], lw=0, marker='o', alpha=0.5)
        pn += 1

    ax[0, 1].legend(fontsize=18)
    ax[1, 1].set_xlabel("Duration of dysplasia (years)")
    ax[1, 1].set_ylabel("")
    ax[0, 1].grid(axis='x')
    ax[1, 1].grid(axis='x')
    ax[1, 1].get_yaxis().set_ticks([])
    ax[1, 1].axhline(y=ccut1['cin1'], ls=':', c='k')
    ax[1, 1].axhline(y=ccut1['cin2'], ls=':', c='k')
    ax[1, 1].axhline(y=ccut1['cin3'], ls=':', c='k')
    ax[1, 1].axhspan(0, ccut1['cin1'], color=cmap[0], alpha=.4)
    ax[1, 1].axhspan(ccut1['cin1'], ccut1['cin2'], color=cmap[1], alpha=.4)
    ax[1, 1].axhspan(ccut1['cin2'], ccut1['cin3'], color=cmap[2], alpha=.4)
    ax[1, 1].axhspan(ccut1['cin3'], 1, color=cmap[3], alpha=.4)
    ax[1, 1].text(-0.3, 0.08, 'CIN1', fontsize=30, rotation=90)
    ax[1, 1].text(-0.3, 0.4, 'CIN2', fontsize=30, rotation=90)
    ax[1, 1].text(-0.3, 0.73, 'CIN3', fontsize=30, rotation=90)

    fig.tight_layout()
    sc.savefig(f'{ut.figfolder}/{location}_prognoses.png', dpi=100)

    fig, ax = pl.subplots(1, 2, figsize=(26, 8))
    cmap = pl.cm.Oranges([0.25, 0.5, 0.75, 1])

    ###### Share of women who develop each CIN grade
    loc_array = np.array([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6])
    w = 0.07
    years = np.arange(1,13)
    for y in years:
        la = loc_array[y - 1] * w + np.sign(loc_array[y - 1]) * (-1) * w / 2
        bottom = np.zeros(ng)
        for gn, grade in enumerate(['CIN1', 'CIN2', 'CIN3', 'Cancer']):
            ydata = sharesdf[sharesdf['Year'] == y][grade]
            ax[0].bar(np.arange(1, ng + 1) + la, ydata, width=w, color=cmap[gn], bottom=bottom, edgecolor='k',
                      label=grade);
            bottom = bottom + ydata

    # ax[1,1].legend()
    ax[0].set_title("Share of women with dysplasia\nby clinical grade and duration")
    ax[0].set_xlabel("")
    ax[0].set_xticks(np.arange(ng) + 1)
    ax[0].set_xticklabels(gtypes, fontsize=30)

    ##### Final outcomes for women
    bottom = np.zeros(ng + 1)
    all_shares = [noneshares + [sum([j * 1 / ng for j in noneshares])],
                  cin1shares + [sum([j * 1 / ng for j in cin1shares])],
                  cin2shares + [sum([j * 1 / ng for j in cin2shares])],
                  cin3shares + [sum([j * 1 / ng for j in cin3shares])],
                  cancershares + [sum([j * 1 / ng for j in cancershares])],
                  ]
    for gn, grade in enumerate(['None', 'CIN1', 'CIN2', 'CIN3', 'Cancer']):
        ydata = np.array(all_shares[gn])
        if len(ydata.shape) > 1: ydata = ydata[:, 0]
        color = cmap[gn - 1] if gn > 0 else 'gray'
        ax[1].bar(np.arange(1, ng + 2), ydata, color=color, bottom=bottom, label=grade)
        bottom = bottom + ydata
    ax[1].set_xticks(np.arange(ng + 1) + 1)
    ax[1].set_xticklabels(gtypes + ['Average'], fontsize=30)
    ax[1].set_ylabel("")
    ax[1].set_title("Eventual outcomes for women\n")
    ax[1].legend(bbox_to_anchor=(0.5, 1.07), loc='upper center', fontsize=20, ncol=5, frameon=False)
    # ax[1].legend(bbox_to_anchor =(1.2, 1.),loc='upper center',fontsize=30,ncol=1,frameon=False)

    pl.figtext(0.04, 0.85, 'A', fontweight='bold', fontsize=30)
    pl.figtext(0.51, 0.85, 'B', fontweight='bold', fontsize=30)

    fig.tight_layout()
    sc.savefig(f'{ut.figfolder}/{location}_cin_outcomes.png', dpi=100)


#%% Run as a script
if __name__ == '__main__':

    T = sc.timer()
    ut.set_font(size=30)

    dwelltimes = []
    dwelltimes_cancer = []
    for ccut in [False, True]:
        sim, dt, dt_cancer = run_sim(location='india', use_ccut=ccut)
        dwelltimes.append(dt)
        dwelltimes_cancer.append(dt_cancer)

    # fig, ax = pl.subplots(3, 2, figsize=(12, 12), sharey=True)
    # ax[0,0].boxplot([dwelltimes[0].dwelltime['hpv'],dwelltimes[1].dwelltime['hpv']], widths=0.6, showfliers=False)
    # ax[0,0].set_title('HPV dwelltime')
    # ax[0,0].set_xticklabels(['', ''])
    #
    # ax[0,1].boxplot([dwelltimes[0].dwelltime['cin1'],dwelltimes[1].dwelltime['cin1']], widths=0.6, showfliers=False)
    # ax[0,1].set_title('CIN1 dwelltime')
    # ax[0,1].set_xticklabels(['', ''])
    #
    # ax[1,0].boxplot([dwelltimes[0].dwelltime['cin2'],dwelltimes[1].dwelltime['cin2']], widths=0.6, showfliers=False)
    # ax[1,0].set_title('CIN2 dwelltime')
    # ax[1,0].set_xticklabels(['Fast prog', 'Slow prog'])
    #
    # ax[1,1].boxplot([dwelltimes[0].dwelltime['cin3'],dwelltimes[1].dwelltime['cin3']], widths=0.6, showfliers=False)
    # ax[1,1].set_title('CIN3 dwelltime')
    # ax[1,1].set_xticklabels(['Fast prog', 'Slow prog'])
    #
    # ax[2,0].boxplot([dwelltimes[0].dwelltime['total'], dwelltimes[1].dwelltime['total']], widths=0.6, showfliers=False)
    # ax[2,0].set_title('Total dwelltime')
    # ax[2,0].set_xticklabels(['Fast prog', 'Slow prog'])
    # fig.suptitle('Dwelltime for non-cancer causing HPV/Dysplasia')
    # pl.tight_layout()
    # fig.show()

    fig, ax = pl.subplots(3, 2, figsize=(12, 12))
    ax[0, 0].boxplot([dwelltimes_cancer[0].dwelltime['hpv'], dwelltimes_cancer[1].dwelltime['hpv']], widths=0.6, showfliers=False)
    ax[0, 0].set_title('HPV dwelltime')
    ax[0, 0].set_xticklabels(['Fast prog', 'Slow prog'])

    ax[0, 1].boxplot([dwelltimes_cancer[0].dwelltime['cin1'], dwelltimes_cancer[1].dwelltime['cin1']], widths=0.6, showfliers=False)
    ax[0, 1].set_title('CIN1 dwelltime')
    ax[0, 1].set_xticklabels(['Fast prog', 'Slow prog'])

    ax[1, 0].boxplot([dwelltimes_cancer[0].dwelltime['cin2'], dwelltimes_cancer[1].dwelltime['cin2']], widths=0.6, showfliers=False)
    ax[1, 0].set_title('CIN2 dwelltime')
    ax[1, 0].set_xticklabels(['Fast prog', 'Slow prog'])

    ax[1, 1].boxplot([dwelltimes_cancer[0].dwelltime['cin3'], dwelltimes_cancer[1].dwelltime['cin3']], widths=0.6, showfliers=False)
    ax[1, 1].set_title('CIN3 dwelltime')
    ax[1, 1].set_xticklabels(['Fast prog', 'Slow prog'])

    ax[2, 0].boxplot([dwelltimes_cancer[0].dwelltime['total'], dwelltimes_cancer[1].dwelltime['total']], widths=0.6, showfliers=False)
    ax[2, 0].set_title('Total dwelltime')
    ax[2, 0].set_xticklabels(['Fast prog', 'Slow prog'])

    ax[-1, -1].axis('off')
    fig.suptitle('Dwelltime for cancer causing HPV/Dysplasia')
    pl.tight_layout()
    fig.show()
    # test_locations = ['nigeria', 'nigeria', 'india', 'india', 'tanzania', 'tanzania']
    # use_ccut = [False, True, False, True, False, True]
    # run_age_causal_analyzer(locations=test_locations, use_ccut=use_ccut, debug=1)
    # for location in ['india', 'nigeria', 'tanzania']:
    #     compute_progs(location=location)
    # test_locations = ['nigeria', 'nigeria']
    # use_ccut = [False, True]

    T.toc('Done')
