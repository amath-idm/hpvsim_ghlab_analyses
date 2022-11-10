'''
List parameters used for genotypes in calibrations
'''

# Define key parameters -- values are [best, low, high]
default_dysp_rate = [1.5, 0.75, 1.75] # Logistic growth curve parameter (assumption)
default_prog_rate = [0.3, 0.1, 0.3] # Logistic growth curve parameter (assumption)
default_dur_disp  = dict(par1=[4.5, 3, 8]) # Mean duration of dysplasia (years) for lognormal distribution

def get_genotype_pars(location):
    ''' Define genotype parameters '''

    all_genotype_pars = dict(
        hpv16=dict(
            dysp_rate=dict(
                india    = default_dysp_rate,
                nigeria  = default_dysp_rate,
                tanzania = default_dysp_rate,
            ),
            prog_rate=dict(
                india    = default_prog_rate,
                nigeria  = default_prog_rate,
                tanzania = default_prog_rate,
            ),
            dur_dysp=dict(
                india    = default_dur_disp,
                nigeria  = default_dur_disp,
                tanzania = default_dur_disp,
            ),
        ),
        hpv18=dict(
            dysp_rate=dict(
                india    = default_dysp_rate,
                nigeria  = default_dysp_rate,
                tanzania = default_dysp_rate,
            ),
            prog_rate=dict(
                india    = default_prog_rate,
                nigeria  = default_prog_rate,
                tanzania = default_prog_rate,
            ),
            dur_dysp=dict(
                india    = default_dur_disp,
                nigeria  = default_dur_disp,
                tanzania = default_dur_disp,
            ),
        ),
        hrhpv=dict(
            dysp_rate=dict(
                india    = default_dysp_rate,
                nigeria  = default_dysp_rate,
                tanzania = default_dysp_rate,
            ),
            prog_rate=dict(
                india    = default_prog_rate,
                nigeria  = default_prog_rate,
                tanzania = default_prog_rate,
            ),
            dur_dysp=dict(
                india    = dict(par1=[1.5, 1, 8]),
                nigeria  = dict(par1=[4.5, 1, 8]),
                tanzania = dict(par1=[1.5, 1, 8]),
            ),
            rel_beta=dict(
                india    = [1, 0.8, 1.2],
                nigeria  = [1, 0.8, 1.2],
                tanzania = [1, 0.8, 1.2],

            ),
        )
    )

    # Pull out parameters for a single location
    genotype_pars = {genotype: {par:value[location] for par,value in all_genotype_pars[genotype].items()} for genotype in all_genotype_pars.keys()}

    return genotype_pars
