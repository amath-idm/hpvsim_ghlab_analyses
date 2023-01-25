'''
List parameters used for genotypes in calibrations
'''

# Define key parameters -- values are [best, low, high]
default_dysp_rate = [0.17, 0.05, 0.3] # Logistic growth curve parameter (assumption)
default_dur_dysp  = dict(par1=[6, 4, 12]) # Mean duration of dysplasia (years) for lognormal distribution
default_transform_prob = [0.0001, 0.01, 0.2]
def get_genotype_pars(location):
    ''' Define genotype parameters '''

    all_genotype_pars = dict(
        hpv16=dict(
            dysp_rate=dict(
                india    = default_dysp_rate,
                nigeria  = default_dysp_rate,
                tanzania = default_dysp_rate,
            ),
            dur_dysp=dict(
                india    = default_dur_dysp,
                nigeria  = default_dur_dysp,
                tanzania = default_dur_dysp,
            ),
        ),
        hpv18=dict(
            dysp_rate=dict(
                india    = default_dysp_rate,
                nigeria  = default_dysp_rate,
                tanzania = default_dysp_rate,
            ),
            dur_dysp=dict(
                india    = default_dur_dysp,
                nigeria  = default_dur_dysp,
                tanzania = default_dur_dysp,
            ),

        ),
        hrhpv=dict(
            dysp_rate=dict(
                india    = default_dysp_rate,
                nigeria  = default_dysp_rate,
                tanzania = default_dysp_rate,
            ),
            dur_dysp=dict(
                india    = default_dur_dysp,
                nigeria  = default_dur_dysp,
                tanzania = default_dur_dysp,
            ),

        )
    )

    # Pull out parameters for a single location
    genotype_pars = {genotype: {par:value[location] for par,value in all_genotype_pars[genotype].items()} for genotype in all_genotype_pars.keys()}

    return genotype_pars
