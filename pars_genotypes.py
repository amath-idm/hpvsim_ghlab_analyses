'''
List parameters used for genotypes in calibrations
'''

# Define key parameters -- values are [best, low, high]
default_sev_rate = [0.5, 0.2, 0.7] # Logistic growth curve parameter (assumption)
default_dur_episomal  = dict(par1=[6, 4, 12]) # Mean duration of dysplasia (years) for lognormal distribution
default_transform_prob = [0.0001, 0.01, 0.2]
def get_genotype_pars(location):
    ''' Define genotype parameters '''

    all_genotype_pars = dict(
        hpv16=dict(
            sev_rate=dict(
                india    = default_sev_rate,
                nigeria  = default_sev_rate,
                tanzania = default_sev_rate,
            ),
            dur_episomal=dict(
                india    = default_dur_episomal,
                nigeria  = default_dur_episomal,
                tanzania = default_dur_episomal,
            ),
        ),
        hpv18=dict(
            sev_rate=dict(
                india    = default_sev_rate,
                nigeria  = default_sev_rate,
                tanzania = default_sev_rate,
            ),
            dur_episomal=dict(
                india    = default_dur_episomal,
                nigeria  = default_dur_episomal,
                tanzania = default_dur_episomal,
            ),

        ),
        hrhpv=dict(
            sev_rate=dict(
                india    = default_sev_rate,
                nigeria  = default_sev_rate,
                tanzania = default_sev_rate,
            ),
            dur_episomal=dict(
                india    = default_dur_episomal,
                nigeria  = default_dur_episomal,
                tanzania = default_dur_episomal,
            ),

        )
    )

    # Pull out parameters for a single location
    genotype_pars = {genotype: {par:value[location] for par,value in all_genotype_pars[genotype].items()} for genotype in all_genotype_pars.keys()}

    return genotype_pars
