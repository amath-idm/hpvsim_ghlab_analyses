'''
List parameters used for genotypes in calibrations
'''

# Define key parameters -- values are [best, low, high]
default_dur_precin = dict(par1=[1.5, 1, 3]) # Mean duration of HPV infection (years)
default_prog_rate = [0.3, 0.05, 0.3] # Logistic growth curve parameter (assumption)
default_dur_dysp  = dict(par1=[10, 7, 15]) # Mean duration of dysplasia (years) for lognormal distribution
default_cancer_prob = [0.025, 0.01, 0.2]
def get_genotype_pars(location):
    ''' Define genotype parameters '''

    all_genotype_pars = dict(
        hpv16=dict(
            dur_precin=dict(
                india    = default_dur_precin,
                nigeria  = default_dur_precin,
                tanzania = default_dur_precin,
            ),
            prog_rate=dict(
                india    = default_prog_rate,
                nigeria  = default_prog_rate,
                tanzania = default_prog_rate,
            ),
            dur_dysp=dict(
                india    = default_dur_dysp,
                nigeria  = default_dur_dysp,
                tanzania = default_dur_dysp,
            ),
            cancer_prob=dict(
                india    = default_cancer_prob,
                nigeria  = default_cancer_prob,
                tanzania = default_cancer_prob,
            )
        ),
        hpv18=dict(
            dur_precin=dict(
                india=default_dur_precin,
                nigeria=default_dur_precin,
                tanzania=default_dur_precin,
            ),
            prog_rate=dict(
                india    = default_prog_rate,
                nigeria  = default_prog_rate,
                tanzania = default_prog_rate,
            ),
            dur_dysp=dict(
                india    = default_dur_dysp,
                nigeria  = default_dur_dysp,
                tanzania = default_dur_dysp,
            ),
            cancer_prob=dict(
                india=default_cancer_prob,
                nigeria=default_cancer_prob,
                tanzania=default_cancer_prob,
            )
        ),
        hrhpv=dict(
            dur_precin=dict(
                india=default_dur_precin,
                nigeria=default_dur_precin,
                tanzania=default_dur_precin,
            ),
            prog_rate=dict(
                india    = default_prog_rate,
                nigeria  = default_prog_rate,
                tanzania = default_prog_rate,
            ),
            dur_dysp=dict(
                india    = default_dur_dysp,
                nigeria  = default_dur_dysp,
                tanzania = default_dur_dysp,
            ),
            cancer_prob=dict(
                india=default_cancer_prob,
                nigeria=default_cancer_prob,
                tanzania=default_cancer_prob,
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
