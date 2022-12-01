'''
Run economic analysis.

'''


#%% General settings

# Standard imports
import numpy as np
import pandas as pd
import sciris as sc
import hpvsim as hpv

# Imports from this repository
import run_sim as rs
import utils as ut
import pars_scenarios as sp

# Comment out to not run
to_run = [
    # 'run_scenarios',
    'plot_scenarios',
]

india = pd.read_csv(f'results/india_econ.csv')
nigeria = pd.read_csv(f'results/nigeria_econ.csv')
tanzania = pd.read_csv(f'results/tanzania_econ.csv')

cost_params = pd.DataFrame()

# Nigeria costs (ref 1)
# HPV DNA testing: financial cost of US$ 36 per service.
# VIA: financial cost of US$ 13 per service.
# Thermal ablation: financial cost of US$ 3.50 per service
# LEEP: financial cost of US$ 107 per service.

# India costs: mean (SD) (ref 4)
# HPV DNA testing: US$ 14.8 ($3.8 SD).
# VIA: $5.2 ($1.3).
# Thermal ablation:
# LEEP: $90.3 ($23).

# References
# 1. (2020 USD) World Health Organization. (2020). Costing the National Strategic Plan on Prevention and Control of Cervical Cancer: Nigeria, 2017 –2021November 2020.
# 2. (2016 USD) Nelson, S., Kim, J., Wilson, F. A., Soliman, A. S., Ngoma, T., Kahesa, C., &#38; Mwaiselage, J. (2016). Cost-Effectiveness of Screening and Treatment for Cervical Cancer in Tanzania: Implications for other Sub-Saharan African Countries. <i>Value in Health Regional Issues</i>, <i>10</i>, 1–6. https://doi.org/10.1016/j.vhri.2016.03.002</div>
# 3. (2019 USD) Hsiao, A., Struckmann, V., Stephani, V., Mbbando, D., Changalucha, J., Baisley, K., Levin, A., Morgan, W., Hutubessy, R., Watson -Jones, D., Whitworth, H., & Quentin, W. (2022). Costs of delivering human papillomavirus vaccination using a one or two dose strategy in Tanzania. https://ssrn.com/abstract=4208243
# 4. (2016 USD) Chauhan, A. S., Prinja, S., Srinivasan, R., Rai, B., Malliga, J. S., Jyani, G., Gupta, N., &#38; Ghoshal, S. (2020). Cost effectiveness of strategies for cervical cancer prevention in India. <i>PLoS ONE</i>, <i>15</i>(9 September). https://doi.org/10.1371/journal.pone.0238291</div>
