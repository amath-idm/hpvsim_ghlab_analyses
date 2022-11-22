# hpvsim_ghlab_analyses

Repository set up for Steve Kern and colleagues at Global Health Labs to model improved triage/diagnostics for cervical cancer screening.

Two scenarios to model:
1. Impact on elimination targets with an AVE with an accurate means to detect CIN2/3 and ablate. Compare to VIA to assess differential value.
2. Value of a diagnostic with NAAT detection ability in a disposable LFA-type package that costs <$2 and gives an answer in 15 mins. Compare to TxVx with mass vaccination of 25-45 year olds. Would it make a difference if this test could do genotyping?

To run analyses and reproduce results:

Part 1: AVE test characteristics
1. Run `run_sweeps.py` with `run_sweeps` uncommented
2. Run `run_sweeps.py` with `plot_sweeps` uncommented

Part 2: POC HPV+AVE vs. TxVx 
TODO
