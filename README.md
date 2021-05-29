# Replicating the experiments
1. Generate the commands for the desired experiment using the `scripts/fcmi_scripts.py` script.
2. Parse the result of the experiment using the `scripts/fcmi_parse_results.py` script.
3. Use the `notebooks/fcmi-plots.ipynb` to generate plots from the parsed results.

# Requirements
* Basic libraries such as `numpy`, `scipy`, `tqdm`, `matplotlib`, and `seaborn`.
* We used `Pytorch 1.7.0`, but higher versions should work too.
