# Dual Algorithmic Reasoning
Official code for the paper `Dual Algorithmic Reasoning`

## Dependencies
`reqs/*` contains requirements files for `linux-cpu`, `macos-cpu` and `linux-gpu`


## Usage

The two main files `build_data.py` and `run.py` have a Typer CLI. Refer to each script's commands by running

`python <filename>.py --help`

### Data
`python build_data.py [PARAMS]` creates training/validation/test data and should be the first command you run.

### Train/Test
`python run.py [COMMAND] [PARAMS]` runs either the model selection (`valid`) or the risk assessment (`test`).

## Structure
- `config/*` contains configurations for data generation, model hyperparameters (included the optimal ones) and general configs.
- `config/exp/*` contains the YAML files for reproducing the experiments shown in the paper (use the `*.one.yaml` versions).
- `nn/*` contains all the model implementations.
