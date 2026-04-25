# WIDSS

AI-driven battery state estimation starter project focused on:

- **SOC (State of Charge)** prediction under dynamic drive cycles
- **SoH (State of Health)** direction-ready structure for future cycle-aging work

## What this repo now includes

- Synthetic EV-like dynamic current profile generator (acceleration, cruise, regen, idle)
- Physics-inspired SOC and terminal-voltage simulation pipeline
- Time-series sequence builder for deep learning
- LSTM training entry script for SOC prediction (TensorFlow optional)
- Basic tests for simulation and dataset preparation

## Project structure

- `/home/runner/work/WIDSS/WIDSS/src/widss/simulation.py` - drive cycle + battery state simulation
- `/home/runner/work/WIDSS/WIDSS/src/widss/dataset.py` - windowed time-series dataset builder
- `/home/runner/work/WIDSS/WIDSS/src/widss/model.py` - LSTM model construction helpers
- `/home/runner/work/WIDSS/WIDSS/scripts/train_soc_lstm.py` - end-to-end training script
- `/home/runner/work/WIDSS/WIDSS/tests/` - basic test coverage

## Setup

```bash
cd /home/runner/work/WIDSS/WIDSS
python -m pip install -r requirements.txt
```

If you want to train the LSTM model, install TensorFlow separately:

```bash
python -m pip install tensorflow
```

## Run tests

```bash
cd /home/runner/work/WIDSS/WIDSS
python -m pytest
```

## Train SOC LSTM

```bash
cd /home/runner/work/WIDSS/WIDSS
PYTHONPATH=src python scripts/train_soc_lstm.py --duration-s 7200 --window-size 30 --epochs 5
```

## Next recommended steps

- Add PyBaMM-based simulation backend and compare with the synthetic generator
- Add Transformer baseline for SOC and long-horizon SoH pipeline
- Add physics-informed loss terms and evaluate robustness under noisy drive cycles
