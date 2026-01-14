# CartPole

Train and evaluate DQN agents for `CartPole-v1`, then test robustness under noisy observations and delayed actions. The notebook includes a baseline agent, a harder evaluation environment, and a domain-randomized training setup for a more robust agent.

## What's inside
- Baseline DQN training on clean `CartPole-v1`.
- Evaluation on clean and noisy/delayed environments.
- Domain randomization with observation noise and action delay.
- Saved models: `dqn_cartpole_baseline` and `dqn_cartpole_robust` (notebook outputs).

## Requirements
- Python 3.9+
- `gymnasium`
- `stable-baselines3`
- `numpy`

Install dependencies:
```bash
pip install gymnasium stable-baselines3 numpy
```

## How to run
Open the notebook and run cells in order:
```bash
jupyter notebook Untitled-1.ipynb
```

Key steps in the notebook:
1. Train baseline DQN and save `dqn_cartpole_baseline`.
2. Evaluate baseline on clean vs noisy/delayed environments.
3. Train a robust DQN with domain randomization and save `dqn_cartpole_robust`.

## Files
- `Untitled-1.ipynb`: training and evaluation code.
- `dqn_cartpole_baseline.zip`: saved baseline model (if present).
- `README.md`: project overview and usage.
