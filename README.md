# Reimagining Blackjack with Double Q-Learning (MLPR S4 Project)

> Automating hit/stand/double decisions in Blackjack with reinforcement learning to minimize risk and outperform basic strategy.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white" />
  <img alt="Gymnasium" src="https://img.shields.io/badge/Gymnasium-Blackjack--v1-1f425f.svg" />
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg" />
</p>

## 🔎 Overview

Blackjack mixes skill, strategy, and uncertainty. We train an RL agent (Double Q-Learning) in the OpenAI Gym/Gymnasium Blackjack environment to learn risk-aware policies that improve on baseline play without relying on a labeled dataset. The agent learns purely from interaction—balancing exploration vs. exploitation and optimizing long-term reward.

**Why RL?** Unlike supervised/unsupervised methods, RL directly optimizes returns under uncertainty, adapts to dynamic gameplay, and learns from consequences of actions.

## 🧠 Method

**Double Q-Learning** reduces the classic Q-learning overestimation bias by maintaining two Q-tables \(Q¹, Q²\).  
At each update, one table selects the greedy action, and the other evaluates it:

- Action selection: \( a^* = \arg\max_a Q^{\text{sel}}(s', a) \)  
- Target evaluation: \( r + \gamma \, Q^{\text{eval}}(s', a^*) \)

We use Gymnasium’s `Blackjack-v1` environment with the state space:
- Player sum, Dealer upcard, Usable Ace flag, Bust flag, Reward signal  
…and action space: **Hit**, **Stand**, **Double Down**.

▶️ Quickstart

Train the agent

Open doubleqfinalmlpr.ipynb.

Set training episodes and random seed.

Run all cells to learn Q-tables and export the learned policy/value heatmaps (if enabled in the notebook).

Hyperparameter tuning

Open hyperparametertuning.ipynb.

Define the search grid for α (alpha), ε (epsilon), and γ (gamma).

Run to evaluate combos; copy the best set back into the training notebook.

Evaluate

Open testingmodel.ipynb.

Run 100k+ simulated rounds using:

Basic strategy (baseline)

Learned policy (ours)

Compare win/loss, return, and policy score metrics.

🔧 Key Training Details

Episodes / Rounds: ~20,000,000 simulated hands during development (configurable).

Best hyperparameters (from tuning):

Alpha (α): 0.1

Epsilon (ε): 0.2

Gamma (γ): 0.8

Exploration schedule: ε-greedy (constant or decayed—see notebook).

Reproducibility: set np.random.seed(...) and environment seeds in notebooks.


For more details look at the attached presenation and codes provided in the repository.
