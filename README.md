# NovoFEL_setup_manage


This package provides a modular framework for applying Bayesian optimization to complex physical systems, such as Free Electron Lasers (FELs), with support for uncertainty-aware modeling, experiment tracking, and intelligent decision making.

## 🚀 Features

- Adaptive Bayesian optimization using Gaussian Processes (via **BoTorch** and **GPyTorch**)
- Centralized experiment tracking (support for **MLflow**)
- History-aware model acceleration using deep neural networks (**PyTorch**)
- Interactive user interface for control and monitoring (**PyQt**)
- Noise-aware optimization in high-cost experimental environments
- Plug-and-play architecture for custom objective functions and constraints

---

## 🧠 Current Modules

| Module            | Description                                                       |
|-------------------|-------------------------------------------------------------------|
| `analysis/`       | Methods to analyse obtained data                                  |
| `manage/`         | Manage system for Free Electron Laser                             |
| `manage/agents`   | Agents to manage parts of Free Electron Laser                     |


---
