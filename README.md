# Lorenz63 HAVOK Control

Modeling the HAVOK forcing signal on Lorenz-63, from simple forecasting baselines to conditional and response-aware chaotic control.

## What This Repo Is About

This repository studies a simple but important question:

Can we learn the HAVOK forcing term well enough to move from passive prediction to controllable chaotic behavior?

The project started with three forcing-model baselines:

- deterministic autoregression
- probabilistic Gaussian autoregression
- Markov modeling

## Current Research Direction

The current thesis framing is:

- HAVOK provides a linear skeleton for chaotic dynamics
- the forcing term acts as the learned closure
- conditioning that forcing on flip schedules turns the problem into a control problem

In short:

- first: predict the forcing
- now: generate the forcing to shape the Lorenz switching pattern

## Repository Structure

```text
papers/        Paper reading notes and summaries
experiments/   Runnable experiments, metrics, figures, and result tables
src/           Reusable Python code for HAVOK control experiments
data/          Raw and processed data files
report/        Thesis/report-style writing and final narrative
```
