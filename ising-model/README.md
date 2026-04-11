# 2D Ising Model

Numerical simulation of the two-dimensional Ising model using Monte Carlo methods.

## Overview

This project simulates a 2D spin system on an N × N lattice to study phase transitions and magnetization behavior as a function of temperature.

The implementation includes an interactive visualization of:
- spin configurations
- magnetization as a function of temperature

## Model

Each lattice site contains a spin.  
The system evolves according to the Metropolis algorithm.

Periodic boundary conditions are applied.

## Features

- Random initialization of spin configurations with a given magnetization
- Monte Carlo updates using the Metropolis criterion
- Interactive control via mouse input:
  - Click in the right plot: set temperature \( \tau \) and initial magnetization \( m \)
  - Click in the left plot: perform Monte Carlo steps
- Real-time visualization of:
  - spin field
  - magnetization vs. temperature

## Requirements

- numpy
- matplotlib
