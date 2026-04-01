# Physics Simulations

This repository contains numerical simulations and visualizations of physical systems, focusing on statistical physics and quantum mechanics.

## Structure

### 1. 2D Ising Model
Simulation of the two-dimensional Ising model for studying phase transitions and magnetization behavior.

Features:
- Monte Carlo simulation (e.g. Metropolis algorithm)
- Visualization of spin configurations
- Analysis of thermodynamic quantities

### 2. Quantum Mechanics
Numerical solutions of the 1D Schrödinger equation.

#### Double-Well Potential
The Schrödinger equation is solved for an asymmetric double-well potential

V(x) = x^4 - x^2 - A x

Includes:
- Time-independent solution:
  - Eigenvalues and eigenfunctions via matrix diagonalization
- Time-dependent solution:
  - Time evolution of Gaussian wave packets
  - Projection onto eigenstates
  - Visualization of dynamics

## Requirements

- numpy
- matplotlib
- scipy
