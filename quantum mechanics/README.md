# Quantum Mechanics

Numerical solutions of the one-dimensional Schrödinger equation with a focus on an asymmetric double-well potential.

## Overview

This project implements numerical methods to study quantum systems in one spatial dimension.  
Both the time-independent and time-dependent Schrödinger equation are solved and visualized.

## Potential

The system is described by an asymmetric double-well potential

V(x) = x^4 - x^2 - A x

where A controls the asymmetry of the potential.

## Periodic Potential

The project also includes a simulation of a particle in a periodic potential

\[
V(x) = A \cos(2\pi x)
\]

The program computes the energy spectrum as a function of the Bloch wave vector and visualizes the corresponding eigenfunctions, leading to the formation of energy bands.

Periodic potentials occur in crystal lattices in solid-state physics and describe the behavior of charge carriers in solids.


## Contents

### 1. Time-independent Schrödinger equation

- Construction of the Hamiltonian using finite difference discretization
- Computation of eigenvalues and eigenfunctions
- Visualization of eigenstates in the potential

### 2. Time-dependent Schrödinger equation

- Initialization of Gaussian wave packets
- Expansion in eigenstates
- Time evolution of the wave function
- Interactive visualization via plot input

### 3. Periodic potential (band structure)

- Calculation of energy bands as a function of k  
- Implementation of Bloch boundary conditions  
- Interactive visualization of eigenfunctions
  
## Files

- `quantenmechanik.py`  
  Core numerical methods:
  - spatial discretization
  - Hamiltonian construction
  - diagonalization
  - plotting of eigenfunctions

- `time-independent_schroedinger_eq.py`  
  Solves the stationary Schrödinger equation and visualizes eigenstates

- `time-dependent_Schrödinger_eq.py`  
  Simulates the time evolution of wave packets in the potential

## Methods

- Finite difference method for spatial discretization
- Matrix representation of the Hamiltonian
- Eigenvalue problem solved using `scipy.linalg.eigh`
- Time evolution via expansion in eigenstates

## Requirements

- numpy
- matplotlib
- scipy

## Usage

Run one of the scripts, for example:

```bash
python time-dependent_Schrödinger_eq.py
