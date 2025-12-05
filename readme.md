# Genetic Algorithm for the Traveling Salesperson Problem (TSP)

This project implements a modular Genetic Algorithm (GA) engine in Python designed to solve the Traveling Salesperson Problem (TSP). It includes a full experimental testbed to compare the performance of different genetic operators (Selection, Crossover, and Mutation).

## Project Overview

The goal of this project is to find a near-optimal route for visiting a set of 50 randomly generated cities exactly once and returning to the start. The system uses an evolutionary approach to search the O(n!) solution space efficiently.

### Key Features

-   Modular Design: The core GA logic (ga_engine.py) is decoupled from the experimental setup, allowing for easy swapping of algorithms.
    
-   Optimized Performance: Pre-computes the $N \times N$ distance matrix to reduce fitness evaluation complexity from $O(n^2)$ to $O(n)$.
    
-   Comparative Analysis: Performs a full-factorial Grid Search to test different operator combinations.
    
-   Visualization: Includes dedicated scripts to plot the final route map and convergence graphs.
    

## Algorithms Implemented

This project compares multiple strategies for each step of the evolutionary process:

Step/epochs/generations

Algorithms Tested


Selection

Tournament vs. Roulette

Compares high selection pressure (Tournament) against probabilistic selection (Roulette).

Crossover

Order (OX) vs. PMX

Compares preserving relative order (OX) against preserving absolute position (PMX).

Mutation

Swap vs. Inversion

Compares swapping two cities against reversing a sub-sequence of the route.

## Installation

1.  Clone the repository:  
    git clone [https://github.com/Bryan-Julius/Genetic_Algorithm_Traveling_Salesman.git](https://github.com/Bryan-Julius/Genetic_Algorithm_Traveling_Salesman.git)  
    cd Genetic_Algorithm_Traveling_Salesman  
      
    
2.  Install dependencies:  
    It is recommended to use a virtual environment.  
    pip install -r requirements.txt  
      
    (Note: Key dependencies are numpy, pandas, matplotlib, and networkx)
    

## Usage Guide

This project is designed to run as a pipeline:

### 1. Run the Experiment

Run the main driver script to execute the Grid Search (32 different experiments) and generate the data.

python experiment_runner.py  
  

-   Output: Generates parameter_impact_results.csv, best_route.pkl, and cities.pkl.
    
-   Note: This script uses a fixed random seed (42) to ensure every algorithm solves the exact same problem instance. Feel free to change this because the script maintains the seed for each test.
    

### 2. Visualize the Route

Once the experiment is complete, run the plotting script to see the best route found across all runs.

python plot_route.py  
  

-   Output: Generates and displays ga_final_route.png.
    

### 3. Analyze Convergence

Run this script to see how the population improved over generations.

python plot_convergence.py  
  

-   Output: Generates and displays ga_convergence_plot.png.
    

## Project Structure

-   ga_engine.py: The Core Library. Contains the City class, distance matrix calculation, and all GA operator functions.
    
-   experiment_runner.py: The Driver. Sets up the parameter grid, runs the GA 32 times, and logs results.
    
-   plot_route.py: Loads the saved best route and plots it using networkx.
    
-   plot_convergence.py: Loads the logbook from the last run and plots best/average/worst distance over time.
    
-   requirements.txt: List of Python dependencies.
    

## Results

Based on our experimental runs with 50 cities, the most effective configuration was found to be:

-   Selection: Tournament
    
-   Crossover: PMX (Partially Mapped Crossover)
    
-   Mutation: Inversion
    

This combination consistently found routes with shorter distances compared to Roulette selection or Swap mutation.
