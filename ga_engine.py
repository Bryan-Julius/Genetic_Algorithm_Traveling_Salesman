import numpy as np
import random
import time
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import namedtuple

# Use a namedtuple for a lightweight, immutable City object
City = namedtuple("City", ['x', 'y'])

def generate_random_cities(num_cities):
    """
    Generates a list of City objects with random coordinates.
    Coordinates are floats between 0 and 100.
    """
    cities = []
    for _ in range(num_cities):
        cities.append(City(x=random.uniform(0, 100), y=random.uniform(0, 100)))
    return cities

def calculate_distance_matrix(cities):
    """
    Calculates the n x n Euclidean distance matrix for n cities.
    This is the O(n^2) pre-computation.
    """
    num_cities = len(cities)
    # Initialize an n x n matrix with zeros
    distance_matrix = np.zeros((num_cities, num_cities))

    for i in range(num_cities):
        for j in range(i, num_cities):
            # Calculate Euclidean distance
            dist = np.sqrt((cities[i].x - cities[j].x)**2 + (cities[i].y - cities[j].y)**2)

            # TSP is symmetric
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist

    return distance_matrix

def calculate_route_distance(route, distance_matrix):
    """
    Calculates the total distance of a given route (permutation of indices).
    This is the O(n) part of the fitness evaluation.
    """
    total_distance = 0
    num_cities = len(route)

    for i in range(num_cities):
        # Get the current city and the next city in the tour
        # The modulo operator (%) handles the wrap-around from the last city to the first
        current_city = route[i]
        next_city = route[(i + 1) % num_cities]

        # Add distance from the pre-computed matrix
        total_distance += distance_matrix[current_city][next_city]

    return total_distance

def calculate_fitness(route, distance_matrix):
    """
    Calculates the fitness of a route.
    Fitness = 1 / Total Distance.
    """
    total_distance = calculate_route_distance(route, distance_matrix)

    # Handle potential division by zero, though distance will always be > 0
    if total_distance == 0:
        return np.inf

    return 1.0 / total_distance



def create_initial_population(pop_size, num_cities):
    """
    Creates an initial population of 'pop_size' random routes.
    Each route is a random permutation of city indices [0, 1... n-1].
    """
    population = []
    base_route = list(range(num_cities))

    for _ in range(pop_size):
        # Create a new random permutation of the base route
        route = random.sample(base_route, num_cities)
        population.append(route)

    return population



def selection_tournament(population, fitnesses, k=5):
    """
    Selects a new parent from the population using k-tournament selection.

    Args:
        population (list): The list of all routes in the current generation.
        fitnesses (list): The list of fitness scores for the current population.
        k (int): The size of the tournament.

    Returns:
        list: The route (chromosome) of the winning parent.
    """
    # Select k random indices from the population
    tournament_indices = random.sample(range(len(population)), k)

    # Find the individual with the best fitness within the tournament
    best_fitness = -1
    best_index = -1

    for index in tournament_indices:
        if fitnesses[index] > best_fitness:
            best_fitness = fitnesses[index]
            best_index = index

    # Return the winning individual (route)
    return population[best_index]


def selection_roulette(population, fitnesses):
    """
    Selects a new parent from the population using Roulette Wheel selection.
    You can also think about this as a lottery or weighted random selection.

    Args:
        population (list): The list of all routes.
        fitnesses (list): The list of fitness scores.

    Returns:
        list: The route (chromosome) of the chosen parent.
    """
    # Calculate the total fitness of the population
    total_fitness = sum(fitnesses)

    # Generate a random "spin" value between 0 and total_fitness
    spin = random.uniform(0, total_fitness)

    # Iterate through the population, accumulating fitness
    current_sum = 0
    for i in range(len(population)):
        current_sum += fitnesses[i]
        if current_sum > spin:
            # This is the chosen individual
            return population[i]

    # Fallback (should rarely be hit)
    return population[-1]


def crossover_order(parent1, parent2):
    """
    Performs Order Crossover (OX) on two parents to create one child.

    1. Select a random slice from parent1.
    2. Copy this slice to the child.
    3. Fill the remaining slots in the child with genes from parent2,
       in the order they appear in parent2, skipping genes already
       present from the parent1 slice.
    """
    num_cities = len(parent1)
    child = [-1] * num_cities  # Initialize child with placeholders

    #  Select a random slice
    start, end = sorted(random.sample(range(num_cities), 2))

    #  Copy the slice from parent1 to the child
    child[start:end] = parent1[start:end]

    #  Fill remaining slots from parent2
    # Create a list of genes from parent2 that are not in the child's slice
    parent2_genes = [gene for gene in parent2 if gene not in child]

    # Use a pointer to fill the remaining slots in the child
    gene_ptr = 0
    for i in range(num_cities):
        if child[i] == -1:  # If the slot is empty
            child[i] = parent2_genes[gene_ptr]
            gene_ptr += 1

    return child


def crossover_pmx(parent1, parent2):
    """
    Performs Partially Mapped Crossover (PMX) on two parents.

    1. Select a random slice (substring).
    2. Copy parent1's slice to the child.
    3. For each gene in parent2's corresponding slice:
       a. Find the gene (let's call it 'gene_A')
       b. Find the gene in parent1's slice at the same position (let's call it 'gene_B')
       c. Find 'gene_A' elsewhere in the child.
       d. Replace it with 'gene_B'.
    4. Fill the remaining slots in the child from parent2.
    """
    num_cities = len(parent1)
    child = parent2.copy()  # Start by making a copy of parent2

    #  Select random slice
    start, end = sorted(random.sample(range(num_cities), 2))

    #  Copy parent1's slice into the child
    child[start:end] = parent1[start:end]

    #  Create mapping and fix collisions
    mapping = {parent1[i]: parent2[i] for i in range(start, end)}

    for i in range(num_cities):
        if i < start or i >= end:  # Only check outside the copied slice
            while child[i] in mapping:
                # This gene was part of the p1 slice and is now a duplicate.
                # We must replace it with its mapped counterpart from p2.
                child[i] = mapping[child[i]]

    return child


def mutate_swap(route, mutation_rate):
    """
    Applies Swap Mutation to a route.
    For each gene, with probability 'mutation_rate', it is swapped
    with another random gene in the same route.
    """
    mutated_route = route.copy()
    for i in range(len(mutated_route)):
        if random.random() < mutation_rate:
            # Select a second, different index to swap with
            j = random.randint(0, len(mutated_route) - 1)

            # Perform the swap
            mutated_route[i], mutated_route[j] = mutated_route[j], mutated_route[i]

    return mutated_route



def mutate_inversion(route, mutation_rate):
    """
    Applies Inversion Mutation.
    With a probability of 'mutation_rate', a random slice of the
    route is selected and inverted (reversed).
    """
    if random.random() < mutation_rate:
        mutated_route = route.copy()

        # Select two random indices
        start, end = sorted(random.sample(range(len(mutated_route)), 2))

        # Reverse the slice in-place
        # Note: Slicing creates a copy, so we must assign it back
        slice_to_reverse = mutated_route[start:end]
        slice_to_reverse.reverse()
        mutated_route[start:end] = slice_to_reverse

        return mutated_route

    return route.copy() # Return a copy if no mutation



def run_ga_engine(cities, pop_size, num_generations,
                  mutation_rate, crossover_prob, elitism_size,
                  selection_fn, crossover_fn, mutation_fn):
    """
    The main engine for the Genetic Algorithm.

    Args:
        cities (list): List of City objects.
        pop_size (int): Number of individuals in the population.
        num_generations (int): Number of generations to run.
        mutation_rate (float): Probability of mutation for an individual.
        crossover_prob (float): Probability of crossover for a pair of parents.
        elitism_size (int): Number of "best" individuals to carry to next gen.
        selection_fn (function): The selection operator to use (e.g., selection_tournament).
        crossover_fn (function): The crossover operator (e.g., crossover_order).
        mutation_fn (function): The mutation operator (e.g., mutate_inversion).

    Returns:
        tuple: (best_route_found, logbook)
               best_route_found (list): The best route (permutation) found.
               logbook (list): A list of dictionaries, one for each generation,
                               containing 'gen', 'best_dist', 'avg_dist', 'worst_dist'.
    """

    #  Initialization
    num_cities = len(cities)
    distance_matrix = calculate_distance_matrix(cities)
    population = create_initial_population(pop_size, num_cities)

    # Calculate initial fitnesses
    fitnesses = [calculate_fitness(route, distance_matrix) for route in population]

    best_route_ever = None
    best_distance_ever = np.inf

    logbook = []

    #  Generation Loop
    start_time = time.time()
    print(f"Starting GA with {selection_fn.__name__}, {crossover_fn.__name__}, {mutation_fn.__name__}")

    for gen in range(num_generations):

        #  Logging & Elitism
        # Find the best route in the current population
        current_best_idx = np.argmax(fitnesses)
        current_best_dist = calculate_route_distance(population[current_best_idx], distance_matrix)

        if current_best_dist < best_distance_ever:
            best_distance_ever = current_best_dist
            best_route_ever = population[current_best_idx]

        # Log statistics for this generation
        distances = [1.0 / f for f in fitnesses]
        logbook.append({
            'gen': gen,
            'best_dist': np.min(distances),
            'avg_dist': np.mean(distances),
            'worst_dist': np.max(distances)
        })

        if gen % 100 == 0:
            print(f"Gen {gen}: Best Distance = {best_distance_ever:.2f}")

        # Elitism: Get the indices of the 'elitism_size' best individuals
        elite_indices = np.argsort(fitnesses)[-elitism_size:]
        new_population = [population[i] for i in elite_indices]

        # Breeding Loop
        while len(new_population) < pop_size:
            #  Selection
            parent1 = selection_fn(population, fitnesses)
            parent2 = selection_fn(population, fitnesses)

            #  Crossover
            if random.random() < crossover_prob:
                child = crossover_fn(parent1, parent2)
            else:
                # No crossover, one parent (e.g., parent1) moves on
                child = parent1.copy()

            #  Mutation
            child = mutation_fn(child, mutation_rate)

            new_population.append(child)

        #  Replacement
        population = new_population

        #  Evaluate New Population
        fitnesses = [calculate_fitness(route, distance_matrix) for route in population]

    end_time = time.time()
    print(f"GA Finished. Time: {end_time - start_time:.2f}s. Final Best Distance: {best_distance_ever:.2f}")

    return best_route_ever, logbook




