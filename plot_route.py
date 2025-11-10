import matplotlib.pyplot as plt
import networkx as nx
import random
import time
import numpy as np
import pandas as pd
from collections import namedtuple
import pickle


# It imports all the functions (like run_ga_engine, City,
# generate_random_cities, etc.) from your other file.
# This MUST be in the same folder as 'ga_engine.py'
from ga_engine import *
def plot_final_route(cities, best_route, title="Final TSP Route"):
    """
    Plots the final optimized route using networkx and matplotlib.

    Args:
        cities (list): The list of City (x, y) objects.
        best_route (list): The permutation of city indices (e.g., [0, 4, 2, ...]).
    """

    # Create a complete graph
    G = nx.Graph()

    # Create node positions dictionary for networkx
    pos = {}
    for i, city in enumerate(cities):
        pos[i] = (city.x, city.y)
        G.add_node(i, pos=(city.x, city.y))


    # Initialize an empty list for the route edges
    route_edges = []


    for i in range(len(best_route)):
        current_city = best_route[i]
        next_city = best_route[(i + 1) % len(best_route)]
        route_edges.append((current_city, next_city))

    # Draw the plot
    plt.figure(figsize=(12, 10))

    # Draw all nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)

    # Draw the optimized route edges
    nx.draw_networkx_edges(G, pos, edgelist=route_edges, edge_color='red', width=2.0)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_color='black')

    plt.title(title, fontsize=16)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.savefig("ga_final_route.png")
    print("\nSaved final route plot to 'ga_final_route.png'")
    plt.show()

if __name__ == "__main__":
    print("Loading experiment results for plotting")

    try:
        # 1. Load the saved cities list
        with open("cities.pkl", "rb") as f:
            cities_to_plot = pickle.load(f)
        print(f"Loaded {len(cities_to_plot)} cities from 'cities.pkl'")

        # 2. Load the saved best route
        with open("best_route.pkl", "rb") as f:
            best_route_to_plot = pickle.load(f)
        print(f"Loaded best route from 'best_route.pkl'")

        # 3. Calculate its distance (for the title)
        dist_matrix = calculate_distance_matrix(cities_to_plot)
        final_dist = calculate_route_distance(best_route_to_plot, dist_matrix)

        # 4. Plot the result
        plot_final_route(cities_to_plot, best_route_to_plot,
                         title=f"Best Route from Experiment ({len(cities_to_plot)} Cities) - Distance: {final_dist:.2f}")

    except FileNotFoundError:
        print("\n--- ERROR ---")
        print("Could not find 'cities.pkl' or 'best_route.pkl'.")
        print("Please run 'experiment_runner.py' first to generate these files.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

    print("\n--- Plotting script complete ---")