

from ga_engine import *  # Import all our GA functions
import pickle


def run_parameter_experiments(base_cities):
    """
    This is the main function to generate parameters for the experiment
    """

    # Define the Parameter Matrix
    # Define the parameter sets to be tested
    param_grid = {
        'pop_size': [50,100],
    'mutation_rate': [0.01, 0.05],
    'crossover_fn': [crossover_order, crossover_pmx],
    'selection_fn': [selection_tournament, selection_roulette],
    'mutation_fn': [mutate_swap, mutate_inversion]
    }

    # Fixed parameters for all runs
    NUM_GENERATIONS = 1000
    CROSSOVER_PROB = 0.8
    ELITISM_SIZE = 2
    NUM_CITIES = len(base_cities)

    results_list = []

    # Keeps track of best run so we can plot it
    best_distance_ever = np.inf
    best_route_ever = None

    print(f" Starting Experimental Runs for {NUM_CITIES} Cities")

    # Loop Through All Combinations
    # This is a simple grid search
    run_id = 0
    for pop_size in param_grid['pop_size']:
        for mut_rate in param_grid['mutation_rate']:
            for cross_fn in param_grid['crossover_fn']:
                for sel_fn in param_grid['selection_fn']:
                    for mut_fn in param_grid['mutation_fn']:
                        run_id += 1
                        print(f"\n--- RUN {run_id} ---")

                        start_time = time.time()

                        # Run the GA Engine
                        best_route, logbook = run_ga_engine(
                            cities=base_cities,
                            pop_size=pop_size,
                            num_generations=NUM_GENERATIONS,
                            mutation_rate=mut_rate,
                            crossover_prob=CROSSOVER_PROB,
                            elitism_size=ELITISM_SIZE,
                            selection_fn=sel_fn,
                            crossover_fn=cross_fn,
                            mutation_fn=mut_fn
                        )

                        end_time = time.time()

                        # --- 4. Capture Results ---
                        final_distance = logbook[-1]['best_dist']
                        total_time = end_time - start_time

                        # Check if best route
                        if final_distance < best_distance_ever:
                            best_distance_ever = final_distance
                            best_route_ever = best_route
                            print(f"!!! New best route found! Distance: {best_distance_ever:.2f}")

                        results_list.append({
                            'run_id': run_id,
                            'pop_size': pop_size,
                            'mutation_rate': mut_rate,
                            'crossover_strategy': cross_fn.__name__,
                            'selection_strategy': sel_fn.__name__,
                            'mutation_strategy': mut_fn.__name__,
                            '_generation': NUM_GENERATIONS,
                                       'final_best_distance': final_distance,
                        'execution_time_s': total_time
                        })

                        # Save the logbook for the *last* run for convergence plotting
                        if run_id == 32: # (2*2*2*2*2)
                            logbook_df = pd.DataFrame(logbook)
                            logbook_df.to_csv("last_run_convergence.csv", index=False)
                            print("Saved last_run_convergence.csv")

    #  5. Save Aggregate Results
    results_df = pd.DataFrame(results_list)
    results_df.to_csv("parameter_impact_results.csv", index=False)

    if best_route_ever:
        with open("best_route.pkl", "wb") as f:
            pickle.dump(best_route_ever, f)
        print(f"\nSaved best route (Distance: {best_distance_ever:.2f}) to 'best_route.pkl'")
    else:
        print("\nNo runs completed, best route not saved.")

    print("\n--- All Experiments Complete ---")
    print(f"Saved results to parameter_impact_results.csv")
    print(results_df.head())

    return results_df


if __name__ == "__main__":
#     # Generate one set of cities to be used for all experiments
#     # This is crucial for a fair comparison!
     NUM_CITIES_FOR_EXPERIMENT = 50
     random.seed(42) # For reproducibility
     CITIES_FOR_EXPERIMENT = generate_random_cities(NUM_CITIES_FOR_EXPERIMENT)

with open("cities.pkl", "wb") as f:
    pickle.dump(CITIES_FOR_EXPERIMENT, f)
print(f"Saved {NUM_CITIES_FOR_EXPERIMENT} cities to 'cities.pkl'")

#     # Run the rig
experimental_results_df = run_parameter_experiments(CITIES_FOR_EXPERIMENT)