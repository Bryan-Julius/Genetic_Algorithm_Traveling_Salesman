
from ga_engine import *
def plot_convergence(logbook_file="last_run_convergence.csv"):
    """
    Reads the logbook CSV file and plots the convergence
    of best, average, and worst distance over generations.
    """
    try:
        log_df = pd.read_csv(logbook_file)
    except FileNotFoundError:
        print(f"Error: Log file '{logbook_file}' not found.")
        print("Please run 'experiment_runner.py' first.")
        return

    plt.figure(figsize=(12, 8))

    plt.plot(log_df['gen'], log_df['best_dist'], label='Best Distance', color='green', linewidth=2)
    plt.plot(log_df['gen'], log_df['avg_dist'], label='Average Distance', color='orange', linestyle='--')
    plt.plot(log_df['gen'], log_df['worst_dist'], label='Worst Distance', color='red', linestyle=':')

    plt.title('GA Convergence for TSP', fontsize=16)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Total Distance', fontsize=12)
    plt.legend()
    plt.grid(True)

    # Set y-axis to log scale for better visualization of convergence
    plt.yscale('log')

    plt.savefig("ga_convergence_plot.png")
    print("Saved convergence plot to 'ga_convergence_plot.png'")
    plt.show()



if __name__ == "__main__":
     plot_convergence()