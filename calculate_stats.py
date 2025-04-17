import numpy as np
from scipy.stats import mannwhitneyu
import os
import glob

# --- Configuration ---
results_directory = './results'
num_runs = 10
# No longer need num_final_games for calculation, but useful for context
num_games_per_average = 50
# --- End Configuration ---

final_performance_dreaming = []
final_performance_no_dreaming = []

print(f"Looking for reward files in: {os.path.abspath(results_directory)}")
print(f"Extracting the final performance metric (last recorded average over {num_games_per_average} games) per run.")

# Loop through each run number
for run_index in range(num_runs):
    # --- Process 'dreaming' run (if_dream_1) ---
    dreaming_file_pattern = os.path.join(results_directory, f"rewards_{run_index}if_dream_1.npy")
    dreaming_files = glob.glob(dreaming_file_pattern)

    if dreaming_files:
        dreaming_file_path = dreaming_files[0]
        try:
            rewards_dreaming_avg = np.load(dreaming_file_path)
            if len(rewards_dreaming_avg) > 0: # Check if the array is not empty
                # Take the LAST value in the array
                final_perf = rewards_dreaming_avg[-1]
                final_performance_dreaming.append(final_perf)
                # print(f"Run {run_index} (Dreaming): Loaded {len(rewards_dreaming_avg)} averages, Final Perf: {final_perf:.4f}")
            else:
                print(f"Warning: Dreaming file {os.path.basename(dreaming_file_path)} is empty. Skipping run {run_index}.")
        except Exception as e:
            print(f"Error processing {os.path.basename(dreaming_file_path)}: {e}")
    else:
        print(f"Error: Dreaming file not found for run {run_index} (pattern: {dreaming_file_pattern})")

    # --- Process 'no dreaming' run (if_dream_0) ---
    no_dreaming_file_pattern = os.path.join(results_directory, f"rewards_{run_index}if_dream_0.npy")
    no_dreaming_files = glob.glob(no_dreaming_file_pattern)

    if no_dreaming_files:
        no_dreaming_file_path = no_dreaming_files[0]
        try:
            rewards_no_dreaming_avg = np.load(no_dreaming_file_path)
            if len(rewards_no_dreaming_avg) > 0: # Check if the array is not empty
                 # Take the LAST value in the array
                final_perf = rewards_no_dreaming_avg[-1]
                final_performance_no_dreaming.append(final_perf)
                # print(f"Run {run_index} (No Dream): Loaded {len(rewards_no_dreaming_avg)} averages, Final Perf: {final_perf:.4f}")
            else:
                 print(f"Warning: No dreaming file {os.path.basename(no_dreaming_file_path)} is empty. Skipping run {run_index}.")
        except Exception as e:
            print(f"Error processing {os.path.basename(no_dreaming_file_path)}: {e}")
    else:
        print(f"Error: No dreaming file not found for run {run_index} (pattern: {no_dreaming_file_pattern})")

# --- Perform Statistical Test ---
print("-" * 30)

# Check if we have data for all runs
if len(final_performance_dreaming) == num_runs and len(final_performance_no_dreaming) == num_runs:
    print(f"Successfully loaded final performance data for {num_runs} dreaming runs and {num_runs} no dreaming runs.")

    # Calculate means and standard deviations for reporting in the paper
    mean_dreaming = np.mean(final_performance_dreaming)
    sd_dreaming = np.std(final_performance_dreaming)
    mean_no_dreaming = np.mean(final_performance_no_dreaming)
    sd_no_dreaming = np.std(final_performance_no_dreaming)

    print(f"\nDescriptive Statistics (Final recorded average over {num_games_per_average} games):")
    print(f"  Dreaming:     Mean = {mean_dreaming:.4f}, SD = {sd_dreaming:.4f}")
    print(f"  No Dreaming:  Mean = {mean_no_dreaming:.4f}, SD = {sd_no_dreaming:.4f}")
    print(f"  (n = {num_runs} per group)")


    # Perform the Mann-Whitney U test
    u_statistic, p_value = mannwhitneyu(
        final_performance_dreaming,
        final_performance_no_dreaming,
        alternative='greater' # Assuming hypothesis is that dreaming performs better
    )

    print("\n--- Mann-Whitney U Test Results ---")
    print(f"  U-statistic: {u_statistic}")
    print(f"  P-value:     {p_value:.6f}") # Print p-value with more precision
    print("------------------------------------")


    # Interpret the result
    alpha = 0.05 # Standard significance level
    if p_value < alpha:
        print(f"\nConclusion: The result is statistically significant (p < {alpha}).")
        print("  Reject the null hypothesis.")
        print("  Evidence suggests the 'dreaming' group performed significantly better than the 'no dreaming' group.")

    else:
        print(f"\nConclusion: The result is not statistically significant (p >= {alpha}).")
        print("  Fail to reject the null hypothesis. There is not enough evidence to conclude a significant difference.")

else:
    print("\nError: Missing data for one or more runs. Cannot perform statistical test.")
    print(f"  Found {len(final_performance_dreaming)} dreaming runs.")
    print(f"  Found {len(final_performance_no_dreaming)} no dreaming runs.")
    print(f"  Expected {num_runs} runs per condition.")