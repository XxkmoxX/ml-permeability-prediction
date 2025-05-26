import os
import csv
import numpy as np
import time
import multiprocessing as mp

# Compute list of all paths
realizations = 10
cwd = os.getcwd()
paths = [os.path.join(cwd, str(ir) + '/') for ir in range(realizations)]
print(paths[0])

# Function to process each path
def process_path(path):
    try:
        # Read SolverRes.txt to obtain Keff
        with open(os.path.join(path, 'SolverRes.txt'), 'r') as keff_file:
            keff_lines = keff_file.readlines()
            keff = [keff_lines[1].strip()]
        return keff
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None

# Use ThreadPoolExecutor for parallel file reading
def process_with_multiprocessing(paths, num_workers):
    
    start_time = time.time()
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(process_path, paths)
    results = [result for result in results if result is not None]
    end_time = time.time()
    print(f"Multiprocessing.Pool Time: {end_time - start_time:.2f} seconds")
    return results

# Main execution
num_workers = mp.cpu_count()
print("\nRunning multiprocessing.Pool...")
process_results = process_with_multiprocessing(paths, num_workers)

# Write the CSV file
csv_file = 'keff_vae.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(process_results)
print(f'{csv_file} plain text file was generated')
