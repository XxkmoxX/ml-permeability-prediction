import os
import csv
import numpy as np
import time
import multiprocessing as mp

cdir = ['/3D_lc12_G1/', '/3D_lc12_G2/', '/3D_lc12_G3/']

# Compute list of all paths
realizations = 12800
cwd = os.getcwd()
paths = [os.path.join(cwd + ic, str(ir) + '/') for ic in cdir for ir in range(realizations)]
print(paths[0])

# Load the first ConnectivityMetrics file to get the labels
ind_keys = np.load(os.path.join(paths[0], 'ConnectivityMetrics/64.npy'), allow_pickle=True).item()
ind_labels = list(ind_keys.keys())
prop_labels = ['L', 'con', 'lc', 'p', 'lcG', 'lcNst', 'lcBin', 'keff']
labels = ind_labels + prop_labels
print(labels)

# Function to process each path
def process_path(path):
    try:
        # Load indicator data
        indicator_dict = np.load(os.path.join(path, 'ConnectivityMetrics/64.npy'), allow_pickle=True).item()
        indicator = [v.item() for v in indicator_dict.values()]

        # Read GenParams.txt
        with open(os.path.join(path, 'GenParams.txt'), 'r') as gen_file:
            lines = gen_file.readlines()
            gen = [lines[i].strip() for i in [3, 5, 6, 7]]

        # Read lc.txt
        with open(os.path.join(path, 'lc.txt'), 'r') as lc_file:
            lc_lines = lc_file.readlines()
            lc = [lc_lines[i].strip() for i in [1, 2, 3]]

        # Read SolverRes.txt
        with open(os.path.join(path, 'SolverRes.txt'), 'r') as keff_file:
            keff_lines = keff_file.readlines()
            keff = [keff_lines[1].strip()]

        return indicator + gen + lc + keff
    
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
csv_file = 'ind_output_3D_ics.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(labels)
    writer.writerows(process_results)
print(f'{csv_file} plain text file was generated')
