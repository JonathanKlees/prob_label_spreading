import argparse
import json
import os
import sys
#import pyarrow.parquet as pq

# if the script is interrupted it might be the case that some dataset is still in shared memory, the following fixes this
import multiprocessing.shared_memory as shm
# remove the shared memory dataframe by name
possible_shared_memory_names = ['TwoMoons_shm', 'EMNIST_shm', 'EEMNIST_shm', 'CIFAR10_shm', 'CIFAR10-H_shm', 'ANIMALS10_shm', 'TinyImageNet_shm', 'MTSD_shm']
for shared_memory_name in possible_shared_memory_names:
    try:
        # Attach to the shared memory segment
        existing_shm = shm.SharedMemory(name=shared_memory_name)
        
        # Unlink to remove it
        existing_shm.unlink()
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Error while attempting to remove shared memory segment: {e}")

def parse_sample_sizes(sample_sizes, total_size):
    # given a list of integers and / or relative frequencies converts this list to integers only via the total dataset size
    parsed_sizes = []

    for size in sample_sizes:
        if isinstance(size, float) and 0 <= size <= 1:
            # Convert decimal to an absolute value (percentage of total_size)
            parsed_sizes.append(int(total_size * size))
        elif isinstance(size, int):
            # Absolute size is already an integer
            parsed_sizes.append(size)
        else:
            raise ValueError(f"Invalid sample size format: {size}")

    return parsed_sizes

# Define available experiments
valid_choices = [s.split(".json")[0] for s in os.listdir("experiments")]

# Check if an argument was passed
if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <choice>")
    print("Choices:", ", ".join(valid_choices))
    sys.exit(1)

# Get the argument and see whether it matches an experiment config
choice = sys.argv[1]
if choice not in valid_choices:
    print(f"Invalid choice: {choice}")
    print("Please choose from:", ", ".join(valid_choices))
    sys.exit(1)

############################
else: # run main code
    experiment = choice
    experiment_config = "experiments/" + experiment + ".json"

    with open(experiment_config, 'r') as json_file:
        config = json.load(json_file)

    # Access the configurations
    datasets = config['datasets']
    dataset_sizes = config['dataset_sizes']
    algorithms = config['algorithms']
    multiple_runs = config['multiple_runs']
    dim_reduction_techniques = config['dim_reduction_techniques']
    dimensions = config['dimensions']
    orig_sample_sizes = config['sample_sizes'] #  is updated below if it contains relative frequencies to map them to integers
    prob_label_columns = config['prob_label_columns']

    # Baseline algorithm configurations
    gammas = config['gammas']

    # K nearest neighbors configurations
    metrics = config['metrics']
    k_neighbors_baseline = config['k_neighbors_baseline']
    # Probabilistic label spreading configurations
    alphas = config['alphas']
    k_neighbors_pls = config['k_neighbors_pls']


    print("experiment:", experiment)
    print("datasets:", datasets)
    print("dim. reduction techniques:", dim_reduction_techniques)
    print("data dimensions:", dimensions)
    print("prob labels to be used:", prob_label_columns)
    print("number of labels via crowdsourcing:", orig_sample_sizes)
    print("gamma values for Gaussian mixture baseline:", gammas)
    print("metrics for k-NN baseline:", metrics)
    print("number of neighbors for k-NN baseline:", k_neighbors_baseline)
    print("number of neighbors for PLS:", k_neighbors_pls)
    print("alpha values for PLS:", alphas)


    from scripts.plot import *
    from scripts.baseline_prob_label_spreading import *
    from scripts.probabilistic_label_spreading import prob_label_spreading
    import time
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import math
    from joblib import Parallel, delayed
    from tqdm import tqdm
    import os
    import psutil
    import threading
    import pickle
    from multiprocessing import shared_memory

    if "gaussian_baseline" in algorithms:
        print("Running baseline algorithm based on Gaussian Mixture")

        start_time = time.time()
        # Set half the cores for parallelization
        n_jobs = os.cpu_count() // 2

        # Function to store the DataFrame in shared memory
        def serialize_df_to_shared_memory(df):
            serialized_df = pickle.dumps(df)
            shm = shared_memory.SharedMemory(create=True, size=len(serialized_df))
            shm.buf[:len(serialized_df)] = np.frombuffer(serialized_df, dtype=np.uint8)
            return shm, len(serialized_df)

        # Function to load DataFrame from shared memory
        def load_shared_df(shm, size):
            buffer = np.ndarray((size,), dtype=np.uint8, buffer=shm.buf)
            df = pickle.loads(buffer.tobytes())
            return df

        def dim_red_technique(data_space):
            # store the dim reduction technique with a + between multiple techniques
            if data_space == "feature":
                dim_reduction_technique = "NONE"
            elif len(data_space.split("_")[:-1]) > 1: # combination of e.g. CLIP_UMAP
                dim_reduction_technique = "+".join(data_space.split("_")[:-1])
            else:
                dim_reduction_technique = data_space.split("_")[0] # just one technique e.g. CLIP

            return dim_reduction_technique

        # Function to be executed in parallel
        def process_combination(dataset_name, prob_label_column, n_data, dim_reduction_technique, dim, gamma, n_samples, i, shm_name, shm_size):
            # Load the DataFrame from shared memory
            shm = shared_memory.SharedMemory(name=shm_name)
            df = load_shared_df(shm, shm_size)

            if dataset_name == "TwoMoons" or dim_reduction_technique == "NONE":
                data_space = "feature"
            elif dim_reduction_technique == "CLIP":
                data_space = "CLIP"
            else:
                data_space = dim_reduction_technique + "_" + str(dim)

            if n_data * n_samples < 1e11: 
                sparse = False
                acc, l2_score, kl, mae, tv = baseline_prob_label_spreading(df, data_space, prob_label_column, n_data, n_samples, gamma)
            else: # if the matrix is too large we apply chunking for distance calculations and sparsify the matrix
                print("Using chunking for distance calculations")
                sparse = True
                acc, l2_score, kl, mae, tv = baseline_prob_label_spreading(df, data_space, prob_label_column, n_data, n_samples, gamma)

            return {
                "dataset": dataset_name,
                "prob_labels": prob_label_column,
                "n_data": n_data,
                "dim_reduction_technique": dim_red_technique(data_space),
                "dimension": dim,
                "gamma": gamma,
                "n_samples": n_samples,
                "Accuracy": acc,
                "RMSE": l2_score,
                "KL": kl,
                "MAE": mae,
                "TV": tv,
                "sparse": sparse
            }

        # Collecting results in a list for DataFrame conversion
        all_results = []

        # Loop to define parameters
        for dataset_name in datasets:

            prob_label_columns = config['prob_label_columns'] # we may overwrite this for some datasets for convenience, that's why it is in the loop

            df = pd.read_pickle(f"data/prob_data/{dataset_name}/{dataset_name}.pkl")
            # keep only the relevant parts of the data for this experiment
            data_spaces = [f"{embedder}_{dim}" for embedder in dim_reduction_techniques for dim in dimensions if embedder != "CLIP"]
            if not all(col in df.columns for col in prob_label_columns): # for TwoMoons and MTSD we have the prob labels stored in the column "prob_label"
                prob_label_columns = ["prob_label"]

            if all(col in df.columns for col in data_spaces):
                columns = data_spaces + ["label","CLIP"] + prob_label_columns
                df = df[columns]
                 
            classes = list(set(df["label"]))
            n_data = dataset_sizes[dataset_name] if dataset_sizes[dataset_name] != "all" else len(df) 

            sample_sizes = parse_sample_sizes(orig_sample_sizes, n_data) # converts relative freqs into integers with n_data

            # Serialize the DataFrame to shared memory
            shm, shm_size = serialize_df_to_shared_memory(df)

            # Prepare combinations for every dataset and run them in parallel
            combinations = []
            
            try:
                if dataset_name == "TwoMoons":
                    # For Two Moons, no dimension reduction is performed
                    dim = df["feature"][0].shape[0]
                    prob_label_column = "prob_label" # there is just one set of prob labels for this dataset
                    combinations.extend([
                        (dataset_name, prob_label_column, n_data, "NONE", dim, gamma, n_samples, i, shm.name, shm_size)
                        for gamma in gammas
                        for n_samples in sample_sizes
                        for i in range(multiple_runs)
                    ])

                else:
                    for dim_reduction_technique in dim_reduction_techniques:
                        if dim_reduction_technique == "NONE": # no need to loop over the dimensions here
                            dim = df["feature"][0].shape[0]
                            combinations.extend([
                                (dataset_name, prob_label_column, n_data, dim_reduction_technique, dim, gamma, n_samples, i, shm.name, shm_size)
                                for gamma in gammas
                                for n_samples in sample_sizes
                                for prob_label_column in prob_label_columns
                                for i in range(multiple_runs)
                            ])
                        elif dim_reduction_technique == "CLIP": # CLIP embeddings have a fixed dimension as well
                            dim = 512
                            combinations.extend([
                                (dataset_name, prob_label_column, n_data, dim_reduction_technique, dim, gamma, n_samples, i, shm.name, shm_size)
                                for gamma in gammas
                                for n_samples in sample_sizes
                                for prob_label_column in prob_label_columns
                                for i in range(multiple_runs)
                            ])
                        else:
                            combinations.extend([
                                (dataset_name, prob_label_column, n_data, dim_reduction_technique, dim, gamma, n_samples, i, shm.name, shm_size)
                                for dim in dimensions
                                for gamma in gammas
                                for n_samples in sample_sizes
                                for prob_label_column in prob_label_columns
                                for i in range(multiple_runs)
                            ])

                # Execute combinations in parallel with a progress bar and dynamic load balancing
                results = Parallel(n_jobs=n_jobs, batch_size="auto")(delayed(process_combination)(*params) for params in tqdm(combinations, desc=f"Processing {dataset_name}", leave=True))
                
                # Append results to the main list
                all_results.extend(results)

            finally:
                # Clean up the shared memory after each dataset
                shm.close()
                shm.unlink()

        # Convert results into a DataFrame
        baseline_results_gaussian_process = pd.DataFrame(all_results)

        baseline_results_gaussian_process.to_csv("results/gaussian_baseline_" + experiment + ".csv")

        baseline_results_gaussian_process.head()


        elapsed_time = time.time() - start_time

        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        # Format the output
        formatted_time = f"{int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds"
        print(f"runtime: {formatted_time}")


    # Serialize DataFrame into shared memory
    def dataframe_to_shared_memory(df, shm_name):
        # Serialize the DataFrame with pickle
        buf = pickle.dumps(df)
        shm = shared_memory.SharedMemory(name=shm_name, create=True, size=len(buf))
        shm.buf[:len(buf)] = buf  # Write the serialized DataFrame into shared memory
        return shm

    # Deserialize DataFrame from shared memory
    def dataframe_from_shared_memory(shm_name, size):
        shm = shared_memory.SharedMemory(name=shm_name)
        buf = bytes(shm.buf[:size])  # Read only the relevant portion
        df = pickle.loads(buf)  # Deserialize back to DataFrame
        return df

    if "knn_baseline" in algorithms:
        print("Running baseline algorithm based on k nearest neighbors")

        start_time = time.time()

        def process_combination_knn(dataset_name, prob_label_column, n_data, dim_reduction_technique, dim, metric, k, n_samples, data_space, shm_name, shm_size):
            df = dataframe_from_shared_memory(shm_name, shm_size)
            
            try:
                acc, l2_score, kl, mae, tv = baseline_prob_label_spreading_kNN(df, data_space, prob_label_column, n_data, n_samples, k)
            except ValueError as e:
                if "n_neighbors <= n_samples_fit" in str(e): # too few samples to aggregate over k nearest neighbors
                    acc, l2_score, kl, mae, tv = (np.nan, np.nan, np.nan, np.nan, np.nan)
                else:
                    raise
            except Exception as e:
                raise

            return {
                "dataset": dataset_name,
                "prob_labels": prob_label_column,
                "n_data": n_data,
                "dim_reduction_technique": dim_red_technique(data_space),
                "dimension": dim,
                "metric": metric,
                "k": k,
                "n_samples": n_samples,
                "Accuracy": acc,
                "RMSE": l2_score,
                "KL": kl,
                "MAE": mae,
                "TV": tv
            }


        all_results = []

        for dataset_name in datasets:
            
            prob_label_columns = config['prob_label_columns'] # we may overwrite this for some datasets for convenience

            # Read DataFrame and determine its size
            df = pd.read_pickle(f"data/prob_data/{dataset_name}/{dataset_name}.pkl")

            # keep only the relevant parts of the data for this experiment
            data_spaces = [f"{embedder}_{dim}" for embedder in dim_reduction_techniques for dim in dimensions if embedder != "CLIP"]
            if not all(col in df.columns for col in prob_label_columns):
                prob_label_columns = ["prob_label"]

            if all(col in df.columns for col in data_spaces):
                columns = data_spaces + ["label","CLIP"] + prob_label_columns
                df = df[columns]

            n_data = dataset_sizes[dataset_name] if dataset_sizes[dataset_name] != "all" else len(df)
            sample_sizes = parse_sample_sizes(orig_sample_sizes, n_data) # converts relative freqs into integers with n_data

            # Serialize the DataFrame into shared memory
            shm_name = f"{dataset_name}_shm"
            shm = dataframe_to_shared_memory(df, shm_name)
            shm_size = len(shm.buf)  # Store the size to read it back

            # Prepare combinations for all dim reduction techniques
            combinations = []

            if dataset_name == "TwoMoons": # for this dataset we do not have dim reduction
                dim = df["feature"][0].shape[0]
                data_space = "feature"
                dim_reduction_technique = "NONE"
                prob_label_column = "prob_label" # there is just one set of prob labels for this dataset
                # Add combinations for this specific technique
                combinations.extend([
                    (dataset_name, prob_label_column, n_data, dim_reduction_technique, dim, metric, k, n_samples, data_space, shm_name, shm_size)
                    for metric in metrics
                    for k in k_neighbors_baseline
                    for n_samples in sample_sizes
                    for _ in range(multiple_runs)
                ])

            else: # else loop over the different dim reduction techniques
                for dim_reduction_technique in dim_reduction_techniques:
                    if dim_reduction_technique == "NONE":
                        dim = df["feature"][0].shape[0]
                        data_space = "feature"
                        # Add combinations for this specific technique
                        combinations.extend([
                            (dataset_name, prob_label_column, n_data, dim_reduction_technique, dim, metric, k, n_samples, data_space, shm_name, shm_size)
                            for metric in metrics
                            for k in k_neighbors_baseline
                            for n_samples in sample_sizes
                            for prob_label_column in prob_label_columns
                            for _ in range(multiple_runs)
                        ])
                    elif dim_reduction_technique == "CLIP": # CLIP embeddings have a fixed dimension as well
                        dim = 512
                        data_space = "CLIP"
                        combinations = [
                            (dataset_name, prob_label_column, n_data, dim_reduction_technique, dim, metric, k, n_samples, data_space, shm.name, shm_size)
                            for metric in metrics
                            for k in k_neighbors_baseline
                            for n_samples in sample_sizes
                            for prob_label_column in prob_label_columns
                            for _ in range(multiple_runs)
                        ]
                    else: #  if there is dim reduction also loop over the dimensions
                        for dim in dimensions:
                            data_space = f"{dim_reduction_technique}_{dim}"
                            combinations.extend([
                                (dataset_name, prob_label_column, n_data, dim_reduction_technique, dim, metric, k, n_samples, data_space, shm_name, shm_size)
                                for metric in metrics
                                for k in k_neighbors_baseline
                                for n_samples in sample_sizes
                                for prob_label_column in prob_label_columns
                                for _ in range(multiple_runs)
                            ])

            # Execute all combinations in parallel with a single tqdm progress bar
            results = Parallel(n_jobs=n_jobs, batch_size="auto")(
                delayed(process_combination_knn)(*params) for params in tqdm(combinations, desc=f"Processing {dataset_name}")
            )

            # Append results to the main list
            all_results.extend(results)

            # Release shared memory
            shm.close()
            shm.unlink()

        # Convert results into a DataFrame
        baseline_results_knn = pd.DataFrame(all_results)

        baseline_results_knn.to_csv("results/knn_baseline_" + experiment + ".csv")

        elapsed_time = time.time() - start_time

        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        # Format the output
        formatted_time = f"{int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds"
        print(f"runtime: {formatted_time}")


    if "pls" in algorithms:
        print("Running probabilistic label spreading")

        from scripts.probabilistic_label_spreading import *

        start_time = time.time()

        batch_size = 1
        n_jobs = 1

        def process_combination_pls(dataset_name, prob_label_column, n_data, dim_reduction_technique, dim, alpha, k, n_samples, data_space, shm_name, shm_size):
            df = dataframe_from_shared_memory(shm_name, shm_size)
            
            try:

                result, _ = prob_label_spreading(dataset_name, df, data_space, prob_label_column, n_data, k, alpha, n_samples) # returns result including RMSE, Accuracy and parameters and the processed dataframe, which is not relevant here

            except ValueError as e:
                if "n_neighbors <= n_samples_fit" in str(e):
                    acc, l2_score = (np.nan, np.nan)
                else:
                    raise
            except Exception as e:
                raise

            return result


        all_results = [] #pd.DataFrame()

        for dataset_name in datasets:
            prob_label_columns = config['prob_label_columns'] # we may overwrite this for some datasets for convenience

            # Read DataFrame and determine its size
            df = pd.read_pickle(f"data/prob_data/{dataset_name}/{dataset_name}.pkl")
            # keep only the relevant parts of the data for this experiment
            data_spaces = [f"{embedder}_{dim}" for embedder in dim_reduction_techniques for dim in dimensions if embedder != "CLIP"]
            if not all(col in df.columns for col in prob_label_columns): # for datasets that have actual prob labels stored in the column "prob_label"
                prob_label_columns = ["prob_label"]

            if all(col in df.columns for col in data_spaces):
                columns = data_spaces + ["label","CLIP"] + prob_label_columns
                df = df[columns] 
            
            n_data = dataset_sizes[dataset_name] if dataset_sizes[dataset_name] != "all" else len(df)
            sample_sizes = parse_sample_sizes(orig_sample_sizes, n_data) # converts relative freqs into integers with n_data

            # Serialize the DataFrame into shared memory
            shm_name = f"{dataset_name}_shm"
            shm = dataframe_to_shared_memory(df, shm_name)
            shm_size = len(shm.buf)  # Store the size to read it back

            # Prepare combinations for all dim reduction techniques
            combinations = []

            if dataset_name == "TwoMoons": # for this dataset we do not have dim reduction
                dim = df["feature"][0].shape[0]
                data_space = "feature"
                dim_reduction_technique = "NONE"
                prob_label_column = "prob_label" # here, we also have just one set of prob labels in precisely this column
                # Add combinations for this specific technique
                combinations.extend([
                    (dataset_name, prob_label_column, n_data, dim_reduction_technique, dim, alpha, k, n_samples, data_space, shm_name, shm_size)
                    for alpha in alphas
                    for k in k_neighbors_pls
                    for n_samples in sample_sizes
                    for _ in range(multiple_runs)
                ])
           
            else: # else loop over the different dim reduction techniques
                for dim_reduction_technique in dim_reduction_techniques:
                    if dim_reduction_technique == "NONE":
                        dim = df["feature"][0].shape[0]
                        data_space = "feature"
                        # Add combinations for this specific technique
                        combinations.extend([
                            (dataset_name, prob_label_column, n_data, dim_reduction_technique, dim, alpha, k, n_samples, data_space, shm_name, shm_size)
                            for alpha in alphas
                            for k in k_neighbors_pls
                            for n_samples in sample_sizes
                            for _ in range(multiple_runs)
                            for prob_label_column in prob_label_columns
                        ])
                    elif dim_reduction_technique == "CLIP":
                        dim = 512
                        data_space = "CLIP"
                        # Add combinations for this specific technique
                        combinations.extend([
                            (dataset_name, prob_label_column, n_data, dim_reduction_technique, dim, alpha, k, n_samples, data_space, shm_name, shm_size)
                            for alpha in alphas
                            for k in k_neighbors_pls
                            for n_samples in sample_sizes
                            for _ in range(multiple_runs)
                            for prob_label_column in prob_label_columns
                        ])
                    else: #  if there is dim reduction also loop over the dimensions
                        for dim in dimensions:
                            data_space = f"{dim_reduction_technique}_{dim}"
                            combinations.extend([
                                (dataset_name, prob_label_column, n_data, dim_reduction_technique, dim, alpha, k, n_samples, data_space, shm_name, shm_size)
                                for alpha in alphas
                                for k in k_neighbors_pls
                                for n_samples in sample_sizes
                                for _ in range(multiple_runs)
                                for prob_label_column in prob_label_columns
                            ])

            # Execute all combinations in parallel with a single tqdm progress bar
            results = Parallel(n_jobs=n_jobs, batch_size=batch_size)(
                delayed(process_combination_pls)(*params) for params in tqdm(combinations, desc=f"Processing {dataset_name}")
            )

            # Append results to the main list
            all_results.extend(results)
            #all_results = pd.concat([all_results, results], ignore_index = True) # extend the results


            # Release shared memory
            shm.close()
            shm.unlink()

        # Convert results into a DataFrame
        results = pd.DataFrame()
        for result in all_results:
            results = pd.concat([results, result], ignore_index = True) # extend the results

        results.to_csv("results/pls_" + experiment + ".csv")

        elapsed_time = time.time() - start_time

        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        # Format the output
        formatted_time = f"{int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds"
        print(f"runtime: {formatted_time}")