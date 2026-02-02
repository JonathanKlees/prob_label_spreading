from scripts.plot import *
from scripts.baseline_prob_label_spreading import *
from scripts.probabilistic_label_spreading import prob_label_spreading

from ConfigSpace import ConfigurationSpace, Configuration, Integer, Categorical, Float, Normal, BetaFloatHyperparameter
from smac import HyperparameterOptimizationFacade, Scenario, RunHistory

import matplotlib.pyplot as plt
import numpy as np
import os

###################################   PARAMETERS  ####################################################
n_trials = 200 # number of trials for each optimization procedure
n_validations = 10 # how often the best parameter is evaluated to get a more robust estimate of the performance

# Set overwrite flag: If True, it will reprocess all combinations even if results already exist -> Deterministic optimization procedure would yield the same results
overwrite = False
# Set datasets and proportions to process
datasets = ["TwoMoons", "CIFAR10", "CIFAR10-H", "ANIMALS10", "EMNIST", "TinyImageNet", "MTSD"] # 
proportions = [0.001, 0.01, 0.1]
n_data = "all"
data_space = "CLIP_UMAP_20"

# File paths
results_path = "results/optimization_results.pkl"
history_path = "results/optimization_history.pkl"

n_workers_baselines = 1 # parallelization did not decrease runtime due to data sharing overhead
n_workers_pls = 1

###################################   DEFINITIONS  ####################################################

##############
# Define optimization objective for PLS
##############

def optimization_objective_pls(config: Configuration, dataset_name, df, data_space, prob_label_column, n_data, n_samples, seed: int = 0):
    k = config["k"]
    alpha = config["alpha"]
    result, processed_data = prob_label_spreading(dataset_name, df, data_space, prob_label_column, n_data, k, alpha, n_samples)
    rmse = result["RMSE"].values[0]
    return rmse

hyperparam_space_pls = ConfigurationSpace({
    "k": Integer("k", (2,100), log = False),
    #"alpha": Float("alpha", (0.0, 1.0 - 1e-6), log = False)
    "alpha": BetaFloatHyperparameter("alpha", lower = 0, upper = 1-1e-6, alpha = 5, beta = 1) # prior belief that alpha is close to 1 if n_samples is small
})
scenario_pls = Scenario(hyperparam_space_pls, deterministic=True, n_workers=n_workers_pls, n_trials=n_trials)

##############
# Define optimization objective for Gaussian Mixture Baseline
##############

def optimization_objective_gm(config: Configuration,  dataset_name, df, data_space, prob_label_column, n_data, n_samples, seed: int = 0):
    gamma = config["gamma"]
    #df = shared_df.read()

    acc, rmse, kl = baseline_prob_label_spreading(df, data_space, prob_label_column, n_data, n_samples, gamma)

    return rmse

hyperparam_space_gm = ConfigurationSpace({
    "gamma": Float("gamma", (1e-6, 1e6) , log = True)
})
scenario_gm = Scenario(hyperparam_space_gm, deterministic=True, n_workers=n_workers_baselines, n_trials=n_trials)

##############
# Define optimization objective for kNN Baseline
##############

def optimization_objective_knn(config: Configuration, dataset_name, df, data_space, prob_label_column, n_data, n_samples, seed: int = 0):
    k = config["k"]
    acc, rmse, kl = baseline_prob_label_spreading_kNN(df, data_space, prob_label_column, n_data, n_samples, k, metric = "euclidean")

    return rmse

    # hyperparam_space_knn and scenario_knn will be defined dynamically below as a suitable range for k depends on n_samples

def optimize(df, dataset_name, data_space, n_data, prob_label_column, n_samples, n_validations = 10):
    
    ############################
    # SMAC optimization for PLS
    ############################
    smac_pls = HyperparameterOptimizationFacade(scenario_pls, 
                                                lambda config, seed: optimization_objective_pls(config, dataset_name, df, data_space, prob_label_column, n_data, n_samples, seed),
                                                overwrite=True)
    incumbent_pls = smac_pls.optimize()
    
    # store tested parameter combinations and their costs
    data = []
    for config in smac_pls.runhistory.get_configs():
        param_values = dict(config)  # Extract parameter values
        cost = smac_pls.runhistory.get_cost(config)  # Get the associated cost
        data.append({"dataset_name": dataset_name, "data_space": data_space, "n_samples": n_samples, "algorithm": "PLS", "parameters": param_values, "rmse": cost})

    optimization_history_pls = pd.DataFrame(data)
    
    # Evaluate the best parameter multiple times
    validation_losses = []
    for _ in range(n_validations):
        loss = smac_pls.validate(incumbent_pls)
        validation_losses.append(loss)

    mean_loss = np.mean(validation_losses)
    std_loss = np.std(validation_losses)

    results_pls = pd.DataFrame([{"dataset_name": dataset_name, "data_space": data_space, "algorithm": "PLS", "n_samples": n_samples,
                                        "parameters": dict(incumbent_pls), "rmse": mean_loss, "rmse_std": std_loss, "list_of_rmses": validation_losses}])  

    ############################
    # SMAC optimization for GM Baseline
    ############################
    smac_gm = HyperparameterOptimizationFacade(scenario_gm,
                                               lambda config, seed: optimization_objective_gm(config, dataset_name, df, data_space, prob_label_column, n_data, n_samples, seed),
                                               overwrite=True)
    incumbent_gm = smac_gm.optimize()
    
    # store tested parameter combinations and their costs
    data = []
    for config in smac_gm.runhistory.get_configs():
        param_values = dict(config)  # Extract parameter values
        cost = smac_gm.runhistory.get_cost(config)  # Get the associated cost
        data.append({"dataset_name": dataset_name, "data_space": data_space, "n_samples": n_samples, "algorithm": "GM", "parameters": param_values, "rmse": cost})

    optimization_history_gm = pd.DataFrame(data)
    
    # Evaluate the best parameter multiple times
    validation_losses = []
    for _ in range(n_validations):
        loss = smac_gm.validate(incumbent_gm)
        validation_losses.append(loss)

    mean_loss = np.mean(validation_losses)
    std_loss = np.std(validation_losses)

    results_gm = pd.DataFrame([{"dataset_name": dataset_name, "data_space": data_space, "algorithm": "GM", "n_samples": n_samples,
                                        "parameters": dict(incumbent_gm), "rmse": mean_loss, "rmse_std": std_loss, "list_of_rmses": validation_losses}])  
    
    ############################
    # SMAC optimization for kNN Baseline 
    # -> dynamically adjust the hyperparameter space as k should not be larger than the number of samples
    ############################
    hyperparam_space_knn = ConfigurationSpace({
        "k": Integer("k", (1, min(n_samples, 1000)), log = False) # in fact n_samples / n_classes would be appropriate but we do not restrict to much here 
    })
    scenario_knn = Scenario(hyperparam_space_knn, deterministic=True, n_workers=n_workers_baselines, n_trials=n_trials)

    smac_knn = HyperparameterOptimizationFacade(scenario_knn,
                                                lambda config, seed: optimization_objective_knn(config, dataset_name, df, data_space, prob_label_column, n_data, n_samples, seed),
                                                overwrite=True)
    incumbent_knn = smac_knn.optimize()
    
    # store tested parameter combinations and their costs
    data = []
    for config in smac_knn.runhistory.get_configs():
        param_values = dict(config)  # Extract parameter values
        cost = smac_knn.runhistory.get_cost(config)  # Get the associated cost
        data.append({"dataset_name": dataset_name, "data_space": data_space, "n_samples": n_samples, "algorithm": "kNN", "parameters": param_values, "rmse": cost})

    optimization_history_knn = pd.DataFrame(data)
    
    # Evaluate the best parameter multiple times
    validation_losses = []
    for _ in range(n_validations):
        loss = smac_knn.validate(incumbent_knn)
        validation_losses.append(loss)

    mean_loss = np.mean(validation_losses)
    std_loss = np.std(validation_losses)

    results_knn = pd.DataFrame([{"dataset_name": dataset_name, "data_space": data_space, "algorithm": "kNN", "n_samples": n_samples,
                                        "parameters": dict(incumbent_knn), "rmse": mean_loss, "rmse_std": std_loss, "list_of_rmses": validation_losses}])  
    ############################
    # gather results
    ############################
    
    optimization_history = pd.concat([optimization_history_pls, optimization_history_gm, optimization_history_knn])
    results = pd.concat([results_pls, results_gm, results_knn])
    
    return results,  optimization_history


def plot(optimization_history, algorithms, dataset_name, n_samples, filename = None):
        
    # Determine the grid size: at most 2x2
    if len(algorithms) == 1:
        rows, cols = 1, 1
    elif len(algorithms) == 2:
        rows, cols = 1, 2
    else:
        rows, cols = 2, 2

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), constrained_layout=True)

    # Flatten axes for easier iteration (even if it's originally 2D)
    axes = np.array(axes).flatten()
    
    for q, algorithm in enumerate(algorithms):
        
        ax = axes[q]     
        optim_history = optimization_history.query(f"algorithm == '{algorithm}' and dataset_name == '{dataset_name}' and n_samples == {n_samples}") 
        
        if len(list(optim_history["parameters"][0].keys())) == 1: # one hyperparameter
            
            parameter = list(optim_history["parameters"][0].keys())[0]   
            X = []
            Y = []
            # Plot all trials
            
            for i in optim_history.index:
                x = optim_history["parameters"][i][parameter]
                y = optim_history["rmse"][i]
                X.append(x), Y.append(y)

            plotdata = pd.DataFrame({parameter: X, "loss": Y})
            
            scatterplot = ax.scatter(plotdata.iloc[:, 0], plotdata.iloc[:, 1])

            xlabel = "\\gamma" if parameter == "gamma" else parameter
            ax.set_xlabel(f"${xlabel}$")
            ax.set_ylabel("RMSE")
            
            # Plot incumbent
            min_rmse_idx = plotdata["loss"].idxmin()
            ax.scatter(plotdata.iloc[min_rmse_idx, 0], plotdata["loss"].min(), c="red", marker="x")
            
            ax.set_xscale("log")
            
            title = "Gaussian Mixture" if algorithm == "GM" else "kNN" if algorithm == "kNN" else "Probabilistic Label Spreading"
            ax.set_title(title)
            
        elif len(list(optim_history["parameters"][0].keys())) == 2: # two hyperparameters
            parameter_1 = list(optim_history["parameters"][0].keys())[0]   
            parameter_2 = list(optim_history["parameters"][0].keys())[1]

            X = []
            Y = []
            Z = []
            # Plot all trials
            for i in optim_history.index:
                x = optim_history["parameters"][i][parameter_1]
                y = optim_history["parameters"][i][parameter_2]
                z = optim_history["rmse"][i]
                X.append(x), Y.append(y), Z.append(z)

            plotdata = pd.DataFrame({parameter_1: X, parameter_2: Y, "loss": Z})
            
            scatterplot = ax.scatter(plotdata.iloc[:, 0], plotdata.iloc[:, 1], c = plotdata.iloc[:, 2], cmap = "Spectral_r") # plotdata, x, y, =z, cmap = "viridis", marker="o"

            xlabel = "\\alpha" if parameter_1 == "alpha" else parameter_1
            ylabel = "\\alpha" if parameter_2 == "alpha" else parameter_2
            ax.set_xlabel(f"${xlabel}$")
            ax.set_ylabel(f"${ylabel}$")
            
            # Plot incumbent
            min_rmse_idx = plotdata["loss"].idxmin()
            ax.scatter(plotdata.iloc[min_rmse_idx, 0], plotdata.iloc[min_rmse_idx, 1], c="purple", marker="x", s = 100)
            
            title = "Gaussian Mixture" if algorithm == "GM" else "kNN" if algorithm == "kNN" else "Probabilistic Label Spreading"
            ax.set_title(title)

            cbar = fig.colorbar(scatterplot, shrink=0.9)
            cbar.set_label("RMSE", labelpad=-25, y=1.05, rotation=0)
            
    
    if len(algorithms) < rows * cols:
        axes[-1].axis("off")
        
    if filename:
        plt.savefig("plots/" + filename)
    #plt.show()
    
###################################   MAIN CODE  ####################################################

# Load previous results if they exist
if os.path.exists(results_path) and os.path.exists(history_path):
    joint_results = pd.read_pickle(results_path)
    joint_optim_history = pd.read_pickle(history_path)
    print("Resuming from existing results...")
else:
    joint_results = pd.DataFrame()
    joint_optim_history = pd.DataFrame()

# Ensure the DataFrame has necessary columns
if joint_results.empty:
    processed_combinations = set()
else:
    processed_combinations = set(zip(joint_results["dataset"], joint_results["proportion"]))

for dataset_name in datasets:
    if dataset_name in ["CIFAR10-H", "MTSD", "TwoMoons"]: # these datasets have different column names for the prob_label
        prob_label_column = "prob_label"
    else:
        prob_label_column = "prob_label_effnetb0"

    if dataset_name == "TwoMoons":
        data_space = "feature"

    df = pd.read_pickle(f"data/prob_data/{dataset_name}/{dataset_name}.pkl")
    
    # keep only relevant columns to reduce memory usage
    df = df[[data_space, prob_label_column, "label"]] # label is only required to determine the accuracy w.r.t. to hard labels

    for proportion in proportions:
        # Check if combination already exists
        if (dataset_name, proportion) in processed_combinations and not overwrite:
            print(f"Skipping {dataset_name} with {proportion * len(df)} samples (already processed).")
            continue

        if proportion * len(df) <= 1:
            print(f"Skipping {dataset_name} with {proportion * len(df)} samples (too few samples).")
            continue

        print(f"Processing {dataset_name} with {proportion * len(df)} samples...")
        n_samples = int(proportion * len(df))
        results, optim_history = optimize(df, dataset_name, data_space, n_data, prob_label_column, n_samples, n_validations)
                
        # create figure to evaluate the optimization results visually
        plot(optim_history, ["GM", "kNN", "PLS"], dataset_name, n_samples, filename=f"optimization_history_{dataset_name}_{n_samples}_samples.png")

        # Add dataset name and proportion to results for tracking
        results["dataset"] = dataset_name
        results["proportion"] = proportion

        # Append new results
        joint_results = pd.concat([joint_results, results], ignore_index=True)
        joint_optim_history = pd.concat([joint_optim_history, optim_history], ignore_index=True)

        # Save intermediate results after processing each dataset-proportion combination
        joint_results.to_pickle(results_path)
        joint_optim_history.to_pickle(history_path)
        print(f"Saved intermediate results after {dataset_name}, proportion={proportion}")

print("Optimization complete. Final results saved!")
