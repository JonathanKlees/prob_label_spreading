import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy.sparse import lil_matrix, csr_matrix
import torch
from torch.nn.functional import kl_div 

def crowdsourcing_experiment(df, prob_label_column, n_samples):
    """based on the dataset containing prob. labels, n_samples crowdsourcing experiments are performed.
    I.e., n_samples times, a data point is sampled and a label is added to it according to its true probabilistic label"""
    classes = list(set(df["label"]))
    feedback = []
    for c in classes:
        feedback.append([]) # list of empty lists

    indices = np.random.choice(df.index, size = n_samples, replace = True) # choose indices randomly and allow for multiple selection
    for ind in indices:
        c = np.random.choice(classes, p = df[prob_label_column][ind]) #  sample a class according to the true prob label
        feedback[c].append(ind)
    return feedback


def baseline_spread(dataset, data_space, feedback, gamma):
    classes = list(set(dataset["label"]))
    data_points = np.array(dataset[data_space].to_list())
    num_classes = len(classes)
    p_hat = np.zeros((len(dataset), num_classes))

    # Calculate impact for each class
    for c in range(num_classes):
        # Get indices of data points labeled with class `c`
        indices = feedback[c]
        
        # Compute distances from all points to the points labeled with class `c`
        if len(indices) > 0:
            # Use scipy's cdist to compute distances in a vectorized way
            distances = cdist(data_points, data_points[indices], metric="sqeuclidean")
            # Compute the impact values
            impact = np.sum(np.exp(-gamma * distances), axis=1)
            # Accumulate the impacts in p_hat
            p_hat[:, c] += impact

    # Normalize p_hat
    p_hat_sum = np.sum(p_hat, axis=1, keepdims=True)
    p_hat = (p_hat + (1e-10 / num_classes)) / (p_hat_sum + 1e-10)

    # Update the dataset
    dataset["p_hat"] = list(p_hat)
    dataset["prediction"] = p_hat.argmax(axis=1) # prediction corresponds to the most probable label

    return dataset


# Baseline 1: Gaussian Process

def baseline_prob_label_spreading(df, data_space, prob_label_column, n_data, n_samples, gamma):
    if n_data == "all":
        n_data = len(df)
    else:
        df = df.sample(n = n_data) # random data subset
        df.reset_index(drop=True, inplace=True)
    
    feedback = crowdsourcing_experiment(df, prob_label_column, n_samples)
    df = baseline_spread(df, data_space, feedback, gamma)

    df["most_probable_class"] = list( np.array(df[prob_label_column].to_list()).argmax(axis = 1) )
    acc = len(df.query("most_probable_class == prediction"))/len(df)
    p = np.array(df[prob_label_column].to_list())
    p_hat = np.array(df["p_hat"].to_list())
    l2_score = sklearn.metrics.root_mean_squared_error(p_hat, p)
    kl = kl_div(torch.tensor(p_hat).log(), torch.tensor(p), reduction = "batchmean").item()
    mae = sklearn.metrics.mean_absolute_error(p_hat, p)
    tv = np.mean(np.max(np.abs(p - p_hat), axis=1))
    
    return acc, l2_score, kl, mae, tv



def k_nn_baseline_spread(dataset, data_space, feedback, k, metric='euclidean'):
    classes = list(set(dataset["label"]))
    # Prepare the data
    features = np.array(dataset[data_space].to_list())
    # Flatten feedback to determine which data points obtained at least one label during crowdsourcing
    labeled_indices = np.unique(np.array([ind for indices in feedback for ind in indices]))

    # track label frequencies in crowdsourcing feedback 
    label_counter = np.zeros((features.shape[0], len(classes)))
    for c in classes:
        indices = feedback[c]
        for i in indices:
            label_counter[i,c]+=1


    # Fit a NearestNeighbors model on the labeled data, i.e. the data points that were assigned at least one label during crowdsourcing
    labeled_data = features[labeled_indices]
    labeled_nbrs = NearestNeighbors(n_neighbors=k, metric=metric).fit(labeled_data)

    # Find k-nearest neighbors for all data points among the labeled data points
    distances, nearest_labeled_neighbors = labeled_nbrs.kneighbors(features)

    neighbors_indices = labeled_indices[nearest_labeled_neighbors]


    # aggregation: average the abs. frequencies of the k nearest neighbors for every unlabeled data point

    # Extract the abs. class frequencies of the nearest neighbors for all data points
    # This will create a matrix where each row contains the abs. class frequencies of the k-nearest neighbors for that row's corresponding data point
    neighbor_abs_freq = label_counter[neighbors_indices]

    # Calculate the sum of the abs. class freq. across the k-nearest neighbors for each data point
    abs_frequencies = neighbor_abs_freq.sum(axis=1)

    p_hat = abs_frequencies / abs_frequencies.sum(axis = 1, keepdims = True)

    dataset["p_hat"] = list(p_hat)
    dataset['prediction'] = p_hat.argmax(axis=1)

    return dataset

# Baseline 2: Perform crowdsourcing and obtain new labels by aggregation over k nearest neighbors out of labeled data points

def baseline_prob_label_spreading_kNN(df, data_space, prob_label_column, n_data, n_samples, k, metric = "euclidean"):
    classes = list(set(df["label"]))

    if n_data == "all":
        n_data = len(df)
    else:
        df = df.sample(n = n_data) # random data subset if not all data points are under consideration here
        df.reset_index(drop=True, inplace=True)

    df["p_hat"] = [[1.0 / len(classes)]*len(classes)]*len(df) # initialize column of estimated probs. with uniform distribution
    
    feedback = crowdsourcing_experiment(df, prob_label_column, n_samples) # crowdsourcing
    df = k_nn_baseline_spread(df, data_space, feedback, k, metric)
    df["most_probable_class"] = list( np.array(df[prob_label_column].to_list()).argmax(axis = 1) )

    acc = len(df.query("most_probable_class == prediction"))/len(df)
    p = np.array(df[prob_label_column].to_list())
    p_hat = np.array(df["p_hat"].to_list())
    l2_score = sklearn.metrics.root_mean_squared_error(p_hat, p)
    kl = kl_div(torch.tensor(p_hat).log(), torch.tensor(p), reduction = "batchmean").item()
    mae = sklearn.metrics.mean_absolute_error(p_hat, p)
    tv = np.mean(np.max(np.abs(p - p_hat), axis=1))

    return acc, l2_score, kl, mae, tv




########## make baseline more efficient using a sparse weight matrix (set very small weights to zero)

def efficient_baseline_spread(dataset, data_space, feedback, gamma, threshold_factor=1e5, chunk_size=100000):
    classes = list(set(dataset["label"]))
    data_points = np.array(dataset[data_space].to_list())
    num_classes = len(classes)
    p_hat = np.zeros((len(dataset), num_classes))

    # Calculate impact for each class
    for c in range(num_classes):
        # Get indices of data points labeled with class `c`
        indices = feedback[c]

        if len(indices) > 0:
            # Initialize a sparse matrix to accumulate impacts
            sparse_impact = lil_matrix((len(dataset), 1))

            # Compute distances in chunks
            for start_idx in range(0, len(dataset), chunk_size):
                end_idx = min(start_idx + chunk_size, len(dataset))

                # Compute distances for the current chunk
                distances_chunk = cdist(data_points[start_idx:end_idx], data_points[indices], metric="sqeuclidean")

                # Compute weights
                weights_chunk = np.exp(-gamma * distances_chunk)

                # Apply thresholding to create a sparse representation
                max_weights = weights_chunk.max(axis=1, keepdims=True)
                sparse_mask = weights_chunk >= (max_weights / threshold_factor)
                sparse_weights_chunk = weights_chunk * sparse_mask

                # Accumulate into the sparse matrix
                sparse_impact[start_idx:end_idx, 0] = sparse_weights_chunk.sum(axis=1).reshape(-1, 1)

            # Add the accumulated sparse impact to p_hat
            p_hat[:, c] += sparse_impact.toarray().flatten()

    # Normalize p_hat
    p_hat_sum = np.sum(p_hat, axis=1, keepdims=True)
    p_hat = (p_hat + (1e-10 / num_classes)) / (p_hat_sum + 1e-10)

    # Update the dataset
    dataset["p_hat"] = list(p_hat)
    dataset["prediction"] = p_hat.argmax(axis=1)  # prediction corresponds to the most probable label

    return dataset



# Baseline 1: Gaussian Process

def efficient_baseline_prob_label_spreading(df, data_space, prob_label_column, n_data, n_samples, gamma, threshold_factor=1e5, chunk_size=100000):
    if n_data == "all":
        n_data = len(df)
    else:
        df = df.sample(n = n_data) # random data subset
        df.reset_index(drop=True, inplace=True)
    
    feedback = crowdsourcing_experiment(df, prob_label_column, n_samples)
    df = efficient_baseline_spread(df, data_space, feedback, gamma, threshold_factor, chunk_size)

    df["most_probable_class"] = list( np.array(df[prob_label_column].to_list()).argmax(axis = 1) )
    acc = len(df.query("most_probable_class == prediction"))/len(df)
    p = np.array(df[prob_label_column].to_list())
    p_hat = np.array(df["p_hat"].to_list())
    l2_score = sklearn.metrics.root_mean_squared_error(p_hat, p)
    kl = kl_div(torch.tensor(p_hat).log(), torch.tensor(p), reduction = "batchmean").item()
    mae = sklearn.metrics.mean_absolute_error(p_hat, p)
    tv = np.mean(np.max(np.abs(p - p_hat), axis=1))

    return acc, l2_score, kl, mae, tv