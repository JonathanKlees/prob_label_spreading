import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scripts.plot import *
import scipy
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import brentq
import argparse
from scipy.stats import binomtest # for Wilson confidence intervals

np.random.seed(42)
plot_params = set_plot_layout(path_to_latex = '/home/jklees/texlive/bin/x86_64-linux') # set plot layout (optional)

def hoeffding_bound(W, q, eps):
    weights = W[q] 
    weights = weights / np.sum(weights) # make sure weights are normalized (should be already)
    prob_bound = 2 * np.exp( - 2 * (eps ** 2) / np.sum(weights**2) ) # sum of squared weights goes to zero, so bound gets tighter when weights are more spread out
    return prob_bound

def find_eps(W, q, delta, lo=0.0, hi=1.0):
    """
    Find smallest eps such that hoeffding_bound(eps) <= delta.
    lo, hi define an interval guaranteed to contain the solution.
    """
    # Ensure the interval brackets a root
    while hoeffding_bound(W, q, hi) > delta:
        hi *= 2   # expand search range exponentially
    
    # Solve p(eps) - delta = 0
    f = lambda eps: hoeffding_bound(W, q, eps) - delta
    return brentq(f, lo, hi)

def compute_bias_bound(X, q, W, L):
    weights = W[q] 
    weights = weights / np.sum(weights) # make sure weights are normalized (should be already

    dists = np.array([abs(X[j] - X[q]) for j in range(len(X))])

    error_bounds = np.minimum(1.0, L * dists) # error is always bounded by 1.0 and also by L * distance due to Lipschitz continuity

    bound = weights @ error_bounds # the bias bound based on the distances and the Lipschitz constant as well as the weight distribution
    return bound

def confidence_interval(X, W, estimates, q, L, delta = 0.05):
    eps = find_eps(W, q, delta) # determine smallest eps such that hoeffding bound <= delta
    bias_bound = compute_bias_bound(X, q, W, L) # bias bound based on all points
    lower_bound = estimates[q] - (eps + bias_bound)
    upper_bound = estimates[q] + (eps + bias_bound)
    return lower_bound, upper_bound, eps, bias_bound


###################### parameters (can be overridden via CLI)


parser = argparse.ArgumentParser(description="Run PLS confidence-interval script")
parser.add_argument("--N", type=int, default=2000, help="number of data points")
parser.add_argument("--k", type=int, default=20, help="number of neighbors")
parser.add_argument("--alpha", type=float, default=0.99, help="spreading intensity")
parser.add_argument("--deltas", type=str, default="0.1,0.01",
                    help='comma-separated list of confidence levels, e.g. "0.01,0.05,0.1"')

args = parser.parse_args()

N = args.N
k = args.k
alpha = args.alpha
deltas = [float(s) for s in args.deltas.split(",") if s.strip() != ""]

###########################

# def gt_function(x):
#     return 0.1 * x
# L = 0.1

def gt_function(x):
    return (np.sin(x) + 1) / 2

L = 0.5 # sin is 1-Lipschitz, here multiplied with 0.5

# X = np.random.uniform(0, 10, size=N)
# X = np.sort(X)
# X1 = np.random.uniform(0, 4, size=int(0.475*N)) 
# X2 = np.random.uniform(4, 6, size=int(0.05*N))
# X3 = np.random.uniform(6, 10, size=int(0.475*N))
# p1 = 0.66
# p2 = 0.06
# X1 = np.random.uniform(0, 6, size=int(p1*N))
# X2 = np.random.uniform(6, 8, size=int(p2*N))
# X3 = np.random.uniform(8, 10, size=N - int(p1*N) - int(p2*N))
# X = np.sort(np.concatenate([X1, X2, X3]))
X = np.random.uniform(0, 10, size=N)
X = np.sort(X)
x_range = np.linspace(0,10, 1000)

Y_true = np.array([gt_function(x) for x in X])
# sampled_indices = np.random.choice(range(N), size= N, replace=True) # draw as many feedbacks as data points (with replacement)
p1 = 0.66
p2 = 0.01

sampled_indices_1 = np.random.choice(range(int(0.6*N)), size= int(p1*N), replace=True) # draw as many feedbacks as data points (with replacement)
sampled_indices_2 = np.random.choice(range(int(0.6*N), int(0.8*N)), size= int(p2*N), replace=True) # draw as many feedbacks as data points (with replacement)
sampled_indices_3 = np.random.choice(range(int(0.8*N), N), size= int((1-p1-p2)*N), replace=True) # draw as many feedbacks as data points (with replacement)

sampled_indices = np.concatenate([sampled_indices_1, sampled_indices_2, sampled_indices_3]) 

abs_sampling_frequencies = np.bincount(sampled_indices, minlength=N) # abs. frequencies of feedbacks for each datapoint for normalization
Y = np.array([np.random.choice([1,0], p=[Y_true[i], 1 - Y_true[i]], size = abs_sampling_frequencies[i]).sum() for i in range(N)]) # for each datapoint conduct the number of experiments and add up the results

# PLS
nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric='euclidean').fit(X.reshape(-1, 1))       
raw_dists, nbr_indices_raw = nbrs.kneighbors(X.reshape(-1, 1))
n_data = X.shape[0]
# preprocessing for setting up a sparse adjacency matrix
indptr = np.asarray(range(0,n_data+1))*(k-1)
raw_dists = (np.asarray( raw_dists )[:,1:]).ravel() ** 2
nbr_indices = (np.asarray( nbr_indices_raw )[:,1:]).ravel()

# transform distances into similarities
mu = np.mean( np.asarray(raw_dists) )
similarity = np.exp( np.asarray( raw_dists ) * ( -1. / (2 * mu) ) )

# form the sparse matrix

C = scipy.sparse.csr_matrix((similarity, nbr_indices, indptr), shape=(n_data,n_data) , dtype=np.float64)

# symmetrize it
W = (C + C.T) / 2.

# compute the column sums to obtain 
D = np.asarray(np.sum(W, axis=-1)).ravel() 

# setup the graph Laplacian
D_inv_sqrt = scipy.sparse.spdiags( D ** (-0.5), 0, len(D), len(D) )
S = D_inv_sqrt @ W @ D_inv_sqrt

I = scipy.sparse.eye(len(D))

M = I - alpha * S

# determine the inverse of M
M_inv = scipy.sparse.linalg.inv(M)

W = M_inv.toarray() # heat kernel weights

# Normalization via absolute frequencies of feedbacks here instead of for loop.
F = W @ Y
C = W @ abs_sampling_frequencies
estimates = F / C

#  compute confidence intervals for all points and plot them together with the estimates

upper_bounds = {}
lower_bounds = {}
epsilons = {}
bias_bounds = {}

spacing = 10
indices_to_evaluate = list(range(0, N, spacing))
if (N - 1) not in indices_to_evaluate:
    indices_to_evaluate.append(N - 1)

for delta in deltas:
    x_values = []
    upper_bounds_delta = []
    lower_bounds_delta = []
    epsilons_delta = []
    bias_bounds_delta = []

    for q in indices_to_evaluate:  # subset of equally spaced points (include last index)
        x_values.append(X[q])
        lower, upper, eps, bias_bound = confidence_interval(X, W, estimates, q, L, delta=delta)
        lower_bounds_delta.append(lower) # max(lower, 0)
        upper_bounds_delta.append(upper) # min(upper, 1)
        epsilons_delta.append(eps)
        bias_bounds_delta.append(bias_bound)

        # exit() # debugging: print only for first point

    upper_bounds[delta] = np.array(upper_bounds_delta)
    lower_bounds[delta] = np.array(lower_bounds_delta)
    epsilons[delta] = np.array(epsilons_delta)
    bias_bounds[delta] = np.array(bias_bounds_delta)

# Wilson Score Confidence Intervals

wilson_upper_bounds = {}
wilson_lower_bounds = {}
for delta in deltas:
    x_values = []
    upper_bounds_delta = []
    lower_bounds_delta = []

    for q in indices_to_evaluate:  # subset of equally spaced points (include last index)
        x_values.append(X[q])
        k = int(np.floor(F[q]))  # number of virtual "successes"
        n = int(np.floor(C[q]))  # number of virtual trials
        result = binomtest(k=k, n=n)
        lower, upper = result.proportion_ci(confidence_level=1-delta, method = "wilson").low, result.proportion_ci(confidence_level=1-delta, method = "wilson").high
        lower_bounds_delta.append(lower)
        upper_bounds_delta.append(upper)

    wilson_upper_bounds[delta] = np.array(upper_bounds_delta)
    wilson_lower_bounds[delta] = np.array(lower_bounds_delta)

# fig, ax = plt.subplots(constrained_layout=True)
# ax.plot(x_range, [gt_function(x) for x in x_range], color = "coral",  label='Prob. Labels', zorder = 4)
# ax.scatter(X, estimates, s=5, label = f"PLS Estimates", zorder = 3)

# # ax.fill_between(x_values, lower_bounds, upper_bounds, color='lightgray', label='Confidence Intervals')
# colors = ["teal", "coral", "gray"]
# for i, delta in enumerate(deltas):
#     ax.fill_between(x_values, lower_bounds[delta], upper_bounds[delta], color=colors[i], alpha = 0.5, label=f'CI ($\delta={delta}$)', zorder = 1)
#     ax.plot(x_values, (lower_bounds[delta] + upper_bounds[delta]) / 2 +  epsilons[delta], color=colors[i], linestyle='dotted', alpha=0.8, zorder = 2, label = "Hoeffding" if delta == deltas[-1] else None)
#     ax.plot(x_values, (lower_bounds[delta] + upper_bounds[delta]) / 2 -  epsilons[delta], color=colors[i], linestyle='dotted', alpha=0.8, zorder = 2)
    
# # bias bound is independent of delta, so only plot once
# ax.plot(x_values, (lower_bounds[deltas[0]] + upper_bounds[deltas[0]]) / 2 + bias_bounds[deltas[0]], color="gray", linestyle='dashed', alpha=0.8, zorder = 2,  label = "Bias Bound")
# ax.plot(x_values, (lower_bounds[deltas[0]] + upper_bounds[deltas[0]]) / 2 - bias_bounds[deltas[0]], color="gray", linestyle='dashed', alpha=0.8, zorder = 2)


# plt.xlabel("X")
# plt.ylabel("Y")
# plt.legend()

# plt.savefig(f"plots/confidence_intervals.png")
# plt.show()

fig, ax = plt.subplots(figsize=(5.8, (7/16)*5.8), constrained_layout=True)
ax.plot(x_range, [gt_function(x) for x in x_range], lw=1.25, color = "coral",  label='Soft Labels', zorder = 4)
ax.scatter(X, estimates, c= "teal", s=1, label = f"PLS Estimates", zorder = 3)

y_text = -0.2
ax.axvspan(6, 8, color = "lightgray", alpha=0.4)
ax.text(
    7,
    y_text,
    "Few Feedbacks",
    ha="center",
    va="bottom"
)


# ax.fill_between(x_values, lower_bounds, upper_bounds, color='lightgray', label='Confidence Intervals')
colors = ["forestgreen"] *3 # , "gray"
linestyles = ['solid', 'dashed', 'dotted']
linewidth = 0.75
# my_colors = ["teal", "coral", "palevioletred", "slategrey", "forestgreen",  "darkmagenta",  "gold", "steelblue", "bisque", "darkseagreen"]

for i, delta in enumerate(deltas):
    # ax.fill_between(x_values, lower_bounds[delta], upper_bounds[delta], color=colors[i], alpha = 0.3, label=f'${int( (1-delta)*100 )}$\,\% Hoeffding CI', zorder = 1) # label=f'CI ($\delta={delta}$)'
    ax.plot(x_values, upper_bounds[delta], color=colors[i], linestyle=linestyles[i], lw = linewidth, zorder = 2)
    ax.plot(x_values, lower_bounds[delta], color=colors[i], linestyle=linestyles[i], lw = linewidth, label=f'${int( (1-delta)*100 )}$\,\% Hoeffding CI', zorder = 2)
    # ax.plot(x_values, (lower_bounds[delta] + upper_bounds[delta]) / 2 +  epsilons[delta], color=colors[i], linestyle='dotted', alpha=0.8, zorder = 2, label = "Hoeffding" if delta == deltas[-1] else None)
    # ax.plot(x_values, (lower_bounds[delta] + upper_bounds[delta]) / 2 -  epsilons[delta], color=colors[i], linestyle='dotted', alpha=0.8, zorder = 2)
    
# # bias bound is independent of delta, so only plot once
# ax.plot(x_values, (lower_bounds[deltas[0]] + upper_bounds[deltas[0]]) / 2 + bias_bounds[deltas[0]], color="gray", linestyle='dashed', alpha=0.8, zorder = 2,  label = "Bias Bound")
# ax.plot(x_values, (lower_bounds[deltas[0]] + upper_bounds[deltas[0]]) / 2 - bias_bounds[deltas[0]], color="gray", linestyle='dashed', alpha=0.8, zorder = 2)

colors = ["slategrey"] *3 # , "gray"
linestyles = ['solid', 'dashed', 'dotted']
for i, delta in enumerate(deltas):
    ax.plot(x_values, wilson_upper_bounds[delta], color=colors[i], linestyle=linestyles[i], lw = linewidth, zorder = 2)
    ax.plot(x_values, wilson_lower_bounds[delta], color=colors[i], linestyle=linestyles[i], lw = linewidth, label=f'${int( (1-delta)*100 )}$\,\% Wilson CI', zorder = 2)

plt.xlabel("X", labelpad = 2)
plt.ylabel("$P(Y | X)$", labelpad = 5)
plt.yticks([0,0.5,1])
plt.grid(axis = "y")
plt.legend( bbox_to_anchor=(0.32, 0.48), 
            frameon=True,
            fontsize=10,
            prop={'size': 7.5},
            handlelength=1,
            handletextpad=0.3,
            markerscale=0.7
            )

plt.savefig(f"plots/confidence_intervals.png")
plt.show()