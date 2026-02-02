import os
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
from matplotlib import colors
from matplotlib.ticker import ScalarFormatter, FuncFormatter

from matplotlib.colors import LogNorm, Normalize
from matplotlib.colors import ListedColormap
import seaborn as sns

def format_percentage(value):
    value *= 100
    if value.is_integer():
        return "{:.0f}".format(value)
    elif value * 10 % 1 == 0:
        return "{:.1f}".format(value)
    else:
        return "{:.2f}".format(value)

my_colors = ["teal", "coral", "palevioletred", "slategrey", "forestgreen",  "darkmagenta",  "gold", "steelblue", "bisque", "darkseagreen"]

# Add the LaTeX bin directory to the PATH
def set_plot_layout(path_to_latex):
    os.environ['PATH'] = os.path.expanduser(path_to_latex) + os.pathsep + os.environ['PATH']
    plt.rc('text', usetex=True)
    plt.rc('font', family='Computer Modern Roman')

    matplotlib.rcParams['lines.linewidth'] = 1
    matplotlib.rcParams['axes.labelpad'] = 10
    plt.rcParams['font.sans-serif'] = "cmr10"
    plt.rcParams['font.size'] = 10.5

    matplotlib.rcParams['axes.titlesize'] = 10.5
    matplotlib.rcParams['axes.labelsize'] = 10.5
    matplotlib.rcParams['xtick.labelsize'] = 10.5
    matplotlib.rcParams['ytick.labelsize'] = 10.5

    matplotlib.rcParams["axes.prop_cycle"] = cycler('color', ["teal", "coral", "palevioletred", "slategrey", "forestgreen",  "darkmagenta",  "gold", "steelblue", "bisque", "darkseagreen"])
    matplotlib.rcParams['legend.framealpha'] = 1
    matplotlib.rcParams['figure.figsize']=[5.8, (9/16)*5.8]
    matplotlib.rcParams['axes.formatter.use_mathtext'] = True
    matplotlib.rcParams['savefig.bbox'] = 'tight'
    matplotlib.rcParams['legend.fontsize'] = 9
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['axes.edgecolor'] = "slategray"
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['figure.titlesize'] = 'medium'

    plot_params = dict()
    plot_params["quadratic_image_size"] = (3.2625, 3.2625)
    plot_params["large_image_size"] = (5.8, 6.525)
    plot_params["large_quadratic_image_size"] = (5.8, 5.8)

    plot_params["small_ms"] = 1
    plot_params["large_ms"] = 2
    plot_params["huge_annot_size"] = 25

    plot_params["colors"] = ["teal", "coral", "palevioletred", "slategrey", "forestgreen",  "darkmagenta",  "gold", "steelblue", "bisque", "darkseagreen"]

    return plot_params


def contour_landscape(results, params, quantity = "RMSE", filename = None):
    """This function draws a contour landscape over the parameter space for either the RMSE or the Accuracy"""
    # extract information from the results dataframe

    X = np.sort(np.array(list(set(results[params[0]]))))
    Y = np.sort(np.array(list(set(results[params[1]]))))
    scores = [[results.query(params[0] + " == "+str(x)+ " and " + params[1] + " == "+str(y))["RMSE"].iloc[0] for x in X] for y in Y]
    accuracies = [[results.query(params[0] + " == "+str(x)+ " and " + params[1] + " == "+str(y))["Accuracy"].iloc[0] for x in X] for y in Y]

    cs = ["teal",  "coral"]
    diverging_cmap = colors.LinearSegmentedColormap.from_list("cmap_name", cs)

    if quantity == "RMSE":
        M = np.array(scores)
    elif quantity == "Accuracy":
        M = np.array(accuracies)
    else:
        print("Quantity not supported. Choose one of RMSE and Accuracy")

    fig = plt.figure()
    ax = plt.gca()
    plot = plt.contourf(M, cmap = diverging_cmap, alpha = 0.7)
    plt.xlabel(params[0])
    plt.ylabel(params[1])

    ax.set_xticks(np.arange(len(X)), X, rotation = 45)
    #ax.set_xticklabels([str(format_percentage(x / n_data)) + "\%" for x in X], rotation = 45)

    plt.yticks(np.arange(len(Y)), Y)

    cbar = fig.colorbar(plot)
    cbar.ax.set_title(quantity)

    if filename:
        plt.savefig("plots/" + filename)

    plt.show()

def performance_heatmap(results, params, quantity = "RMSE", filename = None, percentage = False, n_data = None):
    # extract information from the results dataframe
    X = np.sort(np.array(list(set(results[params[0]]))))
    Y = np.sort(np.array(list(set(results[params[1]]))))
    
    if quantity == "RMSE":
        scores = [[results.query(params[0] + " == "+str(x)+ " and " + params[1] + " == "+str(y))["RMSE"].iloc[0] for x in X] for y in Y]
        M = np.array(scores)
        cmap = "Spectral_r"
    elif quantity == "MAE":
        scores = [[results.query(params[0] + " == "+str(x)+ " and " + params[1] + " == "+str(y))["MAE"].iloc[0] for x in X] for y in Y]
        M = np.array(scores)
        cmap = "Spectral_r"
    elif quantity == "TV":
        scores = [[results.query(params[0] + " == "+str(x)+ " and " + params[1] + " == "+str(y))["TV"].iloc[0] for x in X] for y in Y]
        M = np.array(scores)
        cmap = "Spectral_r"
    elif quantity == "KL":
        scores = [[results.query(params[0] + " == "+str(x)+ " and " + params[1] + " == "+str(y))["KL"].iloc[0] for x in X] for y in Y]
        M = np.array(scores)
        cmap = "Spectral_r"
    elif quantity == "Accuracy":
        accuracies = [[results.query(params[0] + " == "+str(x)+ " and " + params[1] + " == "+str(y))["Accuracy"].iloc[0] for x in X] for y in Y]
        M = np.array(accuracies)
        cmap = "Spectral"
    else:
        print("Quantity not supported. Choose one of RMSE, MAE, KL, TV and Accuracy")

    # make heatmap
    fig = plt.figure()
    ax = plt.gca()
    plot = plt.pcolormesh(M, cmap = cmap, alpha = 0.9)
    xlabel = "$n_{samples}$" if params[0] == "n_samples" else params[0]
    xlabel = "$k$" if params[0] == "k" else xlabel
    ylabel = "$\\alpha$" if params[1] == "alpha" else params[1]
    ylabel = "$k$" if params[1] == "k" else ylabel
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xticks([0.5 +  i for i in range(len(X))], X, rotation = 45) # ,[str(format_percentage(x / n_data)) + "\%" for x in sample_sizes]
    if percentage and n_data:   
        plt.xticks([0.5 + i for i in range(len(X))], [str(format_percentage(x / n_data)) + "\%" for x in X], rotation = 45)
        plt.xlabel("budget")
    plt.yticks([0.5 + i for i in range(len(Y))], Y)

    cbar = fig.colorbar(plot)
    cbar.ax.set_title(quantity)

    if filename:
        plt.savefig("plots/" + filename)

    plt.show()

# def performance_trajectories(results, parameter, parameter_values, quantity = "RMSE", display_std= False, filename = None):
#     # evaluate the dependence on the number of samples for suitable values of alpha

#     sample_sizes = np.sort(np.array(list(set(results["n_samples"]))))

#     fig = plt.figure()
#     ax = plt.gca()

#     for p in parameter_values:

#         accuracies = []
#         scores = []
#         accuracies_std = []
#         scores_std = []

#         for n_samples in sample_sizes: 
#             accuracies.append(results.query(parameter + " == "+str(p)+ " and n_samples == "+str(n_samples))["Accuracy"].iloc[0])
#             scores.append(results.query(parameter + " == "+str(p)+ " and n_samples == "+str(n_samples))["RMSE"].iloc[0])
#             if "Accuracy_std" in results.columns:
#                 accuracies_std.append(results.query(parameter + " == "+str(p)+ " and n_samples == "+str(n_samples))["Accuracy_std"].iloc[0])
#                 scores_std.append(results.query(parameter + " == "+str(p)+ " and n_samples == "+str(n_samples))["RMSE_std"].iloc[0])

#         if parameter == "alpha":
#             label = r"$\alpha$"
#         elif parameter == "gamma":
#             label = r"$\gamma$"
#         else:
#             label = parameter

#         if display_std:

#             if quantity == "RMSE":
#                 ax.plot(sample_sizes, scores, marker = "o", label=f"{label}\,=\,{p:.3g}")
#                 ax.fill_between(sample_sizes, np.array(scores) - np.array(scores_std), np.array(scores) + np.array(scores_std), alpha = 0.2)
#             elif quantity == "Accuracy":
#                 ax.plot(sample_sizes, accuracies, marker = "o", label=f"{label}\,=\,{p:.3g}")
#                 ax.fill_between(sample_sizes, np.array(accuracies) - np.array(accuracies_std), np.array(accuracies) + np.array(accuracies_std), alpha = 0.2)
#             else:
#                 print("Quantity not supported. Choose one of RMSE and Accuracy")
#         else:
#             if quantity == "RMSE":
#                 ax.plot(sample_sizes, scores, marker = "o", label=f"{label}\,=\,{p:.3g}")
#             elif quantity == "Accuracy":
#                 ax.plot(sample_sizes, accuracies, marker = "o", label=f"{label}\,=\,{p:.3g}")
#             else:
#                 print("Quantity not supported. Choose one of RMSE and Accuracy")
            
#     ax.legend()
#     ax.set_xscale("log")
#     plt.ylabel(quantity)
#     plt.xlabel("$n_{samples}$")
#     if filename:
#         plt.savefig("plots/"+filename)
    
#     return fig, ax

def performance_trajectories(results, parameter, parameter_values, quantity = "RMSE", display_std= False, filename = None):
    # evaluate the dependence on the number of samples for suitable values of alpha

    sample_sizes = np.sort(np.array(list(set(results["n_samples"]))))

    fig = plt.figure(figsize=(5.8, (7/16)*5.8))
    ax = plt.gca()

    for p in parameter_values:

        accuracies = []
        scores = []
        accuracies_std = []
        scores_std = []

        for n_samples in sample_sizes: 
            accuracies.append(results.query(parameter + " == "+str(p)+ " and n_samples == "+str(n_samples))["Accuracy"].iloc[0])
            scores.append(results.query(parameter + " == "+str(p)+ " and n_samples == "+str(n_samples))["RMSE"].iloc[0])
            if "Accuracy_std" in results.columns:
                accuracies_std.append(results.query(parameter + " == "+str(p)+ " and n_samples == "+str(n_samples))["Accuracy_std"].iloc[0])
                scores_std.append(results.query(parameter + " == "+str(p)+ " and n_samples == "+str(n_samples))["RMSE_std"].iloc[0])

        if parameter == "alpha":
            label = r"$\alpha$"
        elif parameter == "gamma":
            label = r"$\gamma$"
        else:
            label = parameter

        if display_std:

            if quantity == "RMSE":
                ax.plot(sample_sizes, scores, marker = "o", label=f"{label}\,=\,{p:.3g}")
                ax.fill_between(sample_sizes, np.array(scores) - np.array(scores_std), np.array(scores) + np.array(scores_std), alpha = 0.2)
            elif quantity == "Accuracy":
                ax.plot(sample_sizes, accuracies, marker = "o", label=f"{label}\,=\,{p:.3g}")
                ax.fill_between(sample_sizes, np.array(accuracies) - np.array(accuracies_std), np.array(accuracies) + np.array(accuracies_std), alpha = 0.2)
            else:
                print("Quantity not supported. Choose one of RMSE and Accuracy")
        else:
            if quantity == "RMSE":
                ax.plot(sample_sizes, scores, marker = "o", label=f"{label}\,=\,{p:.3g}")
            elif quantity == "Accuracy":
                ax.plot(sample_sizes, accuracies, marker = "o", label=f"{label}\,=\,{p:.3g}")
            else:
                print("Quantity not supported. Choose one of RMSE and Accuracy")
            
    ax.legend()
    ax.set_xscale("log")
    plt.ylabel(quantity, labelpad = 5)
    plt.xlabel("$n_{samples}$")
    if filename:
        plt.savefig("plots/"+filename, bbox_inches='tight', dpi = 300)
    
    return fig, ax


# def compare_performance_trajectories(df1, param1, paramvalues1, df2, param2, paramvalues2, df3 = None, param3 = None,
#  paramvalues3 = None, quantity = "RMSE", display_std = False, filename = None, linestyles = ["solid", "dashed", "dotted"], markers = ["o", "s", "^"]):
#     # evaluate the dependence on the number of samples for suitable values of alpha

#     sample_sizes_1 = np.sort(np.array(list(set(df1["n_samples"]))))
#     sample_sizes_2 = np.sort(np.array(list(set(df2["n_samples"]))))
#     if param3:
#         sample_sizes_3 = np.sort(np.array(list(set(df3["n_samples"]))))

#     fig = plt.figure()
#     ax = plt.gca()

#     for p in paramvalues1: # plot the first set of lines
#         accuracies = []
#         scores = []
#         accuracies_std = []
#         scores_std = []

#         for n_samples in sample_sizes_1: 
#             accuracies.append(df1.query(param1 + " == "+str(p)+ " and n_samples == "+str(n_samples))["Accuracy"].iloc[0])
#             scores.append(df1.query(param1 + " == "+str(p)+ " and n_samples == "+str(n_samples))["RMSE"].iloc[0])
#             if "Accuracy_std" in df1.columns:
#                     accuracies_std.append(df1.query(param1 + " == "+str(p)+ " and n_samples == "+str(n_samples))["Accuracy_std"].iloc[0])
#                     scores_std.append(df1.query(param1 + " == "+str(p)+ " and n_samples == "+str(n_samples))["RMSE_std"].iloc[0])

#         label = "PLS with $\\alpha$" if param1 == "alpha" else "GKR with $ \\gamma$" if param1=="gamma" else "kNN with $" + param1 + "$"

#         if display_std:

#             if quantity == "RMSE":
#                 ax.plot(sample_sizes_1, scores, marker = markers[0], ls = linestyles[0],  label=f"{label}\,=\,{p:.3g}")
#                 ax.fill_between(sample_sizes_1, np.array(scores) - np.array(scores_std), np.array(scores) + np.array(scores_std), alpha = 0.2)
#             elif quantity == "Accuracy":
#                 ax.plot(sample_sizes_1, accuracies, marker = markers[0], ls = linestyles[0], label=f"{label}\,=\,{p:.3g}")
#                 ax.fill_between(sample_sizes_1, np.array(accuracies) - np.array(accuracies_std), np.array(accuracies) + np.array(accuracies_std), alpha = 0.2)
#             else:
#                 print("Quantity not supported. Choose one of RMSE and Accuracy")
#         else:
#             if quantity == "RMSE":
#                 ax.plot(sample_sizes_1, scores, marker = markers[0], ls = linestyles[0], label=f"{label}\,=\,{p:.3g}")
#             elif quantity == "Accuracy":
#                 ax.plot(sample_sizes_1, accuracies, marker = markers[0], ls = linestyles[0], label=f"{label}\,=\,{p:.3g}")
#             else:
#                 print("Quantity not supported. Choose one of RMSE and Accuracy")

#     for p in paramvalues2: # plot the second set of lines
#         accuracies = []
#         scores = []
#         accuracies_std = []
#         scores_std = []

#         for n_samples in sample_sizes_2: 
#             accuracies.append(df2.query(param2 + " == "+str(p)+ " and n_samples == "+str(n_samples))["Accuracy"].iloc[0])
#             scores.append(df2.query(param2 + " == "+str(p)+ " and n_samples == "+str(n_samples))["RMSE"].iloc[0])
#             if "Accuracy_std" in df2.columns:
#                     accuracies_std.append(df2.query(param2 + " == "+str(p)+ " and n_samples == "+str(n_samples))["Accuracy_std"].iloc[0])
#                     scores_std.append(df2.query(param2 + " == "+str(p)+ " and n_samples == "+str(n_samples))["RMSE_std"].iloc[0])

#         label = "PLS with $\\alpha$" if param2 == "alpha" else "GKR with $ \\gamma$" if param2=="gamma" else "kNN with $" + param2 + "$"


#         if display_std:

#             if quantity == "RMSE":
#                 ax.plot(sample_sizes_2, scores, marker = markers[1], ls = linestyles[1],  label=f"{label}\,=\,{p:.3g}")
#                 ax.fill_between(sample_sizes_2, np.array(scores) - np.array(scores_std), np.array(scores) + np.array(scores_std), alpha = 0.2)
#             elif quantity == "Accuracy":
#                 ax.plot(sample_sizes_2, accuracies, marker = markers[1], ls = linestyles[1], label=f"{label}\,=\,{p:.3g}")
#                 ax.fill_between(sample_sizes_2, np.array(accuracies) - np.array(accuracies_std), np.array(accuracies) + np.array(accuracies_std), alpha = 0.2)
#             else:
#                 print("Quantity not supported. Choose one of RMSE and Accuracy")
#         else:
#             if quantity == "RMSE":
#                 ax.plot(sample_sizes_2, scores, marker = markers[1], ls = linestyles[1], label=f"{label}\,=\,{p:.3g}")
#             elif quantity == "Accuracy":
#                 ax.plot(sample_sizes_2, accuracies, marker = markers[1], ls = linestyles[1], label=f"{label}\,=\,{p:.3g}")
#             else:
#                 print("Quantity not supported. Choose one of RMSE and Accuracy")


#     if param3:
#         for p in paramvalues3: # plot the second set of lines
#             accuracies = []
#             scores = []
#             accuracies_std = []
#             scores_std = []

#             for n_samples in sample_sizes_3: 
#                 accuracies.append(df3.query(param3 + " == "+str(p)+ " and n_samples == "+str(n_samples))["Accuracy"].iloc[0])
#                 scores.append(df3.query(param3 + " == "+str(p)+ " and n_samples == "+str(n_samples))["RMSE"].iloc[0])
#                 if "Accuracy_std" in df3.columns:
#                     accuracies_std.append(df3.query(param3 + " == "+str(p)+ " and n_samples == "+str(n_samples))["Accuracy_std"].iloc[0])
#                     scores_std.append(df3.query(param3 + " == "+str(p)+ " and n_samples == "+str(n_samples))["RMSE_std"].iloc[0])

#             label = "PLS with $\\alpha$" if param3 == "alpha" else "GKR with $ \\gamma$" if param3=="gamma" else "kNN with $" + param3 + "$"

#             if display_std:
#                 if quantity == "RMSE":
#                     ax.plot(sample_sizes_3, scores, marker = markers[2], ls = linestyles[2], label=f"{label}\,=\,{p:.3g}")
#                     ax.fill_between(sample_sizes_3, np.array(scores) - np.array(scores_std), np.array(scores) + np.array(scores_std), alpha = 0.2)
#                 elif quantity == "Accuracy":
#                     ax.plot(sample_sizes_3, accuracies, marker = markers[2], ls = linestyles[2], label=f"{label}\,=\,{p:.3g}")
#                     ax.fill_between(sample_sizes_3, np.array(accuracies) - np.array(accuracies_std), np.array(accuracies) + np.array(accuracies_std), alpha = 0.2)
#                 else:
#                     print("Quantity not supported. Choose one of RMSE and Accuracy")
#             else:
#                 if quantity == "RMSE":
#                     ax.plot(sample_sizes_3, scores, marker = markers[2], ls = linestyles[2], label=f"{label}\,=\,{p:.3g}")
#                 elif quantity == "Accuracy":
#                     ax.plot(sample_sizes_3, accuracies, marker = markers[2], ls = linestyles[2], label=f"{label}\,=\,{p:.3g}")
#                 else:
#                     print("Quantity not supported. Choose one of RMSE and Accuracy")


#     #ax.legend(ncol = 3, loc="lower center", bbox_to_anchor=(0.5, 1), frameon=False)
#     ax.set_xscale("log")
#     plt.ylabel(quantity)
#     plt.xlabel("$n_{samples}$")
#     if filename:
#         plt.savefig("plots/"+filename)
    
#     return fig, ax

def compare_performance_trajectories(df1, param1, paramvalues1, df2, param2, paramvalues2, df3 = None, param3 = None,
 paramvalues3 = None, quantity = "RMSE", display_std = False, filename = None, linestyles = ["solid", "dashed", "dotted"], markers = ["o", "s", "^"]):
    # evaluate the dependence on the number of samples for suitable values of alpha

    sample_sizes_1 = np.sort(np.array(list(set(df1["n_samples"]))))
    sample_sizes_2 = np.sort(np.array(list(set(df2["n_samples"]))))
    if param3:
        sample_sizes_3 = np.sort(np.array(list(set(df3["n_samples"]))))

    fig = plt.figure(figsize=(5.8, (7/16)*5.8))
    ax = plt.gca()

    for p in paramvalues1: # plot the first set of lines
        scores = []
        scores_std = []

        for n_samples in sample_sizes_1: 
            scores.append(df1.query(param1 + " == "+str(p)+ " and n_samples == "+str(n_samples))[quantity].iloc[0])
            if display_std and (quantity + "_std") in df1.columns:                    
                scores_std.append(df1.query(param1 + " == "+str(p)+ " and n_samples == "+str(n_samples))[quantity + "_std"].iloc[0])

        label = "PLS with $\\alpha$" if param1 == "alpha" else "GKR with $ \\gamma$" if param1=="gamma" else "kNN with $" + param1 + "$"

        if display_std:
            ax.plot(sample_sizes_1, scores, marker = markers[0], ls = linestyles[0],  label=f"{label}\,=\,{p:.3g}")
            ax.fill_between(sample_sizes_1, np.array(scores) - np.array(scores_std), np.array(scores) + np.array(scores_std), alpha = 0.2)

        else:
            ax.plot(sample_sizes_1, scores, marker = markers[0], ls = linestyles[0], label=f"{label}\,=\,{p:.3g}")

    for p in paramvalues2: # plot the second set of lines
        scores = []
        scores_std = []

        for n_samples in sample_sizes_2: 
            scores.append(df2.query(param2 + " == "+str(p)+ " and n_samples == "+str(n_samples))[quantity].iloc[0])
            if display_std and (quantity + "_std") in df2.columns:  
                    scores_std.append(df2.query(param2 + " == "+str(p)+ " and n_samples == "+str(n_samples))[quantity +"_std"].iloc[0])

        label = "PLS with $\\alpha$" if param2 == "alpha" else "GKR with $ \\gamma$" if param2=="gamma" else "kNN with $" + param2 + "$"


        if display_std:
            ax.plot(sample_sizes_2, scores, marker = markers[1], ls = linestyles[1],  label=f"{label}\,=\,{p:.3g}")
            ax.fill_between(sample_sizes_2, np.array(scores) - np.array(scores_std), np.array(scores) + np.array(scores_std), alpha = 0.2)
        else:
            ax.plot(sample_sizes_2, scores, marker = markers[1], ls = linestyles[1], label=f"{label}\,=\,{p:.3g}")


    if param3:
        for p in paramvalues3: # plot the optional third set of lines
            scores = []
            scores_std = []

            for n_samples in sample_sizes_3: 
                scores.append(df3.query(param3 + " == "+str(p)+ " and n_samples == "+str(n_samples))[quantity].iloc[0])
                if display_std and (quantity + "_std") in df3.columns:
                    scores_std.append(df3.query(param3 + " == "+str(p)+ " and n_samples == "+str(n_samples))[quantity +"_std"].iloc[0])

            label = "PLS with $\\alpha$" if param3 == "alpha" else "GKR with $ \\gamma$" if param3=="gamma" else "kNN with $" + param3 + "$"

            if display_std:
                ax.plot(sample_sizes_3, scores, marker = markers[2], ls = linestyles[2], label=f"{label}\,=\,{p:.3g}")
                ax.fill_between(sample_sizes_3, np.array(scores) - np.array(scores_std), np.array(scores) + np.array(scores_std), alpha = 0.2)
               
            else:
                ax.plot(sample_sizes_3, scores, marker = markers[2], ls = linestyles[2], label=f"{label}\,=\,{p:.3g}")

    #ax.legend(ncol = 3, loc="lower center", bbox_to_anchor=(0.5, 1), frameon=False)
    ax.set_xscale("log")
    plt.ylabel(quantity, labelpad = 7)
    plt.xlabel("$n_{samples}$")
    if filename:
        plt.savefig("plots/"+filename, bbox_inches='tight', dpi = 300)

    return fig, ax

def two_histograms(df, x, xlabel = "x", ylabel = "relative frequency", stat = "probability", n_bins = 20 , yscale = None, title = None, include_cdf = True, filename = None):
    fig = plt.figure()
    ax = plt.gca()
    sns.histplot(df, x = x, ax = ax, hue="correct_class", hue_order=[True, False], stat = stat, bins = n_bins, common_norm = False, palette = ["teal", "coral"], element = 'step')

    

    if include_cdf:
        axb = ax.twinx()
        sns.ecdfplot(df.query("correct_class == True")[x], ax=axb, color="teal", label="Estimated cdf")
        sns.ecdfplot(df.query("correct_class == False")[x], ax=axb, color="coral", label="Estimated cdf")
        legend = ax.get_legend()
        handles = legend.legend_handles
        legend.remove()

        h1, l1 = handles, ["correctly labeled", "mislabeled"]
        h2, l2 = axb.get_legend_handles_labels()
        plt.legend(h1+h2, l1+l2, fancybox=True, framealpha=0.5)
        axb.set_ylabel("cumulative relative frequency")
        plt.grid(linewidth = 0.5)
    else:
        plt.legend()
        plt.grid(linewidth = 0.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if yscale:
        ax.set_yscale(yscale)
    if title:
        plt.title(title)
    if filename:
        plt.savefig(filename)
    plt.show()



hist_colors = my_colors[5:]
hist_colors.reverse()

# Create a ListedColormap
cmap5 = matplotlib.colors.ListedColormap(hist_colors, name='cmap5')

def two_dimensional_histogram(df, b, x, y, xlabel, ylabel, n_bins, filename = None):

    

    cmap = cmap5

    if b in df.columns:
        fig, ax = plt.subplots(1,2)
        sns.histplot(df.query(f"{b} == prediction"), x=x, y = y, ax = ax[0],\
                    bins=n_bins, stat = 'probability', cbar = True, cmap= cmap, norm=LogNorm(), vmin=None, vmax=None, cbar_kws=dict(shrink=.9))

        ax[0].grid(linewidth = 0.5)
        ax[0].set_xlabel(xlabel)
        ax[0].set_ylabel(ylabel)
        ax[0].set_title("Correctly labeled data points")

        fig.axes[-1].set_title("Prob.")

        r = np.corrcoef(df.query(f"{b} == prediction")[x], df.query(f"{b} == prediction")[y])[0,1]
        ax[0].annotate(
            'correlation:\nr=%.4f'%r,
            xy=(0.5, 0.75), xycoords='axes fraction',
            bbox=dict(boxstyle="round", fc="0.9", ec="teal")
        )

        sns.histplot(df.query(f"{b} != prediction"), x=x, y = y, ax = ax[1],\
                    bins=n_bins, stat = 'probability', cbar = True, cmap= cmap, norm=LogNorm(), vmin=None, vmax=None, cbar_kws=dict(shrink=.9))

        ax[1].grid(linewidth = 0.5)
        ax[1].set_xlabel(xlabel)
        ax[1].set_ylabel(ylabel)
        ax[1].set_title("Mislabeled data points")

        fig.axes[-1].set_title("Prob.")

        r = np.corrcoef(df.query(f"{b} != prediction")[x], df.query(f"{b} != prediction")[y])[0,1]
        ax[1].annotate(
            'correlation:\nr=%.4f'%r,
            xy=(0.5, 0.75), xycoords='axes fraction',
            bbox=dict(boxstyle="round", fc="0.9", ec="teal")
        )
        fig.tight_layout()
        if filename:
            plt.savefig(filename)
        plt.show()
    else:
        fig, ax = plt.subplots()
        sns.histplot(df, x=x, y = y, ax = ax,\
                    bins=n_bins, stat = 'probability', cbar = True, cmap= cmap, norm=LogNorm(), vmin=None, vmax=None, cbar_kws=dict(shrink=.9))

        ax.grid(linewidth = 0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        fig.axes[-1].set_title("Prob.")

        # r = np.corrcoef(df[x], df[y])[0,1]
        # ax.annotate(
        #     'correlation:\nr=%.4f'%r,
        #     xy=(0.5, 0.75), xycoords='axes fraction',
        #     bbox=dict(boxstyle="round", fc="0.9", ec="teal")
        # )
        fig.tight_layout()
        if filename:
            plt.savefig(filename)
        plt.show()

def two_dim_scatter_errors(dataset, data_space, c1 = "neural_network_prediction", c2 = "prediction", title = None, filename = None):

    fig, ax = plt.subplots()
    x, y = np.vstack(dataset[data_space].values).T
    dataset["x"] = x
    dataset["y"] = y
    sns.kdeplot(dataset, x ="x", y = "y", hue = c1, palette = "tab10", alpha = 0.5, fill = True, legend = False)

    # for deeper understanding: edgecolor indicates initial NN prediction

    # edgecolors = matplotlib.colormaps["tab10"](dataset.query("prediction != neural_network_prediction")["neural_network_prediction"])
    # ax.scatter(dataset.query("prediction != neural_network_prediction")["x"], dataset.query("prediction != neural_network_prediction")["y"],  marker = "X", \
    #             c= dataset.query("prediction != neural_network_prediction")["prediction"], cmap = "tab10", edgecolors=edgecolors)

    ax.scatter(dataset.query(c1 + " != "+ c2)["x"], dataset.query(c1 + " != " + c2)["y"],  marker = "X", \
                 c= dataset.query(c1 + " != " + c2)[c2], cmap = "tab10", ec = "lightgrey")

    for i in range(10):
        x_center = dataset.query(c1 + ' == '+str(i))['x'].mean()
        y_center = dataset.query(c1 + ' == '+str(i))['y'].mean()
        ax.annotate(str(i), xy = (x_center, y_center), size = 25, c = "teal", weight = 'bold')

    if title:
        plt.title(title)

    if filename:
        plt.savefig("plots/" + filename)


small_ms = 1
def two_dim_color_by_quantity_plot(dataset, data_space, quantity = "entropy", filename = None):

    fig, ax = plt.subplots()
    x, y = np.vstack(dataset[data_space].values).T
    dataset["x"] = x
    dataset["y"] = y
    if quantity in dataset.columns:
        ax.scatter(dataset["x"], dataset["y"],  marker = "o", s = small_ms, c = dataset[quantity], cmap = "Spectral_r", alpha = 0.5)
    else: 
        print("Quantity not supported / not contained in dataset.")
    for i in range(10):
        x_center = dataset.query('label == '+str(i))['x'].mean()
        y_center = dataset.query('label == '+str(i))['y'].mean()
        ax.annotate(str(i), xy = (x_center, y_center), size = 25, c = "coral", weight = 'bold')

    if filename:
        plt.savefig("plots/"+filename)


from matplotlib_venn import venn2, venn3

def venn_diagram(dataset, columns, labels, filename = None):
    if len(labels) == 2 and len(columns) == 2:
        # two item venn diagram
        c1_wrong = ~dataset[columns[0]]
        c2_wrong = ~dataset[columns[1]]

        # Calculate the counts
        only_c1 = sum(c1_wrong & ~c2_wrong)
        only_c2 = sum(c2_wrong & ~c1_wrong)
        both = sum(c1_wrong & c2_wrong)

        # Plotting Venn diagram
        venn = venn2(subsets=(only_c1, only_c2, both), 
            set_labels=labels)

        venn.get_patch_by_id('10').set_color('teal')
        venn.get_patch_by_id('01').set_color('forestgreen')
        venn.get_patch_by_id('11').set_color('coral')

        venn.get_patch_by_id('10').set_edgecolor('black')
        venn.get_patch_by_id('01').set_edgecolor('black')
        venn.get_patch_by_id('11').set_edgecolor('black')

        plt.title('Overlap of Errors between '+ labels[0] + " and " + labels[1])

        if filename:
            plt.savefig("plots/" + filename)
        plt.show()
    elif len(labels) == 3 and len(columns) == 3:

        # Calculate counts for the Venn diagram
        c1, c2 , c3 = ('correct_NN_prediction', 'correct_PLS_prediction', 'correct_DR_prediction')

        c1_only = sum(~dataset[c1] & dataset[c2] & dataset[c3])
        c2_only = sum(dataset[c1] & ~dataset[c2] & dataset[c3])
        c3_only = sum(dataset[c1] & dataset[c2] & ~dataset[c3])
        c1_c2 = sum(~dataset[c1] & ~dataset[c2] & dataset[c3])
        c1_c3 = sum(~dataset[c1] & dataset[c2] & ~dataset[c3])
        c2_c3 = sum(dataset[c1] & ~dataset[c2] & ~dataset[c3])
        c1_c2_c3 = sum(~dataset[c1] & ~dataset[c2] & ~dataset[c3])

        # Plotting Venn diagram with custom colors
        plt.figure(figsize=(8, 8))
        venn = venn3(subsets=(c1_only, c2_only, c1_c2, c3_only, c1_c3, c2_c3, c1_c2_c3), 
                    set_labels=labels)

        venn.get_patch_by_id('100').set_color('skyblue')
        venn.get_patch_by_id('010').set_color('lightgreen')
        venn.get_patch_by_id('001').set_color('lightcoral')
        venn.get_patch_by_id('110').set_color('orange')
        venn.get_patch_by_id('101').set_color('purple')
        venn.get_patch_by_id('011').set_color('brown')
        venn.get_patch_by_id('111').set_color('grey')

        for patch_id in ['100', '010', '001', '110', '101', '011', '111']:
            venn.get_patch_by_id(patch_id).set_edgecolor('black')

        plt.title('Overlap of Errors')
        plt.show()

    else:
        print("Can only process two or three columns and corresponding labels")

def histogram(df, x, xlabel = "x", ylabel = "relative frequency", stat = "probability", n_bins = 20 , yscale = None, title = None, include_cdf = True, filename = None):
    fig = plt.figure()
    ax = plt.gca()
    sns.histplot(df, x = x, ax = ax, stat = stat, bins = n_bins, element = 'step', color = 'teal')

    if include_cdf:
        axb = ax.twinx()
        sns.ecdfplot(df[x], ax=axb, color="teal", label="Estimated cdf")
        # legend = ax.get_legend()
        # handles = legend.legend_handles
        # legend.remove()

        # h1, l1 = handles, ["correctly labeled", "mislabeled"]
        # h2, l2 = axb.get_legend_handles_labels()
        # plt.legend(h1+h2, l1+l2, fancybox=True, framealpha=0.5)
        axb.set_ylabel("cumulative relative frequency")
        plt.grid(linewidth = 0.5)
    else:
        plt.legend()
        plt.grid(linewidth = 0.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if yscale:
        ax.set_yscale(yscale)
    if title:
        plt.title(title)
    if filename:
        plt.savefig(filename)
    plt.show()
    
def entropy_histogram(df, x, acc = None, ax = None, xlabel = "", ylabel = "", stat = "probability", n_bins = 20 , yscale = None, title = None,
              include_cdf = True, remove_ticks_of_second_yaxis = False, filename = None):
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
        
    sns.histplot(df, x = x, ax = ax, stat = stat, bins = n_bins, element = 'step', color = 'teal')
    ax.grid(linewidth = 0.5)
    if include_cdf:
        axb = ax.twinx()
        sns.ecdfplot(df[x], ax=axb, color="teal", label="Estimated cdf")
        if ylabel!= "":
            axb.set_ylabel("cumulative relative frequency")
        else:
            axb.set(ylabel=None)
        if remove_ticks_of_second_yaxis:
            axb.set_yticks([])
    else:
        ax.legend()
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ax.set_ylim([0, 1])

    if yscale:
        ax.set_yscale(yscale)
    
    if acc:
        plt.text(0.65, 0.6, f"Acc: {100*acc:.2f}\,\%", fontsize=12, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        
    if title:
        ax.set_title(title)
    if filename:
        plt.savefig(filename)

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import matplotlib.lines as mlines

def add_zoom_window(fig, ax, x_limits, y_limits, zoom=2.5, bbox_to_anchor=None, bbox_loc=1, **kwargs):
    """
    Add a zoom window to an existing matplotlib plot with consistent styling,
    supporting multiple lines in a line plot.

    Parameters:
    - fig: The matplotlib figure object.
    - ax: The main plot's axes object.
    - x_limits: Tuple specifying (x_min, x_max) for the zoomed region.
    - y_limits: Tuple specifying (y_min, y_max) for the zoomed region.
    - zoom: Zoom factor for the inset.
    - bbox_to_anchor: Tuple specifying the (x, y, width, height) for positioning the inset.
    - bbox_loc: Location code (1=upper right, 2=upper left, 3=lower left, 4=lower right) for alignment of the inset with bbox_to_anchor.
    """
    # Use bbox_to_anchor for custom positioning, or default to loc-based placement
    if bbox_to_anchor:
        axins = zoomed_inset_axes(ax, zoom=zoom, bbox_to_anchor=bbox_to_anchor, bbox_transform=ax.figure.transFigure, loc=bbox_loc)
    else:
        axins = zoomed_inset_axes(ax, zoom=zoom, loc="upper left")

    # Iterate over all Line2D objects in the axes
    for line in ax.get_lines():
        x, y = line.get_data()  # Get data for the current line
        axins.plot(x, y, label=line.get_label(), color=line.get_color(), linestyle=line.get_linestyle(), marker=line.get_marker())

    # Handle axhline and axvline by checking for identical ydata or xdata
    for line in ax.lines:
        if isinstance(line, mlines.Line2D):
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            if all(y == y_data[0] for y in y_data):  # Horizontal line
                axins.axhline(
                    y=y_data[0],
                    color=line.get_color(),
                    linestyle=line.get_linestyle(),
                    linewidth=line.get_linewidth(),
                    alpha=line.get_alpha(),
                    label=line.get_label(),
                )
            elif all(x == x_data[0] for x in x_data):  # Vertical line
                axins.axvline(
                    x=x_data[0],
                    color=line.get_color(),
                    linestyle=line.get_linestyle(),
                    linewidth=line.get_linewidth(),
                    alpha=line.get_alpha(),
                    label=line.get_label(),
                )

    axins.set_xlim(x_limits)
    axins.set_ylim(y_limits)
    axins.set_xticks([])
    axins.set_yticks([])

    # Match edge color and linewidth with the main axes
    edge_color = ax.spines["left"].get_edgecolor()
    edge_linewidth = ax.spines["left"].get_linewidth()

    for spine in axins.spines.values():
        spine.set_edgecolor(edge_color)
        spine.set_linewidth(edge_linewidth)

    loc1 = kwargs.get('loc1', 2)  # Steers the location of the left line connecting the zoom window to the original segment
    loc2 = kwargs.get('loc2', 4)  # Steers the location of the right line connecting the zoom window to the original segment

    # Mark the zoomed region with lines matching the edge color and linewidth
    mark_inset(ax, axins, loc1=loc1, loc2=loc2, fc="none", ec=edge_color, lw=edge_linewidth)
   
    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 1])

    return fig, ax
