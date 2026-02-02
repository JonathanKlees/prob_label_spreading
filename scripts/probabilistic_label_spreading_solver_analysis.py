import numpy as np
import pandas as pd
import math
import argparse
import ast
import sklearn.metrics
import scipy
import sklearn
import time
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sparse
import json
import pyamgx # amgxenv
import torch
from torch.nn.functional import kl_div 

def prob_label_spreading(dataset_name, dataset, data_space, prob_label_column, n_data, k, alpha, n_samples, train_split = None, config = "configs/FGMRES.json"):
    
    if n_data == "all":
        n_data = len(dataset)
    else:
        dataset = dataset.sample(n = n_data) # random data subset
        dataset.reset_index(drop=True, inplace=True)

    classes = list(set(dataset["label"])) # determine the classes automatically
    n_classes = len(classes)
    start=time.time()

    if train_split is not None:
        indices = np.random.choice(dataset.iloc[train_split].index, size=n_samples, replace=True) # this draws n_samples many indices randomly
    else:
        indices = np.random.choice(dataset.index, size=n_samples, replace=True) # this draws n_samples many indices randomly 

    if alpha != 0: # if alpha is zero we do not need to set up the graph structure nor solve the linear system
        # set graph and linear system
        S, I, time_NN, time_graph = set_graph(k, dataset, data_space, n_data)
        #Y = make_rhs(n_samples, len(classes), dataset, data_space, n_data)

        spreaded_info, trials, ind, time_spread, time_setup, time_solver = sampling_and_spreading_efficient(alpha, S, dataset, prob_label_column, classes, n_classes, config, indices)

        
        dataset["prediction"] = [x.argmax() for x in spreaded_info]
        dataset['trials'] = trials
        dataset["p_hat"] = [ (spreaded_info[i] + ((1e-4)/n_classes)) /(trials[i] + 1e-4) for i in range(len(trials))]    # for k in range(n_classes):

    else:
        feedback = np.zeros( (n_data, len(classes))) # counts the number of sampled classes for every datapoint -> histogram that corresponds to the prob. distr.
        for i in indices: # for every sampled data point update the estimated prob. distr.
             c = np.random.choice(classes, p = dataset[prob_label_column][i]) # sample a class according to the prob. label
             feedback[i, c] += 1
        # insert uniform distr. for data points that have not been sampled
        trials = [x.sum() for x in feedback]
        feedback = [x if x.sum() > 0 else np.ones(len(classes)) for x in feedback]
        # normalize 
        feedback = [x / x.sum() for x in feedback]

        dataset["prediction"] = [x.argmax() for x in feedback] # this will assign class zero if the distr. is uniform
        dataset["p_hat"] = [x for x in feedback]

        # assign prediction

    dataset["most_probable_class"] = list( np.array( dataset[prob_label_column].to_list() ).argmax(axis = 1) )
    acc = len(dataset.query("most_probable_class == prediction"))/len(dataset)
    p = np.array(dataset[prob_label_column].to_list())
    p_hat = np.array(dataset["p_hat"].to_list())
    rmse = sklearn.metrics.root_mean_squared_error(p_hat, p)
    kl = kl_div(torch.tensor(p_hat).log(), torch.tensor(p), reduction = "batchmean").item()

    elapsed_time = time.time() - start

    data_dim = dataset[data_space].iloc[0].shape[0]

    if data_space == "feature":
        dim_reduction_technique = "NONE"
    elif len(data_space.split("_")[:-1]) > 1: # combination of e.g. CLIP_UMAP
        dim_reduction_technique = "+".join(data_space.split("_")[:-1])
    else:
        dim_reduction_technique = data_space.split("_")[0] # just one technique e.g. CLIP 

    result = pd.DataFrame({"dataset": dataset_name, "n_data": n_data, "dim_reduction_technique": dim_reduction_technique, "dimension": data_dim, "prob_labels": prob_label_column,\
         'alpha':alpha,'n_samples': n_samples, "k": k, 'RMSE':rmse,'Accuracy':acc, "KL": kl , "runtime": elapsed_time, "config": config}, index=[0])

    return result, dataset


def set_graph(k_NN, train, data_space, n_data = None):
        if n_data == None:
                n_data = len(train)
        # compute k nearest neighbors and setup a graph where each node has a link to his k nearest neighbors
        time_nn_anfang=time.time()
        x=np.array(list(train[data_space]))
        nbrs = NearestNeighbors(n_neighbors=k_NN, algorithm='ball_tree', metric='euclidean').fit(x)       
        raw_dists, indices_raw = nbrs.kneighbors(x)
        time_nn_ende=time.time()
        time_setup_graph_anfang=time.time()
        # preprocessing for setting up a sparse adjacency matrix
        indptr = np.asarray(range(0,n_data+1))*(k_NN-1)
        raw_dists = (np.asarray( raw_dists )[:,1:]).ravel() ** 2
        indices = (np.asarray( indices_raw )[:,1:]).ravel()

        # transform distances into similarities
        mu = np.mean( np.asarray(raw_dists) )
        similarity = np.exp( np.asarray( raw_dists ) * ( -1. / (2 * mu) ) )

        # form the sparse matrix
     
        C = scipy.sparse.csr_matrix((similarity, indices, indptr), shape=(n_data,n_data) , dtype=np.float64)
       
        # symmetrize it
        W = (C + C.T) / 2.

        # compute the column sums to obtain 
        D = np.asarray(np.sum(W, axis=-1)).ravel() 
       
        # setup the graph Laplacian
        D_inv_sqrt = scipy.sparse.spdiags( D ** (-0.5), 0, len(D), len(D) )
        S = D_inv_sqrt @ W @ D_inv_sqrt
        
        I = scipy.sparse.eye(len(D))
        time_setup_graph_ende=time.time()
        time_NN=time_nn_ende-time_nn_anfang
        time_graph=time_setup_graph_ende-time_setup_graph_anfang
        return S,I, time_NN,time_graph


# construct the right hand side of the linear system for class-wise label spreading     
def make_rhs(n_samples, n_classes, train, data_space, n_data = None):
        if n_data == None:
                n_data = len(train)
        ind=[]
        data_train = ( np.asarray(train[[data_space]]), np.asarray(train[["label"]]) )
        y = np.asarray( train[['label']] ).ravel().astype(int)
        divide=int(n_samples/n_classes) 
        for i in range(n_classes):
                    random_i=np.random.choice( np.where(data_train[1] == i)[0],divide,replace=False )  
                    ind.append( random_i )
        Y=np.zeros((n_data, n_classes))
        for i in ind:
            y_i=y[i]
            Y[i,y_i]=1
        
        return Y 


def solve_probabilistic(Matrix, config, ind, length, dataset, prob_label_column, n_classes, Sampling_matrix, classes, positives_new, trials):
    """ Solves the linear system involved in label spreading with algebraic multigrid methods using the AMGX library."""
    
    ### suppress the repetitive output of the AMGX library
    pyamgx.register_print_callback(lambda *args, **kwargs : None)
    ####
    start_solver=time.time()
    pyamgx.initialize()

    # Initialize config and resources (which solver to use, etc. as specified in config file):
    cfg = pyamgx.Config().create_from_file(config)
    rsc = pyamgx.Resources().create_simple(cfg)

    # Create matrices and vectors:
    A = pyamgx.Matrix().create(rsc)
    b = pyamgx.Vector().create(rsc)
    x = pyamgx.Vector().create(rsc)

    # Create solver:
    solver = pyamgx.Solver().create(rsc, cfg)

    # Upload system:
    M = sparse.csr_matrix(Matrix)
    A.upload_CSR(M)
    t_a=time.time()
    solver.setup(A)
    t_c=time.time()
    t_setup=t_c-t_a
    t_g=time.time()
    t_spread_start=time.time()

    converged = []
    for i in ind:
        e_i = np.zeros(length)
        e_i[i] = 1
        #e_i is the i-th unit vector. Solution ist die Information die der i-te Punkt im Graphen spreaded/verteilt
        #solution = inverse_matrix.dot(e_i)
    
        rhs = np.array(e_i)
        sol = np.zeros(np.size(e_i), dtype=np.float64)
        b.upload(rhs)
        x.upload(sol)
        # Setup and solve system:
        solver.solve(b, x, zero_initial_guess=True)
        # Download solution
        x.download(sol)
        solution=sol

        rel_res = np.linalg.norm(M @ solution - rhs) / np.linalg.norm(rhs)
        if rel_res < json.load(open(config))["solver"]["tolerance"]:
            converged.append(True)
        else:
            converged.append(False)
        #print("Relative residual: ", rel_res)

        if np.isnan(solution).all():
            solution = np.zeros(length) # it might happen that the solver occasionally does not find the solution (for alpha and k very small)
        #normieren der Solution, sodass der maximale verteilte Wert 1 ist. trials immer erhöhen 
        m = np.max(solution)
        if m > 0: # always the case if the solution is found, otherwise solution is set to zero anyways so no update necessary
            trials += solution/m

        #Erst in der unten stehenden for-Schleife wird bestimmt von welcher Klasse die Information gespreaded wird
        u = np.random.uniform(size=1)

        p_sum = 0
        
        #mit der uniform verteilten Zufallsvariabele u wird an Hand der "wahren" probabilistischen Labels namens 'prob_j' eine Klasse gesampled
        for j in range(n_classes):
            
            p_sum = p_sum + dataset[prob_label_column].iloc[i][j]
            if j == 0 and u <=dataset[prob_label_column].iloc[i][j]:
                break
            if j > 0 and u <= p_sum:
                break

        j = classes[j]

        #merken der gesampleten Klassen in einer Samplingmatrix.      
        Sampling_matrix[i][classes.index(j)] += 1

        #spreaden der Information für die Klasse die gezogen (in der Spalte de gezogenen Klasse). Für jeden Punkt ist die Summe der über alle Klasse erhalten Informationen in positives_new gleich dem Wert in trials
        if m > 0:
            positives_new[:,classes.index(j)] += solution/m

    if np.array(converged).all() == False:
        print("Solver did not converge for all samples.")
    if json.load(open(config))["solver"]["tolerance"] < 1e-12:
        print(f'Warning: The specified tolerance is very small ({json.load(open(config))["solver"]["tolerance"]:.0e}). This may exceed the precision limit of float64.')

    t_spread_end=time.time()     
    t_spread=t_spread_end-t_spread_start   
    t_b=time.time()   
    A.destroy()
    x.destroy()
    b.destroy()
    solver.destroy()
    rsc.destroy()
    cfg.destroy()

    pyamgx.finalize()  
    end_solver=time.time()
    t_solver=end_solver-start_solver

    return Sampling_matrix, positives_new, trials, t_setup, t_spread, t_solver

def sampling_and_spreading_efficient(alpha, S, dataset, prob_label_column, classes, n_classes, config, ind):
       
    length = len(dataset)

    #sample zufällig n_sample indizes mit maximalem Index n_data-1. Hier könnte man ggf. auch bestimmen, dass nur ein Teil der Daten gesampled werden dürfen, z.B. 
    #nur die ersten N Punkte dürfen Labelinformation erhalten.

    #der vektor trials gibt an wie viel Feedback jeder Punkt insgesamt erhalten hat   
    trials = np.zeros(length)

    #in positives_new wird das Feedback für die einzelnen Klassen gespeichert
    #Sampling_matrix speichert wie oft jede Klasse pro Sample gezogen wurde und Sampling_probs enthält die entsprechenden relativen Häufigkeiten
    positives_new = np.zeros(length*n_classes).reshape(length,n_classes)
    Sampling_matrix = np.zeros(length*n_classes).reshape(length, n_classes)
    Sampling_probs = np.zeros(length*n_classes).reshape(length, n_classes)

    I = scipy.sparse.eye(length)

    #Lösen des linearen Gleichungssystems aus dem Paper um das Feedback bestimmen zu können --> Hier muss ggf. eine effizientere Implementierung gefunden werden.
    
    M = I - alpha * S
    
    Sampling_matrix, positives_new, trials, time_setup, time_spread, time_solver = solve_probabilistic(M,config,ind,length,dataset,prob_label_column,n_classes,Sampling_matrix,classes,positives_new,trials)
                        
    return positives_new, trials, ind, time_spread, time_setup, time_solver


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Performs probabilistic label spreading on the supplied data set and stores the resulting performance in a CSV file. Supplied dataset should be preprocessed by process_data.py."
    )
    parser.add_argument("--datasets", required = True, nargs='+')
    parser.add_argument("--dataset_sizes", required = True, type = str, help = "Dict as JSON string")
    parser.add_argument("--dim_reduction_techniques", required = True, nargs='+')
    parser.add_argument("--dimensions", required = False, type=str)
    parser.add_argument("--k_neighbors", required = True, type = str)
    parser.add_argument("--alphas", required = True, type = str)
    parser.add_argument("--sample_sizes", required = True, type = str)
    parser.add_argument("--multiple_runs", required = True, type = int)
    parser.add_argument("--experiment", required = True, type = str)


    # read arguments for parameters
    args = parser.parse_args()

    datasets = args.datasets
    dim_reduction_techniques = args.dim_reduction_techniques
    dimensions = ast.literal_eval(args.dimensions)

    dataset_sizes = json.loads(args.dataset_sizes)

    k_neighbors = ast.literal_eval(args.k_neighbors)
    alphas = ast.literal_eval(args.alphas)
    sample_sizes = ast.literal_eval(args.sample_sizes)

    multiple_runs = args.multiple_runs

    experiment = args.experiment

    ####################

    results = pd.DataFrame()

    for dataset_name in datasets: # loop over datasets
        df = pd.read_pickle("data/prob_data/" + dataset_name + "/" + dataset_name + ".pkl") #  read preprocessed dataset
        classes = list(set(df["label"]))
        n_data = dataset_sizes[dataset_name]
        if n_data == "all":
            n_data = len(df)

        if dataset_name == "TwoMoons": # Two Moons is a special case as no dim reduction is applied
            dim_reduction_technique = "NONE"
            dim = df["feature"][0].shape[0]
            data_space = "feature"

            for alpha in alphas:
                    for k in k_neighbors:
                        for n_samples in sample_sizes:
                            for i in range(multiple_runs):
                                # perform prob label spreading and get the l2 score as well as the accuracy w.r.t. the most probable class
                                result, _ = prob_label_spreading(dataset_name, df, data_space, n_data, k, alpha, n_samples) # returns result including RMSE, Accuracy and parameters and the processed dataframe, which is not relevant here
                                results = pd.concat([results, result], ignore_index = True) # extend the results

        else:
            for dim_reduction_technique in dim_reduction_techniques:
                
                if dim_reduction_technique == "NONE": # In that case we do not want to loop over different dimensions
                    data_space = "feature" 
                    dim = df["feature"][0].shape[0]

                    for alpha in alphas:
                        for k in k_neighbors:
                            for n_samples in sample_sizes:
                                for i in range(multiple_runs):
                                    # perform prob label spreading and get the l2 score as well as the accuracy w.r.t. the most probable class
                                    result, _ = prob_label_spreading(dataset_name, df, data_space, n_data, k, alpha, n_samples) # returns result including RMSE, Accuracy and parameters and the processed dataframe, which is not relevant here
                                    results = pd.concat([results, result], ignore_index = True) # extend the results


                else: # consider every one of the selected dimensions
                    for dim in dimensions:
                        data_space = dim_reduction_technique + "_" + str(dim) # current data space

                        for alpha in alphas:
                            for k in k_neighbors:
                                for n_samples in sample_sizes:
                                    for i in range(multiple_runs):
                                        # perform prob label spreading and get the l2 score as well as the accuracy w.r.t. the most probable class
                                        result, _ = prob_label_spreading(dataset_name, df, data_space, n_data, k, alpha, n_samples) # returns result including RMSE, Accuracy and parameters and the processed dataframe, which is not relevant here
                                        results = pd.concat([results, result], ignore_index = True) # extend the results


    filename = "results/pls_" + experiment + ".csv"
    results.to_csv(filename)


    print("results generated and stored in " + filename)