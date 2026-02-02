
from absl import app
from src.algorithms.common.algorithm_skeleton import AlgorithmSkelton
from os.path import join

from PIL import Image
import umap
import clip
import os

import numpy as np
import pandas as pd
import scipy
import time
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sparse
import pyamgx # amgxenv
import torch

from absl import flags

from pathlib import Path


flags.DEFINE_float(name='alpha',default= 0.1 , help='alpha for pls algo.')
FLAGS = flags.FLAGS


class PLS(AlgorithmSkelton):

    def __init__(self):
        AlgorithmSkelton.__init__(self, f'pls_alpha_{FLAGS.alpha}')


    def preprocess(self, dataframe, dataset_info, paths, v_fold, num_annos, percentage_labeled, mode):   #self, dataset_info, paths, gt, v_fold, num_annos, percentage_labeled

        image= []
        image_path = []
        targets = []
        splits = []

        soft_gts = []
        labels = []
        label_names = []
        features = []

        # daectivate preprocess for now, maybe delet images from dataframe before saving to save disc space
        save_preprocess = False


        dataset_root, dataset_name = dataset_info.raw_data_root_directory, dataset_info.name
        
        path_full = [join(dataset_root, path) for path in paths]



        dataframe["path"] = dataframe["path"].apply(lambda x: join(dataset_root, x))

        for img_path in path_full:
            try:
                img = Image.open(img_path).convert("RGB")
                image.append(img)
                image_path.append(img_path)

                row = dataframe[dataframe["path"] == img_path]

                if not row.empty:
                    soft_gts.append(row["soft_gt"].values[0])  # Extract the value and append to targets
                    labels.append(row["gt"].values[0])
                    label_names.append(row["gt_verbose"].values[0])
                    img_np = np.array(img).reshape(-1)
                    features.append(img_np)
                

            except Exception as e:
                print(f"Error processing image {img_path}: {e}")


        dataset = pd.DataFrame({"img_path": image_path, "img": image, "feature": features, 
                                "label":labels, "gt_verbose":label_names, "soft_gt": soft_gts})


        # CLIP
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load('ViT-B/32', device)
        
        # Function to process and reduce image features
        def reduce_image_features(img):
            image_tensor = preprocess(img).unsqueeze(0).to(device)
            # Reduce image with CLIP
            with torch.no_grad():
                reduced_features = model.encode_image(image_tensor)
            return reduced_features.cpu().numpy().flatten()  # Flatten to a 1D array


        imgs = dataset["img"].tolist()
        reduced_features = np.array([reduce_image_features(img) for img in imgs])
        dataset["CLIP"] = list(reduced_features)

        # UMAP
        def UMAP_reduction(dataset, layer, dim = 50, store = False, dir_name = "./"):
            features = np.array(dataset[layer].to_list())
            reducer = umap.UMAP(n_components=dim, random_state= 0)
            reducer.fit(features)

            reduced_features = reducer.transform(features)

            return reduced_features

        # dimensions = [20]

        #  # CLIP + UMAP
        # for dim in dimensions:
        #     reduced_features = UMAP_reduction(dataset, "CLIP", dim = dim, store = False)
        #     dataset["CLIP_UMAP_"+str(dim)] = list(reduced_features)

        reduced_features_umap = UMAP_reduction(dataset, "CLIP", dim = 20, store = False)
        dataset["CLIP_UMAP_"+str(20)] = list(reduced_features_umap)

    # ##################
    #     dir_name = "/home/klees/prob_label_spreading/data/dc_bench/dcic/data/pls_features"
    # ##########################

    #     test = "_test_00"

    #     dataset_full_name = f'{dataset_name}-{self.name}-{num_annos:02d}-{percentage_labeled:0.02f}-{v_fold}-{mode}'

    #     dataset.to_pickle(dir_name + "/" + dataset_full_name + test +".pkl")
    #     print("Dimension reduction was applied. Dataset is stored in " + dir_name + "/" + dataset_full_name + test + ".pkl")

        if save_preprocess:
            dir_name = "/home/heller/projects/probl_dcb/prob_label_spreading/data/dc_bench/dcic/data/pls_features"
            test = "_test_00"
            dataset_full_name = f'{dataset_name}-{self.name}-{num_annos:02d}-{percentage_labeled:0.02f}-{v_fold}-{mode}'
            dataset.to_pickle(dir_name + "/" + dataset_full_name + test +".pkl")
            print("Dimension reduction was applied. Dataset is stored in " + dir_name + "/" + dataset_full_name + test + ".pkl")
        else: print("Preprocess not saved")

        return dataset

    def run(self, ds, oracle, dataset_info, v_fold,num_annos,percentage_labeled):

        # /home/heller/projects/probl_dcb/prob_label_spreading/data/dc_bench/dcic/src/algorithms/probabilistic_label_spreading_dcic.py

        # phase 1 command:
        # ./run_new.sh src.algorithms.probabilistic_label_spreading_dcic --percentage_labeled_data 0.6 --number_annotations_per_image 3

        # phase 2 command:
        # ./run_new.sh src.evaluation.evaluate --folders Benthic-pls-03-0.60,CIFAR10H-pls-03-0.60,MiceBone-pls-03-0.60,Pig-pls-03-0.60,Plankton-pls-03-0.60,QualityMRI-pls-03-0.60,Synthetic-pls-03-0.60,Treeversity#1-pls-03-0.60,Treeversity#6-pls-03-0.60,Turkey-pls-03-0.60 --mode soft

        # phase 2 command:
        # ./run_new.sh src.evaluation.evaluate --folders Benthic-pls_alpha_0.1-01-1.00,Benthic-pls_alpha_0.1-10-1.00,Benthic-pls_alpha_0.5-01-0.10,Benthic-pls_alpha_0.5-01-1.00,Benthic-pls_alpha_0.05-10-1.00,Benthic-pls_alpha_0.5-10-1.00,Benthic-pls_alpha_0.9-01-0.10,Benthic-pls_alpha_0.25-01-1.00,Benthic-pls_alpha_0.25-10-1.00,Benthic-pls_alpha_0.75-01-0.10,Benthic-pls_alpha_0.75-01-1.00,Benthic-pls_alpha_0.99-01-0.10,CIFAR10H-pls_alpha_0.1-01-1.00,CIFAR10H-pls_alpha_0.1-10-1.00,CIFAR10H-pls_alpha_0.5-01-0.10,CIFAR10H-pls_alpha_0.5-01-1.00,CIFAR10H-pls_alpha_0.05-10-1.00,CIFAR10H-pls_alpha_0.5-10-1.00,CIFAR10H-pls_alpha_0.9-01-0.10,CIFAR10H-pls_alpha_0.25-01-1.00,CIFAR10H-pls_alpha_0.25-10-1.00,CIFAR10H-pls_alpha_0.75-01-0.10,CIFAR10H-pls_alpha_0.75-01-1.00,CIFAR10H-pls_alpha_0.99-01-0.10,MiceBone-pls_alpha_0.1-01-1.00,MiceBone-pls_alpha_0.1-10-1.00,MiceBone-pls_alpha_0.5-01-0.10,MiceBone-pls_alpha_0.5-01-1.00,MiceBone-pls_alpha_0.05-10-1.00,MiceBone-pls_alpha_0.5-10-1.00,MiceBone-pls_alpha_0.9-01-0.10,MiceBone-pls_alpha_0.25-01-1.00,MiceBone-pls_alpha_0.25-10-1.00,MiceBone-pls_alpha_0.75-01-0.10,MiceBone-pls_alpha_0.75-01-1.00,MiceBone-pls_alpha_0.99-01-0.10,Pig-pls_alpha_0.1-01-1.00,Pig-pls_alpha_0.1-10-1.00,Pig-pls_alpha_0.5-01-0.10,Pig-pls_alpha_0.5-01-1.00,Pig-pls_alpha_0.05-10-1.00,Pig-pls_alpha_0.5-10-1.00,Pig-pls_alpha_0.9-01-0.10,Pig-pls_alpha_0.25-01-1.00,Pig-pls_alpha_0.25-10-1.00,Pig-pls_alpha_0.75-01-0.10,Pig-pls_alpha_0.75-01-1.00,Pig-pls_alpha_0.99-01-0.10,Plankton-pls_alpha_0.1-01-1.00,Plankton-pls_alpha_0.1-10-1.00,Plankton-pls_alpha_0.5-01-0.10,Plankton-pls_alpha_0.5-01-1.00,Plankton-pls_alpha_0.05-10-1.00,Plankton-pls_alpha_0.5-10-1.00,Plankton-pls_alpha_0.9-01-0.10,Plankton-pls_alpha_0.25-01-1.00,Plankton-pls_alpha_0.25-10-1.00,Plankton-pls_alpha_0.75-01-0.10,Plankton-pls_alpha_0.75-01-1.00,Plankton-pls_alpha_0.99-01-0.10,QualityMRI-pls_alpha_0.1-01-1.00,QualityMRI-pls_alpha_0.1-10-1.00,QualityMRI-pls_alpha_0.5-01-0.10,QualityMRI-pls_alpha_0.5-01-1.00,QualityMRI-pls_alpha_0.05-10-1.00,QualityMRI-pls_alpha_0.5-10-1.00,QualityMRI-pls_alpha_0.9-01-0.10,QualityMRI-pls_alpha_0.25-01-1.00,QualityMRI-pls_alpha_0.25-10-1.00,QualityMRI-pls_alpha_0.75-01-0.10,QualityMRI-pls_alpha_0.75-01-1.00,QualityMRI-pls_alpha_0.99-01-0.10,Synthetic-pls_alpha_0.1-01-1.00,Synthetic-pls_alpha_0.1-10-1.00,Synthetic-pls_alpha_0.5-01-0.10,Synthetic-pls_alpha_0.5-01-1.00,Synthetic-pls_alpha_0.05-10-1.00,Synthetic-pls_alpha_0.5-10-1.00,Synthetic-pls_alpha_0.9-01-0.10,Synthetic-pls_alpha_0.25-01-1.00,Synthetic-pls_alpha_0.25-10-1.00,Synthetic-pls_alpha_0.75-01-0.10,Synthetic-pls_alpha_0.75-01-1.00,Synthetic-pls_alpha_0.99-01-0.10,Treeversity#1-pls_alpha_0.1-01-1.00,Treeversity#1-pls_alpha_0.1-10-1.00,Treeversity#1-pls_alpha_0.5-01-0.10,Treeversity#1-pls_alpha_0.5-01-1.00,Treeversity#1-pls_alpha_0.05-10-1.00,Treeversity#1-pls_alpha_0.5-10-1.00,Treeversity#1-pls_alpha_0.9-01-0.10,Treeversity#1-pls_alpha_0.25-01-1.00,Treeversity#1-pls_alpha_0.25-10-1.00,Treeversity#1-pls_alpha_0.75-01-0.10,Treeversity#1-pls_alpha_0.75-01-1.00,Treeversity#1-pls_alpha_0.99-01-0.10,Treeversity#6-pls_alpha_0.1-01-1.00,Treeversity#6-pls_alpha_0.1-10-1.00,Treeversity#6-pls_alpha_0.5-01-0.10,Treeversity#6-pls_alpha_0.5-01-1.00,Treeversity#6-pls_alpha_0.05-10-1.00,Treeversity#6-pls_alpha_0.5-10-1.00,Treeversity#6-pls_alpha_0.9-01-0.10,Treeversity#6-pls_alpha_0.25-01-1.00,Treeversity#6-pls_alpha_0.25-10-1.00,Treeversity#6-pls_alpha_0.75-01-0.10,Treeversity#6-pls_alpha_0.75-01-1.00,Treeversity#6-pls_alpha_0.99-01-0.10,Turkey-pls_alpha_0.1-01-1.00,Turkey-pls_alpha_0.1-10-1.00,Turkey-pls_alpha_0.5-01-0.10,Turkey-pls_alpha_0.5-01-1.00,Turkey-pls_alpha_0.05-10-1.00,Turkey-pls_alpha_0.5-10-1.00,Turkey-pls_alpha_0.9-01-0.10,Turkey-pls_alpha_0.25-01-1.00,Turkey-pls_alpha_0.25-10-1.00,Turkey-pls_alpha_0.75-01-0.10,Turkey-pls_alpha_0.75-01-1.00,Turkey-pls_alpha_0.99-01-0.10 --mode soft
        
        
        # CIFAR10H-pls-03-0.60,MiceBone-pls-03-0.60,Pig-pls-03-0.60,Plankton-pls-03-0.60,QualityMRI-pls-03-0.60,Synthetic-pls-03-0.60,Treeversity#1-pls-03-0.60,Treeversity#6-pls-03-0.60,Turkey-pls-03-0.60 --mode soft

        print("start new pls dcic")

        print("name:", self.name)

        mode = 'hard'

        #dir_name = "/home/klees/prob_label_spreading/data/dc_bench/dcic/data/pls_features"
        dir_name = "/home/heller/projects/probl_dcb/prob_label_spreading/data/dc_bench/dcic/data/pls_features"

        dataset_full_name = f'{dataset_info.name}-{self.name}-{num_annos:02d}-{percentage_labeled:0.02f}-{v_fold}-{mode}'

        test = "_test_00"

        file_path = dir_name + "/" + dataset_full_name + test +".pkl"

        # get images dict from ds and save as Dataframe
        df_ds = pd.DataFrame(ds.images_dict)
        df_ds = df_ds.T.reset_index()

        if os.path.exists(file_path):
            # load dataset from .pkl file
            dataset = pd.read_pickle(file_path)
            print("Dataset File loaded successfully from:" + file_path)
        else:
            # apply preprocess to ds
            print("Dataset File does not exist:" + file_path)
            print("Load from raw_datasets and apply preprocessing")
            ### from other exampes like het.py, dividemix_dataloader
            # own datasets
            mode = 'hard'
            paths, _ = ds.get_training_subsets('all', mode)
            # gt_train = np.argmax(gt_train, axis=1)

            dataset = self.preprocess(df_ds, dataset_info, paths, v_fold, num_annos, percentage_labeled, mode)


        dataset["annotations"] = [ num_annos * np.array(x) for x in dataset["soft_gt"]]

        data_space = "CLIP_UMAP_20"
        n_data = "all"
        annotation_column = "annotations"


        k = 20

        alpha = FLAGS.alpha
        print("alpha:", alpha)


        result, new_dataset = self.prob_label_spreading(dataset_info.name, dataset, data_space, annotation_column, 
                                                        n_data, k, alpha)


        preds = new_dataset["p_hat"]

        paths, _ = ds.get_training_subsets('all', mode)

        for i, path in enumerate(paths):
            split = ds.get(path,'original_split')  # determine original split before move to unlabeled
            ds.update_image(path, split, [float(temp) for temp in preds[i]])

        return ds


    # Get the path to the current script
    script_dir = Path(__file__).resolve().parent

    path_to_solver_config = script_dir / 'FGMRES.json'


    def prob_label_spreading(self, dataset_name, dataset, data_space, annotation_column, n_data, k, alpha,
                             config =path_to_solver_config): # config =  "/home/heller/projects/probl_dcb/prob_label_spreading/configs/FGMRES.json"

        if n_data == "all":
            n_data = len(dataset)
            print("n_data", n_data)
        else:
            dataset = dataset.sample(n = n_data) # random data subset
            dataset.reset_index(drop=True, inplace=True)

        classes = list(set(dataset.query("label != -1")["label"])) # determine the classes
        n_classes = len(classes)
        start=time.time()

        valid_indices = dataset[dataset["label"] != -1].index # indices that have at least one annotation

        annotations = [] # get annotations as a list
        indices = [] # and store the corresponding indices of the datapoints
        for i in valid_indices:
            annots = np.array(dataset.loc[i, annotation_column], dtype = int)
            for c in classes:
                if annots[c] == 0:
                        continue
                else:
                    annotations.extend([c]*annots[c])
                    indices.extend([i]*annots[c])

        if alpha != 0: # if alpha is zero we do not need to set up the graph structure nor solve the linear system
            # set graph and linear system
            S, I, time_NN, time_graph = self.set_graph(k, dataset, data_space, n_data)
            spreaded_info, trials, ind, time_spread, time_setup, time_solver = self.sampling_and_spreading_efficient(alpha, S, dataset, annotations, classes, n_classes, config, indices)

            dataset["prediction"] = [x.argmax() for x in spreaded_info]
            dataset['trials'] = trials
            dataset["p_hat"] = [ (spreaded_info[i] + ((1e-10)/n_classes)) /(trials[i] + 1e-10) for i in range(len(trials))] 

        else:
            pass # baseline - not necessary to adapt it here

            # feedback = np.zeros( (n_data, len(classes))) # counts the number of sampled classes for every datapoint -> histogram that corresponds to the prob. distr.
            # for i in indices: # for every sampled data point update the estimated prob. distr.
            #     c = np.random.choice(classes, p = dataset[prob_label_column][i]) # sample a class according to the prob. label
            #     feedback[i, c] += 1
            # # insert uniform distr. for data points that have not been sampled
            # trials = [x.sum() for x in feedback]
            # feedback = [x if x.sum() > 0 else np.ones(len(classes)) for x in feedback]
            # # normalize 
            # feedback = [x / x.sum() for x in feedback]

            # dataset["prediction"] = [x.argmax() for x in feedback] # this will assign class zero if the distr. is uniform
            # dataset["p_hat"] = [x for x in feedback]

            # assign prediction

        # computing scores - also not necessary here 

        # dataset["most_probable_class"] = list( np.array( dataset[prob_label_column].to_list() ).argmax(axis = 1) )
        # acc = len(dataset.query("most_probable_class == prediction"))/len(dataset)
        # p = np.array(dataset[prob_label_column].to_list())
        # p_hat = np.array(dataset["p_hat"].to_list())
        # rmse = sklearn.metrics.root_mean_squared_error(p_hat, p)
        # kl = kl_div(torch.tensor(p_hat).log(), torch.tensor(p), reduction = "batchmean").item()

        elapsed_time = time.time() - start

        data_dim = dataset[data_space].iloc[0].shape[0]

        if data_space == "feature":
            dim_reduction_technique = "NONE"
        elif len(data_space.split("_")[:-1]) > 1: # combination of e.g. CLIP_UMAP
            dim_reduction_technique = "+".join(data_space.split("_")[:-1])
        else:
            dim_reduction_technique = data_space.split("_")[0] # just one technique e.g. CLIP 

        result = pd.DataFrame({"dataset": dataset_name, "n_data": n_data, "dim_reduction_technique": dim_reduction_technique, "dimension": data_dim, "annotations": annotation_column,\
            'alpha':alpha, "k": k,
            # 'RMSE':rmse,
            # 'Accuracy':acc, "KL": kl , 
            "runtime": elapsed_time, "config": config}, index=[0])

        return result, dataset


    def set_graph(self, k_NN, train, data_space, n_data = None):

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


    def solve_probabilistic(self, Matrix, cfg, indices, length, dataset, annotations, n_classes, Sampling_matrix, classes, positives_new, trials):
        
        ### suppress the repetitive output of the AMGX library
        pyamgx.register_print_callback(lambda *args, **kwargs : None)
        ####
        start_solver=time.time()
        pyamgx.initialize()

        # Initialize config and resources:
        cfg = pyamgx.Config().create_from_file(cfg)
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
        for enum, i in enumerate(indices): # loops over all annotations received to spread the information
            e_i = np.zeros(length)
            e_i[i] = 1
            #e_i ist der i-te Einheitsvektor. Solution ist die Information die der i-te Punkt im Graphen spreaded/verteilt
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
        
            if np.isnan(solution).all():
                solution = np.zeros(length) # it might happen that the solver occasionally does not find the solution (for alpha and k very small)
            #normieren der Solution, sodass der maximale verteilte Wert 1 ist. trials immer erhöhen 
            m = np.max(solution)
            if m > 0: # always the case if the solution is found, otherwise solution is set to zero anyways so no update necessary
                trials += solution/m
    
            j = annotations[enum] # current annotation for the current datapoint

            #merken der gesampleten Klassen in einer Samplingmatrix.      
            Sampling_matrix[i][classes.index(j)] += 1

            #spreaden der Information für die Klasse die gezogen (in der Spalte de gezogenen Klasse). Für jeden Punkt ist die Summe der über alle Klasse erhalten Informationen in positives_new gleich dem Wert in trials
            if m > 0:
                positives_new[:,classes.index(j)] += solution/m

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

    def sampling_and_spreading_efficient(self, alpha, S, dataset, annotations, classes, n_classes, config, indices):
        
        length = len(dataset)

        # vector trials accounts for cumulative feedback
        trials = np.zeros(length)
        positives_new = np.zeros(length*n_classes).reshape(length,n_classes) #  feedback per class
        Sampling_matrix = np.zeros(length*n_classes).reshape(length, n_classes) #  how often a class was drawn - unimportant here
        Sampling_probs = np.zeros(length*n_classes).reshape(length, n_classes)

        I = scipy.sparse.eye(length)
        
        M = I - alpha * S # heat kernel to solve
        
        Sampling_matrix, positives_new, trials, time_setup, time_spread, time_solver = self.solve_probabilistic(M, config, indices, length, dataset, annotations,
                                                                                                                n_classes, Sampling_matrix, classes, positives_new, trials)
            
        return positives_new, trials, indices, time_spread, time_setup, time_solver
    

def main(argv):
    alg = PLS()
    alg.apply_algorithm()


if __name__ == '__main__':
    app.run(main)
