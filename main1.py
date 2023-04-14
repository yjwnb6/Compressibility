# %%
from scipy.io import loadmat
from scipy.sparse import triu
from scipy.sparse.linalg import eigs
import scipy
import pandas as pd
import numpy as np
import itertools

def avgClusteringCoefficient(graph):
    graph[graph != 0] = 1
    deg = np.sum(graph, axis=1)
    tri = np.triu(graph)
    cn = np.diag(np.dot(np.dot(graph, tri), graph))
    c = np.zeros(deg.shape)
    c[deg > 1] = 2 * cn[deg > 1] / (deg[deg > 1] * (deg[deg > 1] - 1))
    acc = np.mean(c[deg > 1])
    return acc

def rate_distortion(G, heuristic, num_pairs):
    N = G.shape[0]
    E = int(np.sum(G) / 2)
    S = np.zeros(N)
    S_low = np.zeros(N)
    clusters = [list(range(N))]
    Gs = [G]
    P_old = G / np.sum(G, axis=1, keepdims=True)
    P_old_ori = P_old.copy()
    D, p_ss = eigs(P_old.transpose())
    D = np.real(D)
    p_ss = np.real(p_ss)
    ind = np.argmax(D)
    p_ss = p_ss[:,ind] / (np.sum(p_ss[:,ind]))
    p_ss_old = p_ss
    P_old[P_old == 0] = np.finfo(float).eps
    logP_old = np.log2(P_old)
    logP_old[np.isinf(logP_old)] = 0
    S_old = -np.sum(p_ss_old * np.sum(P_old * logP_old, axis=1))
    P_old = P_old_ori.copy()
    P_joint = P_old * (np.tile(p_ss_old, (N, 1)).T)
    P_low = P_old
    S[-1] = S_old
    S_low[-1] = S_old
    clusters.append(list(range(1, N+1)))
    Gs.append(G)
    for n in range(N - 1, 1, -1):
        if heuristic == 1:
            pairs = list(itertools.combinations(range(1, n+1), 2))
            I = [pair[0] for pair in pairs]
            J = [pair[1] for pair in pairs]
        elif heuristic == 2:
            pairs = list(itertools.combinations(range(1, n+1), 2))
            inds = np.random.choice(pairs.shape[0], min(num_pairs, pairs.shape[0]), replace=False)
            I = pairs[inds, 0]
            J = pairs[inds, 1]
        elif heuristic == 3:
            I, J = np.where(np.triu(P_old + P_old.T, k=1))
        elif heuristic == 4:
            I, J = np.where(np.triu(P_old + P_old.T, k=1))
            pair_inds = np.random.choice(len(I), min(num_pairs, len(I)), replace=False)
            I = I[pair_inds]
            J = J[pair_inds]
        elif heuristic == 5:
            P_joint_symm = np.triu(P_joint + P_joint.T, k=1)
            P_joint_symm = np.triu(P_joint + P_joint.T, k=1)
            inds = np.argpartition(-P_joint_symm.flatten(), kth=min(num_pairs, np.sum(P_joint_symm > 0)))[:min(num_pairs, np.sum(P_joint_symm > 0))]
            I, J = np.unravel_index(inds, (n+1, n+1))
        elif heuristic == 6:
            P_joint_symm = np.triu(P_joint + P_joint.T + np.tile(np.diag(P_joint), (n+1, 1)) + np.tile(np.diag(P_joint), (1, n+1)), k=1)
            inds = np.argpartition(-P_joint_symm.flatten(), kth=min(num_pairs, np.sum(P_joint_symm > 0)))[:min(num_pairs, np.sum(P_joint_symm > 0))]
            I, J = np.unravel_index(inds, (n+1, n+1))

        elif heuristic == 7:
            P_ss_temp = np.triu(np.tile(p_ss_old, (n+1, 1)).T + np.tile(p_ss_old, (n+1, 1)).T, 1)
            inds = np.argpartition(-P_ss_temp.flatten(), kth=min(num_pairs, int(scipy.special.comb(n+1, 2))))[:min(num_pairs, int(scipy.special.comb(n+1, 2)))]
            I, J = np.unravel_index(inds, (n+1, n+1))
        elif heuristic == 8:
            I = np.array([0])
            J = np.array([n])
        else:
            raise ValueError('Variable "setting" is not properly defined.')
        
        num_pairs_temp = len(I)
        S_all = np.zeros(num_pairs_temp)
        for ind in range(1, num_pairs_temp, 1):
            i = I[ind]
            j = J[ind] 
            inds_not_ij = list(range(1, i)) + list(range(i+1, j)) + list(range(j+1, n+1))
            p_ss_temp = np.concatenate((p_ss_old[inds_not_ij], [p_ss_old[i] + p_ss_old[j]]), axis=0)
            P_temp_1 = np.sum(np.tile(p_ss_old[inds_not_ij, np.newaxis], (1, 2)) * P_old[np.ix_(inds_not_ij, [i, j])], axis=1)
            P_temp_1 = P_temp_1 / p_ss_temp[:-1]
            # P_temp_2 = np.sum(np.tile(p_ss_old[[i, j]].T, (1, n-1)) * P_old[[i, j], inds_not_ij], axis=0)
            # P_temp_2 = P_temp_2 / p_ss_temp[-1]
            P_temp_3 = np.sum(np.sum(np.tile(p_ss_old[[i, j]], (2, 1)) * P_old[[i, j], [i, j]]))
            P_temp_3 = P_temp_3 / p_ss_temp[-1]
            logP_temp_1 = np.log2(P_temp_1)
            logP_temp_1[np.isinf(logP_temp_1)] = 0
            # logP_temp_2 = np.log2(P_temp_2)
            # logP_temp_2[np.isinf(logP_temp_2)] = 0
            logP_temp_3 = np.log2(P_temp_3)
            # logP_temp_3[np.isinf(logP_temp_3)] = 0
            # assuming p_ss_temp, P_temp_1, P_temp_2, P_temp_3, p_ss_old, P_old, and logP_old are already computed
            #- p_ss_temp[-1] * np.sum(P_temp_2 * logP_temp_2)
            dS = -np.sum(p_ss_temp[:-1] * P_temp_1 * logP_temp_1)  - \
                p_ss_temp[-1] * P_temp_3 * logP_temp_3 + \
                np.sum(p_ss_old * P_old[:, i] * logP_old[:, i]) + np.sum(p_ss_old * P_old[:, j] * logP_old[:, j]) + \
                p_ss_old[i] * np.sum(P_old[i, :] * logP_old[i, :]) + p_ss_old[j] * np.sum(P_old[j, :] * logP_old[j, :]) - \
                p_ss_old[i] * (P_old[i, i] * logP_old[i, i] + P_old[i, j] * logP_old[i, j]) - \
                p_ss_old[j] * (P_old[j, j] * logP_old[j, j] + P_old[j, i] * logP_old[j, i])
            S_temp = S_old + dS
            S_all[ind] = S_temp
    return S, S_low, clusters, Gs


df = pd.read_csv(r'C:\Users\余峻玮\Desktop\Research\Compressibility\pycode\graph.csv')
data = pd.DataFrame(df)
families = pd.DataFrame(data.family.unique())
num_families = len(families)
graphs_topTrans = {'G_FOLDOC', 'G_AstroPh', 'G_Facebook', 'G_karate','G_Clavier', 'G_SC_2', 'G_macaques', 'G_zebras'}
families = data.family.unique()
tmp = families[0]
families[0] = families[7]
families[7] = tmp
# for i in range(0, num_families):
# print(loadmat('graphs_' + families[0] + '_samples_undirected.mat', squeeze_me=True, struct_as_record=False))
# data = read_mat('graphs_info.mat')

#i
for i in range(num_families):

    graph_struct = loadmat(r"C:\Users\余峻玮\Desktop\Research\Compressibility\pycode\graphs\graphs_" + str(families[0]) + "_samples_undirected.mat", squeeze_me=True, struct_as_record=False)
    graph_names = (list(graph_struct.keys()))[3:]
    num_graphs = len(graph_names)
    rate_distortion_upper = np.zeros((num_graphs, 1))
    rate_distortion_lower = np.zeros((num_graphs, 1))
    rate_distortion_upper = np.zeros((num_graphs, 1))
    rate_distortion_lower = np.zeros((num_graphs, 1))
    avg_deg_inside = np.zeros((num_graphs, 1))
    avg_deg_outside = np.zeros((num_graphs, 1))
    num_edges_inside = np.zeros((num_graphs, 1))
    num_edges_boundary = np.zeros((num_graphs, 1))
    num_edges_outside = np.zeros((num_graphs, 1))
    cluster_size = np.zeros((num_graphs, 1))
    compressibility = np.zeros((num_graphs, 1))
    deg_het = np.zeros((num_graphs, 1))
    avg_clustering = np.zeros((num_graphs, 1))
    avg_deg = np.zeros((num_graphs, 1))
    measures_struct = []
    for j in range(num_graphs):
        Gs = graph_struct[graph_names[0]] #j
        num_samples = len(Gs) #not sure len(Gs)
        N = len(Gs[0])
        rate_distortion_upper_temp = np.zeros((num_samples, N))
        rate_distortion_lower_temp = np.zeros((num_samples, N))
        avg_deg_inside_temp = np.zeros((num_samples, N-2))
        avg_deg_outside_temp = np.zeros((num_samples, N-2))
        num_edges_inside_temp = np.zeros((num_samples, N-1))
        num_edges_boundary_temp = np.zeros((num_samples, N-1))
        num_edges_outside_temp = np.zeros((num_samples, N-1))
        cluster_size_temp = np.zeros((num_samples, N))
        compressibility_temp = np.zeros((num_samples, 1))
        deg_het_temp = np.zeros((num_samples, 1))
        avg_clustering_temp = np.zeros((num_samples, 1))
        avg_deg_temp = np.zeros((num_samples, 1))
        for k in range(num_samples):
            G = Gs # lack of k idx
            ks = np.sum(G, axis=0)
            # print(np.array(ks.T).reshape(1, -1))
            # print(abs(np.array(ks).reshape(-1, 1) - np.array(ks.T).reshape(1, -1)))
            #k
            deg_het_temp[0] = np.sum(np.abs(ks - ks.T)) / (N*(N-1)*np.mean(ks)) # don't know
            avg_clustering_temp[0] = avgClusteringCoefficient(G) #k
            avg_deg_temp[0] = np.mean(ks) #k

            if graph_names[0] in graphs_topTrans: #j
                rd_upper, rd_lower, Cs, _ = rate_distortion(G, 5, 100)
            else:
                rd_upper, rd_lower, Cs, _ = rate_distortion(G, 7, 100)

            rate_distortion_upper_temp[k,:] = rd_upper
            rate_distortion_lower_temp[k,:] = rd_lower
            compressibility_temp[k] = np.mean(rd_upper[-1] - rd_upper)

            # Compute statistics about large cluster:
            cluster_size_temp[k,0] = N
            cluster_size_temp[k,-1] = 1

            num_edges_inside_temp[k,0] = np.sum(np.sum(G))/2
            num_edges_boundary_temp[k,0] = 0
            num_edges_outside_temp[k,0] = 0
            # for l in range(1, N - 1):
                
            #     C = Cs[l]
                
            #     sizes = np.zeros(len(C))
            #     ind = np.argmax(sizes)
            #     cluster = C[ind]
            #     not_cluster = np.setdiff1d(np.arange(1, N+1), cluster)
                
            #     # cluster_size_temp[k,l] = len(cluster)
            #     avg_deg_inside_temp[k,l-1] = np.mean(ks[cluster])
                # avg_deg_outside_temp[k,l-1] = np.mean(ks[not_cluster])
                # num_edges_inside_temp[k,l] = np.sum(np.sum(G[cluster][:,cluster]))/2
                # num_edges_boundary_temp[k,l] = np.sum(np.sum(G[cluster][:,not_cluster]))
                # num_edges_outside_temp[k,l] = np.sum(np.sum(G[not_cluster][:,not_cluster]))/2
        
    rate_distortion_upper[j] = rate_distortion_upper_temp
    rate_distortion_lower[j] = rate_distortion_lower_temp
    avg_deg_inside[j] = avg_deg_inside_temp
    avg_deg_outside[j] = avg_deg_outside_temp
    num_edges_inside[j] = num_edges_inside_temp
    num_edges_boundary[j] = num_edges_boundary_temp
    num_edges_outside[j] = num_edges_outside_temp
    cluster_size[j] = cluster_size_temp
    compressibility[j] = compressibility_temp
    deg_het[j] = deg_het_temp
    avg_clustering[j] = avg_clustering_temp
    avg_deg[j] = avg_deg_temp

measures_struct[f'{families[i]}_rate_distortion_upper'] = rate_distortion_upper
measures_struct[f'{families[i]}_rate_distortion_lower'] = rate_distortion_lower
measures_struct[f'{families[i]}_avg_deg_inside'] = avg_deg_inside
measures_struct[f'{families[i]}_avg_deg_outside'] = avg_deg_outside
measures_struct[f'{families[i]}_num_edges_inside'] = num_edges_inside
measures_struct[f'{families[i]}_num_edges_boundary'] = num_edges_boundary
measures_struct[f'{families[i]}_num_edges_outside'] = num_edges_outside
measures_struct[f'{families[i]}_cluster_size'] = cluster_size
measures_struct[f'{families[i]}_compressibility'] = compressibility
measures_struct[f'{families[i]}_deg_het'] = deg_het
measures_struct[f'{families[i]}_avg_clustering'] = avg_clustering
measures_struct[f'{families[i]}_avg_deg'] = avg_deg

with open(f'graphs_{families[i]}_rate_distortion_samples', 'w') as f:
    pickle.dump(measures_struct, f)



# Set the diagonal elements to the eigenvalues
# print(p_ss)
# if graph_names[0] in graphs_topTrans: # j
#     # Use top joint transition probability pairs heuristic:
#     rd_upper, rd_lower, Cs, _ = rate_distortion(G, 5, 100)
# else:
#     # Use top stationary distribution pairs heuristic:
#     rd_upper, rd_lower, Cs, _ = rate_distortion(G, 7, 100)

# print(data["None"])
# annots = loadmat('graphs_info.mat', squeeze_me = True)
# print(data['__function_workspace__'])
# print(annots.keys())
# family = annots['family']
# print (family[: , :])
# print(annots['__function_workspace__'])
# annots['__function_workspace__'][0][0]['family'], annots['__function_workspace__'][0][0]['name']
# %%
