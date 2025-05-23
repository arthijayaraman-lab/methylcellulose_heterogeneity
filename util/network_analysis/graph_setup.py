import numpy as np

import skimage.morphology as ski
import networkx as nx

from scipy.spatial import KDTree
from copy import deepcopy
from skimage.morphology import binary_dilation
import matplotlib.pyplot as plt
import sys
import warnings

def check_number_of_dat_voxels(G): #write a code to compare the lists and see whats missing, which node/edge it comes from
    dat_voxels_list = list(nx.get_node_attributes(G, 'dat_voxels').values())
    node_vox_list = []
    for l in dat_voxels_list:
        node_vox_list.extend(l)
        
    dat_voxels_list = list(nx.get_edge_attributes(G, 'dat_voxels').values())
    edge_vox_list = []
    for l in dat_voxels_list:
        edge_vox_list.extend(l)
    vox_list = node_vox_list.copy()
    vox_list.extend(edge_vox_list)
    return np.unique(vox_list, return_counts = True)

def get_where_discrepancies(l1, l2, G):
    diffs = set(l1) - set(l2)
    node_dict = nx.get_node_attributes(G, 'dat_voxels')
    edge_dict = nx.get_edge_attributes(G, 'dat_voxels')
    
    locs = []
    for vox in diffs:
        for key, val in node_dict.items():
            if vox in val:
                locs.append(key)
        for key, val in edge_dict.items():
            if vox in val:
                locs.append(key)
    return diffs, locs
                
def simplifyGraph(G, debug = False):
    ''' Loop over the graph until all nodes of degree 2 have been removed and their incident edges fused '''
    
    g = G.copy()
    
    # add 'dat_voxels' attribute to all edges
    
    for edge in list(g.edges()):
        if 'dat_voxels' in g[edge[0]][edge[1]].keys(): 
            continue
        else:
            g[edge[0]][edge[1]]['dat_voxels'] = []
        
    ct1 = 0
    while any(degree==2 for _, degree in g.degree):    
        g0 = g.copy() #<- simply changing g itself would cause error `dictionary changed size during iteration` 
        ct1+=1
        ct2 = 0
        for node, degree in g.degree():
            ct2 +=1
            edges = g0.edges(node)
            edges = list(edges.__iter__())
            if len(edges)==2:
                if debug: num_init = check_number_of_dat_voxels(g0)
                a0,b0 = edges[0] 
                a1,b1 = edges[1]
                
                e0 = a0 if a0!=node else b0
                e1 = a1 if a1!=node else b1
                
                wt1 = g0.get_edge_data(a0, b0)['weight'] #get weight of edge 1
                wt2 = g0.get_edge_data(a1, b1)['weight'] #get weight of edge 2
                
                #combine dat_voxels to store in edges
                dat_voxels_to_add = []
                dat_voxels_of_this_node = g0.nodes[node]['dat_voxels']
                dat_voxels_to_add.extend(dat_voxels_of_this_node)
                dat_voxels_of_adjacent_edges1 = g0.get_edge_data(a0, b0)['dat_voxels']
                dat_voxels_to_add.extend(dat_voxels_of_adjacent_edges1)
                dat_voxels_of_adjacent_edges2 = g0.get_edge_data(a1, b1)['dat_voxels']
                dat_voxels_to_add.extend(dat_voxels_of_adjacent_edges2)
                
                g0.remove_node(node)
                g0.add_edge(e0, e1) #add the edge 
                g0[e0][e1]['weight'] = wt1 + wt2 #add the weight as the sum of the two weights of the pervious edges
                
                if 'dat_voxels' in g0[e0][e1].keys(): #handle the fringe case where the edge already exists
                    g0[e0][e1]['dat_voxels'].extend(dat_voxels_to_add)
                else: 
                    g0[e0][e1]['dat_voxels'] = dat_voxels_to_add
                
                if debug: 
                    num_final = check_number_of_dat_voxels(g0)
                    print(ct1, ct2)
                    print(len(num_init), len(num_final))
        g = g0
    return g

def skeletonize_voxel_representation(voxel_rep):
    skeleton = ski.skeletonize_3d(voxel_rep)
    skeleton = skeleton/255
    return skeleton

def make_skel_to_data_dict(skel, dat_inds):
    #make a dictionary with keys referring to the skeleton index and items being a list of all associated data inds
    skel_inds = np.array(np.where(skel)).T
    tree = KDTree(skel_inds)
    Ls, closest_skel_inds = tree.query(dat_inds)
    
    skel_to_data_dict = dict()
    for i in range(len(skel_inds)):
        skel_to_data_dict[i] = []
    
    for i,val in enumerate(closest_skel_inds):
        skel_to_data_dict[int(val)].append(i)
    return skel_to_data_dict
        
def assign_original_voxels_to_skel_graph(G, skel_to_data_dict):
    nx.set_node_attributes(G, skel_to_data_dict, 'dat_voxels')
    return G
    
def expand_and_reskeletonize(skeleton):
    skel = binary_dilation(skeleton)
    skel = skeletonize_voxel_representation(skel)
    return skel

def check_edge(graph, node1, node2, voxel_inds, num_vox):
    #if nodes are not more than half the box apart, return false
    p1 = graph.nodes[node1]['pos']
    p2 = graph.nodes[node2]['pos']
    dist = np.linalg.norm(p2 - p1)
    if dist < 0.5*num_vox:
        return False
    
    #check if original inds for each node are in the edge list.
    #if so, return true
    vox_inds1 = graph.nodes[node1]['dat_voxels']
    vox_inds2 = graph.nodes[node2]['dat_voxels']
    
    pos1 = voxel_inds[vox_inds1]
    pos2 = voxel_inds[vox_inds2]
    
    tree = KDTree(pos1, boxsize = num_vox)
    neighs = tree.query_ball_point(pos2, 1.99)
    
    check = np.any([len(x) > 0 for x in neighs])    

    return check
    
def check_near_edge(rep_size, pos_ind, tol = 5):
    if np.any(pos_ind < tol):
        return True
    if np.any(pos_ind>rep_size - tol):
        return True
    return False

def check_skeleton_edges(G_skel, voxel_inds, vox_rep):
    """
    Sometimes skeletonization algorithm "pulls" the skeleton inwards from 
    the edge of the box. This scans the skeleton voxels near the edge of the box 
    and adds edges to the skeleton graph if the original voxels crossed the 
    periodic boundaries.

    Parameters
    ----------
    G_skel : networkx graph
        graph of the skeleton
    voxel_inds : np array
        array of positions corresponding to voxel positions

    Returns
    -------
    G_skel_upd.
        updated networkx graph with added edges near the periodic boundaries

    """
    G_skel_upd = G_skel.copy()
    num_vox = vox_rep.shape[0]

    nodes = list(G_skel.nodes)
    
    pos = nx.get_node_attributes(G_skel, 'pos')
    
    nodes = [n for n in nodes if check_near_edge(num_vox, pos[n])]
    
    curr_edges = list(G_skel.edges)
    for i,n1 in enumerate(nodes):
        for n2 in nodes[i+1:]:
            if (n1, n2) in curr_edges:
                continue
            check = check_edge(G_skel, n1, n2, voxel_inds, num_vox)
            if check:
                G_skel_upd.add_edge(n1, n2, weight = 1)

    return G_skel_upd
    
    

def make_original_graph(skeleton):
    skel_inds = np.where(skeleton == 1)
    skel_inds = np.transpose(np.array(skel_inds))
    
    num_vox = skeleton.shape[0]
    
    tree = KDTree(skel_inds, boxsize = num_vox)
    edges = tree.query_ball_point(skel_inds, 1.99)
    
    edge_list = []
    for i,edge in enumerate(edges):
        for val in edge:
            if val == i:
                continue
            else:
                edge_list.append([i,val])

    '''Make original graph (each node is a skeleton voxel)'''
    G_orig = nx.from_edgelist(edge_list)
    
    #add a node for any voxel with no adjacent neighbors
    for i in range(int(np.sum(skeleton))):
        if not i in G_orig.nodes:
            G_orig.add_node(i)
    
    node_label = dict()
    for i, ind in enumerate(skel_inds):
        node_label[i] = ind
    
    nx.set_node_attributes(G_orig, node_label, name = 'pos')
    nx.set_edge_attributes(G_orig, values = 1, name = 'weight')
    return G_orig

def compress_edges(G_nodes_along_edges, debug = False):
    G = deepcopy(G_nodes_along_edges)
    G = simplifyGraph(G, debug)
    return G
    
def count_clusters(G):
    n_clusters = 0
    for tup in nx.get_edge_attributes(G, 'weight').items():
        val  = tup[1]
        if val == 1: n_clusters +=1
    return n_clusters

def expand_and_skeleltonize_cluster_removal(skeleton):
    G = make_original_graph(skeleton)
    prev_cluster_nodes = np.inf
    curr_cluster_nodes = count_clusters(G)
    
    while prev_cluster_nodes > curr_cluster_nodes:
        G_last = G.copy()
        skel_last = skeleton.copy()
        print(prev_cluster_nodes, curr_cluster_nodes)
        prev_cluster_nodes = curr_cluster_nodes
        skeleton = expand_and_reskeletonize(skeleton)
        G = make_original_graph(skeleton)
        G = compress_edges(G)
        n_cluster_nodes = count_clusters(G)
        curr_cluster_nodes = n_cluster_nodes
    
    return G_last, skel_last

def remove_cluster_artifacts(G, G_orig, num_vox):
    '''Remove Junk Nodes from Graph'''
    #Skeletonization can lead to clusters of nodes around junctions. 
    #This process removes the junk nodes and replaces them with a single representative node

    G_with_clusters = G.copy()
    edge_list = []
    for edge in G_with_clusters.edges():
        if G_with_clusters.get_edge_data(edge[0], edge[1])['weight'] == 1:
            edge_list.append(edge)
            
    max_node = np.max(G_orig.nodes)
    C = nx.from_edgelist(edge_list)
    clusters = list(nx.connected_components(C))
    val = 1
    diam_dict = dict() #gives value of original node corresponding to the new node, used for diameter calculation later
    extra_edges = [] #keep track of the extra edges you add so as not to use them for analysis later
    ct2 = 0
    ct1 = 0
    for cluster in clusters[:]:
        # for edge in G_with_clusters.edges():
        #     if G_with_clusters[edge[0]][edge[1]]['weight']>1:
        #         cluster.discard(edge[0])
        #         cluster.discard(edge[1])
        #num_init = check_number_of_dat_voxels(G_with_clusters)
        cluster_neighbors = []
        ct1 += 1
        neighbor_edges = []
        for edge in G_with_clusters.edges(cluster):
            ct2 +=1 
            node1 = edge[0]
            node2 = edge[1]
            if not(node1 in cluster):
                cluster_neighbors.append(node1)
                neighbor_edges.append(edge)
            if not(node2 in cluster):
                cluster_neighbors.append(node2)
                neighbor_edges.append(edge)
                
        if len(cluster_neighbors) !=0:
            #find location of added node
            pos = []
            for node in cluster:
                pos.append(np.array(G_with_clusters.nodes[node]['pos']))
                
            pos = np.array(pos)  
            
            # ax = plt.figure().add_subplot(projection = '3d')
            # ax.scatter(*pos.T, color = 'k')
            
            pw = pos.copy()
            #handle unwrapping for cluster spanning pbcs
            pu, _, _, orig_ixs = unwrap3(pos, num_vox, np.array([1 for p in pos]), np.zeros((2, 3)), plot = False, return_ix = True)
            
            shift = pu[0] - pw[orig_ixs[0]]
            
            pu -= shift[0]
            
            avg_loc = np.mean(pu, axis = 0)
            avg_loc = wrap(avg_loc, num_vox) 
            # ax.scatter(*avg_loc, color = 'r')
            
            #get index of added node
            node_no = max_node + val
            
            #SHOULDNT NEED diam_dict WITH NEW CODE
            dist_to_avg = [] #define distance to average to select the closest node to map to for diameter calculation (Since G_orig does not include these added nodes)
            clust_list = []
            for node in cluster:
                inds = np.array(G_with_clusters.nodes[node]['pos'])
                dist_to_avg.append(np.sum((avg_loc - inds)**2))
                clust_list.append(node)
            dist_to_avg = np.array(dist_to_avg)
            closest_ind = np.where(dist_to_avg == min(dist_to_avg))[0]
            if len(closest_ind)>0: closest_ind = closest_ind[0] #if more than one are the same distance, just pick the first in the list
            cluster_val = clust_list[closest_ind]
            diam_dict[node_no] = cluster_val
            
            #Get all data_voxels to add to the new node
            dat_voxel_list = []
            for node in cluster:
                dat_voxel_list.extend(G_with_clusters.nodes[node]['dat_voxels'])
            for edge in G_with_clusters.edges(cluster):
                if edge[0] in cluster and edge[1] in cluster:
                    dat_voxel_list.extend(G_with_clusters[edge[0]][edge[1]]['dat_voxels'])
            
            G_with_clusters.add_node(node_no, pos= avg_loc, old_node= cluster_val, dat_voxels = dat_voxel_list) 
            G_orig.add_node(node_no, pos = avg_loc, old_node = cluster_val, dat_voxels = dat_voxel_list) #add node here so that you can compare lengths later
            val = val+1
            # edge_list = []
            # for node in cluster_neighbors:
            #     edge = [node_no, node]
            #     edge_list.append(edge)
            #     extra_edges.append(edge)
                
            for edge in neighbor_edges:
                if edge[0] in cluster and not(edge[1] in cluster):
                    cluster_node = edge[0]
                    neighbor_node = edge[1]
                elif edge[1] in cluster and not(edge[0] in cluster):
                    cluster_node = edge[1]
                    neighbor_node = edge[0]
                else:
                    print('something is wrong here')
                    
                if not G_with_clusters.has_edge(node_no, neighbor_node):
                    dat_voxel_list = G_with_clusters[cluster_node][neighbor_node]['dat_voxels']
                    G_with_clusters.add_edge(neighbor_node, node_no, weight = 2, dat_voxels = dat_voxel_list)
                else: 
                    dat_voxel_list = G_with_clusters[cluster_node][neighbor_node]['dat_voxels']
                    G_with_clusters[node_no][neighbor_node]['dat_voxels'].extend(dat_voxel_list)
                    

            # G_with_clusters.add_edges_from(edge_list)
            # for e in edge_list:
            #     G_with_clusters[e[0]][e[1]]['weight'] =2
            #     G_with_clusters[e[0]][e[1]]['dat_voxels'] = []
            
            
            #num_before_removal = check_number_of_dat_voxels(G_with_clusters)            
            #G_copy = G_with_clusters.copy()
            G_with_clusters.remove_nodes_from(cluster)
            
            #num_final = check_number_of_dat_voxels(G_with_clusters)            
            
            
            #diffs, locs = get_where_discrepancies(num_init, num_final, G_copy)
        
        # num_final_final = check_number_of_dat_voxels(G_with_clusters)            
        # print(len(num_final_final))
    G_without_clusters = compress_edges(G_with_clusters, debug = False)
    return G_without_clusters
   
    
def make_edgelist(pts, len_box, cutoff, pbc):
    if pbc:
        tree= KDTree(pts, boxsize = len_box)
    else:
        tree= KDTree(pts)

    neigh = tree.query_ball_point(pts, cutoff )
    
    edges = []
    num_neigh = []
    dists = []
    for i, n in enumerate(neigh):
        dist, neighs = tree.query(pts[i], k = len(n))
        
        if len(n) > 1:
            assert set(neighs) == set(n)
        
            edges.extend([[i, x] for x in neighs if i!=x])
            num_neigh.extend([len(n)-1 for x in neighs[1:]])
            dists.extend(dist[1:])
            
    return edges, num_neigh, dists
    
def make_bead_graph(bead_pos, len_box, cutoff = 2, pbc = True):
    '''Make Bead Graph'''
    #Make the graph of all simulation beads, connecting any within a distance cutoff (here it's 2)
    edges = []
    
    G_beads = nx.Graph()
    
    
    for i,pt in enumerate(bead_pos):
        G_beads.add_node(i, pos = bead_pos[i])
    
    edges, num_neigh, dists = make_edgelist(bead_pos, len_box, cutoff, pbc)
    
    edges = np.array(edges).astype(int)
    num_neigh = np.array(num_neigh)
    dists = np.array(dists)
    
    if len(edges) > 0:
        G_beads.add_edges_from(edges[:,:2])
    #store distances between beads as wedge weights for use in strand length calculation later
        comb = 10*dists - num_neigh
        comb -= np.min(comb)
        comb +=1
    else:
        comb = []


    for edge, nn, d, c in zip(edges, num_neigh, dists, comb):
        bead1 = edge[0]
        bead2 = edge[1]
        G_beads[bead1][bead2]['dist'] = d
        G_beads[bead1][bead2]['num_neigh'] = nn
        G_beads[bead1][bead2]['comb'] = c
    return G_beads

def make_main_component_graph(G):
    '''Make Graph of Main Component'''
    #remove extraneous portions that do not connect to the primary connected component
    groups =list(nx.connected_components(G))
    groups= sorted(nx.connected_components(G), key=len, reverse=True)
    
    main_group = groups[0]
    G_main = G.subgraph(main_group)
    return G_main

def make_graph_excluding_boundary_nodes(G, max_ind, boundary_size = 2):
    G_removed = G.copy()
    node_pos = nx.get_node_attributes(G, 'pos')
    nodes_to_remove = []
    for node, pos in node_pos.items():
        if np.any(pos <= boundary_size-1) or np.any(pos >= max_ind - boundary_size + 1):
            nodes_to_remove.append(node)
    G_removed.remove_nodes_from(nodes_to_remove)
    return G_removed

def wrap(pts, len_box):
    return np.mod(pts, len_box)
    
def unwrap(x, y, z, len_box):
    r_unwrap = np.zeros(([3, len(x)]))
    r_unwrap[:,0] = [x[0], y[0], z[0]]
    for i in range(len(x)-1):
        r1 = r_unwrap[:,i] #set 2 active atoms to i and i+1 (all times)
        r2 = np.array([x[i+1], y[i+1], z[i+1]])
        delx=r2[0]-r1[0]
        dely=r2[1]-r1[1]
        delz=r2[2]-r1[2]
        delx=delx-len_box*np.round(delx/len_box) #Periodic boundary conditions
        dely=dely-len_box*np.round(dely/len_box)
        delz=delz-len_box*np.round(delz/len_box)
        delr = np.array([delx, dely, delz])
        r_unwrap[:, i+1] = r1 + delr     
    return(r_unwrap)

def unwrap2(pts, len_box):
    x = pts[:,0]
    y = pts[:,1]
    z = pts[:,2]
    
    xu, yu, zu = unwrap(x, y, z, len_box)
    
    ret = np.vstack((xu, yu, zu)).T
    return ret

def unwrap3(pts, len_box, bead_ix, node_pos, plot = False, max_iter = 10, return_ix = False):
    ncomp = 2
    unwrapped = pts.copy()
    ret_node_pos = node_pos.copy()
    
    if plot:
        colors = ['r','g','b','gold','purple']
        ax = plt.figure().add_subplot(projection = '3d')
        ax.scatter(unwrapped.T[0], unwrapped.T[1], unwrapped.T[2], color = 'k', s = 1)
    ct = 0
    
    best_subset = unwrapped.copy()
    best_node_pos = ret_node_pos.copy()
    best_fraction = 0
    best_no=0
    best_shift = np.zeros((1,3))
    
    ix_out = bead_ix.copy()
    all_shift = np.zeros((1,3))
    while ncomp > 1:
        #TODO make this cutoff not hard coded
        graph = make_bead_graph(unwrapped, len_box, cutoff = 10, pbc = False)
        components = nx.connected_components(graph)
        ncomp = len(list(components))
        sizes = [len(c) for c in sorted(nx.connected_components(graph), key=len, reverse=True)]
        fraction = sizes[0]/sum(sizes)
        beads_in_largest_comp = list(max(nx.connected_components(graph), key=len))
        pts_in_largest_comp = unwrapped[beads_in_largest_comp]
        
        fraction = sizes[0]/sum(sizes)
        if sizes[0] > best_no:
            best_subset = pts_in_largest_comp.copy()
            ix_out= bead_ix[beads_in_largest_comp]
            best_fraction = fraction
            best_no = sizes[0]
            best_node_pos =ret_node_pos
            ret_ixs = beads_in_largest_comp
            
            
        shift = np.mean(pts_in_largest_comp, 0)
        all_shift += (- shift + len_box/2)
        unwrapped = unwrapped - shift + len_box/2
        unwrapped = wrap(unwrapped, len_box)
        
        ret_node_pos = ret_node_pos - shift + len_box/2
        ret_node_pos = wrap(ret_node_pos, len_box)
        
        
        if plot: ax.scatter(unwrapped.T[0], unwrapped.T[1], unwrapped.T[2], color = colors[ct], s = 1)
        ct+=1
        if ct == max_iter+1:
            warnings.warn(f'strand unwrapping: max_iter reached with {best_no}/{sum(sizes)} in largest component')
            break
    
    if return_ix: return best_subset, ix_out, best_node_pos, ret_ixs
    return best_subset, ix_out, best_node_pos
    
