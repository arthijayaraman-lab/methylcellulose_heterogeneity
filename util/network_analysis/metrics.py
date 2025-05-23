import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import warnings
from persistence_length import do_persistence_length
from scipy.spatial import KDTree
import graph_setup as gs
from sympy import divisors
import vox_setup as vs

class strand:
    def __init__(self, pts, bead_ix, graph, len_box, node_pos, trim = True, unwrap = True):
        self.pts = pts
        self.bead_ix = bead_ix
        self.get_uniq_pts()
        self.node_pos = node_pos

        self.get_node_ix()
        
        self.graph = graph
        self.num_beads = len(pts)
        self.len_box = len_box

            
        self.trim = trim
        if trim:
            self.trim_strand()
            
        if len(self.pts) < 10:
            unwrap = False
            warnings.warn('insufficient points in strand. Not unwrapping')

        self.unwrap = unwrap
        if unwrap:
            self.unwrap_strand()
    
    def get_node_ix(self, k = 5000):
        #get closest bead ixs before wrapping
        num_pts = len(self.pts)
        if k > num_pts: #lower k if greater than number of points
            k = num_pts
                
        tree = KDTree(self.pts)
        d, best_beads = tree.query(self.node_pos, k = k)
        #best_beads = best_beads[best_beads!=num_pts] 
            
        node_ixs = self.bead_ix[best_beads]
        self.node_ixs = node_ixs
    
    def get_uniq_pts(self):

        _, uniq = np.unique(self.bead_ix, return_index = True)
        self.pts = np.array(self.pts)[np.array(uniq)]
        self.bead_ix = np.array(self.bead_ix)[np.array(uniq)]

    
    def trim_strand(self):
        comp = list(nx.connected_components(self.graph))
        graph = nx.subgraph(self.graph, comp[0])
        positions = nx.get_node_attributes(graph, 'pos')
        
        self.bead_ix = np.array(list(positions.keys()))
        self.pts = np.array(list(positions.values()))
        self.num_beads = len(self.pts)
        self.graph = graph
    
    def unwrap_strand(self):
        
        # ax = plt.figure().add_subplot(projection = '3d')
        # ax.scatter(*self.pts.T, color = 'k', s = 0.1)
        # ax.scatter(*self.node_pos.T, color = 'r')
        unwrapped, ix, node_pos = gs.unwrap3(self.pts, self.len_box, self.bead_ix, self.node_pos)
        self.pts = unwrapped
        self.bead_ix = ix
        g = self.graph.copy()
        # Remove nodes that are NOT in the keep_nodes list
        self.graph = g.subgraph(self.bead_ix) #warning: 'pos' attributes become inaccurate with wrapping

        
    def get_furthest_points(self, method = 'nearest_to_node_loc'):
        
        if method == 'nearest_to_node_loc':
            ixs = []
            best_beads = []
            for node in self.node_ixs:
                err = True
                for ix in node:
                    if ix in self.bead_ix:
                        ixs.append(ix)
                        err = False
                        best_beads.append(np.where(self.bead_ix == ix)[0])
                        break
                    else: continue
                if err:
                    raise RuntimeError('Ran out of indexes, need to increase k in get_node_ix function')
            ixs = np.array(ixs)
            pts = self.pts[np.array(best_beads)]
            
            max_distance = np.linalg.norm(pts[0]- pts[1])
            
        elif method == 'oneshot':
            # Compute pairwise Euclidean distances
            distances = pdist(self.pts)
        
            # Convert to a square matrix for easier manipulation
            distance_matrix = squareform(distances)
        
            # Find the indices of the two points with the largest distance
            i, j = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)
            ix1, ix2 = self.bead_ix[i], self.bead_ix[j]
            # The two points
            point1 = self.pts[i]
            point2 = self.pts[j]
        
            # The maximum distance
            max_distance = distance_matrix[i, j]
            pts = np.vstack((point1, point2))
            ixs = np.array([ix1, ix2])
            
        elif method == 'iter':
            #for large # of particles, semi-serially process for lower mem  
            points = self.pts
            tree = KDTree(points)
            max_distance = 0
            max_ixs = []
            max_pts = []
            
            for i, point in enumerate(points):
                if i%1000 == 0: print(i)
                distances, indices = tree.query(point, k=len(points))
                max_idx = indices[-1]
                max_dist = distances[-1]
                
                if max_dist > max_distance:
                    max_distance = max_dist
                    max_ixs = [i, max_idx]
                    max_points = [point, points[max_idx]]
                    
            pts = np.array(max_points)
            ix1 = self.bead_ix[max_ixs[0]]
            ix2 = self.bead_ix[max_ixs[1]]
            ixs = np.array([ix1, ix2])
            
        else: 
            raise ValueError(f'furthest point method {method} is not supported.')
                        
        self.furthest_ixs = ixs
        self.end_pos = pts
        return pts, ixs, max_distance
        
    def get_path(self):
        try:
            a = self.furthest_ixs
        except:
            self.get_furthest_points()
            
            # if self.num_beads > 10000:
            #     self.get_furthest_points('iter')
            # else: self.get_furthest_points('oneshot')
            
        if not self.unwrap:
            warnings.warn('Calculating stand length with unwrapped coordinates')
        
        G_bead_node1= self.furthest_ixs[0]
        G_bead_node2 = self.furthest_ixs[1]
        
        pos_dict = {ix:pt for ix,pt in zip(self.bead_ix, self.pts, strict = True)}
        if nx.has_path(self.graph, G_bead_node1, G_bead_node2):
            path = nx.shortest_path(self.graph, source = G_bead_node1, target = G_bead_node2, weight = 'comb')
        else:
            raise ValueError('No Path exists')
        
        if len(path) == 1:
            self.path = []
        else:
            pp = []
            for i in range(len(path) -1): 
                ix1 = path[i]
                ix2 = path[i+1]
                pp.append(pos_dict[ix1])
            pp.append(pos_dict[ix2])
            self.path = np.array(pp)
        
    def get_len(self):
        try: a = self.path
        except: self.get_path()
                    
        if len(self.path) < 2:
            return np.nan
            
        len_path = 0
        for i in range(len(self.path)-1):
            len_path += np.sum((self.path[i] - self.path[i+1])**2)**0.5
        self.len_strand = len_path
        return len_path
        
            
    def get_pl(self):
        try: a = self.path
        except: self.get_path()
        
        plen, tan, auto, bl = do_persistence_length(self.path, plot = False)
        '''technically this should be the bond length corresponding to the mean autocorrelation, 
        but this assumes the distance between most points are relatively similar compared to the 
        overall persistence length
        '''
        bl_sum = [sum(bl[:i]) for i in range(len(bl))] 
        
        self.plen = plen
        
        return plen, bl_sum, auto
    
    def plot_strand(self, ax = None, s= 0.1, color = 'k', box = False, nodes = True, path = True):
        if not ax:
            ax = plt.figure().add_subplot(projection = '3d')
            
        ax.scatter(self.pts.T[0], self.pts.T[1], self.pts.T[2], s =s , color = color)
        if box:
            ax.set_xlim([0,self.len_box])
            ax.set_ylim([0,self.len_box])
            ax.set_zlim([0,self.len_box])
        
        if nodes:
            plot = False
            try:
                pos = self.end_pos
                plot = True
            except: warnings.warn('no node positions calculated, not plotting them')
            if plot:
                ax.scatter(pos.T[0], pos.T[1], pos.T[2], color = 'r', s = 30)
        if path:
            plot = False
            try:
                path = self.path
                plot = True
            except: warnings.warn('no path calculated, not plotting it')
            if plot:
                ax.plot(*path.T, color = 'b', linewidth = 3)


        ax.set_aspect('equal')
        return ax
    
def p2l(PQ,u): #gives the distance from point PQ to vector u
    return np.linalg.norm(np.cross(PQ,u))/np.linalg.norm(u)



def lacunarity(data): 
    n = len(data)
    box_sizes = divisors(n)
    locs = np.where(data > 0)
    voxels = np.array([(x,y,z) for x,y,z in zip(*locs)])

    lacs = []
    for size in box_sizes:
        bin_edges = [np.linspace(0, i, int(i/size+1)) for i in data.shape]
        H1, e = np.histogramdd(voxels, bins = bin_edges)
        counts = np.ndarray.flatten(H1)    
        un, cts = np.unique(counts, return_counts = True)
        Q = cts/sum(cts)
        PQ = un*Q
        P2Q = PQ*un
        Z1 = sum(PQ)
        Z2 = sum(P2Q)
        lac = Z2/Z1**2
        lacs.append(lac)
    return box_sizes, lacs

def lac_calc(data):
    print('Calculating Lacunarity')
    num_vox = data.shape[0]
    cut_vals = [32, 50, 64, 75, 100, 128, 200, 256, 300, 400, 500, 512, 600, 800, 900, 1000, 1024] #pick numbers that have lots of square divisors
    cut = next(x[0] for x in enumerate(cut_vals) if x[1] > num_vox)
    cut = cut_vals[cut-1]
    data = data[:cut, :cut, :cut]
    box_sizes, lacs = lacunarity(data)
    return box_sizes, lacs

def get_nearest_bead_to_node(G, node, dat_inds, nearby_beads, points):
    #get beads in voxel of node position
    vox = G.nodes[node]['pos']
    inds = G.nodes[node]['dat_voxels']
    
    beads = [nearby_beads[ind] for ind in inds]

    beads = beads[0]
    bead_pos = points[beads]
    dists_sq = np.sum((bead_pos - vox)**2, axis = 1)
    closest_bead = beads[np.argmin(dists_sq)]
    return closest_bead
    
def get_beads_from_edge(G, edge, nearby_beads):
    #first do edge
    vox_list = G[edge[0]][edge[1]]['dat_voxels']
    bead_list = []
    for vox in vox_list:
        bead_list.extend(nearby_beads[vox])
    #then do nodes
    for node in edge:
        vox_list = G.nodes[node]['dat_voxels']
        for vox in vox_list:
            bead_list.extend(nearby_beads[vox])

    return bead_list

def get_beads_from_G_orig_path(G_orig, nodes, nearby_beads):
    all_beads = []
    for node in nodes:
        vox_list = G_orig.nodes[node]['dat_voxels']
        bead_list = []
        for vox in vox_list:
            bead_list.extend(nearby_beads[vox])
        all_beads.append(bead_list)
    return all_beads


def get_node(graph, node):
    if 'old_node' in graph.nodes[node].keys():
        node = graph.nodes[node]['old_node']
    return node

def get_strands(G_for_analysis, G_beads, points, len_box, nearby_beads, vox_res, unwrap):
    
    G_edgelist = list(G_for_analysis.edges)
    
    # Get paths on bead graph
    strands = []
    for ie,edge in enumerate(G_edgelist[:]):
        pos1 = vs.ix_to_pos(G_for_analysis.nodes[edge[0]]['pos'], vox_res)
        pos2 = vs.ix_to_pos(G_for_analysis.nodes[edge[1]]['pos'], vox_res)
        node_pos = np.vstack((pos1, pos2))
        
        beads_on_edge = get_beads_from_edge(G_for_analysis, edge, nearby_beads)
        edge_subgraph = G_beads.subgraph(beads_on_edge)
        
        strand_pts = points[beads_on_edge]
        if len(beads_on_edge) < 10: continue
    
        s = strand(strand_pts, beads_on_edge, edge_subgraph, len_box, node_pos, unwrap = unwrap)
        strands.append(s)
        
    return strands
    

def diameters(G_for_analysis,G_orig, res, points, nearby_beads, skeleton, bead_diam, len_box):
    print('Calculating Strand Diameters')

    skel_inds = np.array(np.where(skeleton)).T
    skel_pos = skel_inds*res + res/2

    diams = []
    G_edgelist = list(G_for_analysis.edges)
    
    #you're moving these to make a good plot of vectors and diameters
    # ax = plt.figure().add_subplot(projection='3d')
    # ax.scatter(all_pts[:,0], all_pts[:,1], all_pts[:,2])
    # ax.scatter(d_pts[:,0], d_pts[:,1], d_pts[:,2])
    # ax.plot([0,avg_vec[0]], [0,avg_vec[1]], [0,avg_vec[2]], color = 'k')
    ct = 0
    for edge in G_edgelist[:]:
        node1 = edge[0]
        node2 = edge[1]
        if 'old_node' in G_for_analysis.nodes[node1].keys():
            node1 = G_for_analysis.nodes[node1]['old_node']
        if 'old_node' in G_for_analysis.nodes[node2].keys():
            node2 = G_for_analysis.nodes[node2]['old_node']
        
        vox_path = nx.shortest_path(G_orig, source = node1, target = node2)
        all_beads = get_beads_from_G_orig_path(G_orig, vox_path, nearby_beads)
        
        all_bead_locs = [points[l] for l in all_beads]
        length = len(vox_path)
        
        seg_len = 5
        if length>seg_len:
            diams_seg = []
            for i,vox1 in enumerate(vox_path[:-(seg_len-1)]):
                segment = vox_path[i:i+seg_len]
                avg_vec = skel_pos[segment[-1]] - skel_pos[segment[0]]                
                d_pts = np.concatenate(all_bead_locs[i:i+seg_len], axis = 0) 
                d_pts = gs.unwrap2(d_pts, len_box)
                
                com = np.mean(d_pts, axis = 0)
                d_pts = d_pts - com
                all_r = []
                for pt in d_pts:
                    all_r.append(p2l(pt, avg_vec))
                if all_r:
                    diams_seg.append(max(all_r)*2 + bead_diam/2)
                if ct < 0:
                    ax = plt.figure().add_subplot(projection = '3d')
                    ax.scatter(d_pts.T[0], d_pts.T[1], d_pts.T[2])
                    ax.plot([0,avg_vec[0]], [0,avg_vec[1]], [0,avg_vec[2]], c = 'k')
                    # print(max(all_r)*2 + 1)
                    ct+=1
            diams.append([np.mean(diams_seg), np.std(diams_seg)])
        else:
            continue
    
    diams = np.array(diams)
    if len(diams) ==0 :
        return [np.nan]
    elif len(diams.shape) == 1:
        diams = diams[np.newaxis, :]
    
    mask = np.any(np.isnan(diams), axis = 1)
    diams = diams[~mask]
    
    ratio = diams[:,1]/diams[:,0]
    mask = np.logical_and(ratio<0.5, ratio>0) #only include fibrils whose diameter is relatively constant
    diams = diams[mask]
    
    return diams
