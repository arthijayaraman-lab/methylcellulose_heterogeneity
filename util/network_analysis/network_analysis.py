import networkx as nx
import matplotlib.pyplot as plt
import sys
import numpy as np

import graph_setup as gs
import metrics
import vox_setup as vs
from persistence_length import fit_pl_fibril
import plotting_util as pu

class network:
    def __init__(self, pts, len_box, types, mols, bead_diam = 1, bead_cutoff = None, vox_res = None):
        self.pts = pts
        self.len_box = len_box
        self.types = types
        self.mols = mols
        self.bead_diam = bead_diam
        
        if not vox_res:
            self.vox_res = bead_diam
        else:
            self.vox_res = vox_res
        
        if not bead_cutoff:
            self.bead_cutoff = bead_diam * 1.25
        else:
            self.bead_cutoff = bead_cutoff
        
        self.shift_points()
        
    def shift_points(self):
        minim = np.min(self.pts)
        maxim = np.max(self.pts)
        
        if minim < 0:
            shift = -minim
            self.pts += shift
            self.shift = shift
            print(f'shifting points by +{shift}')
        elif maxim > self.len_box:
            shift = self.len_box - maxim
            self.pts += shift
            self.shift = shift
            print(f'shifting points by {shift}')

        else:
            self.shift = 0
        
        
    def pore_size(self):
        ...
        
    def lacunarity(self):
        vox = vs.make_vox_representation(self.pts, self.len_box, res = self.bead_diam)
        box_sizes, lacs = metrics.lac_calc(vox)
        return box_sizes, lacs
        
    def strand_length(self):
        print('Calculating Strand Lengths')

        if not self.fibrils_formed:
            return np.array([np.nan])
        
        try: a = self.strands
        except: self.identify_strands()
        
        lens = []
        n = len(self.strands)
        for i,s in enumerate(self.strands,1):
            print(f'strand {i}/{n} has {s.num_beads} beads')
            if s.num_beads < 100: continue
            slen = s.get_len()
            lens.append(slen)
        self.strand_lengths = lens
        return lens
    
    def identify_strands(self, unwrap = True):
        print('Identifying Strands')
        try:
            a = self.vox_graph
        except:
            self.make_vox_graph()
        try:
            a = self.bead_graph
        except:
            self.make_bead_graph()
        
        self.strands = metrics.get_strands(self.vox_graph,
                                           self.bead_graph,
                                           self.pts,
                                           self.len_box,
                                           self.nearby_beads,
                                           self.vox_res_upd,
                                           unwrap)
        
    def cross_link_density(self):
        print('Calculating Crosslink Density')
        if not self.fibrils_formed:
            return np.array([np.nan])
        
        try:
            a = self.vox_graph
        except:
            self.make_vox_graph()
            
        degrees = nx.degree(self.vox_graph)
        degrees = [x[1] for x in list(degrees) if x[1] > 2]
        num_xlinks = len(degrees)
        xlink_density = num_xlinks/self.len_box**3
        self.xlink_density = xlink_density
        return xlink_density
        
        
    def strand_persistence_length(self, plot = False):
        print('Calculating Strand Persistence Length')
        if not self.fibrils_formed:
            return np.array([np.nan])
        try: a = self.strands
        except: self.identify_strands()
        
        
        
        x = []
        y = []
        n = len(self.strands)
        for i,s in enumerate(self.strands,1):
            print(f'strand {i}/{n} has {s.num_beads} beads')
            if s.num_beads < 100: continue
            if len(s.path) < 2: continue
            l = s.get_len()    
            if l< 250: continue
            pl, bls, auto = s.get_pl()
            x.extend(bls)
            y.extend(auto)
        x = np.array(x)
        y = np.array(y)

        plen, _, _, params = fit_pl_fibril(x, y, plot = plot)
        return plen
        
        
    def strand_diameter(self):
        if not self.fibrils_formed:
            return np.array([np.nan])
        try:
            a = self.vox_graph
        except:
            self.make_vox_graph()
            
        diams = metrics.diameters(self.vox_graph,
                                  self.G_orig,
                                  self.vox_res,
                                  self.pts,
                                  self.nearby_beads,
                                  self.skeleton,
                                  self.bead_diam,
                                  self.len_box)
        self.strand_diams = diams
        return diams
        
    def is_percolated(self):
        """Not implemented yet"""
        graph = gs.make_bead_graph(self.pts, self.len_box, cutoff = self.bead_cutoff, pbc = False)
            
        px, py, pz = metrics.check_percolated(graph)
        return px, py, pz
        
    def make_bead_graph(self):
        G_beads = gs.make_bead_graph(self.pts, self.len_box, cutoff = self.bead_cutoff)
        self.bead_graph = G_beads
        
    def make_vox_rep(self):
        vox_rep, nearby_beads, vox_inds, max_vix, res_out = vs.make_vox_representation(self.pts, 
                                                                              self.len_box, 
                                                                              self.vox_res)
        self.vox_representation = vox_rep
        self.nearby_beads = nearby_beads
        self.vox_inds = vox_inds
        self.max_vix = max_vix
        self.vox_res_upd = res_out
    
    def make_vox_graph(self, exclude_boundaries = False, main_component = False):
        self.make_vox_rep()
        
        skel = gs.skeletonize_voxel_representation(self.vox_representation)
        self.skeleton = skel
        G_orig = gs.make_original_graph(skel)        
        skel_to_data_dict = gs.make_skel_to_data_dict(skel, self.vox_inds)
        G_orig = gs.assign_original_voxels_to_skel_graph(G_orig, skel_to_data_dict)
        
        G_orig = gs.check_skeleton_edges(G_orig, self.vox_inds, self.vox_representation)
        
        G_comp = gs.compress_edges(G_orig)

        G = gs.remove_cluster_artifacts(G_comp, G_orig, self.max_vix)
        

        if exclude_boundaries:
            G_for_analysis = gs.make_graph_excluding_boundary_nodes(G, self.max_vix)
        else:
            G_for_analysis = G.copy()
            
        if main_component:
            G_for_analysis = gs.make_main_component_graph(G_for_analysis)
        
        
        self.vox_graph = G_for_analysis
        self.G_orig = G_orig
        self.G_comp = G_comp
        self.G_noclust = G
        self.check_fibrils_formed()

    def check_fibrils_formed(self):
        self.make_bead_graph()
        comp = nx.connected_components(self.bead_graph)
        comp_sizes = [len(x) for x in comp]
        max_size = max(comp_sizes)
        self.fibrils_formed = max_size > 100
    
    def plot_graph(self, dpi = 100, vox = True, skel = True, graph = True):
        vi = self.vox_inds
        
        skel_inds  =  np.array(np.where(self.skeleton)).T
        nodes = list(self.vox_graph.nodes())
        if graph:
            colors = ['r' if i in nodes else 'k' for i in range(len(skel_inds)) ]
            sizes = [5 if i in nodes else 1 for i in range(len(skel_inds)) ]
        else:
            colors = ['k' for s in skel_inds]
            sizes = [1 for s in skel_inds]
        
        
        ax = plt.figure(dpi = dpi).add_subplot(projection = '3d')
        if vox:
            ax.scatter(*vi.T, s= 0.01, color = 'b')
        if skel:
            ax.scatter(*skel_inds.T, s = sizes, c = colors)
        num_vox = len(self.vox_representation)
        
        if graph:
            for e in list(self.vox_graph.edges):
                node1 = e[0]
                node2 = e[1]
                if 'old_node' in self.vox_graph.nodes[node1].keys():
                    node1 = self.vox_graph.nodes[node1]['old_node']
                    pos = skel_inds[node1]
                    #ax.scatter(pos[0], pos[1], pos[2], c = 'r', s = 5)
                if 'old_node' in self.vox_graph.nodes[node2].keys():
                    node2 = self.vox_graph.nodes[node2]['old_node']
                    pos = skel_inds[node2]
                    #ax.scatter(pos[0], pos[1], pos[2], c = 'r', s = 5)
                
    
                p1 = skel_inds[node1]
                p2 = skel_inds[node2]
                ax.scatter(p1[0], p1[1], p1[2], c = 'r', s = 5)
                ax.scatter(p2[0], p2[1], p2[2], c = 'r', s = 5)

                edge_ix = self.vox_graph.edges[e]['dat_voxels']
                edge_pts = self.vox_inds[edge_ix]
                p1b, p2b, dist = pu.scan_edge_copies(edge_pts, p1, p2, num_vox)
                if np.any(p1b):
                    p1 = p1b
                    p2 = p2b
                l = np.vstack((p1, p2))
                ax.scatter(*l.T, c = 'r', s = 5)
                ax.plot(l.T[0], l.T[1], l.T[2], color = 'lightcoral')
            
        ax.set_xlim([0,num_vox])
        ax.set_ylim([0,num_vox])
        ax.set_zlim([0,num_vox])
        ax.set_axis_off()
        L = num_vox
        edges = [
            ([L, L], [0, 0], [L, 0]),
            ([0, 0], [L, L], [L, 0]),
            ([L, L], [L, L], [L, 0]),
            ([0, 0], [0, 0], [L, 0]),
            ([0, 0], [0,L], [0, 0]),
            ([L, L], [0,L], [0, 0]),
            ([0, 0], [0,L], [L, L]),
            ([L, L], [0,L], [L, L]),
            ([0, L], [0,0], [0, 0]),
            ([0, L], [0,0], [L, L]),
            ([0, L], [L,L], [0, 0]),
            ([0, L], [L,L], [L, L]),
        ]
        
        # Plot the edges
        for edge in edges:
            ax.plot(edge[0], edge[1], edge[2], color='black', alpha = 0.2)
        
        ax.set_aspect('equal')
        return ax
        
    def write_vmd_vis(self, color_scheme):
        ...
        #filename = sim_id
        #write_tcl_script_for_vmd_plotting_by_diam(G_final, G_orig, res, points, nearby_beads, skel, filename)
        #write_tcl_script_for_vmd_plotting_by_length(G_final, G_beads, G_orig, points, skel, res, nearby_beads, dat_inds, filename)
        
        
    
if __name__ == '__main__':
    x = network(1, 1, 1, 1)
    
