import numpy as np
from scipy.spatial import KDTree


def make_vox_representation(points, len_box, res = 1):
    res = len_box/(np.round(len_box/res, 0))
    num_vox = int(np.round(len_box/res))
    max_ind= num_vox - 1
    x = np.linspace(res/2, len_box - res/2, num_vox)

    xx, yy, zz = np.meshgrid(x, x, x)
    vox_pos = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    vox_inds = np.round((vox_pos - res/2)/res,0).astype(int)
    tree = KDTree(points, boxsize = len_box)
    nearby_beads = tree.query_ball_point(vox_pos, res)
    
    
    data_beads = []    
    vals = []
    for item in nearby_beads:
        if item:
            vals.append(1)
        else:
            vals.append(0)
    data = np.zeros((num_vox, num_vox, num_vox))
    dat_inds = []
    dat_pos = []
    for i,val in enumerate(vals):
        if val == 1:
            #get index
            inds = vox_inds[i]
            dat_pos.append(vox_pos[i])
            data[inds[0],inds[1],inds[2]] = 1
            dat_inds.append(inds)
            data_beads.append(nearby_beads[i])
    
            
    data_beads2 = get_nearest_vox(dat_pos, points, len_box)
    
    return data, data_beads2, np.array(dat_inds), max_ind, res

def get_nearest_vox(vox_pos, bead_pos, len_box):
    """nearby beads" or "data_beads" in the above function maps all beads within 
    the resolution of a voxel, this mapping makes it one bead to one voxel by taking
    the nearest voxel to each bead"""
    
    tree = KDTree(vox_pos, boxsize = len_box)
    d, nearest_vox = tree.query(bead_pos, k = 1)
    
    nearest_beads = [[] for x in vox_pos]
    for i,x in enumerate(nearest_vox):
        nearest_beads[x].append(i)
    
    return nearest_beads

def ix_to_pos(ix, res):
    return ix * res + res/2

    
