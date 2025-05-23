import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import warnings

analysis_path = '../util/network_analysis'
sys.path.append(analysis_path)
from network_analysis import network

#%%
def scan_timesteps(file):
    timesteps = []
    
    with open(file, 'r') as f:
        text = f.read()
    lines = text.splitlines()
    
    
    for l_num, line in enumerate(lines):
        line = line.split()
        if (len(line) ==2) and (line[1] == "TIMESTEP"):
            timesteps.append(lines[l_num+1])
    return timesteps

def read_dump(file, ret_dict = False, tsi = None):
    """
    

    Parameters
    ----------
    file : TYPE
        DESCRIPTION.

    Returns
    -------
    (len_box, timesteps, xpos, ypos, zpos, atype, molecnum, atoms)

    """
    
        
    with open(file, 'r') as f:
        text = f.read()
    lines = text.splitlines()
    
    ts_scan = scan_timesteps(file)
    if np.any(tsi):
        ts_incl = np.array(ts_scan)[np.array(tsi)]
    else:
        ts_incl = np.array(ts_scan)   
    save_timestep = False
    
    atoms = int(lines[3].split()[0])
    ts = 10000 #if you every have a trajectory with more than this many timesteps, this will need to be changed
    xpos = np.zeros([ts, atoms])
    ypos = np.zeros([ts, atoms])
    zpos = np.zeros([ts, atoms])
    
    timesteps = []
    len_box = []
    atype= np.zeros([atoms], dtype = 'int')
    molecnum= np.zeros([atoms], dtype = 'int')
    
    count = 0
    
    
    for l_num, line in enumerate(lines):
        line = line.split()
        if (len(line) ==2) and (line[1] == "TIMESTEP"):
            ts = lines[l_num+1]
            
            if ts in ts_incl:
                save_timestep = True
            else:
                save_timestep = False
                
            if save_timestep:
                timesteps.append(ts)
            
        elif (len(line) ==6) and (line[1] =="BOX"):
            a = lines[l_num+1].split()[1]
            if save_timestep:
                len_box.append(float(a))
                
        elif (len(line) ==6) and (line[0] != "ITEM:"):  
            if save_timestep:
                x = float(line[3])
                y = float(line[4])
                z = float(line[5])            
                if count<atoms:
                    atype[count]= int(line[2])
                    molecnum[count] = int(line[1])
                
                xpos[int(count/atoms), (count - atoms*int(count/atoms))]=x
                ypos[int(count/atoms), (count - atoms*int(count/atoms))]=y
                zpos[int(count/atoms), (count - atoms*int(count/atoms))]=z
                count += 1
            
        else:
            continue
    
    xcnt = 0
    ycnt = 0
    zcnt = 0
    if np.all(xpos[0,:]==0):
        xcnt =1
    if np.all(ypos[0,:]==0):
        ycnt = 1
    if np.all(zpos[0,:]==0):
        zcnt = 1
        
    xpos = xpos[~np.all(xpos == 0, axis=1)]
    ypos = ypos[~np.all(ypos == 0, axis=1)]
    zpos = zpos[~np.all(zpos == 0, axis=1)]
    timesteps = [int(ts) for ts in timesteps]
    if xcnt ==1:
        xpos = np.vstack((np.zeros(atoms), xpos))
    if ycnt ==1:
        ypos = np.vstack((np.zeros(atoms), ypos))
    if zcnt ==1:
        zpos = np.vstack((np.zeros(atoms), zpos))
        
    if ret_dict:
        ret = {'box_lens': len_box,
               'timesteps': timesteps,
               'xpos': xpos,
               'ypos':ypos,
               'zpos':zpos,
               'types':atype,
               'molecs':molecnum,
               'n_atoms':atoms
            }
        return ret
        
        
        
    return (len_box, timesteps, xpos, ypos, zpos, atype, molecnum, atoms)



def reduce_fibrils(pts, mols, types, len_box):
    global ix_g, ix_l
    #wrap points to deal with some numerical stragglers
    tree_pts = pts + len_box/2

    ix_g = np.where(tree_pts > len_box)
    ix_l = np.where(tree_pts < 0)
    
    if len(ix_g[0]) > 0:
        warnings.warn(f"points {tree_pts[ix_g]} greater than {len_box}. Shifting them")
        tree_pts[ix_g] = len_box - 0.0005

    if len(ix_l[0]) > 0:
        warnings.warn(f"points {tree_pts[ix_l]} less than 0. Shifting them")
        tree_pts[ix_l] = 0 + 0.0005

    tree= KDTree(tree_pts, boxsize = len_box)
    neigh = tree.query_ball_point(tree_pts, 8 )
    num_neigh = np.array([len(n) for n in neigh])
    mask = num_neigh > 5
    pts = pts[mask]
    mols = mols[mask]
    types = types[mask]
    return pts, mols, types

def make_example_analysis_plot():
    file = '/home/skronen/Documents/tests_that_went_nowhere/mc_beadspring/misc_systems/large_systems_final/sims/1000_100mers/trial_1/ds2.0_Hs1.0/vary_epshp/prod.lammpstrj'
    dump = read_dump(file, ret_dict=True, tsi = [-1])
    x = dump['xpos'][0]
    y = dump['ypos'][0]
    z = dump['zpos'][0]
    
    points = np.vstack((x, y, z)).T
    len_box = dump['box_lens'][0]*2
    mols = dump['molecs']
    types = dump['types']
    
    
    points, mols, types = reduce_fibrils(points, mols, types, len_box)
    
    a = network(points, len_box, types, mols, bead_diam = 6.2, vox_res=20)
    a.make_vox_graph()
    ax = a.plot_graph(dpi = 400)
    ax.view_init(elev = 10, azim=230)

    plt.savefig('/home/skronen/mc_paper_figs/network_analysis_example.png')
    plt.close()

def plot_all_strands(a):
    a.identify_strands(unwrap = False)
    ax = plt.figure().add_subplot(projection = '3d')
    strands = a.strands
    n = len(strands)
    colors = plt.cm.jet(np.linspace(0,1, n))
    np.random.shuffle(colors)
    
    for i,s in enumerate(strands):
        if s.num_beads < 100: continue
        s.plot_strand(ax = ax, color = colors[i])
    ax.set_xlim([0,a.len_box])
    ax.set_ylim([0,a.len_box])
    ax.set_zlim([0,a.len_box])
    ax.set_aspect('equal')

def get_info(file, ts):
    global points, dump, len_box
    dump = read_dump(file, ret_dict=True, tsi = [-1])
    x = dump['xpos'][0]
    y = dump['ypos'][0]
    z = dump['zpos'][0]
    
    points = np.vstack((x, y, z)).T
    len_box = dump['box_lens'][0]*2
    mols = dump['molecs']
    types = dump['types']
    
    points, mols, types = reduce_fibrils(points, mols, types, len_box)
    return points, len_box, mols, types, 

def do_analysis(file, ts):
    global ntk
    savdir = os.path.dirname(file)

    points, len_box, mols, types = get_info(file, ts)
    ntk = network(points, len_box, types, mols, bead_diam = 6.2, vox_res = 20)
    ntk.make_vox_graph()

    ntk.plot_graph()
    plt.savefig(os.path.join(savdir, 'network.png'))
    plt.close()    
    
    fib_diams = ntk.strand_diameter()
    np.savetxt(os.path.join(savdir, 'diameter.txt'), fib_diams)
    
    fib_lens = ntk.strand_length()
    np.savetxt(os.path.join(savdir, 'lengths.txt'), fib_lens)

    xlink_dens = ntk.cross_link_density()
    np.savetxt(os.path.join(savdir, 'xlink.txt'), [xlink_dens])

    fib_pl = ntk.strand_persistence_length(plot = True)
    plt.savefig(os.path.join(savdir, 'fibril_pl_plot.png'))
    plt.close()
    np.savetxt(os.path.join(savdir, 'persistence.txt'), [fib_pl])

    
    
if __name__ == '__main__':
    #pl = example()

    hd = 'sims/1000_100mers/'
    
    trials = [1, 2, 3]
    dss = [1.8, 2.0, 2.4]
    Hcs = [0.0, 0.5, 1.0]
    Hss = [1.0, 4.0, 7.0]
    
    sds = []
    
    for ds in dss:
        for hc in Hcs:
            d = f'ds{ds:0.1f}_Hc{hc:0.1f}'
            sds.append(d)
        for hs in Hss:
            d = f'ds{ds:0.1f}_Hs{hs:0.1f}'
            sds.append(d)
    
    for trial in trials[:]:
        trial_dir = os.path.join(hd, f'trial_{trial}')
        files = [os.path.join(trial_dir, x, 'vary_epshp', 'prod.lammpstrj') for x in sds]
        for file in files[:]:
            print(file)
            ts = -1
            do_analysis(file, ts)


            savdir = os.path.dirname(file)
            
            points, len_box, mols, types = get_info(file, ts)
            ntk = network(points, len_box, types, mols, bead_diam = 6.2, vox_res = 20)
    
    
