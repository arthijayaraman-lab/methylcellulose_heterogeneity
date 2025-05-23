import sys
import os
import numpy as np
from scipy.spatial import KDTree
from json import load

def get_score(dirname):
    
    filename = os.path.join(dirname, 'vary_epshp', 'post_prod.lammpstrj')
    frac = get_fib_frac(filename)
        
    fibs = frac > 0.1
    
    np.savetxt(os.path.join(dirname, 'score_arr.txt'), [fibs])
    return fibs
        
def test_score():
    desired = [1, 1, 0, 0, 1, 0, 1]
    obtained = [0, 0, 0, 0, 1, 0, 1]
    score = np.sum(~(np.array(desired).astype(bool) ^ np.array(obtained).astype(bool)))
    return score

def read_dump(file, ret_dict = False):
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
    
    atoms = int(lines[3].split()[0])
    ts = 10000 #if you every have a trajectory with more than this many timesteps, this will need to be changed
    xpos = np.zeros([ts, atoms])
    ypos = np.zeros([ts, atoms])
    zpos = np.zeros([ts, atoms])
    
    timesteps = []
    len_box = []
    atype= np.zeros([atoms], dtype = 'int')
    molecnum= np.zeros([atoms], dtype = 'int')

    natoms = 0
    
    count = 0
    for l_num, line in enumerate(lines):
        line = line.split()
        if (len(line) ==6) and (line[0] != "ITEM:"):  
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
            
        elif (len(line) ==2) and (line[1] == "TIMESTEP"):
            timesteps.append(lines[l_num+1])
            if (len(timesteps) == 2):
                natoms = count
        elif (len(line) ==6) and (line[1] =="BOX"):
            a = lines[l_num+1].split()[1]
            len_box.append(float(a))
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

def get_pos(dump, ts_ind, types_incl):
    xpos = dump['xpos']
    ypos = dump['ypos']
    zpos = dump['zpos']
    types = dump['types']
    resid = dump['molecs']
    #comment the following line if sorted by id
    #types = types[ts_ind]
    
    mask = [t in types_incl for t in types]
    
    
    xpos = xpos[ts_ind, mask]
    ypos = ypos[ts_ind, mask]
    zpos = zpos[ts_ind, mask]
    
    pos = np.vstack((xpos, ypos, zpos)).T
    types = types[mask]
    resid = resid[mask]
    return pos, types, resid

def get_fibril_points(pos, len_box, distance_cutoff = 8, num_cutoff = 5):
    pos += len_box/2
    pos[pos > len_box] = len_box - 0.000001
    pos[pos<0]= 0.00001
    tree = KDTree(pos, boxsize = len_box)
    neigh = tree.query_ball_point(pos, r = distance_cutoff)
    num_neigh = np.array([len(n) for n in neigh])
    mask = num_neigh > num_cutoff
    pos -= len_box/2
    fib_pos = pos[mask]
    non_fib_pos = pos[~mask]
    return fib_pos, non_fib_pos

def get_fib_frac(file):
    dump = read_dump(file, ret_dict = True)

    for tsi in [-1]: #tsi, ts in enumerate(dump['timesteps'][-1:]):
        typ = [1, 2, 3, 4]
        pos, types, molecs = get_pos(dump,tsi, typ)
        len_box = 2*dump['box_lens'][tsi]
        fib_pos, non_fib_pos = get_fibril_points(pos, len_box)
        frac_fib = len(fib_pos)/(len(fib_pos) + len(non_fib_pos))
    return frac_fib

def check_complete(dirname):
    bools = []
    file = os.path.join(dirname, 'vary_epshp', 'post_prod.lammpstrj')
    if os.path.isfile(file):
        bools.append(True)
    else:
        bools.append(False)
    return np.all(bools)
    
    

def get_result_points(batch_no):
    batch_dir = f'batch_{batch_no}'
    #batch_dir = 'global_equil_data'
    dirs = []
    for item in os.listdir(batch_dir):
        if item.startswith('eng'):
            check = check_complete(os.path.join(batch_dir, item))
            if check:
                dirs.append(item)
    
    results = []
    for d in dirs:
        file = os.path.join(batch_dir, d, 'params.json')
        with open(file, 'r') as f:
            all_params = load(f)
        score = get_score(os.path.join(batch_dir, d))
    return results
    
    
if __name__ =='__main__':
    print('ANALYZING BATCH')
    batch_no = int(sys.argv[1]) #batch id of batch that you want to analyze
    bo = get_result_points(batch_no)
    
    
        
