import numpy as np
import matplotlib.pyplot as plt
import os
import json
import shutil
from scipy.spatial.transform import Rotation
from mc_markov_chain_heterogeneity import get_monomer_sequence
#updated to add different BB sizes (types) and different LLL angles/different dihedral angles depending on monomer type
import sys
from scipy.interpolate import interp1d
import warnings

temp = 300
pattern = None
#pattern = ['236']*9 + ['23']*5 + ['2']*16 + ['23']*5
random_heterogeneity = 1

if pattern:
    ds_pattern = '_'.join(pattern)
else: 
    ds_pattern = None

DS = 1.8

def scale(series, low, high):
    ran = high - low
    norm = series * ran
    norm += low
    return norm

def interp(temp, high, low, n, stretch, neg = True, plot = False):
    temps = np.array([298, 348])

    scaled_temp = st = (temp - temps[0])/(temps[1] - temps[0])
    if neg:
        y = (-(1-st/stretch)**(n)+ 1)/(-(1-1/stretch)**(n)+ 1)
    else:
        y = (st/stretch)**(n)
    y = scale(y, low, high)
    x = scale(st, *temps)
    
    ret = y
    if plot:
        x = np.linspace(0,1, 100)
        y = -(1-x)**(n)+ 1
        y = scale(y, low, high)
        x = scale(x, *temps)
        plt.plot(x, y)
    return ret
    #plt.figure()
    #plt.plot(x, y)    

def interpolate_engs(all_engs, exp, stretch, temp):
    temps = [298, 348]
    
    ret = []
    for bounds in all_engs.T:
        low = bounds[0]
        high = bounds[1]
        ret.append(interp(temp, high, low, exp, stretch))
    ret = np.array(ret)
    print(f'At {temp}K', ret, sep = '\n')
    return ret


mono_type = ['',
             '2',
             '3',
             '6',
             '26',
             '36',
             '23',
             '236']

def move_files(save_dir):
    d0 = os.path.join(save_dir, 'equil')
    d2 = os.path.join(save_dir, 'vary_epshp')
    
    for d in [save_dir, d0, d2]:
        if not os.path.exists(d):
            os.makedirs(d)
    
    shutil.copy('data_in.data', d0)
    shutil.copy('params.json', save_dir)
    os.remove('params.json')
    os.remove('data_in.data')
    
    shutil.copy('mc_equil.in', d0)
    shutil.copy('mc_vary_epshp.in', os.path.join(d2, 'mc_vary_epshp.in'))
    shutil.copy('bashsub.sh', save_dir)

    
    subfiles =[]
    shutil.copy('mcequil_sub.qs', d0)
    subfiles.append(os.path.join(d0, 'mcequil_sub.qs'))
    shutil.copy('mchp_sub.qs', d2)
    subfiles.append(os.path.join(d2, 'mchp_sub.qs'))
    for file in subfiles:
        change_rseed(file)
    

def change_rseed(file):
    rseed = int(np.random.rand()*1000000)
    
    with open(file, 'r') as f:
        text = f.read()
    text = text.replace('RSEEDREPLACE', str(rseed))
    with open(file, 'w') as f:
        f.write(text)
    
    
    
def generate_monomers_pattern(pattern, even_divide = False, rand_shift = True):
    """
    pattern is a listlike of '', '2', '3', '6', '23', '26', '36', '236'
    """
    if even_divide: 
        assert DP%len(pattern) == 0
        
    monomer_types = []
    for i in range(noc):
        if rand_shift:
            shift = np.random.choice(range(len(pattern)))
        else:
            shift = 0
        chain_pattern = np.roll(pattern, shift)
        for j in range(DP):
            ind = j%len(pattern)
            monomer_types.append(chain_pattern[ind])
    return monomer_types
            
def generate_monomers_random():
    monomer_types = get_monomer_sequence( noc,  DP, random_heterogeneity, DS, uniform)
    return monomer_types

def plot_finalpos(pos):
    ax = plt.figure().add_subplot(projection = '3d')
    colors = ['green', 'k',  [0, 0, 1],   [0, 0, 0.7], [0, 0, 0.4] ,'orange',  [1, 0, 0],  [0.7, 0, 0], [0.4, 0, 0]  ,'purple']
    for i, p in enumerate(pos):
        ax.scatter(p[0], p[1], p[2], s = 50, color = colors[i])
    bead_order = 'L, BB, HB2, HB3, HB6, SB, MB2, MB3, MB6, next_L'.split(', ')
    ax.legend(bead_order)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    from scipy.spatial.distance import pdist, squareform
    dists = squareform(pdist(pos))
    print(dists[1,2], dists[1,3])
    

def get_types_array(monomer_type):
    types = []
    bbconvdict = {'':1, '2':2, '3':2, '6':2,'23':3, '26':3, '36':3,'236':4}
    types.append(bbconvdict[monomer_type]) #BB
    types = np.array(types)
    return types

def get_res_array():
    res = []
    res_num = 0
    for i in range(noc):
        res_num +=1
        for j in range(DP):
            #res_num +=1
            for k in range(nobpm):
                res.append(res_num)
                # if k == 0: #if linker bead, it is part of the next monomer
                #     res.append(res_num + 1)
                # else: #else part of the current monomer
                #     res.append(res_num)
    res = np.array(res)
    return res

def get_bond_list(no_chains, DP):
    aid1 = 1 #BBind
    aid2 = 1  + nobpm #nextBB ind
    bonds = []
    bondid = 1
    ct = 1
    for i in range(no_chains):
        for j in range(DP-1):
            bonds.append(f'{ct} {bondid} {aid1} {aid2}\n')
            aid1 += nobpm
            aid2 += nobpm
            ct+=1
        aid1 += nobpm
        aid2 += nobpm    
    
    return bonds
            
def get_Langle_list(no_chains, DP, add_inds, angid = 1):
    aid1 = 1
    aid2 = 1 + nobpm
    aid3 = 1 + 2 * nobpm
    angs = []
    last_angs = []
    ct = 1
    for i in range(no_chains):
        for j in range(DP-2):
            if j == DP-2:
                saveaid3 = aid3
                aid3 = add_inds[i]
            angs.append(f'{ct} {angid} {aid1} {aid2} {aid3}\n')
            if j == DP -2:
                aid3 = saveaid3
            aid1 += nobpm
            aid2 += nobpm
            aid3 += nobpm
            ct+=1
        aid1 += nobpm*2
        aid2 += nobpm *2   
        aid3 += nobpm*2
    return angs

def create_staggered_grid(spacing, num_rows, zval = -90):
    grid = []
    for row in range(num_rows):
        for col in range(num_rows):
            x = col * spacing
            y = row * spacing

            grid.append((x, y))
            
            if row == num_rows -1 or col == num_rows -1:
                continue
            grid.append((x + spacing/2, y + spacing/2))
    grid_pts = np.array(grid)
    grid_pts -= np.mean(grid_pts, 0)
    z = np.ones((len(grid_pts), 1)) * zval
    grid_pts = np.concatenate((grid_pts, z), axis =1)
    return grid_pts

def make_origins(orientation = 'random'):
    ##set up several starting points of chains
    #orientation is 'random' or 'fibrilx' where x is the number of chains per side of the fibril
    #ie fibril4 gives the 4343434 orientation of 25 chains
    if orientation == 'random':
        origin = []
        for i in np.linspace(-len_box*0.95,len_box*0.95,32):
            for j in np.linspace(-len_box*0.95,len_box*0.95,32):
                for k in [-len_box*0.95]:
                    origin.append([i,j,k])
        origin = np.array(origin)  
    
    elif 'fibril' in orientation:
        dim = int(orientation.replace('fibril', ''))
        nchains = dim**2 + (dim-1)**2
        assert nchains == noc
        spacing = init_spacing
        origin = create_staggered_grid(spacing, dim)
        
    else:
        raise ValueError("origin type")
    
    np.random.shuffle(origin) 
    return origin

def get_additional_pos():
    oxy = origin[:noc][:,:2]
    oz = np.max(origins[:,2])
    oz += z_spacing
    add = np.tile(oz, (oxy.shape[0], 1))
    add_pos = np.concatenate((oxy, add), axis = 1)
    return add_pos

# def modify_dih_type():
#     nmonos = len(mono_types)
#     ct = 0
#     for i in range(nmonos):
#         if i%DP == DP-1: #maybe zero depending on direction -- need to check this...
#             continue
#         if '2' in mono_types[i]:
#             newdih = dihs[ct]
#             foo = newdih.split()
#             foo[1] = '2'
#             dihs[ct] = ' '.join(foo) + '\n'
#         ct+=1
#     return dihs            
        
        
def modify_Lang_type():
    nmonos = len(mono_types)
    ct = 0
    for i in range(nmonos):
        if i%DP == 0 or i%DP == (DP-1): #0 since no LLL angle for first monomer or last monomer
            continue
        if '3' in mono_types[i]:
            newLang = Langs[ct]
            foo = newLang.split()
            foo[1] = '2'
            Langs[ct] = ' '.join(foo) + '\n'
        ct+=1
    
    return Langs

def check_Langs():
    ids = []
    for line in Langs:
        spl = line.split()
        if spl[1] == '2':
            ids.append(int(spl[3])) #-1 for index +1 for move to BB from L
    types = all_types[ids]
    return types            


def replace_bond_potentials(ds, temp, interp = False):
    params = {(300, 'Cell'): np.array([ 11.77796681, 175.09324611]),
     (300, '222'): np.array([  9.83099201, 174.17845849]),
     (300, '444'): np.array([  8.72836445, 174.37798608]),
     (300, '666'): np.array([  7.61981871, 176.85394023]),
     (300, '888'): np.array([  6.53278164, 179.39194787]),
     (348, 'Cell'): np.array([ 13.55563637, 171.90462009]),
     (348, '222'): np.array([ 13.46455677, 168.29120575]),
     (348, '444'): np.array([ 12.60193723, 169.15223754]),
     (348, '666'): np.array([ 11.15345091, 170.56936732]),
     (348, '888'): np.array([ 14.51414517, 168.26011257])}

    params_m = {(300, '222'): np.array([  7.07966611, 180.        ]),
     (300, '444'): np.array([  6.33039127, 180.        ]),
     (300, '666'): np.array([  6.41602608, 180.        ]),
     (300, '888'): np.array([  5.9106433, 180.       ]),
     (300, 'MGF'): np.array([  6.20698675, 180.        ]),
     (348, '222'): np.array([  7.74244247, 177.24193519]),
     (348, '444'): np.array([  6.62429128, 179.50767752]),
     (348, '666'): np.array([  6.6339333, 180.       ]),
     (348, '888'): np.array([  6.46413619, 180.        ]),
     (348, 'MGF'): np.array([  6.9737058, 180.       ])}
    
    dss = ['cell', '222', '444','666','888','mgf']
    temps = [298, 348]
    if ds_str in dss:    
        ds_ind = dss.index(ds_str)
    else:
        warnings.warn(f'DS value {ds!r} does not have bond params. Interpolating them.')
        interp = True
        dss = [0.0, 0.6, 1.2, 1.8, 2.4, 3.0]
    if temp in temps:
        temp_ind = temps.index(temp)
    else:
        warnings.warn(f'temp value {temp!r} does not have bond params. Interpolating them.')
        dss = [0.0, 0.6, 1.2, 1.8, 2.4, 3.0]
        interp = True
        
    bond_params = {
        'lll_rep' : [[params[(300, x)][1] for x in ['Cell', '222', '444', '666', '888', '888'] ], 
                     [params[(348, x)][1] for x in ['Cell', '222', '444', '666', '888', '888'] ]],
        'lll2_rep' : [[params_m[(300, x)][1] for x in ['222', '222', '444', '666', '888', 'MGF'] ], 
                     [params_m[(348, x)][1] for x in ['222', '222', '444', '666', '888', 'MGF'] ]],
        'lll_rep_k' : [[params[(300, x)][0] for x in ['Cell', '222', '444', '666', '888', '888'] ], 
                     [params[(348, x)][0] for x in ['Cell', '222', '444', '666', '888', '888'] ]],
        'lll2_rep_k' : [[params_m[(300, x)][0] for x in ['222', '222', '444', '666', '888', 'MGF'] ], 
                        [params_m[(348, x)][0] for x in ['222', '222', '444', '666', '888', 'MGF'] ]],
                       }

    params = {}
    for key, val in bond_params.items():
        if interp:
            inter = interp1d(dss, val)
            p_ds = inter(ds)
            inter_temp = interp1d(temps, p_ds)
            p = inter_temp(temp)
        else:
            p = val[temp_ind][ds_ind]
        params[key] = p
        
    return params

    
def file_replace(filename, replace_strings, replace_vals):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    
    finlines = []
        
    for line in lines:
        for string, val in zip(replace_strings, replace_vals, strict = True):
            if string in line:
                #print(string, line)
                line = line.replace(string, val, 1)
                #print(line)
        finlines.append(line + '\n')
        
    with open(filename, 'w') as f:
        f.writelines(finlines)    

    
def get_box_size(wt_fraction, number_of_chains, degree_of_polymerization, degree_of_substitution):
    mwt_monomer = 162 + degree_of_substitution * 14
    chain_total_mass = number_of_chains * degree_of_polymerization * mwt_monomer/6.02e23 # in grams
    vbox = chain_total_mass/wt_fraction #Vbox in cm^3
    box_side = vbox**(1/3)
    box_side *= 1e8 #convert to Angstroms
    return box_side





if __name__ == '__main__':
    
    if len(sys.argv) == 12:
        num_dig = 8
        batch_no = int(sys.argv[1])
        param_id = int(sys.argv[2])
        test_id = int(sys.argv[3])
        cell_eps_300 = 0.05
        cell_eps_348 = 0.05
        eps_11_300 = 0.075
        eps_11_348= 0.075
        eps_22_300 = np.round(float(sys.argv[4]), num_dig) #0.5#0.5 #0.4
        eps_22_348 = np.round(float(sys.argv[5]), num_dig)
        mgf_eps_300 = np.round(float(sys.argv[6]), num_dig)
        mgf_eps_348 = np.round(float(sys.argv[7]), num_dig)
        exp = np.round(float(sys.argv[8]), num_dig)
        frac_300= np.round(float(sys.argv[9]), num_dig)
        frac_348 = np.round(float(sys.argv[10]), num_dig) #0.5#0.5 #0.4
        stretch = np.round(float(sys.argv[11]), num_dig)
        all_engs = np.array([[cell_eps_300, eps_11_300, eps_22_300, mgf_eps_300, frac_300],
                             [cell_eps_348, eps_11_348, eps_22_348, mgf_eps_348, frac_348]])
        save_dir = f'batch_{batch_no}/eng_{param_id}/test_{test_id}'
    else:
        raise ValueError("Not 12 params")
        



    degree_of_polymerization = 100
    number_of_chains = 100

    wt_fraction= 0.02 #0.0114

    if test_id == 1: #heterogeneous DS 1.8 at 328C does not gel
        temp = 318
        DS = 1.8
        random_heterogeneity = 1
        pattern = None
        uniform = False
        ramp = True
    elif test_id == 2: #heterogeneous DS 1.8 at 338C does gel
        temp = 328
        DS = 1.8
        random_heterogeneity = 1
        pattern = None
        uniform = False
        ramp = False    
    elif test_id == 3: #homogeneous DS 1.8 at 348C does not gel
        temp = 348
        DS = 1.8
        random_heterogeneity = 0.05
        pattern = None
        uniform = True
        ramp = True
    elif test_id == 4: #heterogeneous DS 2 at 308C does not gel
        temp = 308
        DS = float(2)
        random_heterogeneity = 1
        pattern = None
        uniform = False
        ramp = True
    elif test_id == 5: #heterogeneous DS 2 at 318C does gel
        temp = 318
        DS = float(2)
        random_heterogeneity = 1
        pattern = None
        uniform = False
        ramp = False
    elif test_id == 6: #heterogeneous DS 2.4 at 298C does not gel
        temp = 298
        DS = 2.4
        random_heterogeneity = 1
        pattern = None
        uniform = False
        ramp = False #no ramp because already at 298
    elif test_id == 7: #heterogeneous DS 2.4 at 308C does gel
        temp = 308
        DS = 2.4
        random_heterogeneity = 1
        pattern = None
        uniform = False
        ramp = False
        
        
    engs = interpolate_engs(all_engs,exp,stretch, temp)
    eps00 = engs[0]#0.1#0.1
    eps11 = engs[1]
    eps22= engs[2]
    eps33 = engs[3]#0.35#0.35
    eps_frac = frac = engs[4]

    eps01 = frac * np.sqrt(eps00 * eps11)
    eps02 = frac**2 * np.sqrt(eps00* eps22)
    eps03 = frac**3 * np.sqrt(eps00* eps33)
    eps12 = frac * np.sqrt(eps11* eps22)
    eps13 = frac**2 * np.sqrt(eps11* eps33)
    eps23 = frac**1 * np.sqrt(eps22* eps33)

    epss = np.array([[eps00, eps01, eps02, eps03 ], 
                     [eps01, eps11, eps12, eps13],
                     [eps02, eps12, eps22, eps23],
                     [eps03, eps13, eps23, eps33]])



    ds_m = DS/3
    sub_ratio = {'6':ds_m,'3':ds_m,'2':ds_m}

    flipped = True
    init_shape = 'random'
    init_spacing = 14 #14 and 10.5
    init_rotation = 0
    len_box = (degree_of_polymerization*1.1) * 5.1 /2

    if ds_pattern:
        ds = np.mean([len(x) for x in pattern])
    else:
        ds = DS
        
        
    ds = np.round(ds, 1)
    degree_of_substitution = ds




    box_size = get_box_size(wt_fraction, number_of_chains, degree_of_polymerization, degree_of_substitution) 
    halfbox = box_size/2
    len_box = max([halfbox, len_box])

    ds_dict = {0:'cell',
               0.6:'222',
               1.2:'444',
               1.8:'666',
               2.4:'888',
               3.0: 'mgf'}

    json_vars = ['degree_of_polymerization',
                 'number_of_chains',
                 'temp',
                 'degree_of_substitution',
                 'init_shape',
                 'init_spacing',
                 'ds_pattern',
                 'sub_ratio',
                 'random_heterogeneity',
                 'wt_fraction',
                 'box_size',
                 'eps00',
                 'eps11',
                 'eps22',
                 'eps33',
                 'eps_frac',
                 'cell_eps_300',
                 'eps_11_300',
                 'eps_22_300', 
                 'mgf_eps_300',
                 'frac_300',
                 'cell_eps_348',
                 'eps_11_348',
                 'eps_22_348',
                 'mgf_eps_348',
                 'frac_348',
                 ]

    json_dict = {var: globals()[var] for var in json_vars if var in globals()}
    json_file = 'params.json'
    with open(json_file, 'w') as json_file:
        json.dump(json_dict, json_file, indent=4)
    #%%
    DP = degree_of_polymerization

    noc = number_of_chains
    if ds in ds_dict:
        ds_str = ds_dict[degree_of_substitution]
    else: ds_str = 'intermediate'

    plot = False

    nobpm = 1 #dont change
    filename = 'data_in.data'

        
        
    lazily_named_dict = {'glc':'',
                         'mg2':'2',
                         'mg3':'3',
                         'mg6':'6',
                         'mg23':'23',
                         'mg26':'26',
                         'mg36':'36',
                         'mgf':'236'}

    z_spacing = 5.1#np.linalg.norm(pos_arr[0] - pos_arr[-1]) #spacing between linker beads
    
    no_mono = noc * DP
    
    if ds_pattern:
        mono_types = generate_monomers_pattern(pattern)
    else: 
        mono_types = generate_monomers_random()
    all_pos = []
    all_types = []
    ct = 0
    R180 = Rotation.from_euler('z', [180], degrees = True)


    for typ in mono_types:
        pos = np.array([[0,0,0]])#get_positions_array(typ, pos_arr)
        #print(typ)
        #plot_finalpos(pos)
        types = get_types_array(typ)
        if ct%2 == 0 and flipped: #flip every other monomer
            pos = R180.apply(pos)
        all_pos.append(pos)
        all_types.append(types)
        ct+=1

    all_pos = np.array(all_pos).reshape(-1, 3)
    all_types = np.array(all_types).reshape(-1)
    all_res = get_res_array()


    noat_t = 4 #number of atom types
    nob_t = 1 #number of bond types
    noan_t = 2 #number of angle types
    nod_t = 2 #number of dihedral types


    box_size = [[-len_box,len_box],[-len_box, len_box],[-len_box, len_box]]
    noat = noc*(DP*nobpm) #number of atoms
    nob = noc*(DP-1) #number of bonds
    noan = noc * ((DP-2))   #number of angles
    nod = noc*(DP-1) #number of dih


    #random initial position for chains, now add zspacing 
    
    origin = make_origins(init_shape)
    origins = []
    for i in range(noc): #loop over chains
        for j in range(DP): #loop over monomers
            for k in range(nobpm): #loop over beads in monomer
                origins.append(origin[i] + j * np.array([0,0,z_spacing]))
    
    origins = np.array(origins)
    
    all_pos = all_pos + origins

    add_inds = [noat + 1 + i for i in range(noc)]
    
    bonds = get_bond_list(noc, DP)
    Langs = get_Langle_list(noc, DP, add_inds)
    #angs = get_other_angle_list(mono_types, noc, DP, add_inds)
    #dihs = get_dih_list(noc, DP)
    
    
    Langs = modify_Lang_type()
    #dihs = modify_dih_type()
    check_Lang_types = check_Langs()
    #check_dih_types = check_dihs()
    #add additional L beads at the end
    #add_pos = get_additional_pos()
    #add_types = np.ones(add_pos.shape[0], dtype = int)
    #add_res = np.array(range(1, noc+1))*DP
    
    #all_pos = np.concatenate((all_pos, add_pos), axis = 0)
    #all_types = np.concatenate((all_types, add_types), axis = 0)
    #all_res = np.concatenate((all_res, add_res), axis = 0)
    
    
    if plot:
        ax = plt.figure().add_subplot(projection = '3d')
        ax.scatter(all_pos[:,0], all_pos[:, 1], all_pos[:,2])

    """WRITE DATA FILE"""
    
    box_str = ''
    for dim,i in (zip(['x','y','z'],range(3))):
        for j in (range(2)):
            box_str+= (str(box_size[i][j]) + ' ')
        box_str += (dim+'lo '+dim+'hi\n')
    
    masses = np.array([162, 176, 190, 204])
    #masses = np.array(masses) * 23.4
    
    #noat += noc
    #noan += 4 * noc
    
    
    with open(filename,'w') as F:
        F.write('Polymer Data\n\n')
                    
        F.write(str(noat)+' atoms\n')
        F.write(str(nob)+' bonds\n')
        F.write(str(noan)+' angles\n')
        # F.write(str(nod)+' dihedrals\n\n')
        
        F.write(str(noat_t)+' atom types\n')
        F.write(str(nob_t)+' bond types\n')
        F.write(str(noan_t)+' angle types\n')
        # F.write(str(nod_t)+' dihedral types\n\n')
        
        F.write(box_str)
        
        F.write('\nMasses\n\n')
        for i, mass in enumerate(masses):
            F.write(f'{i+1} {mass}\n')
        
        F.write('\nAtoms\n\n')
        
        ind = 0
        for p,t, r in zip(all_pos, all_types, all_res):
            ind +=1
            F.write(f'{ind} {r} {t} {p[0]} {p[1]} {p[2]}\n')
            
        F.write('\nBonds\n\n')
        for line in bonds:
            F.write(line)
        
        F.write('\nAngles\n\n')
        for line in Langs:
            F.write(line)
        # for line in angs:
        #     F.write(line)
        
        # F.write('\nDihedrals\n\n')
        # for line in dihs:
        #     F.write(line)
    
    move_files(save_dir)
    
    params = replace_bond_potentials(ds, temp)
    replace_strings = ['TEMP_REPLACE',
                       'L1KREPL',
                       'L2KREPL',
                       'L1REPL',
                       'L2REPL',
                       'eps11_298REPLACE',
                       'eps22_298REPLACE',
                       'eps33_298REPLACE',
                       'eps44_298REPLACE',
                       'frac_298REPLACE',
                       'eps11_348REPLACE',
                       'eps22_348REPLACE',
                       'eps33_348REPLACE',
                       'eps44_348REPLACE',
                       'frac_348REPLACE',
                       'EXP_REPLACE',
                       'STRETCH_REPLACE',
                       'BOXLENREPLACE',
                       'RSEEDREPLACE',
                       'variable DP equal 100',
                       ]
    replace_vals = [temp,
                    params['lll_rep_k'],
                    params['lll2_rep_k'],
                    params['lll_rep'],
                    params['lll2_rep'],
                    cell_eps_300,
                    eps_11_300,
                    eps_22_300,
                    mgf_eps_300,
                    frac_300,
                    cell_eps_348,
                    eps_11_348,
                    eps_22_348,
                    mgf_eps_348,
                    frac_348,
                    exp,
                    stretch,
                    halfbox,
                    int(np.random.rand() * 10000000),
                    f'variable DP equal {DP}',
                    ]
    
    replace_vals = [str(f) for f in replace_vals]
    if ramp:
        replace_strings.append('TEMPRAMP_REPLACE')
        replace_vals.append('1')
        #replace_strings.append('variable annealtime equal 5000000')
        #replace_vals.append('variable annealtime equal 15000000')
    else:
        replace_strings.append('TEMPRAMP_REPLACE')
        replace_vals.append('0')
        
        #replace_str1 = ["#SBATCH --time=0-03:00:00   #Do this "]
        #replace_vals1 = ["#SBATCH --time=0-09:00:00   #Do this "]
        #file_replace(os.path.join(save_dir, 'vary_epshp', 'mchp_sub.qs'), replace_str1, replace_vals1)

        
    equil_lmp = os.path.join(save_dir, 'equil', 'mc_equil.in')
    prod_lmp = os.path.join(save_dir, 'vary_epshp', 'mc_vary_epshp.in')
    for lmp_infile in [equil_lmp, prod_lmp]:
        file_replace(lmp_infile, replace_strings, replace_vals)
        
