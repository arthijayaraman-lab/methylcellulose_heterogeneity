import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap

import json

import sys
from bis import BIS

bigE = 10 #how many energy sets per batch

def pts_to_arr(dicts):
    pts = []
    for d in dicts:
        add = []
        add.append(d['target'])
        p = d['params']
        add.extend([x for x in p.values()])
        pts.append(add)
    return np.array(pts)

def get_batches(rundir):
    batches = []
    for thing in os.listdir(rundir):
        if os.path.isdir(os.path.join(rundir, thing)) and thing.startswith('batch_'):
            check_files = [os.path.join(rundir, thing, f'eng_{x}', 'score_arr.txt') for x in range(bigE)]
            check = [os.path.isfile(file) for file in check_files]
            if np.any(check):
                batches.append(int(thing.split('_')[-1]))
    batches.sort()
    return batches

def get_bound_params(rundir):
    file = os.path.join(rundir, 'bound_params.json')
    with open(file, 'r') as f:
        a = json.load(f)
    return a

param_ids = {'temp': 0,
             'ds': 1, 
             'H': 2,
             'dp':3,
             'nchains':4,
             'conc':5}

def sample_boundaries(rundir, N, ix = -1):
    batches = get_batches(rundir) 
    bound_params = get_bound_params(rundir)
    cwd = os.getcwd()

    bounds = []
    for i in range(N):
        print(i)
        ix = np.random.choice([-2], 1)[0]
        batch = batches[ix]

        print(ix)
        os.chdir(rundir)
        bis = BIS(batch, bound_params)
        
        os.chdir(cwd)
        b =bis.get_boundary(plot_bound_w_pts= False, boot = True)
        probs = bis.get_probs()
        bounds.append(probs)
    return np.array(bounds)
        

def plot_multiple_boundaries(rundirs, colors, ids= None, labs = [], title = '', loc = 'upper right'):
    if len(labs)==0:
        labs = rundirs
    if not np.any(ids):
        ids = [-1 for i in rundirs]
    cwd = os.getcwd()
    f, ax = plt.subplots(figsize = (4, 4), dpi = 400)
    biss = []
    for rundir, c, ix, lab in zip(rundirs, colors, ids, labs, strict = True):
        batches = get_batches(rundir) 
        bound_params = get_bound_params(rundir)
        
        batch = batches[ix]
        print(batch)
        os.chdir(rundir)
        bis = BIS(batch, bound_params)
        os.chdir(cwd)
        b =bis.get_boundary()
        bis.plot_bound_w_err(Nboot = 1, ax = ax, lev = 0.1, color = c)
        biss.append(bis)
    patches= []
    for color, lab in zip(colors, labs):
        patches.append(mlines.Line2D([], [], color=color, label=lab, linewidth=5))
    plt.legend(handles = patches, fontsize = 12, title = title, loc = loc)
    
    plt.tight_layout()
    
    return biss

def get_all_biss(rundir):
    batches = get_batches(rundir)  
    cwd = os.getcwd()
    bound_params = get_bound_params(rundir)

    bs = []
    for batch in batches:
        os.chdir(rundir)
        bis = BIS(batch, bound_params)
        os.chdir(cwd)
        bs.append(bis)
    return bs

def check_convergence(rundir):
    biss = get_all_biss(rundir)
    bounds = []
    for bis in biss[:-1]:
        bounds.append(bis.get_boundary())
    
    bools = []
    for i in range(len(bounds)-1):
        print(i)
        b1 = bounds[i]
        b2 = bounds[i+1]
        bools.append(bis.check_done(b1, b2))
    return bools
    
    
if __name__ == '__main__':
    
    try:
        os.makedirs('plots')
    except:
        pass
    
    color_map= ['#000000', '#004488', '#BB5566', '#228833', '#DDAA33']
    
    color1_s = [91/255,110/255,193/255,1]
    color2_s = [255/255, 97/255, 97/255, 1]

    color1_c = [108/255,183/255,206/255,1]
    color2_c = [240/255, 148/255, 86/255, 1]
    
    cmap_s = LinearSegmentedColormap.from_list("custom_cmap", [color1_s, color2_s])
    cmap_c = LinearSegmentedColormap.from_list("custom_cmap", [color1_c, color2_c])

    color_map_s = cmap_s(np.linspace(0,1, 4))
    color_map_c = cmap_c(np.linspace(0,1, 5))

#%% Vary HS
    xs = [1, 3, 5, 7]
    rundirs = [f'sims/temp_ds_Hs{x}' for x in xs]
    ids = [-1 for x in rundirs]
    ids[-2] = -2
    colors = color_map_s[:len(xs)]
    labs = [f'{x}' for x in xs]
    title = r'$H_S$'
    bis = plot_multiple_boundaries(rundirs, colors, ids, labs, title, loc = 'center left')
    plt.ylabel(r'Temperature, K')
    plt.xlabel('DS')
    
    plt.savefig('plots/temp_ds_Hs.png')
    plt.close()
#%% Vary Hc
    xs = [0, 0.25, 0.5, 0.75, 1]
    rundirs = [f'sims/temp_ds_Hc{x}' for x in xs]
    colors = color_map_c[:len(xs)]

    ids = [-1 for x in rundirs]
    labs = [f'{x}' for x in xs]
    title = r'$H_C$'
    bis = plot_multiple_boundaries(rundirs, colors, ids, labs, title, loc = 'center left')
    plt.ylabel(r'Temperature, K')
    plt.xlabel('DS')
    plt.savefig('plots/temp_ds_Hc.png')
    plt.close()
    
#%%
    xs = [1.8, 2.0, 2.4]
    rundirs = [f'sims/temp_Hc_ds{x}' for x in xs]
    colors = color_map[:len(xs)]
    ids = [-1 for x in rundirs]
    labs = [f'{x}' for x in xs]
    title = 'DS'
    biss = plot_multiple_boundaries(rundirs, colors, ids, labs, title, loc = 'center right')
    plt.xlabel(r'$H_C$')
    plt.ylabel('Temperature, K')
    plt.savefig('plots/temp_Hc_ds.png')
    plt.close()
#%%

    xs = [1.8, 2.0, 2.4]
    rundirs = [f'sims/temp_Hs_ds{x}' for x in xs]
    colors = color_map[:len(xs)]
    ids = [-1 for x in rundirs]
    labs = [f'{x}' for x in xs]
    title = 'DS'
    bis = plot_multiple_boundaries(rundirs, colors, ids, labs, title, loc = 'center right')
    plt.xlabel(r'$H_S$')
    plt.ylabel('Temperature, K')

    
    plt.savefig('plots/temp_Hs_ds.png')
    plt.close()
