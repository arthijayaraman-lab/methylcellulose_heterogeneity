import numpy as np
import os
import matplotlib.pyplot as plt
import json
import sys
from post_batch import read_dump
import warnings
from scipy.interpolate import interp1d

if 'bayopt_util' in sys.modules:
    del sys.modules['bayopt_util']

from bayopt_util import normalize_points, unnormalize_points, sorted_params

warnings.filterwarnings("ignore", message = "invalid value encountered in cast")

bigE = 10 #how many energy sets per batch
nparam = 8


def pts_to_arr(dicts):
    pts = []
    for d in dicts:
        add = []
        add.append(d['target'])
        p = d['params']
        add.extend([x for x in p.values()])
        pts.append(add)
    return np.array(pts)

def get_scores(batch, eng, test = False):
    if test:
        scores = []
        for t in range(1, 8):
            fname = os.path.join(rundir, f'batch_{batch}', f'eng_{eng}', f'test_{t}', 'score_arr.txt')
            if os.path.isfile(fname):
                scores.append(np.loadtxt(fname))
            else: scores.append(np.nan)
    else:
        fname = os.path.join(rundir, f'batch_{batch}', f'eng_{eng}', 'score_arr.txt')
        if os.path.isfile(fname):
            scores = np.loadtxt(fname)
        else: scores = np.array([np.nan]*7)
    return scores

def get_param(batch, eng):
    fname = os.path.join(rundir, f'batch_{batch}', f'eng_{eng}', 'params.json')
    with open(fname, 'r') as f:
        data = json.load(f)
    params = np.array([float(x) for x in data.values()])
    params = normalize_points(params)
    return params

def get_all_scores(batches, engnum = bigE):
    scores = []
    for batch in batches:
        bs = []
        for eng in range(engnum):
            try:
                s = get_scores(batch, eng)
                bs.append(s)
            except:
                bs.append([np.nan]*7)
        scores.append(bs)
    return np.array(scores)

def get_all_params(batches, engnum = bigE):
    params = []
    for batch in batches:
        bs = []
        for eng in range(engnum):
            try:
                p = get_param(batch, eng)
                bs.append(p)
            except:
                bs.append([np.nan]*nparam)
        params.append(bs)
    return np.array(params)

def plot_pts(by = 'batch', sub = list(range(nparam)), p = None, hist = False):
    if not p:
        if by == 'batch':
            plot_pts = all_pts
            labs = batches
        elif by == 'score':
            plot_pts = uniq_params
            labs = uniq
    else:
        if len(p) == 2:
            plot_pts = p
            labs = [0, 1]
        else:
            plot_pts = p
            labs = range(len(p))
    
    f, ax = plt.subplots(len(sub), len(sub), figsize = (10,10))
    for k,pt in enumerate(plot_pts):
        lab = labs[k]
        for i, x in enumerate(sub):
            for j, y in enumerate(sub):
                
                if j < i: continue
                if hist:
                    ax[i,j].hist2d(pt[:,y], pt[:,x], density = True, bins = np.linspace(0,1, 20))
                else:
                    ax[i,j].scatter(pt[:,y], pt[:,x], s = 1, label = lab)
    
    for i, x in enumerate(sub):
        for j, y in enumerate(sub):
            ax[i,j].set_xlim([0,1])
            ax[i,j].set_ylim([0,1])
            
            if i == len(sub)-1:
                ax[i,j].set_xlabel(sorted_params[y])
            if j == 0:
                ax[i,j].set_ylabel(sorted_params[x])
    if not hist:
        plt.legend()


def get_ds(file):
    dump = read_dump(file, ret_dict = True)
    types = dump['types']
    ds = np.mean(types -1)
    return ds
    
def check_ds():
    dss = []
    for batch in batches:
        dsb = []
        for eng in range(10):
            dse = []
            for test in range(1,8):
                file = os.path.join(rundir, f'batch_{batch}', f'eng_{eng}', f'test_{test}', 'vary_epshp/prod.lammpstrj')
                ds = get_ds(file)
                dse.append(ds)
            dsb.append(dse)
        dss.append(dsb)
    return np.array(dss)

def unscale(series, low, high):
    ran = high - low
    norm = series * ran
    norm += low
    return norm

def func1(temp, high, low, n, stretch ):
    
    
    temps = np.array([298, 348])

    scaled_temp = st = (temp - temps[0])/(temps[1] - temps[0])
    y = (-(1-st/stretch)**(n)+ 1)/(-(1-1/stretch)**(n)+ 1)
    y = unscale(y, low, high)
    
    ret = y
    return ret

def get_curve(ntemps = 1000, func = func1, ret_interp = False, **kwargs):
    temps = temps = np.linspace(298, 348, ntemps)
    low_22 = kwargs.get('22_300')
    high_22 = kwargs.get('22_348')
    low_33 = kwargs.get('33_300')
    high_33 = kwargs.get('33_348')
    low_frac = kwargs.get('frac_300')
    high_frac = kwargs.get('frac_348')
    
    

    exp = kwargs.get('exp')
    stretch = kwargs.get('stretch')
    exp22= exp33 = expf = exp
    stretch22 = stretch33 = stretchf = stretch
    
    two = func(temps, high_22, low_22, exp22, stretch22)
    thr = func(temps, high_33, low_33, exp33, stretch33)
    f = func(temps, high_frac, low_frac, expf, stretchf)
    pts = np.vstack([two, thr, f]).T
    if ret_interp:
        interp = interp1d(temps, pts.T)
        return interp
    
    return pts
                

def get_unique_scores():
    global uniq
    a = scores.reshape(-1, 7)
    uniq, ids = np.unique(a, axis = 0, return_inverse = True)
    
        
    uniq_params = [[] for i in range(len(uniq))]
    params = all_pts.reshape(-1,nparam)
    
    for i, param in zip(ids, params):
        uniq_params[i].append(param)
    
    uniq_params = [np.array(x) for x in uniq_params]
    
    return uniq, uniq_params

def get_best_scores(uniq_scores, uniq_params):
    sum_scores = np.sum(uniq_scores, 1)
    best_score = np.nanmax(sum_scores)
    mask = sum_scores == best_score
    best_scores_arr = uniq_scores[mask]
    
    params = all_pts.reshape(-1,nparam)

    best_params = []
    for bs in best_scores_arr:
        mask = np.all(scores.reshape(-1, 7) == bs, 1)
        best_params.append(params[mask])
    
    return best_scores_arr, best_params
    
def plot_best_score_histos():
    f, ax = plt.subplots(2, 5, figsize = (10, 5))
    for i, (s, p) in enumerate(zip(best_scores, best_params)):
        for j in range(nparam):
            row = j//5
            col = j%5
            ax[row, col].set_title(sorted_params[j])
            ax[row, col].hist(p[:,j], density = True, bins = np.linspace(0,1, 20), label = str(s))
            
    plt.legend()
    plt.tight_layout()
    
def plot_test_histos():
    ap = all_pts.reshape(-1,nparam)
    s = scores.reshape(-1, 7)
    
    mask = ~np.isnan(s[:,0])
    s = s[mask]
    ap = ap[mask]
    
    
    for i in range(7):
        mask = s[:,i].astype(bool)
        f, ax = plt.subplots(2, 5, figsize = (10, 5))
        for j in range(nparam):
            row = j//5
            col = j%5
            ax[row, col].set_title(sorted_params[j])
            p = ap[mask]
            ax[row, col].hist(p[:,j], density = True, bins = np.linspace(0,1, 20), label = str(s), alpha = 0.5, color = 'b')
            p2 = ap[~mask]
            ax[row, col].hist(p2[:,j], density = True, bins = np.linspace(0,1, 20), label = str(s), alpha = 0.5, color = 'r')
            
            
        plt.suptitle(f'test_{i+1}')
        plt.suptitle(f'test_{i+1}')

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

def arr_to_dict(x):
    return {'22_300': x[0],
              '22_348': x[1],
              '33_300': x[2],
              '33_348': x[3],
              'exp': x[4],
              'frac_300':x[5],
              'frac_348':x[6],
              'stretch':x[7]}

if __name__ == '__main__':    
    rundir = 'sims/check_bo_params'
    batches = [1, 2, 3 , 4, 5, 6]#get_batches(rundir)    
    all_pts = get_all_params(batches)

    ap = all_pts.reshape(-1, nparam)
    scores = get_all_scores(batches)
    ac = scores.reshape(-1, 7)
    
    mask = ~np.isnan(ac[:,0])
    ap = ap[mask]
    ac = ac[mask]
    
    
    if not 'cont' in rundir:
        uniq_scores, uniq_params = get_unique_scores()
        best_scores, best_params = get_best_scores(uniq_scores, uniq_params)

        print(best_scores, len(best_params[0]))

    else:
        print(np.mean(np.sum(scores, -1), -1))
        
    
    
    for bs in scores:
        correct = np.sum(bs, 0)
    
    batch_scores = np.mean(np.sum(scores, -1), -1)

    bp = unnormalize_points(best_params[0])
    mp = np.mean(bp, 0)
    dicts = [arr_to_dict(x) for x in bp]
    
    f, ax = plt.subplots(dpi = 400)
    ax2 = ax.twinx()
    md = arr_to_dict(mp)
    mean_curve = get_curve(**md)
    for d in dicts:
        curve = get_curve(**d)
        for i in range(3):
            if i !=2:
                ax2.plot(np.linspace(298, 348, 1000), curve[:,i], c = ['r', 'b', 'k'][i], alpha = 0.05,)
            else:
                ax.plot(np.linspace(298, 348, 1000), curve[:,i], c = ['r', 'b', 'k'][i], alpha = 0.05,)
    for i in range(3):
        if i !=2:
            ax2.plot(np.linspace(298, 348, 1000), mean_curve[:,i], c = ['r', 'b', 'k'][i], alpha = 1)
        else:
            ax.plot(np.linspace(298, 348, 1000), mean_curve[:,i], c = ['r', 'b', 'k'][i], alpha = 1)

    ax.set_xlabel('Temperature, K')
    ax.set_ylabel(r'$f$')


    ax2.set_ylabel(" ")
    ax2.text(1.15, 0.4, r"$\varepsilon_{22}$   ", color="red", fontsize=14, transform=ax2.transAxes, va='top', ha='center', rotation = 90)
    ax2.text(1.15, 0.4, r"$\varepsilon_{33}$", color="blue", fontsize=14, transform=ax2.transAxes, va='bottom', ha='center', rotation = 90)
    ax2.text(1.15, 0.5, "(kcal/mol)", color="k", fontsize=14, transform=ax2.transAxes, va='bottom', ha='center', rotation = 90)

    ax2.set_ylim([0,0.7])
    plt.tight_layout()
    plt.savefig('/home/skronen/mc_paper_figs/good_params.png')
    plt.close()
