import numpy as np
import pydtmc.markov_chain as mark
import matplotlib.pyplot as plt
import random
from scipy.stats import norm
from itertools import groupby
from scipy.interpolate import interp1d

def zero_stretches(lst):
    return [len(list(group)) for value, group in groupby(lst) if value == 1]

def iy(l):
    return [int(x) for x in l]


def get_substitutions(n_chains, DP, hetero, DS):
    
    DS_single = DS/3
    
    if DS_single ==1:
        mean_block_len_sticky = 1
    elif DS_single == 0: #if cellulose, make transition to substitute very small
        mean_block_len_sticky = 1e16
    else:
        mean_block_len_ds = 1/DS_single
        mean_block_len_sticky = hetero * mean_block_len_ds
    
    
    b = 1/mean_block_len_sticky
    if DS_single !=0:
        c = b * (1-DS_single)/DS_single
    else:
        c = 1
    a = 1-b
    d = 1-c

    probs = [[a, b],
              [c,d]]
    print(probs)
    states = ['0','1']
    
    mc = mark.MarkovChain(probs, states)
    print(mc.steady_states)
    chains = []
    for i in range(n_chains):
        p = []
        for i in range(3):
            if DS == 0:
                init = '0'
            elif DS ==3:
                init = '1'
            else:
                init = np.random.choice(states, p=[1-DS_single, DS_single])
            p.append(iy(mc.simulate(DP-1, initial_state = init)))
        p= np.sum(p, axis = 0)
        chains.append(p)
    return chains

def convert_to_monomer_sequence(chains):
    chains = np.array(chains).flatten()
    ds0 = ['']
    ds1 = ['2', '3', '6']
    ds2 = ['23','36','26']
    ds3 = ['236']
    
    dss = [ds0,ds1,ds2,ds3]
    
    all_mono = []
    for n in chains:
        mono = np.random.choice(dss[n])
        all_mono.append(mono)
    return all_mono

def get_monomer_sequence(n_chains, DP, hetero, DS, uniform = False):
    if uniform:
        chains = random_uniform(n_chains, DP, DS, hetero)
    else:
        chains = get_substitutions(n_chains, DP, hetero, DS)

    monos = convert_to_monomer_sequence(chains)
    return monos

def random_uniform(n_chains, DP, DS, hetero, ret_p = False):
    #hetero between 0 and 1
    dss = [0,1,2, 3]
    lowds = int(DS)
    highds = lowds+1
    
    rem = DS-lowds
    p = [0,0,0,0]
    p[highds] =  rem
    p[lowds] = (1-rem)


    #minvar = sum((np.array([0,1,2,3])-DS)**2*np.array(p))
    pmaxvar = np.array([1-DS/3, 0, 0, DS/3])
    #maxvar = sum((np.array([0,1,2,3])-DS)**2*np.array(pmaxvar))
    p_interp = interp1d(np.array([0,1]), np.array([p, pmaxvar]), axis = 0)
    finalp = p_interp(hetero)
    if ret_p: return finalp

    monos = np.random.choice(dss, size = DP*n_chains, p=finalp)
    return monos.reshape(n_chains, DP)

def max_tri_block_length(n_chains, DP, DS, num):
    ##NOT COMPLETE.... don't think this is worth it...
    chains = get_substitutions(n_chains, DP, 1, DS)
    ids= []
    for chain in chains:
        mask = chain == 3
        #ids = []
        ct = 0
        si = 0
        for i,foo in enumerate(mask):
            if foo:
                ct +=1
            else:            
                if ct > num:
                    ids.append([si+1, i -1]) 
                ct = 0
                si = i 
    return chain, ids
    
def get_dimer_probs(probs):
    dprobs = np.zeros(7)
    dprobs[0] = probs[0]**2
    dprobs[1] = probs[0]*probs[1]*2
    dprobs[2] = probs[0]*probs[2]*2 + probs[1]**2
    dprobs[3] = probs[0]*probs[3]*2 + probs[1]*probs[2]*2
    dprobs[4] = probs[1]*probs[3]*2 + probs[2]**2
    dprobs[5] = probs[2]*probs[3]*2
    dprobs[6] = probs[3]**2
    return dprobs

if __name__ == '__main__':
    fs = 8
    f, ax = plt.subplots(1,2, dpi=400, figsize = (8, 4))
    hcs = [0,0.25, 0.5, 0.75, 1]
    colors = plt.cm.PRGn(np.linspace(0,1, len(hcs)))
    colors[2] = [0.7, 0.7, 0.7, 1]

    from matplotlib.colors import LinearSegmentedColormap

    color_map= ['#000000', '#004488', '#BB5566', '#228833', '#DDAA33']
    
    
    color1_s = [91/255,110/255,193/255,1]

    color2_s = [255/255, 97/255, 97/255, 1]

    color1_c = [108/255,183/255,206/255,1]
    color2_c = [240/255, 148/255, 86/255, 1]
    
    cmap_s = LinearSegmentedColormap.from_list("custom_cmap", [color1_s, color2_s])
    cmap_c = LinearSegmentedColormap.from_list("custom_cmap", [color1_c, color2_c])
    colors = cmap_c(np.linspace(0,1, len(hcs)))
    markers = ['o', '^', 's', '*', '+']
    
    for hc, col,m in zip(hcs, colors,markers):
        ds = 1.8
        probs = random_uniform(1, 1, ds, hc, ret_p = True)
        dimer_probs = get_dimer_probs(probs)
        print(sum(dimer_probs))
        ax[0].plot(range(4), probs, label = str(hc), color = col, marker = m)
        ax[1].plot(range(7), dimer_probs, label = str(hc), color = col, marker = m)

    ax[1].set_xticks(range(7))
    
    ax[1].legend(fontsize = fs, title = r'$H_C$')
    ax[1].set_xlabel('Dimer DS')
    ax[1].set_ylabel('Frequency')        
    ax[0].set_xlabel('Monomer DS')
    ax[0].set_ylabel('Frequency')
    ax[0].set_xticks(range(3 + 1))
    ax[0].legend(fontsize = fs, title = r'$H_C$')
        
    
    plt.tight_layout()
    plt.savefig('/home/skronen/mc_paper_figs/vary_hc_monodimer_distributions.png')
    plt.close()
