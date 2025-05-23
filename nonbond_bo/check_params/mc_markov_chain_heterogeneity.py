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

def random_uniform(n_chains, DP, DS, hetero):
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
    
    

if __name__ == '__main__':
    #test_max_block_length
    a, b = max_tri_block_length(1, 100, 2.5, 2)
        
    #test uniform
    # a = random_uniform(1, 100, 1.8, 1)
    # print([np.sum(a[0]==i) for i in range(4)])
    # print(np.mean(a))
    # plt.plot(a[0])
    # plt.ylim([0,3])
    
    
    
    # #%%
    # n_chains = 2
    # DP=100
    # hetero = 1
    # DS= 1.8
    # f, ax = plt.subplots(4, 1, figsize = [10, 10])
    # for i,hetero in enumerate([1, 3, 5, 10]): 
    #     chains = get_substitutions(n_chains, DP, hetero, DS)
    #     print(np.mean(chains))
    #     ax[i].plot(chains[1], color = 'k')
    #     ax[i].set_title(f'Heterogeneity = {hetero}')
    #     ax[i].axhline(np.mean(chains[1]), color = 'gray', ls ='--')
    #     #plt.hist(np.mean(chains, 1))
        
    #     a = convert_to_monomer_sequence(chains)
    
    #     h, b = np.histogram(np.array(chains).flatten(), bins = 4)
    #     h =h/sum(h)
    #     #plt.plot(range(4), h)
    #     ax[i].set_xlabel('Monomer ID')
    #     ax[i].set_ylabel('Monomer DS')
    #     ax[i].set_yticks(range(4), range(4))
    #     ax[i].set_xlim([0,100])
    # plt.tight_layout()
