import pydtmc.markov_chain as mark
import numpy as np
import matplotlib.pyplot as plt

def make_mc(DS, hetero):
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
    probs = [
            [a**3, 3*a**2*b, 3*a*b**2, b**3],
            [a**2*c, d*a**2 + 2*c*b*a, 2*d*b*a+c*b**2, d*b**2],
            [c**2*a, 2*c*d*a + c**2*b, 2*c*b*d + d**2*a, d**2*b],
            [c**3, 3*c**2*d, 3*c*d**2, d**3],
            ]
    
    states = ['0','1', '2', '3']
    
    mc = mark.MarkovChain(probs, states)
    return mc, probs


#%%

from itertools import product
def find_combinations(N, X):
    # Generate all combinations of length N from numbers 0 to 3
    combinations = product(range(4), repeat = N)
    
    # Filter combinations that sum up to X
    valid_combinations = [combo for combo in combinations if sum(combo) == X]
    
    return valid_combinations

def get_N_distribution(N, mc, probs):
    p = np.array(probs)
    ss = mc.steady_states[0]
    
    probs_out = []
    for i in range(3*N+1):
        combs = find_combinations(N, i)
        p_num = 0
        for c in combs:
            prob = ss[c[0]]
            for j in range(len(c)-1):
                prob*= p[c[j], c[j+1]]
            p_num += prob
        probs_out.append(p_num)
    return probs_out

#%%
if __name__=='__main__':
    DS = 1.8
    heteros = [1, 2, 3, 4]
    colors = plt.cm.autumn(np.linspace(0,1,len(heteros)+1))
    f, ax = plt.subplots(1,2, dpi=400, figsize = (8, 4))
    fs = 8
    exptl = [0.015, 0.043, 0.123, 0.255, 0.309, 0.195, 0.06]#from Fractionation of Methyl Cellulose According to polarity Fig 6 top left
    ax[1].plot(exptl, label = 'Experimental', color= 'k')
    
    from matplotlib.colors import LinearSegmentedColormap

    color_map= ['#000000', '#004488', '#BB5566', '#228833', '#DDAA33']
    
    
    color1_s = [91/255,110/255,193/255,1]
    #color1_s = [70/255,80/255,190/255,1]

    color2_s = [255/255, 97/255, 97/255, 1]
    #color2_s = [255/255, 80/255, 80/255, 1]

    color1_c = [108/255,183/255,206/255,1]
    color2_c = [240/255, 148/255, 86/255, 1]
    
    cmap_s = LinearSegmentedColormap.from_list("custom_cmap", [color1_s, color2_s])
    cmap_c = LinearSegmentedColormap.from_list("custom_cmap", [color1_c, color2_c])
    colors = cmap_s(np.linspace(0,1, len(heteros)))
    
    for j,hetero in enumerate(heteros):        
        mc, probs = make_mc(DS, hetero)
        N = 2
        ax[1].plot(get_N_distribution(N, mc, probs),color = colors[j], label = r'Simulated, $H_S$ ' + f'= {hetero}')
    ax[1].set_xticks(range(3*N + 1))
    
    ax[1].legend(fontsize = fs)
    ax[1].set_xlabel('Dimer DS')
    ax[1].set_ylabel('Frequency')
    
    exp_data = [0.06, 0.28, 0.42, 0.24] #from Fractionation of Methyl Cellulose According to polarity Fig 6 top right
    ax[0].plot(exp_data, color = 'k', label = 'Experimental')
    ax[0].plot(mc.steady_states[0], color = 'r', label = 'Simulated')
    ax[0].set_xlabel('Monomer DS')
    ax[0].set_ylabel('Frequency')
    ax[0].set_xticks(range(3 + 1))
    ax[0].legend(fontsize = fs)
    plt.tight_layout()
    plt.savefig('/home/skronen/mc_paper_figs/heterogeneity_comp_with_exp.png')
    plt.close()
    #%%
    heteros = np.linspace(0.9, 1.3, 1000)
    maes = []
    for j,hetero in enumerate(heteros):  
        print(j)
        mc, probs = make_mc(DS, hetero)
        N = 2
        sim = np.array(get_N_distribution(N, mc, probs))
        mae = np.sum(abs(exptl - sim))
        maes.append(mae)
    #%%
    minid = np.argmin(maes)
    hetero = heteros[minid]
    
    f, ax = plt.subplots(1,1, dpi=400, figsize = (4, 4))
    fs = 8
    ax.plot(exptl, label = 'Experimental', color= 'k')
    mc, probs = make_mc(DS, hetero)
    N = 2
    ax.plot(get_N_distribution(N, mc, probs),color = colors[0], label = r'Simulated, $H_S$ ' + f'= {hetero:0.4f}')
    ax.set_xticks(range(3*N + 1))
    
    ax.legend(fontsize = fs)
    ax.set_xlabel('Dimer DS')
    ax.set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('/home/skronen/mc_paper_figs/optimal_heterogeneity.png')
    plt.close()
    
    
