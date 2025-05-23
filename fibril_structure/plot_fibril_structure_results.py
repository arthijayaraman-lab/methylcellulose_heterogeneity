import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns


def get_dirs(trial):
    hd = '/home/skronen/Documents/tests_that_went_nowhere/mc_beadspring/misc_systems/large_systems_final/sims/1000_100mers/'
    
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
    

    trial_dir = os.path.join(hd, f'trial_{trial}')
    files = [os.path.join(trial_dir, x, 'vary_epshp') for x in sds]
    return files, sds

def read_data(directory):
    diam_file = os.path.join(directory, 'diameter.txt')
    len_file = os.path.join(directory, 'lengths.txt')
    xlink_file = os.path.join(directory, 'xlink.txt')
    pl_file = os.path.join(directory, 'persistence.txt')

    diams = np.loadtxt(diam_file)
    if len(diams.shape) == 0:
        diams = np.array([[np.nan, np.nan]])
    elif len(diams.shape) ==1:
        diams = diams[np.newaxis, :]
    
    lens = np.loadtxt(len_file)
    xlink = np.loadtxt(xlink_file)
    pl = np.loadtxt(pl_file)
    return diams, lens, xlink, pl
    

if __name__ == '__main__':
    trials = [1, 2, 3]
    df_all = pd.DataFrame()
    for t in trials[:]:
        dirs, sds = get_dirs(t)
    
        df_dict = {x: [] for x in ['ds','ht','h','diam','diam_std','len','len_std', 'xlink','pl','trial']}
        for d, ids in zip(dirs[:], sds, strict = True):
            try:
                diams, lens, xlink, pl = read_data(d)
                spl = ids.split('_')
                ds = float(spl[0].replace('ds', ''))
                ht = spl[1][1]
                h = float(spl[1].replace(f'H{ht}', ''))            
                dm = np.mean(diams[:,0])
                lm = np.mean(lens)
                dst = np.std(diams[:,0])
                lst = np.std(lens)
                
                df_dict['ds'].append(ds)
                df_dict['ht'].append(ht)
                df_dict['h'].append(h)
                df_dict['diam'].append(dm)
                df_dict['diam_std'].append(dst)
                df_dict['len'].append(lm)
                df_dict['len_std'].append(lst)
                df_dict['xlink'].append(np.mean(xlink))
                df_dict['pl'].append(np.mean(pl))
                df_dict['trial'].append(t)
            except:
                print(ids)
            
        df = pd.DataFrame(df_dict)
        df_all = pd.concat((df_all, df), axis = 0)    
    
    df = df_all.copy()

    df['len'] = df['len']/10
    df['pl'] = df['pl']/10
    df['diam'] = df['diam']/10
    colors = {'s': [plt.cm.autumn(0.1), plt.cm.autumn(0.5), plt.cm.autumn(0.8)],
              'c': [plt.cm.PRGn(0.1), [0.7, 0.7, 0.7, 1], plt.cm.PRGn(0.9) ]}
    
    """ ################ """
    from matplotlib.colors import LinearSegmentedColormap

    color_map= ['#000000', '#004488', '#BB5566', '#228833', '#DDAA33']
    
    
    color1_s = [91/255,110/255,193/255,1]

    color2_s = [255/255, 97/255, 97/255, 1]

    color1_c = [108/255,183/255,206/255,1]
    color2_c = [240/255, 148/255, 86/255, 1]
    
    cmap_s = LinearSegmentedColormap.from_list("custom_cmap", [color1_s, color2_s])
    cmap_c = LinearSegmentedColormap.from_list("custom_cmap", [color1_c, color2_c])
    
    
    colors = {'s': [cmap_s(0), cmap_s(0.5), cmap_s(0.999)],
              'c': [cmap_c(0), cmap_c(0.5), cmap_c(0.999)]}
    
    """ ################ """


    leg_titles = {'s': r'$H_S$',
                  'c': r'$H_C$'}
    
    ylabs = {'diam': 'Fibril Diameter (nm)',
             'len': 'Fibril Contour Length (nm)',
             'pl': 'Fibril Persistence Length (nm)'}

    
    vals = ['len', 'pl', 'diam']
    fs = 12
    for val in vals:
        f, ax = plt.subplots(1, 2, figsize = (10, 5), dpi = 400)
        for i,ht in enumerate(['s', 'c']):
            df_plot = df[df['ht']==ht]
            if val == 'pl':
                df_plot = df_plot[df_plot['pl'] < 500]
            if val == 'len':
                df_plot = df_plot[df_plot['len'] < 500]
            hue = f'H{ht}'
            df_plot[hue] = df_plot['h']
            
            sns.pointplot(df_plot, x = 'ds', y = val, hue = hue, palette= colors[ht],
                          linestyles = 'none', errorbar='sd', errwidth = 1,
                          capsize = 0.15, ax = ax[i], markers = ['o', '^', 's'], scale = 1.5)
            
            for artist in ax[i].collections:
                artist.set_edgecolor('k')
                artist.set_linewidth(1.2)
                
            ax[i].set_xlabel('DS', fontsize = fs)
            ax[i].set_ylabel(ylabs[val], fontsize = fs)
            legend = ax[i].get_legend()
            for handle in legend.legend_handles:
                handle.set_edgecolor('k')
                handle.set_linewidth(1.2)
                
            legend.set_title(leg_titles[ht], prop={'size': 14})
            
            
            
        plt.tight_layout()
        plt.savefig(f'/home/skronen/mc_paper_figs/{val}_plot.png')
        plt.close()
    
        
