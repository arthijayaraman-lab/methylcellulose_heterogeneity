import pandas as pd
from joblib import load
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
from scipy.optimize import curve_fit

#Perform BI for bonds and backbone angles for CG model from atomistic fibril simulations


def separate_lists(orig_list, trial_removed_list):

    data = zip(orig_list, trial_removed_list)
    i = 0
    out = []
    
    for key, group in groupby(data, lambda x: x[0]):
            number, word = next(group)
            elems = len(list(group)) + 1
            if number == 9 and elems > 1:
                out.append((key, elems, i))
            i += elems

    return out

def compute_dihedral_angle(point1, point2, point3, point4):
    # Convert points to NumPy arrays for easier calculations
    p1, p2, p3, p4 = np.array(point1), np.array(point2), np.array(point3), np.array(point4)

    # Calculate vectors for the three bonds
    bond1 = p2 - p1
    bond2 = p3 - p2
    bond3 = p4 - p3

    # Calculate the normal vectors of the two planes formed by the bonds
    plane1_normal = np.cross(bond1, bond2)
    plane2_normal = np.cross(bond2, bond3)

    # Compute the dihedral angle using the dot product and arctan2
    angle = np.arctan2(np.dot(np.cross(plane1_normal, plane2_normal), bond2), np.dot(plane1_normal, plane2_normal))

    # Convert the angle from radians to degrees
    dihedral_angle_degrees = np.degrees(angle)

    return dihedral_angle_degrees

def compute_bond_length(coms, chain_nos):
    bonds = []
    for chain_no in np.unique(chain_nos):
        mask = chain_nos == chain_no
        pos = coms[mask]
        for i in range(len(pos)-1):
            p1 = pos[i]
            p2 = pos[i + 1]
            b = np.linalg.norm(p2 - p1)
            bonds.append(b)
    return bonds

def make_bins():
    num_bins = 35
    bbins = np.linspace(0.45, 0.55, num_bins)
    abins = np.linspace(120, 180, num_bins)
    dibins = np.linspace(-180, 180, num_bins)
    return [bbins, abins, dibins]

def get_metas(file):
    sp = file.split('_')
    if '33' in sp[0]: 
        nch = 13
    else: 
        nch = 25
    
    if '10' in sp[0]:
        lch = 20
    else: 
        lch = 30
    
    if 'highertemp' in file:
        temp = 348
    else: 
        temp = 300
    
    if "Cell" in file:
        ds = 0
    elif "MGF" in file:
        ds = 3
    else:
        val = int(sp[1][0])
        ds = val * 0.3
    return nch, lch, temp, ds

def change_monomer_names(name):
    if name == 'MG32':
        name = 'MG23'
    if name == 'MG63':
        name = 'MG36'
    if name == 'MG62':
        name = 'MG26'
    return name

def get_bin_mids(bins):
    mids = []
    for i in range(len(bins)-1):
        lb = bins[i]
        ub = bins[i+1]
        mids.append(np.mean([lb, ub]))
    return mids

def make_bond_histograms(pos, chain_nos, use_kde = False):
    bonds = compute_bond_length(pos, chain_nos)
    return bonds

def harmonic(x, A, x0):
    return A* (x - x0)**2
def cosine(x, A):
    return A * (1 + np.cos(x/180 * np.pi))
    
def P2U(p, x, bond_or_angle, temp):
    global denom
    boltzmann = 1.9858285706E-3 # Kcal/K
    T = temp # K
    if bond_or_angle == 'bond':
        denom = x**2
    elif bond_or_angle == 'angle':
        denom = np.sin(x/180 * np.pi)
    else:
        raise ValueError("bond_or_angle")
    U = - boltzmann * T * np.log(p / denom)
    return U - np.min(U)

def boltmann_invert_harmonic_fit(p, R, bond_or_angle,temp, anglefunc = harmonic, fitbounds = (-np.inf, np.inf), plot = True):
    U = P2U(p, R, bond_or_angle, temp)
    min_r = R[np.argmin(U)]
    if anglefunc == cosine:
        p0 = [3]
    else:
        p0= [1000, min_r]
    if bond_or_angle == 'bond':
        bounds = [(1e-3, min_r* 0.9), (np.inf, min_r * 1.1)]
    else:
        if anglefunc == cosine:
            bounds = [(0), (100)]
        else:
            bounds = [(1e-3, 130), (np.inf, 180)]
    
    mask1 = np.logical_and(R>fitbounds[0], R< fitbounds[1])
    mask2 = np.logical_and(~np.isnan(U), ~np.isinf(U))
    mask = np.logical_and(mask1, mask2)
    rfit = R[mask]
    ufit = U[mask]
    popt, _ = curve_fit(anglefunc, rfit, ufit, p0, bounds = bounds, maxfev = 100000)
    if plot:
        plt.figure()
        plt.plot(rfit, ufit, color = 'k', marker = 'o', label = 'BI')
        plt.plot(rfit, anglefunc(rfit, *popt), color = 'r', ls = '--', label = 'fit')
        plt.ylabel('U, kcal/mol')
        if bond_or_angle == 'bond':
            plt.xlabel('r, nm')
        elif bond_or_angle == 'angle':
            plt.xlabel(r'$\theta$, degrees')
        plt.title('Harmonic BI')
        plt.legend()
    return popt


def check_file_temp(file, temps):
    if temps == [348] and ('highertemp' not in file):
        return False
    if ('highertemp' in file) and (348 not in temps):
        return False
    return True

def check_file_ds(file, dss):
    b = [x in file for x in dss]
    return np.any(b)

def check_file_size(file, size):
    b = [x in file for x in size]
    return np.any(b)

def check_file(file, dss, temps, size):
    b = [
        check_file_temp(file, temps),
        check_file_ds(file, dss),
        check_file_size(file, size)]
    return np.all(b)

    
def load_data_bond(dss = ['Cell', '222', '444', '666', '888', 'MGF'], temps = [300, 348], size = ['3310', '3315', '4410', '4415']):
    storage_dir = r'/home/skronen/Documents/methylcellulose/MC_atomistic_analysis/for_caviness/proc_monomer_output/'
    
    files = []
    for i,file in enumerate(os.listdir(storage_dir)[:]):
        if check_file(file, dss, temps, size):
            files.append(file)
    files.sort()
    
    file_notrial = ['_'.join([*f.split('_')[:2], f.split('_')[-1]]) for f in files]
    lens = [len(list((y))) for x, y in groupby(file_notrial)]
    files_split = []
    ct = 0
    for val in lens:
        files_split.append(files[ct: ct + val])
        ct += val

    temps = []
    
    dss = []
    params = []
    print('loading files')
    all_bonds = []
    for files in files_split:
        for i,file in enumerate(files[:]):
            #print(f'loading file {i+1}/{len(files)}')
            #print(file)
            all_monomers = load(os.path.join(storage_dir, file, 'all_monomers.pkl'))
            nch, lch, temp, ds = get_metas(file)
            if i == 0: params.append([nch, lch, temp, ds])
            dss.append(ds)
            typs = []
            coms = []
            chains = []
            for monomer in all_monomers:
                com = monomer.mono_com
                chain = (int(monomer.mono_num)-1)//monomer.chain_len
                typ = monomer.mono_type  
                
                typs.append(typ)
                coms.append(com)
                chains.append(chain)
        
            coms = np.array(coms)
            chains = np.array(chains)
            
            bonds = make_bond_histograms(coms, chains)
            all_bonds.extend(bonds)

    print('done loading files')    
    bbins = np.linspace(0.4, 0.6, 35)
    bond_p, b = np.histogram(all_bonds, bins = bbins, density = True)
    bond_R = get_bin_mids(b)

    return np.array(bond_p), np.array(bond_R)


def load_data_angle(dss = ['Cell', '222', '444', '666', '888', 'MGF'], temps = [300, 348], size = ['3310', '3315', '4410', '4415']):
    global df
    direct = 'atomistic_angle_dfs'
    
    cd = {0:'Cell',
      0.6:'222',
      1.2:'444', 
      1.8:'666', 
      2.4:'888', 
      3:'MGF',}
    
    cd2 = {
        (13, 20):'3310',
        (13, 30):'3315',
        (25, 20):'4410',
        (25, 30):'4415'
        }
    
    df = pd.DataFrame()
    
    for file in os.listdir(direct):
        df_add = pd.read_csv(os.path.join(direct, file))
        df_add['ds'] = float(file.split('_')[-2])
        df = pd.concat((df, df_add), axis = 0)
        
        
    mask1 = [cd[x] in dss for x in df['ds']] 
    mask2 = [x in temps for x in list(df['temp'])]
    mask3 = [cd2[(x, y)] in size for x, y in zip(df['n_chains'], df['chain_len'])]
    
    mask = np.logical_and.reduce((mask1, mask2, mask3))
    df = df[mask]
    return df

def get_hist(df, filt = 'B3m', meth = True):
    
    if filt:
        mask = df[filt].astype(bool)
        if not meth:
            mask= ~mask
        print(np.sum(mask))
        df = df[mask]
    
    data = df['lll']

    abins = np.linspace(120, 180, 35)
    h, b = np.histogram(data, bins = abins, density = True)
    bin_mids = get_bin_mids(b)
    return bin_mids, h

def make_histogram(dss = ['Cell', '222', '444', '666', '888', 'MGF'], temps = [300, 348], size = ['3310', '3315', '4410', '4415'], filt = 'B3m', meth = True):
    df = load_data_angle(dss, temps, size)
    b,h = get_hist(df, filt, meth)
    return np.array(b),h

ds_lab_to_ds = {'Cell': 0.0,
                '222':0.6,
                '444':1.2,
                '666': 1.8,
                '888': 2.4,
                'MGF': 3.0}

#%%
if __name__ == '__main__':
    #tried both harmonic and cosine angles; harmonic generally performed better
    anglefunc = harmonic
    sav = True
    savdir = 'atomistic_hists'
    if sav: 
        try: os.mkdir(savdir)
        except: pass
    params_m = {}
    params = {}
    for temp in [300,348]:
        for ds in ['Cell', '222', '444', '666', '888', 'MGF']:
            if ds != 'Cell': 
                b, h = make_histogram([ds], [temp], meth = True)
                popt_angle_m = boltmann_invert_harmonic_fit(h, b, 'angle', anglefunc = anglefunc, temp = temp, fitbounds = (140, np.inf))
                plt.title(f'DS{ds_lab_to_ds[ds]}_T{temp}K_C3methyl')
                if anglefunc==harmonic:
                    popt_angle_m[0] *= 180**2/np.pi**2
                params_m[(temp, ds)] = popt_angle_m
                
                if sav: np.savetxt(f'{savdir}/anghist_{ds_lab_to_ds[ds]}_{temp}_methyl.txt', np.vstack((b,h)).T)
                
            if ds!= 'MGF':
                b2, h2 = make_histogram([ds], [temp], meth = False)
                popt_angle = boltmann_invert_harmonic_fit(h2, b2, 'angle', anglefunc = anglefunc, temp = temp, fitbounds = (140, np.inf))
                plt.title(f'DS{ds_lab_to_ds[ds]}_T{temp}K_C3nomethyl')
                if anglefunc==harmonic:
                    popt_angle[0] *= 180**2/np.pi**2
                params[(temp, ds)] = popt_angle
                
                if sav: np.savetxt(f'{savdir}/anghist_{ds_lab_to_ds[ds]}_{temp}_nomethyl.txt', np.vstack((b2,h2)).T)

    #%% 300K
    bond_p, bond_r = load_data_bond(temps = [300])
    popt_bond = boltmann_invert_harmonic_fit(bond_p, bond_r, 'bond', temp = 300, fitbounds = (0.46, 0.56))   
    plt.title(f'AllDS_T300K')
    print(popt_bond)
    if sav: np.savetxt(f'{savdir}/bondhist_300.txt', np.vstack((bond_r,bond_p)).T)

    #%% 348k
    bond_p, bond_r = load_data_bond(temps = [348])
    popt_bond = boltmann_invert_harmonic_fit(bond_p, bond_r, 'bond', temp = 348, fitbounds = (0.46, 0.58))   
    plt.title(f'AllDS_T348K')
    print(popt_bond)
    if sav: np.savetxt(f'{savdir}/bondhist_348.txt', np.vstack((bond_r,bond_p)).T)

    #%% ALL
    bond_p, bond_r = load_data_bond() #all data
    
    popt_bond = boltmann_invert_harmonic_fit(bond_p, bond_r, 'bond', temp = 348, fitbounds = (0.46, 0.58))   
    plt.title(f'AllDS_AllT')
    print(popt_bond)
    if sav: np.savetxt(f'{savdir}/bondhist_alltemp.txt', np.vstack((bond_r,bond_p)).T)
