from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import numpy as np
import matplotlib.pyplot as plt
import os
from bayes_opt.util import load_logs
import warnings

def black_box_function(x, y):
    return -x ** 2  + 1 - (1-y)**2
    
def plot_bo(bo):
    global arr
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-3, 3, 100)
    xv,yv = np.meshgrid(x,y)

    arr = np.stack((np.ravel(xv), np.ravel(yv)), axis=-1)
    mean = bo._gp.predict(arr, return_std=False)
    mean = mean.reshape(len(y), len(x))
    plt.figure()
    plt.pcolormesh(x, y, mean)
    plt.title('bo')
    return mean
    

def plot_f(f):
    global arr
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-3, 3, 100)
    xv,yv = np.meshgrid(x,y)

    arr = np.stack((np.ravel(xv), np.ravel(yv)), axis=-1)
    mean = f(*arr.T)
    mean = mean.reshape(len(y), len(x))
    plt.figure()
    plt.pcolormesh(x, y, mean)
    plt.title('function')
    return mean

def make_arr(pt):
    pt = np.array([x for k,x in pt.items()])
    return pt
    
def probe_pts(optimizer, n):
    global next_point, pt, target
    
    pts = []
    pts_a = []
    pt_strings = []
    ct = 0
    while ct < n:
        next_point = optimizer.suggest(n_random=100000, n_l_bfgs_b=0)
        pt= make_arr(next_point)
        pt_string = '_'.join([f'{x:0.4f}' for x in pt])
        
        if pt_string not in pt_strings:
            pt_strings.append(pt_string)
            pts.append(pt)
            pts_a.append(next_point)
            ct +=1
    return np.array(pts), pts_a

def run_bo(optimizer, func, n, npt, plot = False):
    if plot:
        colors = plt.cm.jet(np.linspace(0,1,n))
        plt.figure()
    for i in range(n):
        print(i)
        pts, pts_a = probe_pts(optimizer, npt)
            
        for pt, pt_a in zip(pts, pts_a):
            target = func(**pt_a)
            optimizer.register(params=pt_a, target=target, )

        if plot:plt.scatter(pts[:,0], pts[:,1], label = str(i), color = colors[i])
    #plt.scatter(0,1, color = 'k', label = 'opt')
    if plot:
        #plt.axvline(x = 0, color = 'k')
        plt.legend()

    #a = plot_bo(optimizer)
    #plt.scatter(pts[:,0], pts[:,1], color = 'k')
    #b = plot_f(black_box_function)
    
def test_multi():
    global pts
    pts = probe_pts(optimizer, 50)

def test_cging():
    ds = np.array([0, 0.6, 1.2, 1.8, 2.4, 3.0])
    
    def bbf3(hb, hp, bb, bbhp, ret_parts = False):
        atom = np.array([-2, -2.3, -2.7, -2.9, -3.2, -3])
        cg = -(5 * bb + 0.005 * (hb-3.5) * (3- ds)**6 + 10 * ds * (hp-0.02) + 10 * ds * bbhp )
        diff = -np.sum((atom- cg)**2)
        if ret_parts:
            return atom, cg, diff
        return diff
    
    def cf_func(x, hb, hp, bb, bbhp):
        cg = -(5 * bb + 0.005 * (hb-3.5) * (3- x)**6 + 10 * x * (hp-0.02) + 10 * x * bbhp )
        return cg
    
    
    acq = acquisition.UpperConfidenceBound(kappa = 0.01)

    optimizer = BayesianOptimization(
        f=None,
        acquisition_function = acq,
        pbounds={'hb': (1.5, 5.5),
                  'hp': (0.01, 0.14),
                  'bb': (0.01, 1.5),
                 'bbhp':(0.01, 0.06), 
                     },
        verbose=2,
        random_state=1,
        allow_duplicate_points = True
    )
    
    
    run_bo(optimizer, bbf3,  5, 50, plot = True)
    print(optimizer.max)
    
    params = optimizer.max['params']
    atom, cg, diff = bbf3(**params, ret_parts = True)
    plt.figure()
    plt.scatter(ds, atom)
    plt.scatter(ds, cg)
    
    from scipy.optimize import curve_fit
    
    bounds = [[1.5, 0.005, 0.01, 0.005], [5.5, 0.14, 1.5, 0.06]]
    
    x = ds
    y = np.array([-2, -2.3, -2.7, -2.9, -3.2, -3])
    params_cf, _ = curve_fit(cf_func, x, y, bounds = bounds)
    
    print(params_cf)
    
    
    
def bbf2(hb, hp, bb, ret_parts = False):
    ds = np.array([0, 0.6, 1.2, 1.8, 2.4, 3.0])

    atom = np.array([-2, -2.3, -2.7, -2.9, -3.2, -3])
    bbhp = hp
    cg = -(5 * bb + 0.005 * (hb-3.5) * (3- ds)**6 + 10 * ds * (hp-0.02) + 10 * ds * bbhp )
    diff = -np.sum((atom- cg)**2)
    if ret_parts:
        return atom, cg, diff
    return diff

def test_cging2():
    global optimizer 
    ds = np.array([0, 0.6, 1.2, 1.8, 2.4, 3.0])

    # atom, cg, diff = bbf2(3.5, 0.1, 0.4, ret_parts = True)
    # plt.scatter(ds, atom)
    # plt.scatter(ds, cg)
    # atom, cg, diff = bbf2(6.5, 0.2, 0.2, ret_parts = True)    
    # plt.plot(ds, cg)
    # atom, cg, diff = bbf2(3.5, 0.2, 0.2, ret_parts = True)
    # plt.plot(ds, cg)
    # atom, cg, diff = bbf2(6.5, 0.02, 0.02, ret_parts = True)
    # plt.plot(ds, cg)
    
    acq = acquisition.UpperConfidenceBound(kappa = 0.01)
    #acq = acquisition.ExpectedImprovement(xi = 0)

    optimizer = BayesianOptimization(
        f=None,
        acquisition_function = acq,
        pbounds={'hb': (1.5, 5.5),
                  'hp': (0.01, 0.14),
                  'bb': (0.01, 1.5),
                      },
        verbose=2,
        random_state=1,
        allow_duplicate_points = True
    )
    
    
    run_bo(optimizer, bbf2,  10, 25, plot = True)
    print(optimizer.max)
    
    params = optimizer.max['params']
    atom, cg, diff = bbf2(**params, ret_parts = True)
    plt.figure()
    plt.scatter(ds, atom)
    plt.scatter(ds, cg)
    
def read_bsprof(fname):
    with open(fname, 'r') as f:
        lines = f.read().splitlines()
    
    data = []
    bs = []
    for line in lines[17:]:
        if line == '&':
            data.append(np.array(bs))
            bs = []
        spl = line.split()
        if len(spl) ==2:
            bs.append([float(spl[0]), float(spl[1])])
    return data

def get_delf(profile):
    x = np.abs(profile[:,0])
    u = profile[:,1]
    norm = np.mean(u[x > 3])
    u -= norm
    return np.min(u)
    
def get_bootstrap_std(fname, plot = False):
    bs_profs = read_bsprof(fname)
    delfs = []
    for prof in bs_profs:
        delfs.append(get_delf(prof))
    if plot:
        plt.figure()
        plt.hist(delfs)
    return np.mean(delfs), np.std(delfs)    

def make_aaref_file(temp, savename, bootstrap = True):
    delfs =[]
    dirname = '/home/skronen/Documents/Methylcellulose/energy_mapping_direction'

    chemlist = ['cell', '222', '444','666','888','mgf']
            
    for atom_chem in chemlist:
        if atom_chem == '444' and temp == 348: atom_chem = '444_3'
        if atom_chem == '222' and temp == 300: atom_chem = '222_2'
        if atom_chem == '888' and temp == 300: atom_chem = '888_3' #NEED TO UPDATE THIS
    
        if bootstrap:
            fname = f'{dirname}/{atom_chem}_{temp}/bsProfs.xvg'
            delf, _ = get_bootstrap_std(fname)
        else:
            fname = f'{dirname}/{atom_chem}_{temp}/profile.xvg'
            delf = read_aa_pmf(fname)
            
        delfs.append(delf)
    
    np.savetxt(savename, delfs)
    
def read_aa_pmf(fname):
    data = np.loadtxt(fname, skiprows = 17)
    x = data[:,0]
    u = data[:,1]
    norm = np.mean(u[x > 3])
    u -= norm
    return np.min(u)
    
    
def read_cg_pmf(fname):    
    data = np.loadtxt(fname)
    x = data[:,0]/10
    u = data[:,1]
    norm = np.mean(u[x > 3])
    u -= norm
    return np.min(u)

def read_prev_cg_data(aaref_fname):
    dirname = '/home/skronen/Documents/Methylcellulose/Coarse-Graining/CODE/Zijie_model/v4/umbrella/sims/current'
    
    chemlist = ['cell', '222', '444','666','888','mgf']
    
    aa_ref = np.loadtxt(aaref_fname)
    
    points_to_register = []
    for hb in np.arange(3.5, 6.6, 0.1):
        for hp in np.arange(0.02, 0.22, 0.02):
            cg_delfs = []
            for chem in chemlist:
                cg_file = os.path.join(dirname, chem, f'hb_{hb:0.1f}_hp_{hp:0.2f}', 'prod', 'PMF.dat')
                delf = read_cg_pmf(cg_file)
                cg_delfs.append(delf)
            cg_delfs = np.array(cg_delfs)
            target = mae(aa_ref, cg_delfs)
            point = {'target':target,
                     'params':
                     {'hp': hp,
                     'hb': hb,
                     'bb': hp,
                     'bbhp': hp}
                     }
            if np.isnan(target): continue
            points_to_register.append(point)
    return points_to_register        

def get_params(foldername):
    spl = foldername.split('_')
    if not foldername.startswith('hb_') or len(spl) != 8:
        print(foldername)
        raise ValueError('folder name not correct format for getting params')
    
    hb = float(spl[1])
    hp = float(spl[3])
    bb = float(spl[5])
    bbhp = float(spl[7])
    return hb, hp, bb, bbhp
    
def read_prev_cgbo_data(run_folder_name, aaref_fname):
    aa_ref = np.loadtxt(aaref_fname)

    chemlist = ['cell', '222', '444','666','888','mgf']
    items = os.listdir(run_folder_name)
    points_to_register = []
    for item in items:
        if not item.startswith('batch_'): continue
    
        d = os.path.join(run_folder_name, item)
        for dirname in os.listdir(d):
            if not dirname.startswith('hb_'): continue
            hb, hp, bb, bbhp = get_params(dirname)

            cg_delfs = []
            for chem in chemlist:
                cg_file = os.path.join(d, dirname, chem, 'prod', 'PMF.dat')
                delf = read_cg_pmf(cg_file)
                cg_delfs.append(delf)
            cg_delfs = np.array(cg_delfs)
            target = mae(aa_ref, cg_delfs)
            point = {'target':target,
                     'params':
                     {'hp': hp,
                     'hb': hb,
                     'bb': bb,
                     'bbhp': bbhp}
                     }
            if np.isnan(target): continue
            points_to_register.append(point)
    
    return points_to_register
    

def mae(aa_ref, cg):
    return -np.mean(np.abs(aa_ref- cg))

def check_pt_in_bounds(pt, optimizer):
    bounds = optimizer._space.bounds
    
    for i,foo in enumerate(sorted_params):
        if pt[foo] > bounds[i,1] or pt[foo] < bounds[i,0]:
            return False
    return True
        

    
def setup(ix, init_points = None, normed = True, bounds_in = None, alpha = 1e-6):
    acq = acquisition.UpperConfidenceBound(kappa = 10)
    
    if normed:
        bounds = {
                  '22_300': (0, 1),
                 '33_300':(0, 1),
                 'frac_300': (0,1),
                 '22_348': (0, 1),
                '33_348':(0, 1),
                'frac_348': (0,1),
                'exp':(0,1),
                'stretch': (0,1)

                     }
        
    elif bounds_in != None:
        bounds = bounds_in
    else:
        bounds = pbd
        
    optimizer = BayesianOptimization(
        f=None,
        acquisition_function = acq,
        pbounds=bounds,
        verbose=2,
        random_state=1,
        allow_duplicate_points = True
    )
    logfile = f'bo_log_{ix}.log'
    logger = JSONLogger(path = logfile)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.set_gp_params(normalize_y = True, alpha = alpha )

    if init_points:
        for point in init_points:
            target = point['target']
            params= point['params']
            if check_pt_in_bounds(params, optimizer):
                optimizer.register(params=params, target=target,)
            
    return optimizer
          
def load_bo(ix, normed = True, bounds_in = None, dirname = None, alpha = 1e-6):
    acq = acquisition.UpperConfidenceBound(kappa = 10)
    
    if normed:
        bounds = {
                  '22_300': (0, 1),
                 '33_300':(0, 1),
                 'frac_300': (0,1),
                 '22_348': (0, 1),
                '33_348':(0, 1),
                'frac_348': (0,1),
                'exp':(0,1),
                'stretch': (0,1)
                     }
    elif bounds_in != None:
        bounds = bounds_in
    else:
        bounds = pbd
    
    optimizer = BayesianOptimization(
        f=None,
        acquisition_function = acq,
        pbounds= bounds,
        verbose=2,
        random_state=1,
        allow_duplicate_points = True
    )

    for i in range(ix+1):
        #print(i)
        #sleep(0.5)
        
        logfile = f'bo_log_{i}.log'
        if dirname:
            logfile = os.path.join(dirname, logfile)
        if os.path.exists(logfile):
            load_logs(optimizer, logs = [logfile])
        else:
            warnings.warn(f'{logfile} does not exist. It is skipped.')
        #load_logs(optimizer, logs = [logfile])
    logger = JSONLogger(path = f'./bo_log_{ix}.log', reset = False)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.set_gp_params(normalize_y = False, alpha = alpha )

    return optimizer

def test_load(ix):
    opt = load_bo(ix)
    print(str(len(opt.space)) + ' points')
    opt.register(params = {'hb': 5.1541, 'hp': 0.111234, 'bb': 0.1413, 'bbhp': 0.02}, target = np.random.rand())
    print(str(len(opt.space)) + ' points')
    return opt

def test_setup(ix):
    opt = setup(ix, add_init_points = True)
    return opt

# data = np.array([[0.10606272, 0.14894714, 0.20481343, 0.31260712],
#        [0.09687268, 0.15517449, 0.32305383, 0.37870631]]) 

pbounds_dict=pbd=bounds = {
          '22_300': (0.15, 0.25),
         '33_300':(0.26, 0.36),
         'frac_300': (0.05,1),
         '22_348': (0.2, 0.3),
        '33_348':(0.33, 0.43),
        'frac_348': (0.2,1),
        'exp': (1, 6),
        'stretch': (1,2)
             }

sorted_params = list(pbd.keys())
sorted_params.sort()
pbounds = np.array([[ pbd[key][0],pbd[key][1]]  for key in sorted_params]) 

def normalize_points(x):
    bounds = pbounds
    norm_x = (x - bounds[:,0])/(bounds[:,1] - bounds[:,0])
    return norm_x

def unnormalize_points(norm_x):
    bounds = pbounds
    x = norm_x * (bounds[:,1] - bounds[:,0]) + bounds[:,0]
    return x
    
    

if __name__ == '__main__':
    pass
    #ds = np.array([0, 0.6, 1.2, 1.8, 2.4, 3.0])
    #aa_ref_file = './aaref.txt'
    #make_aaref_file(348, aa_ref_file)
    #test_cging()

    #pts = read_prev_cgbo_data('sims/run_cav_fake', aa_ref_file)


    #logname = ''    
    #opt = test_setup(0) 
    
    #test_load(1)
    #test_load(2)
    
    
