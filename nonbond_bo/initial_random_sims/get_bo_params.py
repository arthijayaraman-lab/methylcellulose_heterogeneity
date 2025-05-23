import numpy as np
import os
import matplotlib.pyplot as plt
import json
from post_batch import read_dump
import warnings

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from scipy.interpolate import interp1d

from bayes_opt import BayesianOptimization
from bayes_opt import acquisition



warnings.filterwarnings("ignore", message = "invalid value encountered in cast")

nparam = 3

def read_log(ix):
    file = os.path.join(rundir, f'bo_log_{ix}.log')
    with open(file, 'r') as f:
        lines = f.read().splitlines()
    
    pts = []
    for line in lines:
        pts.append(json.loads(line))
    return pts

def pts_to_arr(dicts):
    pts = []
    for d in dicts:
        add = []
        add.append(d['target'])
        p = d['params']
        add.extend([x for x in p.values()])
        pts.append(add)
    return np.array(pts)

def get_subdirs(maindir, startstring):
    ret = []
    for d in os.listdir(maindir):
        if d.startswith(startstring):
            ret.append(d)
    return ret
    
def get_scores(rundir, batch, test = True):
    if test:
        scores = {}
        
        bdir = os.path.join(rundir, f'batch_{batch}')
        tests = get_subdirs(bdir, 'test')
        
        
        for test in tests:
            tdir = os.path.join(bdir, test)
            engs = get_subdirs(tdir, 'eng')
            for eng in engs:    
                fname = os.path.join(tdir, eng, 'score_arr.txt')
                if os.path.isfile(fname):
                    scores[int(eng.split('_')[-1])] = [np.loadtxt(fname), int(test.split('_')[-1])]
                else: scores[int(eng.split('_')[-1])] = [np.nan, int(test.split('_')[-1])]
    else:
        fname = os.path.join(rundir, f'batch_{batch}', f'eng_{eng}', 'score_arr.txt')
        if os.path.isfile(fname):
            scores.append(np.loadtxt(fname))
        else: scores = np.array([np.nan]*7)
    return scores

def get_param(rundir, batch):
    params = {}
    
    bdir = os.path.join(rundir, f'batch_{batch}')
    tests = get_subdirs(bdir, 'test')
    
    
    for test in tests:
        tdir = os.path.join(bdir, test)
        engs = get_subdirs(tdir, 'eng')
        for eng in engs:
            fname = os.path.join(tdir, eng, 'eng_params.json')
            if os.path.isfile(fname):
                with open(fname, 'r') as f:
                    data = json.load(f)
                p = np.array([float(x) for x in data.values()])
                params[int(eng.split('_')[-1])] = p
    #params = normalize_points(params)
    return params

def get_all_scores(rundir, batches):
    scores = []
    for batch in batches:
        s = get_scores(rundir, batch)
        scores.append(s)
    return scores

def get_all_params(rundir, batches):
    params = []
    for batch in batches:
        p = get_param(rundir, batch)
        params.append(p)
    return params

def get_ds(file):
    dump = read_dump(file, ret_dict = True)
    types = dump['types']
    ds = np.mean(types -1)
    return ds


def file_exists_in_directory(directory, filename):
    for root, _, files in os.walk(directory):
        if filename in files:
            return True  # File found
    return False  # File not found

def get_batches(rundir):
    batches = []
    for test_sims in os.listdir(rundir):
        if os.path.isdir(os.path.join(rundir, test_sims)) and test_sims.startswith('batch_'):
            check_files = file_exists_in_directory(os.path.join(rundir, test_sims), 'score_arr.txt')
            if check_files:
                batches.append(int(test_sims.split('_')[-1]))
    batches.sort()
    return batches


def process_sp(pts, scores):
    N = 70
    print(len(pts))
    pret = np.zeros((len(pts), N, nparam))
    sret = np.zeros((len(pts), N))
    tret = np.zeros((len(pts), N))

    for j,(p, s) in enumerate(zip(pts, scores, strict = True)):    
        ct = 0
        for i in range(N):
            if i not in p.keys(): continue
            pret[j,i]= p[i]
            sret[j,i] = s[i][0]
            tret[j,i] = s[i][1]
            ct+=1
            
    return pret, sret, tret
            
        
def svc_pred(X, y):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel="linear", probability=True)),
    ])
    param_grid = {
        'svc__gamma': ['scale']
    }
        
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X, y)
    
    best_model = grid_search.best_estimator_
    svc = best_model
    
    Ngrid = 40
    xl = np.linspace(scale_bounds[0,0], scale_bounds[0,1], Ngrid)
    yl = np.linspace(scale_bounds[1,0], scale_bounds[1,1], Ngrid)
    zl = np.linspace(scale_bounds[2,0], scale_bounds[2,1], Ngrid)

    xx, yy, zz = np.meshgrid(xl,yl, zl, indexing = 'ij')
    pts = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    Z = best_model.predict_proba(pts)
    Z = Z[:,1]
    Z = Z.reshape(xx.shape)
   
    return Z, xl, yl, zl, svc

test_params = {1: {'ds':1.8, 'hs': 1, 'temp':318},
             2: {'ds':1.8, 'hs': 1, 'temp':328},
             3: {'ds':1.8, 'hs': 0.05, 'temp':348},
             4: {'ds':2, 'hs': 1, 'temp':308},
             5: {'ds':2, 'hs': 1, 'temp':318},
             6: {'ds':2.4, 'hs': 1, 'temp':298},
             7: {'ds':2.4, 'hs': 1, 'temp':308}
             } 

class test_sims:
    def __init__(self, test, want_fibrils = True):
        self.test = test
        self.temp = test_params[test]['temp']
        self.h = test_params[test]['hs']
        self.ds = test_params[test]['ds']
        self.want_fibrils = want_fibrils
        
        mask = at == self.test
        scores = ac[mask]
        params = ap[mask]
        
        nanmask = ~np.isnan(scores)
        
        maskc = scores[nanmask].astype(bool)
        if not self.want_fibrils:
            maskc = ~maskc
        maskc = maskc.astype(int)
        
        eps22 =a22 =  params[nanmask, 0]
        eps33= a33 = params[nanmask, 1]
        frac =af =  params[nanmask, 2]
        
        X = np.vstack((eps22, eps33, frac)).T
        y = maskc.copy()
        
        true_centroid = np.mean(X[y.astype(bool)], axis = 0)
        false_centroid = np.mean(X[~y.astype(bool)], axis = 0)
        
        self.af = af
        self.a22 = a22
        self.a33 = a33
        self.true_centroid = true_centroid
        self.false_centroid = false_centroid
        
        self.colors = np.array(['r', 'b'])[maskc]
        self.tc = true_centroid
        self.min = np.min(X[y.astype(bool)], axis = 0)
        self.max = np.max(X[y.astype(bool)], axis = 0)    

        self.X = X
        self.y = y

        Z, xl, yl, zl, svc = svc_pred(X, y)
        self.svc = svc
    
    def plot(self, fact = 1, centroid = False):
        ax = plt.figure().add_subplot(projection = '3d')
        ax.scatter(np.array(self.a22)* fact, np.array(self.a33)* fact, np.array(self.af)* fact, c = self.colors, s = 3)
        if centroid:
            ax.scatter(*(np.array(self.true_centroid)* fact), c = 'b')
            ax.scatter(*(self.false_centroid* fact), c = 'r')
        ax.set_xlabel(r'$\varepsilon_{22}$')
        ax.set_ylabel(r'$\varepsilon_{33}$')
        ax.set_zlabel('f')
        ax.set_title(f'test_{self.test}')
       
        self.ax = ax
        return ax

def unscale(series, low, high):
    ran = high - low
    norm = series * ran
    norm += low
    return norm

def scale(series, low, high):
    series = np.array(series)
    low = np.array(low)
    high = np.array(high)
    num = series - low
    den = high - low
    return num/den

def temp_interp(temp, high, low, n, stretch ):
    temps = np.array([298, 348])

    scaled_temp = st = (temp - temps[0])/(temps[1] - temps[0])
    y = (-(1-st/stretch)**(n)+ 1)/(-(1-1/stretch)**(n)+ 1)
    y = unscale(y, low, high)
    
    ret = y
    return ret

def loss_fn(test_sims, two, thr, f, curve):
    probs = test_sims.svc.predict_proba([[two, thr, f]])
    return [int(probs[0][1] > 0.5) + int(probs[0][1] < 0.5)* probs[0][1]]  
    
def get_curve(ntemps = 1000, func = temp_interp, ret_interp = False, **kwargs):
    temps = temps = np.linspace(298, 348, ntemps)
    low_22 = kwargs.get('22_300')
    high_22 = kwargs.get('22_348')
    low_33 = kwargs.get('33_300')
    high_33 = kwargs.get('33_348')
    low_frac = kwargs.get('frac_300')
    high_frac = kwargs.get('frac_348')
    
    
    if sep_scales:
        exp22 = kwargs.get('exp_22')
        stretch22 = kwargs.get('stretch_22')
        exp33 = kwargs.get('exp_33')
        stretch33 = kwargs.get('stretch_33')
        expf = kwargs.get('exp_f')
        stretchf = kwargs.get('stretch_f')
    else:
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

def bo_func(func = temp_interp, plot = False, ret_pts = False, **kwargs):
    global NNs
    low_22 = kwargs.get('22_300')
    high_22 = kwargs.get('22_348')
    low_33 = kwargs.get('33_300')
    high_33 = kwargs.get('33_348')
    low_frac = kwargs.get('frac_300')
    high_frac = kwargs.get('frac_348')
    
    if low_22 > high_22:
        return 1
    
    if low_33 > high_33:
        return 1
    
    if low_frac > high_frac:
        return 1
    
    curve = get_curve(100, **kwargs)
    
    if sep_scales:
        exp22 = kwargs.get('exp_22')
        stretch22 = kwargs.get('stretch_22')
        exp33 = kwargs.get('exp_33')
        stretch33 = kwargs.get('stretch_33')
        expf = kwargs.get('exp_f')
        stretchf = kwargs.get('stretch_f')
    else:
        exp = kwargs.get('exp')
        stretch = kwargs.get('stretch')
        exp22= exp33 = expf = exp
        stretch22 = stretch33 = stretchf = stretch
        
    NNs = []
    pts = []
    temps = []
    for test_sims in bo_test_simss:
        temp = test_sims.temp
        temps.append(temp)
        two = func(temp, high_22, low_22, exp22, stretch22)
        thr = func(temp, high_33, low_33, exp33, stretch33)
        f = func(temp, high_frac, low_frac, expf, stretchf)
        pts.append([two, thr, f])
        
        loss = loss_fn(test_sims, two, thr, f, curve)
        NNs.append(loss)
        
    NNs = np.array(NNs)
    pts = np.array(pts)

    if ret_pts: return pts
    
    if plot:
        plt.figure()
        tempsx = np.linspace(298, 348, 100)
        two = func(tempsx, high_22, low_22, exp22, stretch22)
        thr = func(tempsx, high_33, low_33, exp33, stretch33)
        f = func(tempsx, high_frac, low_frac, expf, stretchf)
        plt.plot(tempsx, two, label = '22', color = 'r')
        plt.plot(tempsx, thr, label = '33', color = 'b')
        plt.plot(tempsx, f, label = 'frac', color = 'k')
        
        colors = ['r', 'b', 'k']
        for i in range(3):
            tcs = np.array([a.tc for a in bo_test_simss])
            mins = np.array([a.min for a in bo_test_simss])
            maxs = np.array([a.max for a in bo_test_simss])
            
            le = tcs[:,i] - mins[:,i]
            ue = maxs[:,i]- tcs[:,i]
            plt.errorbar(temps, tcs[:,i], yerr = (le, ue), ls = 'none', marker = 'o', capsize = 10, c = colors[i])
            
        plt.legend()
    NNs = np.array(NNs)
    return np.sum(NNs)

def plot_bo(bp):    
    a = bo_func(plot = True, **bp)
    pts = bo_func(ret_pts = True, **bp)
    
    curve_t = np.linspace(298, 348, 1000)
    curve = get_curve(temps= curve_t, **bp)
    
    
    for i,(test_sims, pt) in enumerate(zip(bo_test_simss, pts)):
        ax = test_sims.plot()
        ax.scatter(*pt, color = 'k')
        ax.scatter(curve[:,0], curve[:,1], curve[:,2], c = np.linspace(0, 1, len(curve)), cmap = 'coolwarm',s = 0.1, alpha = 0.3)

def do_bo(bounds, test_simss, plot_results =True, rs = None):
    if not rs:
        rs = 1
    acq = acquisition.UpperConfidenceBound(kappa = 1)
    
    optimizer = BayesianOptimization(
                f=bo_func,
                acquisition_function = acq,
                pbounds=bounds,
                verbose=0,
                random_state=rs,
                allow_duplicate_points = True
            )
    
    #init
    optimizer.maximize(init_points = 100, n_iter = 0)
    
    TARGET_SCORE = 6.25
    for i in range(200):  # Up to 200 iterations
        if i%10 == 0: 
            print(i)
        optimizer.maximize(init_points=0, n_iter=1)
        if optimizer.max['target'] >= TARGET_SCORE:
            print(f"Stopping optimization: target {TARGET_SCORE} reached")
            break
        
    m = optimizer.max
    bp = optimizer.max['params']
    
    if plot_results:
        plot_bo(bp)

    return optimizer, m 

class sim_data:
    def __init__(self, rundir):
        batches = get_batches(rundir)    
        all_pts = get_all_params(rundir, batches)
        scores = get_all_scores(rundir, batches)

        pts, scores, tests = process_sp(all_pts, scores)
        
        ap = pts.reshape(-1, nparam)
        ac = scores.ravel()
        at = tests.ravel().astype(int)

        self.batches = batches
        self.all_pts = all_pts
        self.scores = scores
        
        mask = ~np.isnan(ac)
        
        self.ap = ap[mask]
        self.ac = ac[mask].astype(int)
        self.at = at[mask]
    
    
#%%
if __name__ == '__main__':  
    sav = False
    rundir = 'sims/init_data'
    data = sim_data(rundir)
    all_tests = at = data.at
    all_scores = ac = data.ac
    all_params = ap = data.ap
    
    pbounds_dict=pbd=bounds = {
              '22_300': (0.10, 0.2),
             '33_300':(0.2, 0.4),
             'frac_300': (0.05,1),
             '22_348': (0.10, 0.2),
            '33_348':(0.3, 0.5),
            'frac_348': (0.2,1),
            'exp': (1, 15),
            'stretch': (1,5)
                 }
    
    bounds = pbd
    
    scale_bounds = np.array([[pbd['22_300'][0], pbd['22_348'][1]],
                    [pbd['33_300'][0], pbd['33_348'][1]],
                    [pbd['frac_300'][0], pbd['frac_348'][1]],
                    ])
    
    sep_scales = False
    
    if sep_scales:
        bounds.update({
            'exp_22': (1, 10),
            'stretch_22': (1,2),
            'exp_33': (1, 10),
            'stretch_33': (1,2),
            'exp_f': (1, 10),
            'stretch_f': (1,2),})
        bounds.pop('exp', None)
        bounds.pop('stretch', None)
    
    
    bounds = {key: value for key, value in sorted(bounds.items())}
    
    want_fibrils = wfs = np.array([0,1,0,0,1,0,1]).astype(bool)
    tests = np.array([1, 2, 3, 4, 5, 6, 7]) # use 50C as gel pt for ds 1.8    

    bo_test_simss = [test_sims(test, want_fibrils = wf) for test, wf in zip(tests, wfs, strict = True)]
    
    max_params = []
    maxims = []
    for j in range(50):
        print(j)
        br = 0   

        while br < 6.25:
            rs =  int(np.random.rand()*10000000)
            bo, maxim = do_bo(bounds, bo_test_simss, rs = rs, plot_results = False )
            br = maxim['target']
            print(f'seed: {rs}, best_result : {br}')
        
        max_params.append(list(maxim['params'].values()))
        maxims.append(maxim)
    max_params = np.array(max_params)
    
    plot_bo(maxim['params'])
    
    if sav: 
        np.savetxt('max_params.txt', max_params)


