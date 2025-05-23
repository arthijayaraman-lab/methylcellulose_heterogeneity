from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import os
from bayes_opt.util import load_logs
import warnings


          
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
        logfile = f'bo_log_{i}.log'
        if dirname:
            logfile = os.path.join(dirname, logfile)
        if os.path.exists(logfile):
            load_logs(optimizer, logs = [logfile])
        else:
            warnings.warn(f'{logfile} does not exist. It is skipped.')
    logger = JSONLogger(path = f'./bo_log_{ix}.log', reset = False)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.set_gp_params(normalize_y = False, alpha = alpha )

    return optimizer


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
