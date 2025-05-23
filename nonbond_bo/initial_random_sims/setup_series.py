import subprocess
import numpy as np
import os
import shutil
import sys
import json

overwrite_dir = True

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

def add_num_arraytasks(array_script):
    with open(array_script, 'r') as f:
        lines = f.read().splitlines()
    
    with open(array_script, 'w') as f:
        for line in lines:
            if line == '#SBATCH --array=1-N_ARRAY_TASKS_REPLACE':
                line = line.replace('N_ARRAY_TASKS_REPLACE', str(len(index_list)))
            
            f.write(line + '\n')

def get_pts(batch_no):
    bounds = np.array([(0.1, 0.3), (0.2, 0.4), (0.05, 1)])
    r = bounds[:,1] - bounds[:,0]
    m = bounds[:,0]
    pts = np.random.rand(70, 3) * r + m
    
    test = np.random.choice(range(1, 15), 70)
    return pts, test

def make_json():
    json_dict = {k:v for k, v in zip(sorted_params, params)}
    print(json_dict)
    json_file = f'batch_{batch_no}/test_{test}/eng_{param_id}/eng_params.json'
    with open(json_file, 'w') as json_file:
        json.dump(json_dict, json_file, indent=4)
    
    
if __name__== '__main__':
    print('SETTING UP BATCH')
    batch_no = int(sys.argv[1]) #batch id that will run next
    
    run_name = f'batch_{batch_no}'
    try: 
        shutil.rmtree(run_name)
    except:pass
    
    try:os.makedirs(run_name)
    except:pass
    
    index_list = []
    dirnames = []
    ct = 1
    
    pts, tests = get_pts(batch_no)
    
    for param_id,(pt, test) in enumerate(zip(pts, tests)):
        dirname = f'batch_{batch_no}/test_{test}/eng_{param_id}'
            
        if os.path.exists(os.path.join('sims', dirname)) and overwrite_dir:
            shutil.rmtree(os.path.join('sims', dirname))
            
        params = [str(p) for p in pt]
        test = str(test)
        subprocess.call(['python3', 'write_data_mc.py', str(batch_no), str(param_id), str(test), *params])
            
        equil_dir = dirname + '/equil'

        index_list.append(f'test_{test}/eng_{param_id}\n')
        make_json()

    shutil.copy('mdrun.qs', run_name)
    add_num_arraytasks(os.path.join(run_name, 'mdrun.qs'))
    
    index_file = f'{run_name}/index.txt'
    with open(index_file, 'w' ) as f:
        f.writelines(index_list)
    
        
    
