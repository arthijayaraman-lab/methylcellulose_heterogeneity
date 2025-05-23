import subprocess
import numpy as np
import os
import shutil
import sys
from bayopt_util import load_bo, probe_pts, unnormalize_points, pbounds, sorted_params
import json

overwrite_dir = True

def add_num_arraytasks(array_script):
    with open(array_script, 'r') as f:
        lines = f.read().splitlines()
    
    with open(array_script, 'w') as f:
        for line in lines:
            if line == '#SBATCH --array=1-N_ARRAY_TASKS_REPLACE':
                line = line.replace('N_ARRAY_TASKS_REPLACE', str(len(index_list)))
            
            f.write(line + '\n')

def get_pts(batch_no):
    max_params = np.loadtxt('max_params.txt')
    ixs = np.array(range(50)) #np.array([4,6,7,8,9,14,15,16,17,18,19,24,26,28,30,31,34,36,37,38,39,40,43])
    ixs = ixs[(batch_no-1)*10: (batch_no)*10]
    unnormed_pts = max_params[ixs]
    print(unnormed_pts.shape)

    return unnormed_pts

def make_json():
    json_dict = {k:v for k, v in zip(sorted_params, params)}
    json_file = f'batch_{batch_no}/eng_{param_id}/params.json'
    with open(json_file, 'w') as json_file:
        json.dump(json_dict, json_file, indent=4)
        
def test():
    global pts
    pts = get_pts(1)
    
    
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
    
    batch_params = get_pts(batch_no)
    for param_id, params in enumerate(batch_params):
        for test_id in range(1, 8):
            dirname = f'batch_{batch_no}/eng_{param_id}/test_{test_id}'
            
            if os.path.exists(os.path.join('sims', dirname)) and overwrite_dir:
                shutil.rmtree(os.path.join('sims', dirname))
            
            params = [str(p) for p in params]
            subprocess.call(['python3', 'write_data_mc.py', str(batch_no), str(param_id), str(test_id), *params])
            
            equil_dir = dirname + '/equil'
            #shutil.copy(f'global_equils/test_{test_id}/post_equil.rs', equil_dir)
            #shutil.copy(f'global_datas/test_{test_id}/post_equil.data', equil_dir)

            index_list.append(f'eng_{param_id}/test_{test_id}\n')
        make_json()

    shutil.copy('mdrun.qs', run_name)
    add_num_arraytasks(os.path.join(run_name, 'mdrun.qs'))
    
    index_file = f'{run_name}/index.txt'
    with open(index_file, 'w' ) as f:
        f.writelines(index_list)
    
        
    
