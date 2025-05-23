import subprocess
import numpy as np
import os
import shutil
import sys
from bis import BIS
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

def check_continue():
    with open('bound_params.json', 'r') as f:
        bound_params = json.load(f)
    bis = BIS(batch_no-1, bound_params) #bis from previous batch
    boundary = bis.get_boundary()
    bis.write_boundary_file(batch_no-1)
    
    bis.check_N_batches()
    

def get_pts(batch_no):
    with open('bound_params.json', 'r') as f:
        bound_params = json.load(f)
    bs = BIS(batch_no, bound_params)
    pts = bs.get_new_points(10) #change number of points

    return pts



def make_json():
    # with open('bound_params.json', 'r') as f:
    #     foo = json.load(f)
    paramnames = ['temp', 'ds', 'H', 'dp', 'nchains','conc']#list(foo.keys())

    
    json_dict = {k:v for k, v in zip(paramnames, params)}
    json_file = f'batch_{batch_no}/eng_{param_id}/params.json'
    with open(json_file, 'w') as json_file:
        json.dump(json_dict, json_file, indent=4)
        
def test():
    global pts
    pts = get_pts(1)
    
    
if __name__== '__main__':
    print('SETTING UP BATCH')
    batch_no = int(sys.argv[1]) #batch id that will run next
    
    if batch_no > 1:
        check_continue()
    
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
        dirname = f'batch_{batch_no}/eng_{param_id}'
        
        if os.path.exists(os.path.join('sims', dirname)) and overwrite_dir:
            shutil.rmtree(os.path.join('sims', dirname))
        
        params = [str(p) for p in params]
        
        subprocess.call(['python3', 'write_data_mc.py', str(batch_no), str(param_id), *params])
        
        equil_dir = dirname + '/equil'

        index_list.append(f'eng_{param_id}\n')
        make_json()

    shutil.copy('mdrun.qs', run_name)
    add_num_arraytasks(os.path.join(run_name, 'mdrun.qs'))
    
    index_file = f'{run_name}/index.txt'
    with open(index_file, 'w' ) as f:
        f.writelines(index_list)
    
        
    
