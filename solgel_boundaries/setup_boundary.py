import os
import shutil
import json

def write_jsons():
    json_file = os.path.join(run_name, 'bound_params.json')
    with open(json_file, 'w') as json_file:
        json.dump(boundary_params, json_file, indent=4)
    
    json_file = os.path.join(run_name, 'chain_params.json')
    with open(json_file, 'w') as json_file:
        json.dump(chain_design_params, json_file, indent=4)
        
    json_file = os.path.join(run_name, 'eng_params.json')
    with open(json_file, 'w') as json_file:
        json.dump(eng_params, json_file, indent=4)
    
def make_bashrun():
    fname = os.path.join(run_name, 'bashrun.sh')
    lines = ['#!bin/bash',
             '',
             'function get_jobdep() {',
             'if [[ "$1" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then',
             '    echo "${BASH_REMATCH[1]}"',
             '    exit 0',
             'else',
             '    echo "submission failed"',
             '    exit 1',
             'fi',
             '}',
             '',
             '#Run first batch',
             'sb="$(sbatch setup.qs 1)"',
             'dep="$(get_jobdep "$sb")"',
             'sb="$(sbatch -D batch_1 --dependency=afterok:$dep --kill-on-invalid-dep=yes mdrun.qs)"',
             'dep="$(get_jobdep "$sb")"',
             'sb="$(sbatch --dependency=afterany:$dep --kill-on-invalid-dep=yes post.qs 1)"',
             'dep="$(get_jobdep "$sb")"',
             '',
             '#run other batches',
             'for i in {2..' + str(n_batches) + '}',
             'do',
             '    sb="$(sbatch --dependency=afterok:$dep --kill-on-invalid-dep=yes setup.qs $i)"',
             '    dep="$(get_jobdep "$sb")"',
             '    sb="$(sbatch -D "batch_$i" --dependency=afterok:$dep --kill-on-invalid-dep=yes mdrun.qs)"',
             '    dep="$(get_jobdep "$sb")"',
             '    sb="$(sbatch --dependency=afterany:$dep --kill-on-invalid-dep=yes post.qs $i)"',
             '    dep="$(get_jobdep "$sb")"',
             'done',
             
             
             ]
    lines = [l + '\n' for l in lines]
    with open(fname, 'w') as f:
        f.writelines(lines)

run_name = 'sims/temp_ds_Hs1.0'
n_batches = 20
batch_size = 10
number_random_batches = 3

#two variables defining the exploration space
#because of the sloppy way I coded this, H needs to come first/second and Hs or Hc need to be last.
boundary_params = {
    #'H' : [1,7],
    'temp': [298, 348],
    'ds': [0, 3],
    #'Hs': [1, 7],
    #'conc': [0.01, 0.04]
    }

#constant params
chain_design_params = {
    #'ds': 2.4, 
    'dp': 100 ,
    'nchains': 100,
    'conc': 0.02,
    'H':1,
    'Hs':1
    }

#nonbonded params
eng_params = {
    '00_298': 0.05,
    '00_348': 0.05,
    '11_298': 0.075,
    '11_348': 0.075,
    '22_298': 0.13094258,
    '22_348': 0.17041429,
    '33_298': 0.24950568,
    '33_348': 0.47670943,
    'exp': 8.74973208,
    'frac_298': 0.21287682,
    'frac_348': 0.94973001,
    'stretch': 2.75170937
    }


try: shutil.rmtree(run_name)
except:pass

try: os.makedirs(run_name)
except:pass

main_files = ['bis.py',
              'setup_series.py',
              'post_batch.py',
              'write_data_mc.py',
              'setup.qs',
              'mdrun.qs',
              'post.qs',
              'dobash.sh',
              '../heterogeneities/mc_markov_chain_heterogeneity.py',
              'mc_equil.in',
              'mc_vary_epshp.in',
              ]

for file in main_files:
    if os.path.isfile(file):
        shutil.copy(file, run_name)
    elif os.path.isdir(file):
        shutil.copytree(file, os.path.join(run_name, file))

def file_replace(filename, replace_strings, replace_vals):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    
    finlines = []
        
    for line in lines:
        for string, val in zip(replace_strings, replace_vals, strict = True):
            if string in line:
                #print(string, line)
                line = line.replace(string, val)
                #print(line)
        finlines.append(line + '\n')
        
    with open(filename, 'w') as f:
        f.writelines(finlines)

make_bashrun()

write_jsons()


file_replace(os.path.join(run_name, 'bis.py'),
             ['if self.batch_id < 5:'],
             [f'if self.batch_id < {number_random_batches}:'])
    
file_replace(os.path.join(run_name, 'setup_series.py'),
             ['pts = bs.get_new_points(10) #change number of points'],
             [f'pts = bs.get_new_points({batch_size}) #change number of points'])


