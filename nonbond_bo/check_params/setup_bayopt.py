import os
from bayopt_util import setup
import shutil

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

run_name = 'sims/check_bo_params'
n_batches = 5
loss = 'discrete'


try: shutil.rmtree(run_name)
except:pass

try: os.makedirs(run_name)
except:pass

main_files = ['bayopt_util.py',
              'setup_series.py',
              'post_batch.py',
              'write_data_mc.py',
              'setup.qs',
              'mdrun.qs',
              'post.qs',
              'dobash.sh',
              'mc_markov_chain_heterogeneity.py',
              'mc_equil.in',
              'mc_vary_epshp.in',
              'mcequil_sub.qs',
              'mchp_sub.qs',
              'bashsub.sh',
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
                line = line.replace(string, val)
        finlines.append(line + '\n')
        
    with open(filename, 'w') as f:
        f.writelines(finlines)


make_bashrun()
cwd = os.getcwd()
os.chdir(run_name)
bo = setup(0)#, init_points = init_points)
os.chdir(cwd)

if loss == 'continuous':
    post_batch_file = os.path.join(run_name, 'post_batch.py')
    file_replace(post_batch_file, ["def get_score(dirname, loss = 'discrete'):"], ["def get_score(dirname, loss = 'continuous'):"])
    
    



