#!/bin/bash -l  

#reqs 1 arg that is the batch number that will be run after this setup

#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --partition=_workgroup_  
# SBATCH --partition=standard  
#SBATCH --job-name=batch_setup
#SBATCH --cpus-per-task=1  
#SBATCH --mem=4G  
#SBATCH --time=0-00:15:00   #Do this 
# SBATCH --time-min=0-00:30:00  #Do this
# SBATCH --output=%x-%j.out  
# SBATCH --error=%x-%j.out  
#SBATCH --mail-user='skronen@udel.edu'  
#SBATCH --mail-type=END,FAIL  
#SBATCH --export=NONE  
#SBATCH --requeue
#UD_QUIET_JOB_SETUP=YES  
#UD_MACHINE_FILE_FORMAT='%h%[:]C'  
#export UD_JOB_EXIT_FN_SIGNALS="SIGTERM EXIT"  


###VALET### 
vpkg_require python/3.7.4  
conda activate bo_env

###RUN BATCH SETUP###
python3 setup_series.py $1

mpi_rc=$?

exit $mpi_rc
