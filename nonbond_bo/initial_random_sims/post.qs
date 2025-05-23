#!/bin/bash -l  

#reqs 1 arg is the batch number of the batch that is to be processed

#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --partition=_workgroup_  
# SBATCH --partition=standard  
#SBATCH --job-name=batch_post
#SBATCH --cpus-per-task=1  
#SBATCH --mem=1G  
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

###RUN BATCH POST###
python3 post_batch.py $1

mpi_rc=$?

exit $mpi_rc
