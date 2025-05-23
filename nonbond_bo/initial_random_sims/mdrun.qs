#!/bin/bash -l  

# SBATCH --nodes=1 
#SBATCH --ntasks=16
# SBATCH --partition=_workgroup_  
#SBATCH --partition=standard  
#SBATCH --job-name=mc_beadspring.job #
#SBATCH --cpus-per-task=1  
#SBATCH --mem=1G  
#SBATCH --time=0-03:00:00
# SBATCH --time-min=0-00:10:00  #Do this
# SBATCH --output=%x-%j.out  
# SBATCH --error=%x-%j.out  
#SBATCH --mail-user='skronen@udel.edu'  
#SBATCH --mail-type=END,FAIL  
#SBATCH --export=NONE  
#UD_QUIET_JOB_SETUP=YES  
#UD_MACHINE_FILE_FORMAT='%h%[:]C'  
#export UD_JOB_EXIT_FN_SIGNALS="SIGTERM EXIT"  
#SBATCH --requeue
#SBATCH --array=1-70


FOLDER_FILE='./index.txt'
FOLDER=$(sed -n "$SLURM_ARRAY_TASK_ID p" "$FOLDER_FILE")
cd "$FOLDER"

###VALET### 
vpkg_require openmpi
. /opt/shared/slurm/templates/libexec/openmpi.sh  

###INPUT PARAMETERS###
lmpfile="mc_equil.in"
rs_file="./equil.rs.*"

###RUN###
if compgen -G "equil/post_equil.rs" > /dev/null; then
	echo "skipping equil, already completed"
else 
	cd equil
	mpirun lmp_2024 -in $lmpfile
	mpi_rc=$?  

	if [ $mpi_rc -eq "0" ]; #if run finishes, 
	then rm $rs_file
	else exit 1
	fi
	cd ..
fi

###INPUT PARAMETERS###
lmpfile="mc_vary_epshp.in"
rs_file="./prod.rs.*"

###DETERMINE IF RESTARTING### 
if compgen -G "vary_epshp/prod.rs.*" > /dev/null; then
	rsval=1 #if restart exists, set restart value to 1
else
	rsval=0
fi
 
###RUN### 

cd vary_epshp
mpirun lmp_2024 -in $lmpfile -v restarting $rsval
mpi_rc=$?  

if [ $mpi_rc -eq "0" ]; #if run finishes, 
then rm $rs_file
else exit 1
fi 

cd ..

exit 0 

