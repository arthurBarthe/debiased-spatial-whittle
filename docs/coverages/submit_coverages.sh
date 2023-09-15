#!/bin/bash

# Run this as "qsub {this_file_name}.sh"

# Set a name for this run and the resource requirements,
# 1 CPU, 5 GB memory and 5 minutes wall time.
# ncpus is number of cores/threads!
#PBS -N coverages 
#PBS -l ncpus=100
#PBS -l mem=10GB
#PBS -l walltime=00:15:00

# Send an email when this job aborts, begins or ends.
#PBS -m abe 
#PBS -M thomas.goodwin@uts.edu.au


### Problems running? OR ###
### Writing to large files using /SCRATCH? ###
### Look at docs typical_submit.sh! ###

# Run your program.
cd ${PBS_O_WORKDIR}
source $HOME/virtualenvs/dewhittle/bin/activate
./bayesian_coverages_hpc.py
deactivate


