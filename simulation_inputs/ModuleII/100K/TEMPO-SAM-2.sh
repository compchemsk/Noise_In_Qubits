#!/bin/bash
#SBATCH -J TEMPO-SAM                    #Job name(--job-name)
#SBATCH -o %j-out-myjob          #Name of stdout output file(--output)
#SBATCH -e %j-err-myjob          #Name of stderr error file(--error)
#SBATCH -p medium        #Queue (--partition) name
#SBATCH --nodes=1                   #number of nodes should be 1 for serial)
#SBATCH -n 40                #(--cpus-per-task) Number of Threads
#SBATCH -A YOUR_ACCOUNT 
#SBATCH -t 72:00:00
##SBATCH --mem=23000        # Memory per node specification is in MB. It is optional.
##SBATCH --mail-user=mishra@chem.iitkgp.ac.in        # user email ID where job status info will be sent
##SBATCH --mail-type=ALL        # Send Mail for all type of event regarding the job
# Thread Reservation Section
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
             omp_threads=$SLURM_CPUS_PER_TASK
           else
             omp_threads=1
fi
# Load Application/Compiler/Library modules, for example
# module load apps/lammps/12.12.2018/intel
module load compiler/openmpi/4.0.1
export AMBERHOME=$HOME/amber18
export PATH=$PATH:$AMBERHOME/bin
export PATH=$PATH:$HOME/software/bison/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/software/bison/lib
export CUDA_HOME=/home/opt_ohpc_pub/apps/cuda/cuda-10.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib:$CUDA_HOME/lib64
# name of the executable
g16root="$HOME"
GAUSS_SCRDIR="/scratch/$USER/gaussian"
export g16root GAUSS_SCRDIR
. $g16root/g16/bsd/g16.profile
export OMP_NUM_THREADS=$omp_threads
# Executables along with right Parameters and right input files
#minimization
#mpirun -np 40 sander.MPI -O -i mini1.in -o mini1.out -p TEMPO-SAM-11.parm7 -c TEMPO-SAM-11.rst7 -r mini1.rst7 -ref TEMPO-SAM-11.rst7
#heating
#mpirun -np 40 sander.MPI -O -i heat1.in -o heat1.out -p TEMPO-SAM-11.parm7 -c mini1.rst7 -r heat1.rst7 -ref mini1.rst7 -x heat1.nc -inf heat1.mdinfo
#equilibration
mpirun -np 40 sander.MPI -O -i equil1-2.in -o equil1-2.out -p TEMPO-SAM-11.parm7 -c equil1.rst7 -r equil1-2.rst7 -ref equil1.rst7 -x equil1-2.nc -inf equil1-2.mdinfo
