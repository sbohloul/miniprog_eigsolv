module --force purge
module load StdEnv/2020
module load gcc/11.3.0
module load openmpi/4.1.4
module load mpi4py/3.1.4
module load python/3.11.2
module load cmake/3.23.1


export LD_LIBRARY_PATH=/home/sbohloul/.local/scalapack/2.2.0-o2/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/sbohloul/.local/openblas/0.3.21-o2/lib64:$LD_LIBRARY_PATH

export OPENBLAS_NUM_THREADS=1