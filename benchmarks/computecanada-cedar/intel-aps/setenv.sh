#!/bin/bash

script_dir=$(dirname $(realpath ${BASH_SOURCE[0]}))

source ~/Projects/miniprog_eigsolv/setenv-intel.sh
source ~/Projects/miniprog_eigsolv/venv-intel/bin/activate
export PYTHONPATH=${script_dir}/bin:$PYTHONPATH

module load vtune/2022.2
export LD_PRELOAD=$(find $I_MPI_ROOT -name "libmpi.so" | head -n 1)