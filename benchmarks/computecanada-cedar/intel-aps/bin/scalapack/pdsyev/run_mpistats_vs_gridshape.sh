#!/bin/bash

# Input data and blkc parameters
export DATADIR=/scratch/sbohloul/benchminiapp/data

export M=10400
export FNAME=$DATADIR/Si_${M}_HR.h5
export MB=128
export NB=128

# mpi launch parameters
export NNODE=4
export PPR=48
export NP=$(($NNODE * $PPR))
export NTHREADS=1

# results directory
export WORKDIR=wrkdir-si-${M}-n$NNODE-np$NP-nt$NTHREADS
export RESULTSDIR=mpistatvsgs-si-${M}-n$NNODE-np$NP-nt$NTHREADS
export APSRESULTSDIR=aps_result

# 
export OMP_NUM_THREADS=$NTHREADS
export MKL_NUM_THREADS=$NTHREADS

# create workdir
if [ -d $WORKDIR ]; then
    echo "Removing $WORKDIR"
    rm -rf $WORKDIR
fi
mkdir $WORKDIR && cd $WORKDIR
mpiexec -n $NP --ppn $PPR -print-rank-map \
    -genv I_MPI_PIN_DOMAIN=auto \
    -genv I_MPI_PIN_ORDER=scatter \
    aps -r ${APSRESULTSDIR} -L 4 -I 2 \
    python ../mpistats_vs_gridshape.py $MB $NB $FNAME | tee log.out

python ../mpistats_vs_gridshape_gen_aps_report.py ./
lscpu | tee lscpu.out
cd ..
mv $WORKDIR $RESULTSDIR
