#!/bin/bash

export DATADIR=/scratch/sbohloul/benchminiapp/data

export M=4992
export FNAME=$DATADIR/Si_${M}_HR.h5
export NITER=1
export MB=128
export NB=128

export NNODE=1
export NP=48
export PPR=48
export NTHREADS=1
export RESULTSDIR=tvsgs-si-${M}-n$NNODE-np$NP-nt$NTHREADS

export OPENBLAS_NUM_THREADS=$NTHREADS
export OMP_NUM_THREADS=$NTHREADS

mpiexec -np $NP --map-by ppr:$PPR:node:pe=$NTHREADS --report-bindings python time_vs_gridshape.py $NITER $MB $NB $FNAME | tee log.out

mkdir $RESULTSDIR
lscpu | tee lscpu.log
mv *.svg *.csv *.out *.log $RESULTSDIR
