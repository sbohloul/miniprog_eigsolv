#!/bin/bash

export DATADIR=/scratch/sbohloul/benchminiapp/data

export FNAME=$DATADIR/Si_10400_HR.h5
export NITER=2
export MB=64
export NB=64
export NP=48
export NTHREADS=1
export RESULTSDIR=si-10400-np$NP-nt$NTHREADS

export OMP_NUM_THREADS=$NTHREADS
mpiexec -np $NP --report-bindings python time_vs_gridshape.py $NITER $MB $NB $FNAME | tee log.out

mkdir $RESULTSDIR
lscpu | tee lscpu.log
mv *.svg *.csv *.out *.log $RESULTSDIR



