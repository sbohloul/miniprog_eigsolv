#!/bin/bash

SRC_DIR=./CloverLeaf/CloverLeaf_MPI

export COMPILER=GNU
export MPI_COMPILER=mpif90
export C_MPI_COMPILER=mpicc
export OPTIONS="-g -O3 -ftree-vectorize -fopt-info"
export C_OPTIONS="-g -O3 -ftree-vectorize -fopt-info"

cd $SRC_DIR
make clean
make IEEE=1n COMPILER=$COMPILER OPTIONS="$OPTIONS" C_OPTIONS="$C_OPTIONS" MPI_COMPILER=$MPI_COMPILER C_MPI_COMPILER=$C_MPI_COMPILER