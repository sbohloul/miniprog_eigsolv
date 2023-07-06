#!/bin/bash

# SRC_DIR=./CloverLeaf/CloverLeaf_Serial
SRC_DIR=./CloverLeaf/CloverLeaf_Serial

COMPILER=GNU
OPTIONS="-g -fno-tree-vectorize -fopt-info"
C_OPTIONS="-g -fno-tree-vectorize -fopt-info"

cd $SRC_DIR
make clean
make COMPILER=$COMPILER OPTIONS="$OPTIONS" C_OPTIONS="$C_OPTIONS"