#!/bin/bash

LIKWID_PATH=/home/sbohloul/.local/likwid/5.2.2
export PATH=$LIKWID_PATH/bin:$PATH
export LD_LIBRARY_PATH=$LIKWID_PATH/lib:$LD_LIBRARY_PATH
likwid-perfctr -C 0 -g FLOPS_DP -g FLOPS_AVX -g DATA -g MEM_DP -m ./kernel
