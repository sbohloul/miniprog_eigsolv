#!/bin/bash

mpicxx -Wall -std=c++11 \
    -I./ \
    -I/home/sbohloul/Projects/scicomp/miniprog_eigsolv/src/utils \
    -I/home/sbohloul/Projects/scicomp/miniprog_eigsolv/src/interfaces \
    /home/sbohloul/Projects/scicomp/miniprog_eigsolv/src/utils/timer.cpp \
    -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl