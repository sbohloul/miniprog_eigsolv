#!/bin/bash


# N=50

echo "N B T"

for N in $(seq 1000 1000 10000); do
    for blkSize in $(seq 4 4 40); do        
        T=$(./driver_matrix_add.x $N $N $blkSize $blkSize | grep duration | awk '{print $2}')
        echo $N $blkSize $T   
    done
    echo "-----------------"
done