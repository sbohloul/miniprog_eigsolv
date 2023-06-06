#!/bin/bash


N=10000

echo "N B T"
for blkSize in $(seq 4 4 512); do
    
    T=$(./driver_matrix_add.x $N $N $blkSize $blkSize | grep duration | awk '{print $2}')
    echo $N $blkSize $T   
done