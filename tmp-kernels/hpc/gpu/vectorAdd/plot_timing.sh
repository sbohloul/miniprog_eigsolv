#!/bin/bash

cat > time.dat << EOF
EOF
for block_size in $(seq 16 16 5000); do
    output=$(./vectorAdd.cu.x $block_size | cut -d" " -f2 | head -n 2 | tr '\n' ' ')
    echo $output | tee -a time.dat
done

# python plot_timing.py &