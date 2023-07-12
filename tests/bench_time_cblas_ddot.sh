#!/bin/bash

NITER=100
STEPSIZE=10000
MAXSIZE=1000000
PROG=./driver_time_cblas_ddot.x
FNAME=driver_time_cblas_ddot.txt

cat > $FNAME << EOF
EOF

printf "niter \t nelem \t time(s)\n" | tee $FNAME
for NELEM in $(seq $STEPSIZE $STEPSIZE $MAXSIZE); do
    TIME=$($PROG $NITER $NELEM | awk '{print $3}')
    printf "%s \t %s \t %s\n" $NITER $NELEM $TIME | tee -a $FNAME
done

PYSCRIPT="
import numpy as np
import matplotlib.pyplot as plt
fname = \"$FNAME\"
print(fname)
data = np.loadtxt(fname, skiprows=1)
x = data[:, 1]
y = data[:, 2]
plt.semilogy(x, y, '-o')
plt.xlabel('nelem')
plt.ylabel('time(s)')
plt.grid(True)
plt.show()
"

python -c "$PYSCRIPT"