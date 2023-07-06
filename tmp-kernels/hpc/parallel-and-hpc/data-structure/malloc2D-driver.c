#include <stdio.h>
#include <stdlib.h>
#include <malloc2D.h>

int main()
{
    int i, j;
    int jmax = 5;
    int imax = 3;

    double **x = (double **)malloc2D(jmax, imax);

    for (j = 0; j < jmax; j++)
    {
        printf("%p\n", x[j]);
    }
    for (i = 0; i < imax * jmax; i++)
    {
        printf("%f\n", x[0][i]);
    }

    free(x);
    return 0;
}